"""Folding a finished chat into what we know about someone.

The tests that matter here are the two that protect against unrecoverable
damage: a merge that writes into the wrong person's memory, and a merge that
empties an existing one. Everything a user has ever told this product
accumulates in these rows, and neither failure announces itself.
"""

import json

import pytest

from app.core.config import MEMORY_MAX_FACTS
from app.services.context import ChatContext
from app.services.memory import MemoryOwnershipError, merge_from_chat
from tests.conftest import unwrapped

OWNER = "user-1"
STRANGER = "user-2"

CHAT = {
    "id": "chat-1",
    "user_id": OWNER,
    "category": "financial",
    "created_at": "2026-03-14T09:00:00+00:00",
}

SAID = [
    {
        "type": "free_text",
        "answer_text": "I send 12k home every month for my brother Ravi's school fees.",
        "metadata": {},
    }
]


def context_with(messages=SAID, chat=None):
    return ChatContext(chat={**CHAT, **(chat or {})}, profile={}, messages=messages)


def reply(global_facts=(), topic_facts=(), global_summary="", topic_summary=""):
    return json.dumps(
        {
            "global": {"summary": global_summary, "facts": list(global_facts)},
            "topic": {"summary": topic_summary, "facts": list(topic_facts)},
        }
    )


class TestOwnership:
    def test_a_chat_is_never_folded_into_someone_elses_memory(self, model, db):
        """The reason merge_from_chat takes user_id at all.

        user_memory is scoped to a person, not a chat. Getting this wrong would
        not leak one conversation — it would graft a stranger's life onto
        someone's permanent record, and every future chat would be answered out
        of it.
        """
        with pytest.raises(MemoryOwnershipError):
            merge_from_chat(context_with(), user_id=STRANGER)

        # Nothing was generated and nothing was written.
        assert model.calls == []
        assert db.memory == []
        assert db.merged == []

    def test_the_owner_gets_through(self, model, db):
        model.replies = [reply(global_facts=["Sends 12k home monthly."])]

        assert merge_from_chat(context_with(), user_id=OWNER) is True
        assert db.memory[0]["user_id"] == OWNER


class TestTheMerge:
    def test_facts_are_split_between_global_and_the_chats_category(self, model, db):
        model.replies = [
            reply(
                global_facts=["Has a younger brother, Ravi, in school."],
                topic_facts=["Sends 12k home every month for Ravi's fees."],
                global_summary="Supports family.",
            )
        ]

        merge_from_chat(context_with(), user_id=OWNER)

        rows = {row["category"]: row for row in db.memory}
        assert rows[None]["facts"] == ["Has a younger brother, Ravi, in school."]
        assert rows[None]["summary"] == "Supports family."
        assert rows["financial"]["facts"] == [
            "Sends 12k home every month for Ravi's fees."
        ]

    def test_what_we_already_knew_is_given_to_the_model(self, model, db):
        db.remember(OWNER, None, facts=["Lives in Pune with her mother."])
        db.remember(OWNER, "financial", facts=["Earns 32k a month."])
        model.replies = [reply(global_facts=["Lives in Pune with her mother."])]

        merge_from_chat(context_with(), user_id=OWNER)

        # Without this the merge is an overwrite wearing a merge's name.
        asked = model.calls[0]["user"]
        assert "Lives in Pune with her mother." in asked
        assert "Earns 32k a month." in asked

    def test_a_fact_can_be_corrected_rather_than_contradicted(self, model, db):
        db.remember(OWNER, None, facts=["Studying at Pune University."])
        model.replies = [reply(global_facts=["Dropped out of Pune University in June."])]

        merge_from_chat(context_with(), user_id=OWNER)

        # The whole reason the model returns the full memory rather than a list
        # of additions: the stale fact is gone, not sitting next to its
        # replacement disagreeing with it.
        assert db.memory[0]["facts"] == ["Dropped out of Pune University in June."]

    def test_other_categories_are_left_alone(self, model, db):
        db.remember(OWNER, "relationship", facts=["Engaged to Meera."])
        model.replies = [reply(topic_facts=["Saving for a deposit."])]

        merge_from_chat(context_with(), user_id=OWNER)

        rows = {row["category"]: row for row in db.memory}
        assert rows["relationship"]["facts"] == ["Engaged to Meera."]

    def test_the_chat_is_marked_merged_so_it_is_not_folded_in_twice(self, model, db):
        model.replies = [reply(global_facts=["Something durable."])]

        merge_from_chat(context_with(), user_id=OWNER)

        assert db.merged == ["chat-1"]

    def test_a_chat_with_nothing_in_it_is_not_remembered(self, model, db):
        assert merge_from_chat(context_with(messages=[]), user_id=OWNER) is False
        assert model.calls == []
        assert db.memory == []

    def test_duplicate_facts_are_collapsed(self, model, db):
        model.replies = [
            reply(global_facts=["Sends money home.", "sends money home", "Sends money home."])
        ]

        merge_from_chat(context_with(), user_id=OWNER)

        assert db.memory[0]["facts"] == ["Sends money home."]

    def test_memory_cannot_grow_without_limit(self, model, db):
        model.replies = [reply(global_facts=[f"Fact number {i}." for i in range(100)])]

        merge_from_chat(context_with(), user_id=OWNER)

        assert len(db.memory[0]["facts"]) == MEMORY_MAX_FACTS


class TestNotLosingMemory:
    def test_an_empty_merge_does_not_erase_what_we_knew(self, model, db):
        """The failure this guard exists for.

        The model is asked to return everything it still believes, so a
        generation that comes back empty is indistinguishable from one that
        lost the lot. Keeping stale memory costs one chat's learning. Accepting
        the empty one silently erases every chat before it.
        """
        db.remember(OWNER, None, summary="Supports her family.", facts=["Sends 12k home."])
        model.replies = [reply()]

        merge_from_chat(context_with(), user_id=OWNER)

        assert db.memory[0]["facts"] == ["Sends 12k home."]
        assert db.memory[0]["summary"] == "Supports her family."
        # Not marked merged: completion.py's _wrap_up only retries a chat
        # whose memory_merged_at is unset, and this generation lost whatever
        # this chat specifically could have added. Marking it here would
        # close off the only chance to recover that on a later visit.
        assert db.merged == []

    def test_a_malformed_section_does_not_erase_what_we_knew(self, model, db):
        db.remember(OWNER, None, facts=["Sends 12k home."])
        model.replies = [json.dumps({"global": "I could not do this", "topic": {}})]

        merge_from_chat(context_with(), user_id=OWNER)

        assert db.memory[0]["facts"] == ["Sends 12k home."]
        assert db.merged == []

    def test_an_empty_merge_for_a_new_user_writes_nothing_rather_than_a_blank_row(
        self, model, db
    ):
        model.replies = [reply()]

        assert merge_from_chat(context_with(), user_id=OWNER) is False
        assert db.memory == []
        # Still marked: the model saw the chat and had nothing to learn from it.
        # Re-running it on every visit to history would buy the same answer.
        assert db.merged == ["chat-1"]


class TestThePrompt:
    def test_it_tells_the_model_to_carry_facts_forward(self, model, db):
        """The instruction that makes this a merge rather than an overwrite.

        Asserted as *sent*, because it is the whole behaviour and a silent
        prompt regression would look exactly like a working feature until
        someone noticed their history had been quietly forgotten.
        """
        model.replies = [reply(global_facts=["x"])]

        merge_from_chat(context_with(), user_id=OWNER)

        system = unwrapped(model.calls[0]["system"].lower())
        assert "carry every existing fact forward" in system
        assert "if you leave one out, it is gone for good" in system

    def test_it_bans_recording_inferences_as_facts(self, model, db):
        model.replies = [reply(global_facts=["x"])]

        merge_from_chat(context_with(), user_id=OWNER)

        system = unwrapped(model.calls[0]["system"].lower())
        assert "if you are guessing, do not write" in system

    def test_the_category_is_named_in_the_prompt(self, model, db):
        model.replies = [reply(global_facts=["x"])]

        merge_from_chat(context_with(), user_id=OWNER)

        # The global/topic split is the model's call, so it has to know which
        # topic it is deciding against.
        assert "financial" in model.calls[0]["system"]
        assert "FINANCIAL" in model.calls[0]["user"]
