"""The adaptive questioner: when it stops, what it persists, what it trusts."""

import json

import pytest

from app.services.adaptive_engine import MAX_OPTIONS, _clean_options, next_question
from app.services.context import ChatContext
from tests.conftest import unwrapped

CHAT = {"id": "chat-1", "user_id": "user-1", "category": "education"}


def context(messages=None, profile=None):
    return ChatContext(
        chat=CHAT,
        profile=profile or {"onboarding_answers": {"location": "Pune, India"}},
        messages=messages or [],
    )


def question(**kwargs):
    payload = {
        "done": False,
        "question": "What is actually stopping you?",
        "options": ["The money", "My family", "I lost interest"],
        "reason": "need the why",
    }
    payload.update(kwargs)
    return json.dumps(payload)


class TestStopping:
    def test_stops_when_the_model_says_it_has_enough(self, model, db):
        model.replies = [json.dumps({"done": True, "reason": "enough"})]

        turn = next_question(context())

        assert turn.done is True
        # Nothing written: there is no question to record.
        assert db.messages == []

    def test_stops_at_the_round_cap_without_asking_the_model(self, model, db):
        # Eight already asked and answered.
        asked = [
            {
                "type": "adaptive_question",
                "question_text": f"q{i}",
                "answer_text": f"a{i}",
                "metadata": {},
            }
            for i in range(8)
        ]

        turn = next_question(context(messages=asked))

        assert turn.done is True
        # The cap is a hard stop, not a suggestion the model gets to argue with.
        assert model.calls == []

    def test_keeps_going_below_the_cap(self, model, db):
        model.replies = [question()]
        asked = [
            {
                "type": "adaptive_question",
                "question_text": "q1",
                "answer_text": "a1",
                "metadata": {},
            }
        ]

        turn = next_question(context(messages=asked))

        assert turn.done is False
        assert turn.round == 2

    def test_tells_the_model_how_many_rounds_are_left(self, model, db):
        model.replies = [question()]
        asked = [
            {
                "type": "adaptive_question",
                "question_text": "q1",
                "answer_text": "a1",
                "metadata": {},
            }
        ]

        next_question(context(messages=asked))

        assert "asked 1 question(s)" in model.calls[0]["user"]
        assert "at most 7 more" in model.calls[0]["user"]


class TestPersistence:
    def test_the_question_is_written_before_it_is_returned(self, model, db):
        model.replies = [question()]

        turn = next_question(context())

        rows = db.of_type("adaptive_question")
        assert len(rows) == 1
        row = rows[0]
        assert row["question_text"] == "What is actually stopping you?"
        # Unanswered: the client sends the answer back against this id, so the
        # turn is one row rather than two.
        assert row["answer_text"] is None
        assert row["metadata"]["options"] == [
            "The money",
            "My family",
            "I lost interest",
        ]
        assert turn.message_id == row["id"]

    def test_an_unanswered_question_is_handed_back_rather_than_replaced(
        self, model, db
    ):
        """A retry after a dropped response must not leave an orphan."""
        pending = [
            {
                "id": "msg-9",
                "type": "adaptive_question",
                "question_text": "Which part is the problem?",
                "answer_text": None,
                "metadata": {"options": ["A", "B"]},
            }
        ]

        turn = next_question(context(messages=pending))

        assert turn.done is False
        assert turn.message_id == "msg-9"
        assert turn.question == "Which part is the problem?"
        assert turn.options == ["A", "B"]
        # The model was never asked, and no second question was written.
        assert model.calls == []
        assert db.messages == []


class TestModelOutput:
    def test_an_unusable_question_stops_the_flow_rather_than_showing_nothing(
        self, model, db
    ):
        # One option is not a choice. Rather than invent options or crash,
        # move on — the recommendation sees everything this would have.
        model.replies = [question(options=["The money"])]

        turn = next_question(context())

        assert turn.done is True
        assert db.messages == []

    def test_an_empty_question_stops_the_flow(self, model, db):
        model.replies = [question(question="   ")]

        turn = next_question(context())

        assert turn.done is True

    def test_non_json_is_retried_then_gives_up(self, model, db):
        from app.core.llm import ModelError

        model.replies = ["I think we should ask about money", "still not json"]

        with pytest.raises(ModelError):
            next_question(context())


class TestMultiSelect:
    """Whether a question takes one answer or several.

    The product failure this fixes: asking "why do you feel like that?"
    single-select, when depressed AND confused AND not valued are all true at
    once, and keeping a quarter of the answer.
    """

    def test_a_multi_question_is_carried_to_the_client_and_the_row(self, model, db):
        model.replies = [question(multi=True)]

        turn = next_question(context())

        assert turn.multi is True
        # On the row as well as in the response: the transcript reads it back
        # long after the client that drew the tick boxes has gone.
        assert db.of_type("adaptive_question")[0]["metadata"]["multi"] is True

    def test_single_select_is_the_default_when_the_model_says_nothing(self, model, db):
        model.replies = [question()]

        turn = next_question(context())

        assert turn.multi is False
        assert db.of_type("adaptive_question")[0]["metadata"]["multi"] is False

    def test_an_explicit_false_stays_false(self, model, db):
        model.replies = [question(multi=False)]

        assert next_question(context()).multi is False

    @pytest.mark.parametrize("junk", ["true", "yes", 1, ["yes"], {"multi": True}, None])
    def test_anything_that_is_not_the_literal_true_is_single_select(
        self, model, db, junk
    ):
        # A model that writes "true" has drifted out of the contract rather than
        # decided something. The safe reading of a drifted field is the default:
        # a wrongly-multi question invites a self-contradicting answer.
        model.replies = [question(multi=junk)]

        assert next_question(context()).multi is False

    def test_a_pending_question_is_handed_back_still_multi(self, model, db):
        """A retry must not turn a multi-select question into a single-select
        one — the user would lose the answer they were halfway through."""
        pending = [
            {
                "id": "msg-9",
                "type": "adaptive_question",
                "question_text": "Why does it feel like that?",
                "answer_text": None,
                "metadata": {"options": ["A", "B"], "multi": True},
            }
        ]

        turn = next_question(context(messages=pending))

        assert turn.multi is True
        assert model.calls == []


class TestPromptInstructions:
    """The anti-restatement rule is the fix. Assert it is actually sent."""

    def test_the_prompt_bans_asking_the_answer_back(self, model, db):
        model.replies = [question()]

        next_question(context())

        system = unwrapped(model.calls[0]["system"])
        # The failure it exists to stop: "I am tired" -> "why are you tired?"
        assert "Never ask for a cause they have already named" in system
        assert "Never ask them to elaborate on the last thing they said" in system
        assert "Why are you tired?" in system
        assert "Why is your head aching?" in system
        # And the replacement: ask for the fact that explains it.
        assert "Ask for the FACT that would explain the thing" in system

    def test_the_prompt_explains_when_a_question_takes_several_answers(
        self, model, db
    ):
        model.replies = [question()]

        next_question(context())

        system = unwrapped(model.calls[0]["system"])
        assert '"multi": true when the options are not mutually exclusive' in system
        assert '"multi": false when the question has exactly one true answer' in system

    def test_the_prompt_is_ruthless_without_being_cruel(self, model, db):
        model.replies = [question()]

        next_question(context())

        system = unwrapped(model.calls[0]["system"])
        assert "BE RUTHLESS" in system
        assert "Ask the question they are avoiding" in system
        # The bound on it. Blunt is the product; cruelty is a bug.
        assert "It does not mean cruel" in system


class TestCleanOptions:
    def test_strips_an_other_option_the_model_added_anyway(self):
        # The client always renders its own free-text escape hatch. A second one
        # from the model reads as a bug.
        assert _clean_options(["The money", "Other", "Something else"]) == ["The money"]

    def test_deduplicates_case_insensitively(self):
        assert _clean_options(["The money", "the money  ", "Family"]) == [
            "The money",
            "Family",
        ]

    def test_collapses_whitespace(self):
        assert _clean_options(["The\n  money"]) == ["The money"]

    def test_caps_the_list(self):
        assert len(_clean_options([f"option {i}" for i in range(20)])) == MAX_OPTIONS

    def test_survives_junk(self):
        assert _clean_options(None) == []
        assert _clean_options("not a list") == []
        assert _clean_options([1, None, "ok", ""]) == ["ok"]


class TestContextGivenToTheModel:
    def test_the_prompt_carries_the_profile_and_the_conversation(self, model, db):
        model.replies = [question()]
        messages = [
            {
                "type": "intake",
                "question_text": "What are you trying to work out?",
                "answer_text": "Whether to stay on it",
                "metadata": {},
            },
            {
                "type": "free_text",
                "answer_text": "I do not think this course is mine.",
                "metadata": {"input_method": "voice"},
            },
        ]

        next_question(context(messages=messages))

        sent = model.calls[0]["user"]
        assert "Pune, India" in sent
        assert "Whether to stay on it" in sent
        assert "I do not think this course is mine." in sent
        # Typed by role: a tapped option and a spoken confession carry different
        # weight, and the model should be able to tell them apart.
        assert "said aloud" in sent
