"""The adaptive questioner: when it stops, what it persists, what it trusts."""

import json

import pytest

from app.services.adaptive_engine import MAX_OPTIONS, _clean_options, next_question
from app.services.context import ChatContext

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
