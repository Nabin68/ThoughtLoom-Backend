"""The recommendation: when it searches, what it persists, what it says."""

import json

import pytest

from app.core.web_search import SearchResult
from app.services.context import ChatContext
from app.services.recommendation_engine import (
    STATUS_AWAITING_FOLLOW_UP,
    follow_up,
    generate,
)

CHAT = {"id": "chat-1", "user_id": "user-1", "category": "education"}


def context(messages=None):
    return ChatContext(
        chat=CHAT,
        profile={
            "display_name": "Ada",
            "onboarding_answers": {
                "location": "Pune, India",
                "education_level": "Partway through an undergraduate degree",
            },
        },
        messages=messages or [
            {
                "type": "free_text",
                "answer_text": "I do not want to finish this degree.",
                "metadata": {},
            }
        ],
    )


NO_SEARCH = json.dumps({"search": False, "queries": []})
DO_SEARCH = json.dumps({"search": True, "queries": ["fees 2026 pune"]})

ANSWER = json.dumps(
    {
        "recommendation": "Finish the degree, but stop pretending it is the point.",
        "next_steps": ["Talk to your head of department this week", "Apply anyway"],
        "confidence": "Fairly sure, unless the money is worse than you said.",
    }
)


class TestSearchDecision:
    def test_does_not_search_when_the_model_says_it_is_not_needed(
        self, model, search, db
    ):
        model.replies = [NO_SEARCH, ANSWER]

        result = generate(context())

        assert search.queries == []
        assert result.sources == []

    def test_searches_when_the_model_asks_for_it(self, model, search, db, hit):
        model.replies = [DO_SEARCH, ANSWER]
        search.results = [hit]

        result = generate(context())

        assert search.queries == ["fees 2026 pune"]
        assert result.sources == [hit]

    def test_results_reach_the_prompt(self, model, search, db, hit):
        model.replies = [DO_SEARCH, ANSWER]
        search.results = [hit]

        generate(context())

        # The second call is the recommendation itself.
        sent = model.calls[1]["user"]
        assert "2.4 lakh" in sent
        assert "https://example.edu/fees" in sent

    def test_no_research_block_when_nothing_was_searched(self, model, search, db):
        model.replies = [NO_SEARCH, ANSWER]

        generate(context())

        assert "CURRENT INFORMATION FROM THE WEB" not in model.calls[1]["user"]

    def test_a_broken_search_decision_does_not_sink_the_answer(
        self, model, search, db
    ):
        # Search is a nice-to-have. Ungrounded advice beats no advice.
        model.replies = ["not json at all", "still not json", ANSWER]

        result = generate(context())

        assert result.text.startswith("Finish the degree")
        assert result.sources == []

    def test_a_search_that_returns_nothing_is_not_an_error(self, model, search, db):
        model.replies = [DO_SEARCH, ANSWER]
        search.results = []

        result = generate(context())

        assert result.sources == []
        assert result.text

    def test_duplicate_urls_across_queries_are_kept_once(self, model, search, db, hit):
        model.replies = [
            json.dumps({"search": True, "queries": ["a", "b"]}),
            ANSWER,
        ]
        search.results = [hit]

        result = generate(context())

        assert len(search.queries) == 2
        assert len(result.sources) == 1

    def test_at_most_three_queries(self, model, search, db):
        model.replies = [
            json.dumps({"search": True, "queries": ["a", "b", "c", "d", "e"]}),
            ANSWER,
        ]

        generate(context())

        assert len(search.queries) == 3


class TestPersistence:
    def test_the_recommendation_is_written_and_the_chat_moves_on(
        self, model, search, db
    ):
        model.replies = [NO_SEARCH, ANSWER]

        result = generate(context())

        rows = db.of_type("recommendation")
        assert len(rows) == 1
        assert rows[0]["answer_text"].startswith("Finish the degree")
        assert rows[0]["metadata"]["next_steps"] == [
            "Talk to your head of department this week",
            "Apply anyway",
        ]
        assert rows[0]["metadata"]["searched"] is False
        assert result.message_id == rows[0]["id"]

        # Not 'completed': the user can still push back, and usually should.
        # Only leaving the chat completes it, and that is the client's call.
        assert db.chats["chat-1"]["status"] == STATUS_AWAITING_FOLLOW_UP

    def test_sources_are_recorded_on_the_row(self, model, search, db, hit):
        model.replies = [DO_SEARCH, ANSWER]
        search.results = [hit]

        generate(context())

        metadata = db.of_type("recommendation")[0]["metadata"]
        assert metadata["searched"] is True
        assert metadata["sources"] == [
            {"title": "Fees 2026", "url": "https://example.edu/fees"}
        ]

    def test_an_empty_recommendation_is_an_error_not_a_blank_screen(
        self, model, search, db
    ):
        from app.core.llm import ModelError

        model.replies = [NO_SEARCH, json.dumps({"recommendation": "   "})]

        with pytest.raises(ModelError):
            generate(context())

        assert db.of_type("recommendation") == []


class TestFollowUp:
    def test_the_users_message_is_saved_before_the_model_is_called(
        self, model, search, db
    ):
        """A model failure must cost the reply, not what they said.

        The reply they can retry. What they typed, once it is gone from the
        text box, they cannot.
        """
        from app.core.llm import ModelError

        model.replies = [ModelError("Cohere is having a day")]

        with pytest.raises(ModelError):
            follow_up(context(), "But I cannot afford another year.")

        saved = db.of_type("free_text")
        assert saved[-1]["answer_text"] == "But I cannot afford another year."
        assert db.of_type("assistant_reply") == []

    def test_a_reply_is_saved_as_its_own_type(self, model, search, db):
        model.replies = ["Then do not do another year. Here is what I would do."]

        result = follow_up(context(), "But I cannot afford another year.")

        replies = db.of_type("assistant_reply")
        assert len(replies) == 1
        assert replies[0]["answer_text"] == result["reply"]
        # Distinct from 'recommendation' so the advice itself stays findable in
        # a chat that ran on for another twenty turns.
        assert db.of_type("recommendation") == []

    def test_the_whole_conversation_is_context_for_the_reply(self, model, search, db):
        model.replies = ["Fair."]
        prior = [
            {
                "type": "recommendation",
                "answer_text": "Finish the degree.",
                "metadata": {},
            }
        ]

        follow_up(context(messages=prior), "But the money.")

        sent = model.calls[0]["user"]
        assert "Finish the degree." in sent
        assert "But the money." in sent
        assert "Pune, India" in sent


class TestPromptInstructions:
    """The 'be direct' instruction is the product. Assert it is actually sent."""

    def test_the_recommendation_prompt_bans_hedging(self, model, search, db):
        model.replies = [NO_SEARCH, ANSWER]

        generate(context())

        system = model.calls[1]["system"]
        assert "It depends" in system
        assert "Only you can decide" in system
        assert "Take a position" in system

    def test_the_follow_up_prompt_keeps_the_ban(self, model, search, db):
        model.replies = ["ok"]

        follow_up(context(), "hm")

        system = model.calls[0]["system"]
        assert "it depends" in system.lower()
        assert "position" in system.lower()
