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
from tests.conftest import unwrapped

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
        "headline": "Finish the degree — but stop pretending it is why you are unhappy.",
        "recommendation": "Finish the degree, but stop pretending it is the point.",
        "next_steps": ["Talk to your head of department this week", "Apply anyway"],
        "confidence": "Fairly sure, unless the money is worse than you said.",
    }
)


def answer(**kwargs):
    payload = json.loads(ANSWER)
    payload.update(kwargs)
    return json.dumps(payload)


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


class TestHeadline:
    """The verdict, printed large above the body.

    It exists because the answer rendered as one grey paragraph reads like
    nothing was decided. Some people will read this line and nothing else.
    """

    def test_it_is_carried_back_and_recorded(self, model, search, db):
        model.replies = [NO_SEARCH, ANSWER]

        result = generate(context())

        assert result.headline == (
            "Finish the degree — but stop pretending it is why you are unhappy."
        )
        # Persisted so history can render the verdict rather than guess at it
        # from the body's first sentence.
        assert db.of_type("recommendation")[0]["metadata"]["headline"] == result.headline

    def test_it_is_trimmed(self, model, search, db):
        model.replies = [NO_SEARCH, answer(headline="  Just finish it.  ")]

        assert generate(context()).headline == "Just finish it."

    @pytest.mark.parametrize("missing", ["", "   ", None])
    def test_a_missing_headline_costs_the_headline_and_not_the_answer(
        self, model, search, db, missing
    ):
        """The body is what they came for.

        Failing a whole recommendation because its title did not arrive would
        throw away the only part worth having.
        """
        model.replies = [NO_SEARCH, answer(headline=missing)]

        result = generate(context())

        assert result.headline == ""
        assert result.text.startswith("Finish the degree")
        assert db.of_type("recommendation")[0]["metadata"]["headline"] == ""

    def test_a_headline_of_the_wrong_shape_is_no_headline(self, model, search, db):
        # Trust the model for wording, never for shape.
        model.replies = [NO_SEARCH, answer(headline=["Finish it.", "Or do not."])]

        result = generate(context())

        assert result.headline == ""
        assert result.text

    def test_an_answer_from_before_headlines_existed_still_works(
        self, model, search, db
    ):
        # The field is additive: a model that ignores it returns the old shape,
        # and the old shape is still an answer.
        model.replies = [
            NO_SEARCH,
            json.dumps({"recommendation": "Finish the degree."}),
        ]

        assert generate(context()).headline == ""


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

        system = unwrapped(model.calls[1]["system"])
        assert "It depends" in system
        assert "Only you can decide" in system
        assert "Take a position" in system

    def test_the_recommendation_prompt_is_ruthless(self, model, search, db):
        model.replies = [NO_SEARCH, ANSWER]

        generate(context())

        system = unwrapped(model.calls[1]["system"])
        assert "Be ruthless" in system
        assert "Name the thing they are avoiding" in system
        # The part that makes it useful rather than merely blunt: the user is
        # sometimes the problem, and hearing so is the reason they came.
        assert "If THEY are the problem, say so, plainly and early" in system
        assert "Do not validate reflexively" in system
        # And the bound on it.
        assert "Be hard on the situation, never on the person" in system

    def test_the_recommendation_prompt_is_ruthless_about_relationships(
        self, model, search, db
    ):
        model.replies = [NO_SEARCH, ANSWER]

        generate(context())

        system = unwrapped(model.calls[1]["system"])
        assert 'Do not soften a bad relationship into "communication issues"' in system
        assert "If someone is being treated badly, say they are being treated badly" in system
        # Pointed both ways: the person talking does not get the benefit of the
        # doubt for being the one talking.
        assert "If they have described themselves doing the mistreating" in system

    def test_the_recommendation_prompt_bans_the_markdown_the_client_cannot_render(
        self, model, search, db
    ):
        """The renderer handles one small subset and prints the rest literally.

        Anything banned here that the model emits anyway arrives on the user's
        screen as raw characters.
        """
        model.replies = [NO_SEARCH, ANSWER]

        generate(context())

        system = unwrapped(model.calls[1]["system"])
        banned = system.split("BANNED. The renderer prints these")[1]
        assert "tables" in banned
        assert "links and images of any kind" in banned
        assert "code fences and backticks" in banned
        assert "HTML tags" in banned
        # Underscores are literal because real text has snake_case in it.
        assert "_underscores_ for emphasis" in banned
        assert "nesting of any kind" in banned

    def test_the_recommendation_prompt_describes_the_markdown_it_does_allow(
        self, model, search, db
    ):
        model.replies = [NO_SEARCH, ANSWER]

        generate(context())

        system = unwrapped(model.calls[1]["system"])
        assert "**bold** — the load-bearing phrase" in system
        # Emphasis only works if it is rationed, and a model told to bold things
        # will bold everything.
        assert "Everything bold is nothing bold" in system
        assert "> line — ONE callout, at most" in system

    def test_the_follow_up_prompt_keeps_the_ban(self, model, search, db):
        model.replies = ["ok"]

        follow_up(context(), "hm")

        system = model.calls[0]["system"]
        assert "it depends" in system.lower()
        assert "position" in system.lower()

    def test_the_follow_up_prompt_stays_ruthless_under_pushback(
        self, model, search, db
    ):
        model.replies = ["ok"]

        follow_up(context(), "hm")

        system = unwrapped(model.calls[0]["system"])
        assert "Stay ruthless" in system
        assert "Do not fold to be liked" in system
        # Still a chat bubble: bold is the only markup that survives.
        assert "Match their length" in system
        assert "**Bold** is available" in system
