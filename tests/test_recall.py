"""Which past conversations this one connects to."""

import pytest

from app.services import recall
from app.services.recall import find_related, keywords_from

USER = "user-1"
NOW = "chat-now"


def chat(id, category, title, created="2026-03-01T09:00:00+00:00"):
    return {
        "id": id,
        "category": category,
        "title": title,
        "status": "completed",
        "created_at": created,
        "updated_at": created,
    }


@pytest.fixture
def past(monkeypatch):
    """The user's past chats, and what we advised in each."""

    state = {"chats": [], "mentioning": set(), "gists": {}, "searched": []}

    def fetch_past_chats(user_id, *, exclude_chat_id, limit=20):
        return [c for c in state["chats"] if c["id"] != exclude_chat_id][:limit]

    def fetch_chats_mentioning(chat_ids, keywords):
        state["searched"].append((list(chat_ids), list(keywords)))
        return {c for c in state["mentioning"] if c in chat_ids}

    monkeypatch.setattr(recall, "fetch_past_chats", fetch_past_chats)
    monkeypatch.setattr(recall, "fetch_chats_mentioning", fetch_chats_mentioning)
    monkeypatch.setattr(
        recall, "fetch_recommendations", lambda ids: {i: state["gists"].get(i, "") for i in ids}
    )
    return state


def related(category="financial", keywords=("brother",)):
    return find_related(
        user_id=USER, chat_id=NOW, category=category, keywords=list(keywords)
    )


class TestColdStart:
    def test_a_first_chat_recalls_nothing(self, past):
        assert related() == []

    def test_a_first_chat_does_not_even_search(self, past):
        # No candidates, no queries. This runs in front of a user waiting on a
        # question, and a new user should pay nothing for a feature that has
        # nothing to tell them.
        related()

        assert past["searched"] == []


class TestWhatConnects:
    def test_the_same_part_of_their_life_connects(self, past):
        past["chats"] = [chat("chat-1", "financial", "Lending Ravi money")]

        assert [r.chat_id for r in related()] == ["chat-1"]

    def test_an_unrelated_chat_is_left_out(self, past):
        past["chats"] = [chat("chat-1", "education", "Which masters to apply for")]

        # Different topic, no shared words, nothing said in common. Dragging it
        # in would have the model reaching for a connection that is not there.
        assert related() == []

    def test_the_same_words_connect_across_topics(self, past):
        past["chats"] = [chat("chat-1", "relationship", "Talking to Ravi about school")]
        past["mentioning"] = {"chat-1"}

        # The cross-topic half: a brother named in a relationship chat and again
        # in a financial one is a connection category matching alone would miss.
        assert [r.chat_id for r in related()] == ["chat-1"]

    def test_a_matching_title_connects(self, past):
        past["chats"] = [chat("chat-1", "education", "Paying for my brother's school")]

        assert [r.chat_id for r in related()] == ["chat-1"]

    def test_what_we_told_them_last_time_comes_along(self, past):
        past["chats"] = [chat("chat-1", "financial", "Lending Ravi money")]
        past["gists"] = {"chat-1": "Keep paying, but tell him it stops in June."}

        assert related()[0].gist == "Keep paying, but tell him it stops in June."

    def test_a_long_recommendation_is_cut_to_a_gist(self, past):
        past["chats"] = [chat("chat-1", "financial", "Lending Ravi money")]
        past["gists"] = {"chat-1": "word " * 500}

        # This lands in a prompt alongside the whole current conversation. Three
        # past recommendations at full length would crowd out the thing the user
        # actually asked.
        assert len(related()[0].gist) < 260


class TestRanking:
    def test_the_most_connected_chats_come_first(self, past):
        past["chats"] = [
            chat("weak", "education", "Which masters to apply for"),
            chat("strong", "financial", "Lending my brother money"),
        ]
        past["mentioning"] = {"strong"}

        assert [r.chat_id for r in related()] == ["strong"]

    def test_only_a_handful_reach_the_prompt(self, past):
        past["chats"] = [chat(f"chat-{i}", "financial", f"Money {i}") for i in range(10)]

        assert len(related()) == recall.MAX_RELATED

    def test_ties_break_on_recency(self, past):
        # fetch_past_chats orders by updated_at and sort is stable, so equally
        # connected chats keep that order rather than shuffling between reads.
        past["chats"] = [
            chat("newest", "financial", "Money three"),
            chat("middle", "financial", "Money two"),
            chat("oldest", "financial", "Money one"),
            chat("ancient", "financial", "Money zero"),
        ]

        assert [r.chat_id for r in related()] == ["newest", "middle", "oldest"]


class TestNaming:
    def test_an_untitled_chat_is_still_recalled(self, past):
        # Titling is best-effort and runs after the user has left. A chat whose
        # title never landed is still a conversation that happened.
        past["chats"] = [chat("chat-1", "financial", None)]

        assert related()[0].title == "An earlier financial conversation"

    def test_the_month_comes_from_the_chat(self, past):
        past["chats"] = [
            chat("chat-1", "financial", "Money", created="2025-11-02T10:00:00+00:00")
        ]

        assert related()[0].when == "November 2025"


class TestKeywords:
    def test_filler_is_not_a_topic(self):
        # Everything a person writes about a hard decision is "really about"
        # something and "thinking" something. Matching on those would connect
        # every chat to every other chat.
        assert keywords_from("I really think I should maybe talk to them about it") == []

    def test_real_topics_survive(self):
        assert "scholarship" in keywords_from("The scholarship deadline is in March")

    def test_only_plain_words_survive(self):
        """These are interpolated into a PostgREST filter.

        A comma or a parenthesis reaching that string changes the query's shape
        rather than being matched literally — so the filter is built only from
        [a-z]+, and fetch_chats_mentioning asserts the same thing at the far end.
        """
        keywords = keywords_from("mother's, salary (approx) 32,000 — engineering")

        assert all(k.isalpha() for k in keywords)
        assert "engineering" in keywords

    def test_the_list_is_capped(self):
        # One rambling description must not turn the content filter into "match
        # anything".
        rambling = " ".join(
            [
                "scholarship", "engineering", "hostel", "stipend", "deadline",
                "internship", "placement", "semester", "counsellor", "admission",
                "transfer", "brother",
            ]
        )

        assert len(keywords_from(rambling)) == recall.MAX_KEYWORDS
