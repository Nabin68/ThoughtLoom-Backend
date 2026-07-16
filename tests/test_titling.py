"""Naming a finished chat."""

import json

from app.services.context import ChatContext
from app.services.titling import generate_title
from tests.conftest import unwrapped

CHAT = {
    "id": "chat-1",
    "user_id": "user-1",
    "category": "financial",
    "created_at": "2026-03-14T09:00:00+00:00",
}


def context_with(messages=None, chat=None):
    return ChatContext(
        chat={**CHAT, **(chat or {})},
        profile={},
        messages=messages
        if messages is not None
        else [
            {
                "type": "free_text",
                "answer_text": "I keep lending my brother money and he never pays it back.",
                "metadata": {},
            }
        ],
    )


class TestTheTitle:
    def test_it_names_the_topic_and_dates_it_from_the_chat(self, model, db):
        model.replies = [json.dumps({"title": "Lending my brother money again"})]

        title = generate_title(context_with())

        # The month comes from the chat's created_at, not from the model — see
        # titling._month. A model does not know what day it is.
        assert title == "Lending my brother money again — March 2026"
        assert db.chats["chat-1"]["title"] == title

    def test_the_date_is_the_chats_own_not_todays(self, model, db):
        model.replies = [json.dumps({"title": "An old question"})]

        title = generate_title(context_with(chat={"created_at": "2024-11-02T10:00:00Z"}))

        assert title.endswith("— November 2024")

    def test_a_chat_with_nothing_in_it_is_not_named(self, model, db):
        # The dashboard opens a chat row on the category tap, so an abandoned
        # tap is a real row with no messages. Naming it would be inventing a
        # conversation that never happened.
        title = generate_title(context_with(messages=[]))

        assert title is None
        assert model.calls == []
        assert db.chats == {}

    def test_a_model_that_declines_to_name_it_writes_nothing(self, model, db):
        model.replies = [json.dumps({"title": None})]

        assert generate_title(context_with()) is None
        assert db.chats == {}

    def test_quotes_and_full_stops_are_stripped(self, model, db):
        model.replies = [json.dumps({"title": '"Whether to move out."'})]

        assert generate_title(context_with()) == "Whether to move out — March 2026"

    def test_a_rambling_title_is_cut_to_one_line(self, model, db):
        model.replies = [
            json.dumps({"title": "Whether " + "to move out of the family house ".upper() * 10})
        ]

        title = generate_title(context_with())

        # Long enough to be a sentence, short enough for a phone row next to a
        # date.
        assert len(title) < 90
        assert title.endswith("— March 2026")

    def test_an_unreadable_created_at_still_produces_a_title(self, model, db):
        model.replies = [json.dumps({"title": "Something happened"})]

        title = generate_title(context_with(chat={"created_at": "not a date"}))

        # No date beats no title: history has its own timestamp column anyway.
        assert title == "Something happened"


class TestThePrompt:
    def test_it_bans_naming_a_chat_after_its_category(self, model, db):
        """The one thing this feature exists to stop.

        A titler that returns "Financial" has changed nothing — the history
        screen already fell back to the category label. That instruction is the
        product, so a silent prompt regression would be invisible.
        """
        model.replies = [json.dumps({"title": "x"})]

        generate_title(context_with())

        system = unwrapped(model.calls[0]["system"].lower())
        assert "that is the category, and every education chat has it" in system
