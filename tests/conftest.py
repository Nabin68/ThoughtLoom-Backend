"""Fakes for the two things these tests must not really touch: the model and
Supabase.

Nothing here calls Cohere or the network. What is under test is the logic
*around* the model — the round cap, the option cleaning, the search decision,
what gets written and when — which is where the bugs actually live and the only
part that can be tested deterministically.
"""

import os

import pytest

# Set before app imports: the Cohere client is built lazily but still checks.
os.environ.setdefault("COHERE_API_KEY", "test-key")

from app.core import llm as llm_module  # noqa: E402
from app.core import web_search as search_module  # noqa: E402
from app.core.llm import LanguageModel  # noqa: E402
from app.core.web_search import SearchResult, WebSearch  # noqa: E402


def unwrapped(text: str) -> str:
    """Text with its line wrapping collapsed.

    The prompts are wrapped to fit a source file, so any instruction in one can
    break across a line. A test that asserted on the wrapping would fail the
    next time someone reflowed a paragraph — which is not a regression. What
    these tests assert is that an instruction was *sent*, not where it wrapped.
    """
    return " ".join(text.split())


class FakeModel(LanguageModel):
    """Returns queued replies in order. Records what it was asked.

    Queue an Exception instead of a string to make that call fail — which is how
    the failure paths get tested without pretending a network exists.
    """

    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.calls = []

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        if not self.replies:
            raise AssertionError(
                "FakeModel ran out of queued replies — the code under test "
                "called the model more times than the test expected"
            )
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeSearch(WebSearch):
    def __init__(self, results=None):
        self.results = list(results or [])
        self.queries = []

    def search(self, query: str, *, limit: int = 4):
        self.queries.append(query)
        return list(self.results)


class FakeDb:
    """Stands in for the Supabase tables these services touch."""

    def __init__(self):
        self.messages: list[dict] = []
        self.chats: dict[str, dict] = {}
        self.profiles: dict[str, dict] = {}
        self.memory: list[dict] = []
        self.merged: list[str] = []
        self._next_id = 1

    def insert_message(
        self, *, chat_id, type, question_text=None, answer_text=None, metadata=None
    ):
        row = {
            "id": f"msg-{self._next_id}",
            "chat_id": chat_id,
            "seq": len([m for m in self.messages if m["chat_id"] == chat_id]) + 1,
            "type": type,
            "question_text": question_text,
            "answer_text": answer_text,
            "metadata": metadata or {},
        }
        self._next_id += 1
        self.messages.append(row)
        return row

    def answer_message(self, message_id, answer_text, *, selections=None):
        for row in self.messages:
            if row["id"] == message_id:
                row["answer_text"] = answer_text
                if selections:
                    row["metadata"] = {**row["metadata"], "selected": selections}
                return row
        raise AssertionError(f"no such message {message_id}")

    def set_chat_status(self, chat_id, status):
        self.chats.setdefault(chat_id, {})["status"] = status

    def set_chat_title(self, chat_id, title):
        self.chats.setdefault(chat_id, {})["title"] = title

    def mark_memory_merged(self, chat_id):
        self.merged.append(chat_id)

    def of_type(self, kind):
        return [m for m in self.messages if m["type"] == kind]

    # --- user_memory ------------------------------------------------------
    #
    # Mirrors the real table's shape, including the bit that makes it awkward:
    # one row per (user, category) with a *separate* row for category IS NULL.

    def fetch_memory_rows(self, user_id):
        return [row for row in self.memory if row["user_id"] == user_id]

    def upsert_memory(self, *, user_id, category, summary, facts):
        for row in self.memory:
            if row["user_id"] == user_id and row["category"] == category:
                row.update({"summary": summary, "facts": facts})
                return row
        row = {
            "id": f"mem-{len(self.memory) + 1}",
            "user_id": user_id,
            "category": category,
            "summary": summary,
            "facts": facts,
        }
        self.memory.append(row)
        return row

    def remember(self, user_id, category, summary="", facts=()):
        """Seed prior memory, as though earlier chats had already run."""
        return self.upsert_memory(
            user_id=user_id, category=category, summary=summary, facts=list(facts)
        )


@pytest.fixture
def model(monkeypatch):
    fake = FakeModel()
    llm_module.set_model(fake)
    yield fake
    llm_module.set_model(None)


@pytest.fixture
def search():
    fake = FakeSearch()
    search_module.set_search(fake)
    yield fake
    search_module.set_search(None)


@pytest.fixture
def db(monkeypatch):
    """Redirects the reads and writes each service performs at a fake.

    Patched at each module's own name rather than at supabase_client, because
    that is how they import them — patching the source would leave every
    `from ... import insert_message` still bound to the real one.
    """
    fake = FakeDb()

    from app.api import completion
    from app.services import adaptive_engine, memory, recommendation_engine, titling

    monkeypatch.setattr(adaptive_engine, "insert_message", fake.insert_message)
    monkeypatch.setattr(recommendation_engine, "insert_message", fake.insert_message)
    monkeypatch.setattr(recommendation_engine, "set_chat_status", fake.set_chat_status)

    monkeypatch.setattr(titling, "set_chat_title", fake.set_chat_title)

    monkeypatch.setattr(memory, "fetch_memory_rows", fake.fetch_memory_rows)
    monkeypatch.setattr(memory, "upsert_memory", fake.upsert_memory)
    monkeypatch.setattr(memory, "mark_memory_merged", fake.mark_memory_merged)

    monkeypatch.setattr(completion, "set_chat_status", fake.set_chat_status)
    return fake


@pytest.fixture
def hit():
    return SearchResult(
        title="Fees 2026",
        snippet="The annual fee is 2.4 lakh for the 2026 intake.",
        url="https://example.edu/fees",
    )
