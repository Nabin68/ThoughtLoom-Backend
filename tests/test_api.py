"""The endpoints: authorisation, and turning failures into honest statuses.

The authorisation tests are the important ones. These endpoints hold a key that
bypasses Row Level Security, so they are the only thing standing between a
guessed uuid and someone's private situation.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.core import auth as auth_module
from app.main import app

CHAT_ID = "chat-1"
OWNER = "user-1"
STRANGER = "user-2"

CHAT = {
    "id": CHAT_ID,
    "user_id": OWNER,
    "category": "education",
    "status": "in_progress",
    # Present because the real row has it and the titler dates a chat from it.
    "created_at": "2026-03-14T09:00:00+00:00",
}


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def as_owner(monkeypatch):
    """Signs every request in as the chat's owner."""
    _sign_in_as(monkeypatch, OWNER)


def _sign_in_as(monkeypatch, user_id):
    app.dependency_overrides[auth_module.current_user_id] = lambda: user_id


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def chat_row(monkeypatch):
    """The chat the endpoints look up when checking ownership."""
    from app.core import auth

    monkeypatch.setattr(auth, "fetch_chat", lambda chat_id: CHAT if chat_id == CHAT_ID else None)


@pytest.fixture
def loaded(monkeypatch):
    """Context loading, without Supabase.

    Includes the cross-chat reads Prompt 6 added: an endpoint that assembles
    context now also asks for the user's memory and their past chats, and both
    would otherwise reach for a real Supabase client.
    """
    from app.services import context as context_module

    monkeypatch.setattr(context_module, "fetch_profile", lambda user_id: {})
    monkeypatch.setattr(context_module, "fetch_messages", lambda chat_id: [])
    monkeypatch.setattr(context_module, "fetch_memory_rows", lambda user_id: [])
    monkeypatch.setattr(context_module, "find_related", lambda **kwargs: [])


@pytest.fixture
def talked(monkeypatch, loaded):
    """A chat with something actually said in it.

    [loaded] alone is an empty transcript, which titling and memory both
    correctly refuse to work on — there is nothing to name or learn. Anything
    testing what they *do* needs a real turn in the chat.
    """
    from app.services import context as context_module

    monkeypatch.setattr(
        context_module,
        "fetch_messages",
        lambda chat_id: [
            {
                "type": "free_text",
                "answer_text": "I want to drop out of my BTech and do design.",
                "metadata": {},
            }
        ],
    )


class TestAuthorisation:
    def test_no_token_is_rejected(self, client, chat_row):
        response = client.post("/api/adaptive-question", json={"chat_id": CHAT_ID})
        assert response.status_code == 401

    def test_a_junk_authorization_header_is_rejected(self, client, chat_row):
        response = client.post(
            "/api/adaptive-question",
            json={"chat_id": CHAT_ID},
            headers={"Authorization": "Basic hunter2"},
        )
        assert response.status_code == 401

    def test_someone_elses_chat_is_not_readable(
        self, client, chat_row, monkeypatch, model, db, loaded
    ):
        """The whole reason app/core/auth.py exists.

        Without the ownership check this would return a stranger's situation to
        anyone who could guess a uuid — the service-role key does not care whose
        row it is.
        """
        _sign_in_as(monkeypatch, STRANGER)

        response = client.post("/api/adaptive-question", json={"chat_id": CHAT_ID})

        assert response.status_code == 404
        # Nothing was generated and nothing was written.
        assert model.calls == []
        assert db.messages == []

    def test_a_chat_that_does_not_exist_looks_the_same_as_one_you_cannot_have(
        self, client, chat_row, monkeypatch, model
    ):
        # Deliberately indistinguishable: 'this id exists but is not yours' is a
        # fact an attacker should not get.
        _sign_in_as(monkeypatch, OWNER)

        response = client.post("/api/adaptive-question", json={"chat_id": "nope"})

        assert response.status_code == 404

    def test_the_owner_gets_through(
        self, client, chat_row, as_owner, model, db, loaded
    ):
        model.replies = [
            json.dumps(
                {
                    "done": False,
                    "question": "What is stopping you?",
                    "options": ["Money", "Family"],
                }
            )
        ]

        response = client.post("/api/adaptive-question", json={"chat_id": CHAT_ID})

        assert response.status_code == 200
        assert response.json()["question"] == "What is stopping you?"

    def test_every_ai_endpoint_is_guarded(self, client, chat_row):
        # A new endpoint that forgets the dependency is the failure mode this
        # catches.
        for path, body in [
            ("/api/adaptive-question", {"chat_id": CHAT_ID}),
            ("/api/recommendation", {"chat_id": CHAT_ID}),
            ("/api/follow-up", {"chat_id": CHAT_ID, "message": "hi"}),
            ("/api/complete-chat", {"chat_id": CHAT_ID}),
        ]:
            assert client.post(path, json=body).status_code == 401, path


class TestAdaptiveEndpoint:
    def test_an_answer_is_recorded_before_the_next_question(
        self, client, chat_row, as_owner, model, db, loaded, monkeypatch
    ):
        recorded = {}
        monkeypatch.setattr(
            "app.api.adaptive.answer_message",
            lambda mid, text: recorded.update({"id": mid, "text": text}),
        )
        model.replies = [json.dumps({"done": True})]

        response = client.post(
            "/api/adaptive-question",
            json={
                "chat_id": CHAT_ID,
                "answer": {"message_id": "msg-3", "text": "The money"},
            },
        )

        assert response.status_code == 200
        assert recorded == {"id": "msg-3", "text": "The money"}
        assert response.json()["done"] is True

    def test_a_model_failure_is_a_502_with_a_readable_message(
        self, client, chat_row, as_owner, model, db, loaded
    ):
        from app.core.llm import ModelError

        model.replies = [ModelError("cohere is down")]

        response = client.post("/api/adaptive-question", json={"chat_id": CHAT_ID})

        assert response.status_code == 502
        # The client shows this verbatim, so it must read like a sentence and
        # must not leak the provider's error.
        detail = response.json()["detail"]
        assert "try again" in detail.lower()
        assert "cohere" not in detail.lower()


class TestRecommendationEndpoint:
    def test_it_returns_the_answer_and_its_sources(
        self, client, chat_row, as_owner, model, search, db, loaded, hit
    ):
        search.results = [hit]
        model.replies = [
            json.dumps({"search": True, "queries": ["fees"]}),
            json.dumps(
                {
                    "recommendation": "Finish it.",
                    "next_steps": ["Talk to them"],
                    "confidence": "Sure.",
                }
            ),
        ]

        response = client.post("/api/recommendation", json={"chat_id": CHAT_ID})

        assert response.status_code == 200
        body = response.json()
        assert body["recommendation"] == "Finish it."
        assert body["next_steps"] == ["Talk to them"]
        assert body["sources"] == [
            {"title": "Fees 2026", "url": "https://example.edu/fees"}
        ]


class TestFollowUpEndpoint:
    def test_it_replies(self, client, chat_row, as_owner, model, db, loaded):
        model.replies = ["Then do not."]

        response = client.post(
            "/api/follow-up",
            json={"chat_id": CHAT_ID, "message": "I cannot afford it."},
        )

        assert response.status_code == 200
        assert response.json()["reply"] == "Then do not."

    def test_an_empty_message_is_rejected_before_anything_runs(
        self, client, chat_row, as_owner, model
    ):
        response = client.post(
            "/api/follow-up", json={"chat_id": CHAT_ID, "message": ""}
        )

        assert response.status_code == 422
        assert model.calls == []


class TestCompleteChatEndpoint:
    """Closing a chat: the status write, and the work that happens after it.

    TestClient runs background tasks after the response, synchronously, so
    asserting on what the task did is asserting on the real wiring rather than
    on a mock of it.
    """

    def test_leaving_completes_the_chat_and_names_it(
        self, client, chat_row, as_owner, model, db, talked
    ):
        model.replies = [
            json.dumps({"title": "Whether to drop out"}),
            json.dumps({"global": {"summary": "", "facts": ["Studies in Pune."]},
                        "topic": {"summary": "", "facts": []}}),
        ]

        response = client.post("/api/complete-chat", json={"chat_id": CHAT_ID})

        assert response.status_code == 200
        assert response.json() == {"status": "completed", "scheduled": True}
        assert db.chats[CHAT_ID]["status"] == "completed"
        # Dated from the chat's own created_at, not from today.
        assert db.chats[CHAT_ID]["title"] == "Whether to drop out — March 2026"
        assert db.memory[0]["facts"] == ["Studies in Pune."]

    def test_someone_elses_chat_cannot_be_completed(
        self, client, chat_row, monkeypatch, model, db, loaded
    ):
        _sign_in_as(monkeypatch, STRANGER)

        response = client.post("/api/complete-chat", json={"chat_id": CHAT_ID})

        assert response.status_code == 404
        # Nothing was written, and — the part that matters — no stranger's
        # conversation was folded into anyone's memory.
        assert db.memory == []
        assert db.chats == {}
        assert model.calls == []

    def test_a_chat_already_named_and_remembered_does_no_work_twice(
        self, client, chat_row, as_owner, model, db, loaded, monkeypatch
    ):
        """Idempotence, which this endpoint needs rather than merely enjoys.

        The client fires it when the user leaves and the history screen asks
        again for any completed chat that never got a title. An AI merge run
        twice over one conversation folds the same facts in twice.
        """
        from app.core import auth

        done = {**CHAT, "title": "Already named", "memory_merged_at": "2026-03-14T09:00:00Z"}
        monkeypatch.setattr(auth, "fetch_chat", lambda chat_id: done)

        response = client.post("/api/complete-chat", json={"chat_id": CHAT_ID})

        assert response.json()["scheduled"] is False
        assert model.calls == []

    def test_a_title_that_never_landed_is_backfilled_without_touching_memory(
        self, client, chat_row, as_owner, model, db, talked, monkeypatch
    ):
        from app.core import auth

        half = {**CHAT, "title": None, "memory_merged_at": "2026-03-14T09:00:00Z"}
        monkeypatch.setattr(auth, "fetch_chat", lambda chat_id: half)
        model.replies = [json.dumps({"title": "Whether to drop out"})]

        client.post("/api/complete-chat", json={"chat_id": CHAT_ID})

        # Exactly one call: the title. The memory merge is the expensive half
        # and it is already done.
        assert len(model.calls) == 1
        assert db.chats[CHAT_ID]["title"].startswith("Whether to drop out")

    def test_a_model_failure_does_not_fail_the_request(
        self, client, chat_row, as_owner, model, db, talked
    ):
        """The user has already left. There is nobody to show an error to.

        What must not happen is the status write being rolled back over a title
        that could not be written — the chat is over either way, and history
        renders an untitled chat perfectly well.
        """
        from app.core.llm import ModelError

        model.replies = [ModelError("cohere is down"), ModelError("still down")]

        response = client.post("/api/complete-chat", json={"chat_id": CHAT_ID})

        assert response.status_code == 200
        assert db.chats[CHAT_ID]["status"] == "completed"

    def test_naming_failing_does_not_stop_the_memory_merge(
        self, client, chat_row, as_owner, model, db, talked
    ):
        from app.core.llm import ModelError

        model.replies = [
            ModelError("no title for you"),
            json.dumps({"global": {"summary": "", "facts": ["Studies in Pune."]},
                        "topic": {"summary": "", "facts": []}}),
        ]

        client.post("/api/complete-chat", json={"chat_id": CHAT_ID})

        # The two halves are independent. Losing both because one threw would be
        # a worse trade than losing either.
        assert db.memory[0]["facts"] == ["Studies in Pune."]


class TestHealth:
    def test_health_needs_no_configuration(self, client):
        # Render polls this, and it must answer on a box with no Supabase keys.
        assert client.get("/health").status_code == 200

    def test_root_lists_the_endpoints(self, client):
        body = client.get("/").json()
        assert "adaptive_question" in body["endpoints"]
        assert "recommendation" in body["endpoints"]
        assert "complete_chat" in body["endpoints"]
