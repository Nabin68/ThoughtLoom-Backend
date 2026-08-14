"""Supabase access, with the service-role key.

This key bypasses Row Level Security. Everything in here can read and write any
user's rows, so nothing in here decides *whether* it should — see
`app/core/auth.py`, which proves the caller owns the chat before any of this is
reached.
"""

import logging
from functools import lru_cache

from supabase import Client, create_client

from app.core.config import require_supabase

logger = logging.getLogger(__name__)


class SupabaseError(RuntimeError):
    """A read or write against Supabase failed."""


@lru_cache(maxsize=1)
def service_client() -> Client:
    """The service-role client, built once.

    Cached rather than global so that import does not require configuration —
    /health runs without Supabase at all.
    """
    url, key = require_supabase()
    return create_client(url, key)


# --- reads ----------------------------------------------------------------


def fetch_chat(chat_id: str) -> dict | None:
    try:
        result = (
            service_client()
            .table("chats")
            .select("*")
            .eq("id", chat_id)
            .maybe_single()
            .execute()
        )
    except Exception as exc:  # noqa: BLE001
        raise SupabaseError(f"Could not load chat {chat_id}: {exc}") from exc
    return result.data if result else None


def fetch_profile(user_id: str) -> dict:
    """The basic profile from onboarding.

    An empty dict rather than an error when absent: a missing profile makes the
    advice less grounded, not impossible, and refusing to answer over it would
    be a worse outcome than answering with less.
    """
    try:
        result = (
            service_client()
            .table("user_profiles")
            .select("*")
            .eq("id", user_id)
            .maybe_single()
            .execute()
        )
    except Exception as exc:  # noqa: BLE001
        raise SupabaseError(f"Could not load profile {user_id}: {exc}") from exc
    return (result.data if result else None) or {}


def fetch_messages(chat_id: str) -> list[dict]:
    """Every turn in the chat, in the order it happened."""
    try:
        result = (
            service_client()
            .table("messages")
            .select("*")
            .eq("chat_id", chat_id)
            .order("seq", desc=False)
            .execute()
        )
    except Exception as exc:  # noqa: BLE001
        raise SupabaseError(f"Could not load messages for {chat_id}: {exc}") from exc
    return result.data or []


def fetch_memory_rows(user_id: str) -> list[dict]:
    """Every user_memory row for one person: the global one, plus per-category.

    Soft-fails to no memory. A user whose memory could not be read should get
    the conversation a new user gets, not an error — starting cold is the
    product working slightly worse, and this read sits in front of every
    generated question.
    """
    try:
        result = (
            service_client()
            .table("user_memory")
            .select("*")
            .eq("user_id", user_id)
            .execute()
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load memory for %s: %s", user_id, exc)
        return []
    return result.data or []


def fetch_past_chats(user_id: str, *, exclude_chat_id: str, limit: int = 20) -> list[dict]:
    """This user's other chats, most recently active first.

    Only the ones that got far enough to be worth recalling: a chat abandoned
    during the scripted opening has no advice in it and nothing to connect to.
    """
    try:
        result = (
            service_client()
            .table("chats")
            .select("id, category, title, status, created_at, updated_at")
            .eq("user_id", user_id)
            .neq("id", exclude_chat_id)
            .in_("status", ["completed", "awaiting_follow_up"])
            .order("updated_at", desc=True)
            .limit(limit)
            .execute()
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load past chats for %s: %s", user_id, exc)
        return []
    return result.data or []


def fetch_recommendations(chat_ids: list[str]) -> dict[str, str]:
    """The advice given in each of [chat_ids], keyed by chat id.

    One query for the lot rather than one per chat — this runs in front of a
    user waiting on a question.
    """
    if not chat_ids:
        return {}
    try:
        result = (
            service_client()
            .table("messages")
            .select("chat_id, answer_text")
            .in_("chat_id", chat_ids)
            .eq("type", "recommendation")
            .execute()
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load past recommendations: %s", exc)
        return {}
    return {
        row["chat_id"]: row.get("answer_text") or ""
        for row in (result.data or [])
        if row.get("chat_id")
    }


def fetch_chats_mentioning(chat_ids: list[str], keywords: list[str]) -> set[str]:
    """Which of [chat_ids] mention any of [keywords] in what was actually said.

    The cross-topic half of the recall lookup: a brother named in a financial
    chat and again in a relationship one is a connection that category matching
    alone would never find.

    [keywords] must already be plain words — see recall.keywords_from, which
    only ever emits [a-z]+. They are interpolated into a PostgREST filter, where
    a comma or a parenthesis would change the query's shape rather than being
    matched literally.
    """
    if not chat_ids or not keywords:
        return set()

    assert all(k.isalpha() for k in keywords), "keywords must be plain words"
    matches = ",".join(f"answer_text.ilike.*{k}*" for k in keywords)
    try:
        result = (
            service_client()
            .table("messages")
            .select("chat_id")
            .in_("chat_id", chat_ids)
            .or_(matches)
            .execute()
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not search past chats: %s", exc)
        return set()
    return {row["chat_id"] for row in (result.data or []) if row.get("chat_id")}


# --- writes ---------------------------------------------------------------


def _next_seq(chat_id: str) -> int:
    """The next turn number for a chat.

    Read-then-write, exactly as the Flutter client does it. Two concurrent
    writers could collide on the (chat_id, seq) unique constraint — but the
    writers here are one user answering one question at a time, and a hard
    failure on a genuine race is better than silently reordering someone's
    conversation.
    """
    try:
        result = (
            service_client()
            .table("messages")
            .select("seq")
            .eq("chat_id", chat_id)
            .order("seq", desc=True)
            .limit(1)
            .execute()
        )
    except Exception as exc:  # noqa: BLE001
        raise SupabaseError(f"Could not read sequence for {chat_id}: {exc}") from exc

    rows = result.data or []
    return (rows[0]["seq"] if rows else 0) + 1


def insert_message(
    *,
    chat_id: str,
    type: str,
    question_text: str | None = None,
    answer_text: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Append a turn."""
    try:
        result = (
            service_client()
            .table("messages")
            .insert(
                {
                    "chat_id": chat_id,
                    "seq": _next_seq(chat_id),
                    "type": type,
                    "question_text": question_text,
                    "answer_text": answer_text,
                    "metadata": metadata or {},
                }
            )
            .execute()
        )
    except Exception as exc:  # noqa: BLE001
        raise SupabaseError(f"Could not save a message to {chat_id}: {exc}") from exc

    if not result.data:
        raise SupabaseError(f"Could not save a message to {chat_id}")
    touch_chat(chat_id)
    return result.data[0]


def _metadata_of(message_id: str) -> dict:
    """One message's metadata, for a caller about to write it back."""
    try:
        result = (
            service_client()
            .table("messages")
            .select("metadata")
            .eq("id", message_id)
            .maybe_single()
            .execute()
        )
    except Exception as exc:  # noqa: BLE001
        raise SupabaseError(f"Could not read message {message_id}: {exc}") from exc
    return ((result.data if result else None) or {}).get("metadata") or {}


def answer_message(
    message_id: str,
    answer_text: str,
    *,
    selections: list[str] | None = None,
) -> dict:
    """Fill in the answer on a question that was already asked.

    The question row is written when the model asks it, so the answer lands on
    that same row rather than a new one — one row per turn, and an unanswered
    question is visibly unanswered rather than missing.

    [selections] are the options a multi-select answer ticked. [answer_text]
    already carries them joined and remains what every reader uses; these are
    kept so a later reader does not have to find the joins in a sentence.
    """
    updates: dict = {"answer_text": answer_text}
    if selections:
        # An update replaces jsonb wholesale rather than merging into it, so the
        # question's own options and round have to be carried across by hand or
        # the answer erases the question it belongs to.
        updates["metadata"] = {**_metadata_of(message_id), "selected": selections}

    try:
        result = (
            service_client()
            .table("messages")
            .update(updates)
            .eq("id", message_id)
            .execute()
        )
    except Exception as exc:  # noqa: BLE001
        raise SupabaseError(f"Could not save that answer: {exc}") from exc

    if not result.data:
        raise SupabaseError("That question no longer exists.")
    return result.data[0]


def set_chat_status(chat_id: str, status: str) -> None:
    try:
        service_client().table("chats").update({"status": status}).eq(
            "id", chat_id
        ).execute()
    except Exception as exc:  # noqa: BLE001
        raise SupabaseError(f"Could not update chat {chat_id}: {exc}") from exc


def set_chat_title(chat_id: str, title: str) -> None:
    try:
        service_client().table("chats").update({"title": title}).eq(
            "id", chat_id
        ).execute()
    except Exception as exc:  # noqa: BLE001
        raise SupabaseError(f"Could not title chat {chat_id}: {exc}") from exc


def mark_memory_merged(chat_id: str) -> None:
    """Record that this chat is now in the user's memory, so it is not folded
    in twice."""
    from datetime import datetime, timezone

    try:
        service_client().table("chats").update(
            {"memory_merged_at": datetime.now(timezone.utc).isoformat()}
        ).eq("id", chat_id).execute()
    except Exception as exc:  # noqa: BLE001
        raise SupabaseError(f"Could not mark chat {chat_id} merged: {exc}") from exc


def upsert_memory(
    *,
    user_id: str,
    category: str | None,
    summary: str,
    facts: list[str],
) -> dict:
    """Write one user_memory row: the global one when [category] is None.

    Read-then-write rather than an upsert, for the same reason the Flutter
    client does it that way: uniqueness here comes from two *partial* indexes
    (one for `category is null`, one for the rest), and on_conflict cannot
    express their WHERE predicate. A blind upsert would arbitrate on the
    synthetic primary key, which never collides, and insert a second global row
    that the index then rejects.

    [user_id] must be the caller's own, proven from their token. This row is
    scoped to a person, not to a chat, so it is the one write here that chat
    ownership alone does not authorise. See app/api/completion.py.
    """
    try:
        query = (
            service_client().table("user_memory").select("id").eq("user_id", user_id)
        )
        # `category = null` and `category is null` are different queries in SQL.
        query = (
            query.is_("category", "null")
            if category is None
            else query.eq("category", category)
        )
        existing = query.maybe_single().execute()
        row = existing.data if existing else None

        if row:
            result = (
                service_client()
                .table("user_memory")
                .update({"summary": summary, "facts": facts})
                .eq("id", row["id"])
                .execute()
            )
        else:
            result = (
                service_client()
                .table("user_memory")
                .insert(
                    {
                        "user_id": user_id,
                        "category": category,
                        "summary": summary,
                        "facts": facts,
                    }
                )
                .execute()
            )
    except Exception as exc:  # noqa: BLE001
        raise SupabaseError(f"Could not save memory for {user_id}: {exc}") from exc

    if not result.data:
        raise SupabaseError(f"Could not save memory for {user_id}")
    return result.data[0]


def touch_chat(chat_id: str) -> None:
    """Bump the chat's activity time so history stays ordered by real use.

    Best-effort: the message is already saved by the time this runs, and losing
    a sort key is not worth failing a request the user is waiting on. The
    database's own touch_updated_at trigger does the real work; this just has to
    make it a real update.
    """
    try:
        from datetime import datetime, timezone

        service_client().table("chats").update(
            {"updated_at": datetime.now(timezone.utc).isoformat()}
        ).eq("id", chat_id).execute()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not touch chat %s: %s", chat_id, exc)
