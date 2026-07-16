"""What we know about a person, updated one finished conversation at a time.

### The write this module makes is the one chat ownership does not authorise

Everything else in this service is scoped to a chat: prove the chat is yours and
the service-role key may touch it. `user_memory` is scoped to a *person*. A bug
that let the wrong id through here would not leak one conversation — it would
graft a stranger's life onto someone's permanent record, and every future chat
would be answered out of it.

So [merge_from_chat] takes the caller's own id, proven from their token, as a
required argument, and refuses to run if the chat it was handed does not belong
to that same person. The check is redundant today — app/core/auth.py already
proved exactly that — and it stays anyway, because "redundant" here means "the
second of two independent reasons this cannot go wrong", and the cost of the
first one ever being wrong is unrecoverable.
"""

import logging

from app.core.config import MEMORY_MAX_FACTS
from app.core.llm import get_model
from app.core.supabase_client import (
    fetch_memory_rows,
    mark_memory_merged,
    upsert_memory,
)
from app.prompts import memory_prompt
from app.services.context import ChatContext

logger = logging.getLogger(__name__)


class MemoryOwnershipError(RuntimeError):
    """A chat was about to be folded into someone else's memory."""


def _existing(rows: list[dict], category: str | None) -> dict | None:
    for row in rows:
        if row.get("category") == category:
            return row
    return None


def _render(row: dict | None) -> str:
    """What we already know, in the shape the prompt reads it back in."""
    if not row:
        return memory_prompt.NOTHING_YET

    lines = [
        f"- {fact.strip()}"
        for fact in (row.get("facts") or [])
        if isinstance(fact, str) and fact.strip()
    ]
    summary = (row.get("summary") or "").strip()
    if summary:
        lines.insert(0, f"In summary: {summary}")
    return "\n".join(lines) if lines else memory_prompt.NOTHING_YET


def _clean_facts(raw) -> list[str]:
    if not isinstance(raw, list):
        return []
    seen: set[str] = set()
    facts: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        text = " ".join(item.split()).strip()
        if not text:
            continue
        key = text.lower().rstrip(".")
        if key in seen:
            continue
        seen.add(key)
        facts.append(text)
    return facts[:MEMORY_MAX_FACTS]


def _merged(section, existing: dict | None) -> tuple[str, list[str]] | None:
    """The updated memory, or None to leave the row alone.

    The guard here is against one specific failure: a generation that comes back
    empty or malformed for a person who already had memory. The merge asks the
    model to return everything it still believes, so "no facts" is
    indistinguishable from "I lost them" — and the two have very different
    costs. Keeping a stale memory loses one conversation's worth of learning;
    accepting an empty one silently erases every conversation before it.
    """
    if not isinstance(section, dict):
        section = {}

    facts = _clean_facts(section.get("facts"))
    summary = " ".join((section.get("summary") or "").split()).strip()

    had_something = bool(existing) and bool(
        (existing.get("facts") or []) or (existing.get("summary") or "").strip()
    )
    if not facts and not summary:
        if had_something:
            logger.warning("Memory merge came back empty for a row that had facts — keeping what we had")
        return None

    return summary, facts


def merge_from_chat(context: ChatContext, *, user_id: str) -> bool:
    """Fold one finished chat into this user's memory. True if anything changed.

    [user_id] is the caller's own, from their token. See the module docstring.
    """
    owner = context.chat.get("user_id")
    if owner != user_id:
        raise MemoryOwnershipError(
            f"Refusing to merge chat {context.chat.get('id')} (owned by {owner}) "
            f"into the memory of {user_id}"
        )

    if not context.messages:
        logger.info("Chat %s has nothing in it to remember", context.chat["id"])
        return False

    category = context.category
    rows = fetch_memory_rows(user_id)
    global_row = _existing(rows, None)
    topic_row = _existing(rows, category)

    result = get_model().complete_json(
        system=memory_prompt.SYSTEM.format(category=category),
        user=memory_prompt.USER.format(
            global_memory=_render(global_row),
            topic_memory=_render(topic_row),
            category_upper=category.upper(),
            summary=context.summary(),
        ),
    )

    changed = False
    for section_key, scope, existing in (
        ("global", None, global_row),
        ("topic", category, topic_row),
    ):
        merged = _merged(result.get(section_key), existing)
        if merged is None:
            continue
        summary, facts = merged
        upsert_memory(user_id=user_id, category=scope, summary=summary, facts=facts)
        changed = True
        logger.info(
            "Chat %s: merged %d facts into %s memory",
            context.chat["id"],
            len(facts),
            scope or "global",
        )

    # Marked whatever the model produced. A merge that legitimately learned
    # nothing is finished with this chat, and re-running it on every visit to
    # history would pay for the same answer forever.
    mark_memory_merged(context.chat["id"])
    return changed
