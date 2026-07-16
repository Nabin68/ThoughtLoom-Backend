"""Naming a finished chat, so history is a list of conversations rather than a
list of the word "Education"."""

import logging
from datetime import datetime

from app.core.llm import get_model
from app.core.supabase_client import set_chat_title
from app.prompts import title_prompt
from app.services.context import ChatContext

logger = logging.getLogger(__name__)

# The phrase the model writes, before the date is added. Long enough for "Whether
# to tell my father about the Bangalore offer", short enough to survive one line
# of a phone screen next to a date.
MAX_PHRASE = 64


def _clean(raw) -> str:
    """Trust the model for the words, never for the shape."""
    if not isinstance(raw, str):
        return ""
    # Models wrap titles in quotes about half the time, whatever you ask.
    text = " ".join(raw.split()).strip().strip("\"'").rstrip(".").strip()
    if len(text) > MAX_PHRASE:
        text = text[:MAX_PHRASE].rsplit(" ", 1)[0] + "..."
    return text


def _month(chat: dict) -> str:
    """The chat's own month, from its created_at.

    Appended here rather than asked of the model, which does not know what day
    it is and will confidently write the wrong year onto a conversation from
    this morning.
    """
    raw = chat.get("created_at") or ""
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).strftime("%B %Y")
    except ValueError:
        logger.warning("Chat %s has an unreadable created_at: %r", chat.get("id"), raw)
        return ""


def generate_title(context: ChatContext) -> str | None:
    """Name the chat and write it back. None when there is nothing to name.

    A chat with no messages is not named: the dashboard opens a chat row the
    moment a category is tapped, so an abandoned tap is a real row with nothing
    in it, and "Untitled — July 2026" is worse than the honest "Unfinished" the
    history screen already shows.
    """
    if not context.messages:
        logger.info("Chat %s has nothing in it to name", context.chat["id"])
        return None

    result = get_model().complete_json(
        system=title_prompt.SYSTEM,
        user=title_prompt.USER.format(summary=context.summary()),
    )

    phrase = _clean(result.get("title"))
    if not phrase:
        # The model was asked to return null when a chat is too thin to name,
        # and this is that answer. Not an error.
        logger.info("Chat %s: model declined to name it", context.chat["id"])
        return None

    month = _month(context.chat)
    title = f"{phrase} — {month}" if month else phrase

    set_chat_title(context.chat["id"], title)
    logger.info("Chat %s titled %r", context.chat["id"], title)
    return title
