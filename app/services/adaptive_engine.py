"""Generating the next question, and knowing when to stop."""

import logging
from dataclasses import dataclass

from app.core.config import MAX_ADAPTIVE_ROUNDS
from app.core.llm import ModelError, get_model
from app.core.supabase_client import insert_message
from app.prompts import adaptive_prompt
from app.services.context import ChatContext

logger = logging.getLogger(__name__)

# An option list longer than this is the model padding rather than thinking, and
# it turns a tap into a reading exercise.
MAX_OPTIONS = 6


@dataclass
class AdaptiveTurn:
    """What the client needs to render one step. [done] means stop asking."""

    done: bool
    round: int
    message_id: str | None = None
    question: str | None = None
    options: list[str] | None = None


def _clean_options(raw) -> list[str]:
    """Trust the model for wording, never for shape.

    Also drops any "other"/"something else" the model slipped in against
    instructions: the client always renders its own free-text escape hatch, and
    two of them side by side looks like a bug.
    """
    if not isinstance(raw, list):
        return []

    seen: set[str] = set()
    options: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        text = " ".join(item.split()).strip()
        if not text:
            continue
        lowered = text.lower().rstrip(".").strip()
        if lowered in ("other", "something else", "none of the above", "other (please specify)"):
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        options.append(text)
    return options[:MAX_OPTIONS]


def next_question(context: ChatContext) -> AdaptiveTurn:
    """Decide and persist the next question, or report that we have enough.

    The question row is written here, unanswered, and its id goes back to the
    client — which returns it with the answer. That keeps one row per turn and
    means an abandoned chat still shows what it was in the middle of asking.
    """
    asked = context.adaptive_rounds

    # A question is already on the table. Almost always a retry after a dropped
    # response — hand back the same one rather than asking something new and
    # leaving an orphan.
    pending = context.last_unanswered
    if pending is not None:
        return AdaptiveTurn(
            done=False,
            round=asked,
            message_id=pending["id"],
            question=pending.get("question_text"),
            options=(pending.get("metadata") or {}).get("options") or [],
        )

    remaining = MAX_ADAPTIVE_ROUNDS - asked
    if remaining <= 0:
        logger.info("Chat %s hit the round cap", context.chat["id"])
        return AdaptiveTurn(done=True, round=asked)

    result = get_model().complete_json(
        system=adaptive_prompt.SYSTEM,
        user=adaptive_prompt.USER.format(
            summary=context.summary(),
            rounds=asked,
            remaining=remaining,
        ),
    )

    if result.get("done") is True:
        logger.info(
            "Chat %s: model has enough after %d rounds (%s)",
            context.chat["id"],
            asked,
            result.get("reason", ""),
        )
        return AdaptiveTurn(done=True, round=asked)

    question = (result.get("question") or "").strip()
    options = _clean_options(result.get("options"))

    # A question with nothing to tap is not a question this UI can show. Rather
    # than invent options or crash the flow, treat it as the model having run
    # out of useful things to ask — the recommendation is the next step anyway,
    # and it sees everything this would have.
    if not question or len(options) < 2:
        logger.warning(
            "Chat %s: unusable question from model (q=%r, options=%r) — stopping",
            context.chat["id"],
            question[:120],
            options,
        )
        return AdaptiveTurn(done=True, round=asked)

    row = insert_message(
        chat_id=context.chat["id"],
        type="adaptive_question",
        question_text=question,
        answer_text=None,
        metadata={
            "options": options,
            "round": asked + 1,
            # Kept for debugging why the model asked what it asked. Never shown.
            "reason": (result.get("reason") or "")[:200],
        },
    )

    return AdaptiveTurn(
        done=False,
        round=asked + 1,
        message_id=row["id"],
        question=question,
        options=options,
    )


__all__ = ["AdaptiveTurn", "next_question", "ModelError"]
