"""The recommendation: optionally research it, then take a position."""

import logging
from dataclasses import dataclass, field

from app.core.llm import get_model
from app.core.supabase_client import insert_message, set_chat_status
from app.core.web_search import SearchResult, get_search
from app.prompts import recommendation_prompt as prompts
from app.services.context import ChatContext

logger = logging.getLogger(__name__)

MAX_QUERIES = 3

# The chat is not over when the recommendation lands — the user can push back,
# and usually should. Only leaving the chat completes it, and that is the
# client's call to make.
STATUS_AWAITING_FOLLOW_UP = "awaiting_follow_up"


@dataclass
class Recommendation:
    text: str

    # The verdict in one sentence, rendered large above the body. Optional on
    # purpose — see [generate]: a missing headline is a plainer-looking answer,
    # and the answer is the thing worth having.
    headline: str = ""

    next_steps: list[str] = field(default_factory=list)
    confidence: str = ""
    sources: list[SearchResult] = field(default_factory=list)
    message_id: str | None = None


def _decide_searches(context: ChatContext) -> list[str]:
    """Ask the model whether this turns on real-world facts, and what to look up.

    Failing soft: if this step falls over, we simply do not search. Ungrounded
    advice is worse than grounded advice and much better than no advice.
    """
    try:
        result = get_model().complete_json(
            system=prompts.SEARCH_SYSTEM,
            user=prompts.SEARCH_USER.format(summary=context.summary()),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not decide on search, continuing without: %s", exc)
        return []

    if result.get("search") is not True:
        return []

    queries = result.get("queries")
    if not isinstance(queries, list):
        return []
    return [q.strip() for q in queries if isinstance(q, str) and q.strip()][:MAX_QUERIES]


def _research(context: ChatContext) -> list[SearchResult]:
    queries = _decide_searches(context)
    if not queries:
        logger.info("Chat %s: no search needed", context.chat["id"])
        return []

    search = get_search()
    results: list[SearchResult] = []
    seen: set[str] = set()
    for query in queries:
        for hit in search.search(query):
            if hit.url in seen:
                continue
            seen.add(hit.url)
            results.append(hit)

    logger.info(
        "Chat %s: searched %d queries, kept %d results",
        context.chat["id"],
        len(queries),
        len(results),
    )
    return results


def _render_research(results: list[SearchResult]) -> str:
    if not results:
        return ""
    body = "\n\n".join(
        f"[{i}] {hit.title}\n{hit.snippet}\nSource: {hit.url}"
        for i, hit in enumerate(results, start=1)
    )
    return prompts.RESEARCH_BLOCK.format(results=body)


def generate(context: ChatContext) -> Recommendation:
    """Research if needed, answer, then persist the answer and the chat's status."""
    sources = _research(context)

    result = get_model().complete_json(
        system=prompts.RECOMMENDATION_SYSTEM,
        user=prompts.RECOMMENDATION_USER.format(
            summary=context.summary(),
            research=_render_research(sources),
        ),
    )

    text = (result.get("recommendation") or "").strip()
    if not text:
        from app.core.llm import ModelError

        raise ModelError("Model returned an empty recommendation")

    # Unlike the body, a missing headline is not worth failing over: the answer
    # renders without one, and throwing away a good recommendation because its
    # title did not arrive would cost the user the only part they came for.
    raw_headline = result.get("headline")
    headline = raw_headline.strip() if isinstance(raw_headline, str) else ""
    if not headline:
        logger.info("Chat %s: no headline from the model", context.chat["id"])

    steps = [
        s.strip()
        for s in (result.get("next_steps") or [])
        if isinstance(s, str) and s.strip()
    ][:4]
    confidence = (result.get("confidence") or "").strip()

    row = insert_message(
        chat_id=context.chat["id"],
        type="recommendation",
        answer_text=text,
        metadata={
            # Persisted rather than derived, so that history can render the
            # verdict the same way the live screen did rather than guessing at
            # a first sentence.
            "headline": headline,
            "next_steps": steps,
            "confidence": confidence,
            # Kept so the client can show its work, and so a later reader can
            # tell whether the advice was grounded or from the model's memory.
            "sources": [
                {"title": s.title, "url": s.url} for s in sources
            ],
            "searched": bool(sources),
        },
    )

    set_chat_status(context.chat["id"], STATUS_AWAITING_FOLLOW_UP)

    return Recommendation(
        text=text,
        headline=headline,
        next_steps=steps,
        confidence=confidence,
        sources=sources,
        message_id=row["id"],
    )


def follow_up(context: ChatContext, message: str) -> dict:
    """One turn of the continued conversation.

    The user's message is persisted before the model is called, so a model
    failure loses the reply — which they can retry — rather than what they
    said, which they cannot get back.
    """
    insert_message(
        chat_id=context.chat["id"],
        type="free_text",
        answer_text=message,
        metadata={"input_method": "typed"},
    )

    reply = get_model().complete(
        system=prompts.FOLLOW_UP_SYSTEM,
        user=prompts.FOLLOW_UP_USER.format(
            summary=context.summary(),
            message=message,
        ),
    ).strip()

    if not reply:
        from app.core.llm import ModelError

        raise ModelError("Model returned an empty reply")

    row = insert_message(
        chat_id=context.chat["id"],
        type="assistant_reply",
        answer_text=reply,
    )
    return {"reply": reply, "message_id": row["id"]}
