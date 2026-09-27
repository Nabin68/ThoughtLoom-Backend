"""The recommendation: optionally research it, then take a position."""

import logging
from dataclasses import dataclass, field

from app.core.concurrency import gather
from app.core.llm import get_model
from app.core.supabase_client import insert_message, set_chat_status
from app.core.timing import timed
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
        with timed("recommendation: search decision"):
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
    # Three lookups against an external search engine, each a second or two of
    # pure waiting and none of them depending on the others. Run in a loop they
    # were the second-largest thing on this endpoint after the generation
    # itself; run together they cost one lookup.
    with timed(f"recommendation: {len(queries)} searches"):
        # A default argument, not a closure over `query` — a lambda capturing
        # the loop variable would have every thread search the last query.
        per_query = gather(*[(lambda q=q: search.search(q)) for q in queries])

    # Flattened in query order, so the same queries always produce the same
    # prompt whatever order the threads happened to finish in.
    results: list[SearchResult] = []
    seen: set[str] = set()
    for hits in per_query:
        for hit in hits:
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


def _existing_recommendation(context: ChatContext) -> Recommendation | None:
    """The recommendation already on this chat, if a retry finds one waiting.

    A client retry after a dropped response — Render's cold start plus a long
    generation is exactly the kind of call that can time out on the client
    side after the server has already finished — would otherwise run the whole
    research-and-generate pipeline again and insert a second, possibly
    conflicting piece of advice into the same chat.
    """
    for message in context.messages:
        if message.get("type") != "recommendation":
            continue
        metadata = message.get("metadata") or {}
        return Recommendation(
            text=message.get("answer_text") or "",
            headline=metadata.get("headline") or "",
            next_steps=metadata.get("next_steps") or [],
            confidence=metadata.get("confidence") or "",
            sources=[
                SearchResult(title=s.get("title") or "", snippet="", url=s.get("url") or "")
                for s in (metadata.get("sources") or [])
                if s.get("url")
            ],
            message_id=message.get("id"),
        )
    return None


def generate(context: ChatContext) -> Recommendation:
    """Research if needed, answer, then persist the answer and the chat's status."""
    existing = _existing_recommendation(context)
    if existing is not None:
        return existing

    sources = _research(context)

    with timed("recommendation: model"):
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


def _already_recorded(context: ChatContext, message: str) -> bool:
    """Whether [message] looks like it was already written and is only
    waiting on a reply.

    A dropped response after the user's turn was saved and before the model
    replied looks, on retry, identical to a first attempt: same chat, same
    text, sent again. The last message being that same free_text with nothing
    after it is the signal — anything else (a reply already came back, or the
    chat's last turn was something else entirely) means this is a genuinely
    new message and must be written.
    """
    if not context.messages:
        return False
    last = context.messages[-1]
    return (
        last.get("type") == "free_text"
        and (last.get("answer_text") or "") == message
    )


def follow_up(context: ChatContext, message: str) -> dict:
    """One turn of the continued conversation.

    The user's message is persisted before the model is called, so a model
    failure loses the reply — which they can retry — rather than what they
    said, which they cannot get back. [_already_recorded] is what keeps that
    retry from writing the same turn twice.
    """
    if not _already_recorded(context, message):
        insert_message(
            chat_id=context.chat["id"],
            type="free_text",
            answer_text=message,
            metadata={"input_method": "typed"},
        )

    with timed("follow-up: model"):
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
