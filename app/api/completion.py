"""POST /api/complete-chat — the user left, so name it and remember it.

### Why the work happens after the response

Naming a chat and folding it into memory are two model calls. The client calls
this at the exact moment the user pressed Back, and the user is *leaving* — they
will not wait twenty seconds to find out what we decided to call something they
have stopped looking at.

So the response carries only the status write, and the two model calls run in a
background task afterwards. That is not just a latency trick: once the request
has reached this service, the work happens whether or not the app survives the
next second. Firing the model calls from the client and hoping the process lives
long enough would lose the memory of every chat a user ended by closing the app.
"""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from app.core.auth import authorize_chat, current_user_id
from app.core.supabase_client import SupabaseError, set_chat_status
from app.schemas.request_response import CompleteChatRequest, CompleteChatResponse
from app.services.context import load_context
from app.services.memory import merge_from_chat
from app.services.titling import generate_title

logger = logging.getLogger(__name__)

router = APIRouter()

UNAVAILABLE = "Could not close that conversation just now."

STATUS_COMPLETED = "completed"


def _wrap_up(chat: dict, user_id: str) -> None:
    """Name the chat, then remember it. Runs after the response has gone.

    The two halves are independent and fail independently: a model that cannot
    write a title has no bearing on whether we can learn from the conversation,
    and losing both because one of them threw would be a worse trade than losing
    either. Nothing here is retried — this is best-effort work on behalf of a
    user who has already walked away, and the client re-requests it if the title
    never appeared.
    """
    try:
        context = load_context(chat, recall=False)
    except Exception:
        logger.exception("Could not load chat %s to wrap it up", chat.get("id"))
        return

    if not chat.get("title"):
        try:
            generate_title(context)
        except Exception:
            logger.exception("Could not name chat %s", chat.get("id"))

    if not chat.get("memory_merged_at"):
        try:
            merge_from_chat(context, user_id=user_id)
        except Exception:
            logger.exception("Could not fold chat %s into memory", chat.get("id"))


@router.post("/complete-chat", response_model=CompleteChatResponse)
async def complete_chat(
    request: CompleteChatRequest,
    background: BackgroundTasks,
    user_id: str = Depends(current_user_id),
) -> CompleteChatResponse:
    """Close a chat: mark it completed, then name it and learn from it.

    Safe to call more than once, which it will be — the client fires this when
    the user leaves and the history screen asks again for any completed chat
    that never got a title. Titling is skipped for a chat that has one, and the
    memory merge for a chat already folded in, so a repeat call is a status
    write and nothing more.
    """
    chat = await run_in_threadpool(authorize_chat, request.chat_id, user_id)

    try:
        # The client sets this too, directly, under RLS — this is the guarantee
        # for the case where that write never landed. It is an idempotent write
        # of a value that only moves one way.
        if chat.get("status") != STATUS_COMPLETED:
            await run_in_threadpool(set_chat_status, chat["id"], STATUS_COMPLETED)
    except SupabaseError:
        logger.exception("Could not complete chat %s", request.chat_id)
        raise HTTPException(status_code=503, detail=UNAVAILABLE)

    outstanding = not chat.get("title") or not chat.get("memory_merged_at")
    if outstanding:
        # user_id is the *verified caller*, not chat["user_id"]. They are the
        # same value — authorize_chat just proved it — and this is the one that
        # is provably the person whose memory is about to be written. See
        # app/services/memory.py.
        background.add_task(_wrap_up, chat, user_id)

    return CompleteChatResponse(status=STATUS_COMPLETED, scheduled=outstanding)
