"""POST /api/adaptive-question — the next generated question, or "enough"."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from app.core.auth import authorize_chat, current_user_id
from app.core.llm import ModelError
from app.core.supabase_client import SupabaseError, answer_message
from app.schemas.request_response import (
    AdaptiveQuestionRequest,
    AdaptiveQuestionResponse,
)
from app.services.adaptive_engine import next_question
from app.services.context import load_context

logger = logging.getLogger(__name__)

router = APIRouter()

UNAVAILABLE = "Could not think of a follow-up just now. Please try again."


@router.post("/adaptive-question", response_model=AdaptiveQuestionResponse)
async def adaptive_question(
    request: AdaptiveQuestionRequest,
    user_id: str = Depends(current_user_id),
) -> AdaptiveQuestionResponse:
    """Record the last answer, then decide what to ask next.

    One endpoint for both halves of the loop: the client posts an answer and
    gets the next question in the same round trip, so a step is one request
    rather than two that can half-succeed.

    The first call of a chat omits `answer`.
    """
    chat = await run_in_threadpool(authorize_chat, request.chat_id, user_id)

    try:
        if request.answer is not None:
            # Written before the model is called, so a model failure costs the
            # user a retry rather than the answer they just gave.
            await run_in_threadpool(
                answer_message,
                request.answer.message_id,
                request.answer.text,
                selections=request.answer.selections,
            )

        context = await run_in_threadpool(load_context, chat)
        # Blocks on Cohere; keep it off the event loop.
        turn = await run_in_threadpool(next_question, context)
    except SupabaseError:
        logger.exception("Supabase failed during adaptive questioning")
        raise HTTPException(status_code=503, detail=UNAVAILABLE)
    except ModelError:
        logger.exception("Model failed during adaptive questioning")
        raise HTTPException(status_code=502, detail=UNAVAILABLE)
    except Exception:
        logger.exception("Unexpected error during adaptive questioning")
        raise HTTPException(status_code=500, detail=UNAVAILABLE)

    return AdaptiveQuestionResponse(
        done=turn.done,
        round=turn.round,
        message_id=turn.message_id,
        question=turn.question,
        options=turn.options or [],
        multi=turn.multi,
    )
