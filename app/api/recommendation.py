"""POST /api/recommendation and POST /api/follow-up."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from app.core.auth import authorize_chat, current_user_id
from app.core.llm import ModelError
from app.core.supabase_client import SupabaseError
from app.schemas.request_response import (
    FollowUpRequest,
    FollowUpResponse,
    RecommendationRequest,
    RecommendationResponse,
    Source,
)
from app.services.context import load_context
from app.services.recommendation_engine import follow_up, generate

logger = logging.getLogger(__name__)

router = APIRouter()

UNAVAILABLE = "Could not work out an answer just now. Please try again."
FOLLOW_UP_UNAVAILABLE = "Could not reply just now. Please try again."


@router.post("/recommendation", response_model=RecommendationResponse)
async def recommendation(
    request: RecommendationRequest,
    user_id: str = Depends(current_user_id),
) -> RecommendationResponse:
    """The actual advice, researched first where the question turns on facts.

    Slow by nature: a search decision, up to three searches, and a long
    generation. The client is built to wait, and Render's free tier may add a
    cold start on top.
    """
    chat = await run_in_threadpool(authorize_chat, request.chat_id, user_id)

    try:
        context = await run_in_threadpool(load_context, chat)
        result = await run_in_threadpool(generate, context)
    except SupabaseError:
        logger.exception("Supabase failed while recommending")
        raise HTTPException(status_code=503, detail=UNAVAILABLE)
    except ModelError:
        logger.exception("Model failed while recommending")
        raise HTTPException(status_code=502, detail=UNAVAILABLE)
    except Exception:
        logger.exception("Unexpected error while recommending")
        raise HTTPException(status_code=500, detail=UNAVAILABLE)

    return RecommendationResponse(
        headline=result.headline,
        recommendation=result.text,
        next_steps=result.next_steps,
        confidence=result.confidence,
        sources=[Source(title=s.title or s.url, url=s.url) for s in result.sources],
        message_id=result.message_id,
    )


@router.post("/follow-up", response_model=FollowUpResponse)
async def follow_up_route(
    request: FollowUpRequest,
    user_id: str = Depends(current_user_id),
) -> FollowUpResponse:
    """One more turn of the conversation, with the whole chat as context."""
    chat = await run_in_threadpool(authorize_chat, request.chat_id, user_id)

    try:
        context = await run_in_threadpool(load_context, chat)
        result = await run_in_threadpool(follow_up, context, request.message.strip())
    except SupabaseError:
        logger.exception("Supabase failed during follow-up")
        raise HTTPException(status_code=503, detail=FOLLOW_UP_UNAVAILABLE)
    except ModelError:
        logger.exception("Model failed during follow-up")
        raise HTTPException(status_code=502, detail=FOLLOW_UP_UNAVAILABLE)
    except Exception:
        logger.exception("Unexpected error during follow-up")
        raise HTTPException(status_code=500, detail=FOLLOW_UP_UNAVAILABLE)

    return FollowUpResponse(reply=result["reply"], message_id=result["message_id"])
