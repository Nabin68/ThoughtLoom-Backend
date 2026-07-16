import logging

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from app.schemas.request_response import AnalyzeRequest, AnalyzeResponse
from app.services.reasoning_engine import InsightGenerationError, generate_insight

logger = logging.getLogger(__name__)

router = APIRouter()

UNAVAILABLE = "Could not generate insights right now. Please try again."


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Analyze user input and generate personalized insights.

    Returns structured JSON with summary and insights.
    """
    payload = {
        "reason": request.reason,
        "mcq_answers": request.mcq_answers,
        "additional_context": request.additional_context,
    }

    try:
        # generate_insight blocks on a network call to Cohere; keep it off the event loop.
        result = await run_in_threadpool(generate_insight, payload)
    except InsightGenerationError:
        logger.exception("Insight generation failed")
        raise HTTPException(status_code=502, detail=UNAVAILABLE)
    except Exception:
        logger.exception("Unexpected error while generating insights")
        raise HTTPException(status_code=500, detail="Something went wrong. Please try again.")

    try:
        return AnalyzeResponse(**result)
    except (ValidationError, TypeError):
        logger.exception("Model returned JSON with an unexpected shape: %r", result)
        raise HTTPException(status_code=502, detail=UNAVAILABLE)
