import json
import logging

from app.core.llm import llm
from app.prompts.reasoning_prompt import PROMPT

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 2


class InsightGenerationError(RuntimeError):
    """Raised when the model does not return usable structured insights."""


def _extract_json(raw: str) -> dict:
    """Parse the model's reply, tolerating markdown fences around the JSON."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise json.JSONDecodeError("No JSON object in response", raw, 0)
    return json.loads(raw[start : end + 1])


def generate_insight(payload: dict) -> dict:
    """
    Generate structured insights from user input.
    Returns a dictionary with 'summary' and 'insights' keys.

    Raises InsightGenerationError if the model never returns valid JSON.
    """
    chain = PROMPT | llm

    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = chain.invoke(
            {
                "reason": payload["reason"],
                "mcq_answers": payload["mcq_answers"],
                "additional_context": payload["additional_context"],
            }
        )
        raw_content = response.content.strip()

        try:
            return _extract_json(raw_content)
        except json.JSONDecodeError:
            logger.warning(
                "Model returned non-JSON on attempt %d/%d: %r",
                attempt,
                MAX_ATTEMPTS,
                raw_content[:500],
            )

    raise InsightGenerationError(
        f"Model did not return valid JSON after {MAX_ATTEMPTS} attempts"
    )
