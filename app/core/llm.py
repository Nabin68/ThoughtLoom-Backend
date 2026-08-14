"""The model, behind an interface.

Every AI call in this service goes through [LanguageModel]. Swapping Cohere for
something else means adding one class here and changing [get_model] — no call
site moves, and no prompt changes.

The Cohere implementation is the original `ChatCohere` setup this service
started with, moved behind the interface rather than replaced.
"""

import json
import logging
import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import COHERE_MODEL

load_dotenv()

logger = logging.getLogger(__name__)


class ModelError(RuntimeError):
    """The model could not be reached, or would not answer usably."""


class LanguageModel(ABC):
    """What this service needs from an LLM. Deliberately small."""

    @abstractmethod
    def complete(self, *, system: str, user: str) -> str:
        """One turn in, raw text out."""

    def complete_json(self, *, system: str, user: str, attempts: int = 2) -> dict:
        """[complete], but insisting on a JSON object.

        Retries once by default. Models that drift out of JSON usually drift
        back on a second ask, and a retry is far cheaper than failing a user's
        request — but a model that has decided to write prose will do it twice,
        so the ceiling is low.
        """
        last_raw = ""
        for attempt in range(1, attempts + 1):
            last_raw = self.complete(system=system, user=user)
            try:
                return _extract_json(last_raw)
            except json.JSONDecodeError:
                logger.warning(
                    "Model returned non-JSON on attempt %d/%d: %r",
                    attempt,
                    attempts,
                    last_raw[:500],
                )
        raise ModelError(
            f"Model did not return valid JSON after {attempts} attempts"
        )


def _extract_json(raw: str) -> dict:
    """Parse a JSON object out of a reply, tolerating markdown fences.

    Same tolerance the original reasoning engine had: models fence their JSON
    about half the time however plainly you ask them not to.
    """
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise json.JSONDecodeError("No JSON object in response", raw, 0)
    parsed = json.loads(raw[start : end + 1])
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("Top-level JSON is not an object", raw, 0)
    return parsed


class CohereModel(LanguageModel):
    """Cohere via LangChain — what this service has always used."""

    def __init__(self, model: str = COHERE_MODEL):
        # Imported here rather than at module scope so that a test or a build
        # without the key can still import this module.
        from langchain_cohere import ChatCohere

        if not os.getenv("COHERE_API_KEY"):
            raise RuntimeError(
                "COHERE_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        self._llm = ChatCohere(model=model)

    def complete(self, *, system: str, user: str) -> str:
        try:
            response = self._llm.invoke(
                [SystemMessage(content=system), HumanMessage(content=user)]
            )
        except Exception as exc:  # noqa: BLE001 — provider errors are not a taxonomy
            raise ModelError(f"Model call failed: {exc}") from exc

        content = response.content
        # Some providers return a list of content blocks rather than a string.
        if isinstance(content, list):
            content = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        return str(content).strip()


_model: LanguageModel | None = None


def get_model() -> LanguageModel:
    """The process-wide model. Built on first use, not at import."""
    global _model
    if _model is None:
        _model = CohereModel()
    return _model


def set_model(model: LanguageModel | None) -> None:
    """Swap the model. For tests, and for wiring a different provider in."""
    global _model
    _model = model
