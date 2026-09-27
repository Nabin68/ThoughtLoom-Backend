from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


# ---------------------------------------------------------------------------
# Adaptive questioning
# ---------------------------------------------------------------------------

class AdaptiveAnswer(BaseModel):
    """The answer to the question the previous call handed back.

    [text] is the canonical answer whether one option was picked or several: a
    multi-select client sends the ticked options joined by "; ", in the order it
    displayed them. That keeps "what the user said" in one required field, so
    every reader of a chat — the transcript, the recommendation, memory — works
    on multi-select answers without knowing multi-select exists.
    """
    message_id: str = Field(..., description="id of the question row being answered")
    text: str = Field(..., min_length=1, description="A chosen option, or free text")

    @field_validator("text")
    @classmethod
    def _not_just_whitespace(cls, value: str) -> str:
        # min_length counts characters, not content — " " passes it. A blank
        # answer_text is falsy, which is exactly what the retry-idempotency in
        # adaptive_engine.last_unanswered checks for: a whitespace-only answer
        # would look identical to no answer at all, and the question would be
        # marked answered forever without ever having been.
        if not value.strip():
            raise ValueError("text must not be blank")
        return value

    selections: Optional[List[str]] = Field(
        default=None,
        description=(
            "The individual options ticked, when the question took several. "
            "Redundant with `text` by construction, and kept so a later reader "
            "does not have to infer where one choice ended and the next began."
        ),
    )


class AdaptiveQuestionRequest(BaseModel):
    chat_id: str = Field(..., description="The chat to continue")
    answer: Optional[AdaptiveAnswer] = Field(
        default=None,
        description="Omitted on the first call of a chat; present on every later one.",
    )


class AdaptiveQuestionResponse(BaseModel):
    done: bool = Field(..., description="True when the model has enough to advise")
    round: int = Field(..., description="How many questions have been asked")
    message_id: Optional[str] = Field(
        default=None, description="Send this back with the answer"
    )
    question: Optional[str] = None
    options: List[str] = Field(
        default_factory=list,
        description=(
            "Model-generated, specific to this user. The client always adds its "
            "own free-text fallback, so no 'other' option appears here."
        ),
    )
    multi: bool = Field(
        default=False,
        description=(
            "Whether this question takes more than one answer. Decided per "
            "question by the model: several, where the options are not mutually "
            "exclusive and forcing one would throw away most of the answer."
        ),
    )


# ---------------------------------------------------------------------------
# Recommendation and the conversation after it
# ---------------------------------------------------------------------------

class Source(BaseModel):
    title: str
    url: str


class RecommendationRequest(BaseModel):
    chat_id: str


class RecommendationResponse(BaseModel):
    headline: str = Field(
        default="",
        description=(
            "The verdict in one sentence, rendered large above the body. Empty "
            "when the model gave none — the body is what matters, and a missing "
            "headline is a worse-looking answer rather than no answer."
        ),
    )
    recommendation: str = Field(
        description=(
            "The body, in a restricted Markdown subset: **bold**, *italic*, "
            "## headings, - bullets, > callout. Nothing else renders."
        ),
    )
    next_steps: List[str] = Field(default_factory=list)
    confidence: str = ""
    sources: List[Source] = Field(
        default_factory=list, description="Empty when the answer needed no research"
    )
    message_id: Optional[str] = None


class FollowUpRequest(BaseModel):
    chat_id: str
    message: str = Field(..., min_length=1)


class FollowUpResponse(BaseModel):
    reply: str
    message_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Finishing a chat: the title, and what we learned from it
# ---------------------------------------------------------------------------

class CompleteChatRequest(BaseModel):
    chat_id: str = Field(..., description="The chat the user has just left")


class CompleteChatResponse(BaseModel):
    """Deliberately thin.

    The naming and the memory merge happen *after* this response is sent, so
    there is nothing here to report about them. The client has already left the
    screen; it is not waiting on a title.
    """
    status: str = Field(..., description="The chat's status now")
    scheduled: bool = Field(
        ...,
        description=(
            "Whether naming and memory extraction were queued. False when this "
            "chat has already been through both."
        ),
    )
