import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.adaptive import router as adaptive_router
from app.api.completion import router as completion_router
from app.api.recommendation import router as recommendation_router
from app.core.supabase_client import SupabaseError

load_dotenv()

logger = logging.getLogger(__name__)

@asynccontextmanager
async def _lifespan(_: FastAPI):
    """Start warming the moment the process is up. See [_warm_up] below."""
    threading.Thread(target=_warm_up, name="warm-up", daemon=True).start()
    yield


app = FastAPI(
    title="ThoughtLoom API",
    description="Weave clarity into your decisions",
    version="1.0.0",
    lifespan=_lifespan,
)

# --- waking up ------------------------------------------------------------
#
# On Render's free tier this process is not running until someone asks for it,
# and "booted" is not the same as "ready". The two expensive things here are
# built lazily on first use, by design — a build without the keys still has to
# import — but that means the *first real request* was paying for them:
#
#   `from langchain_cohere import ChatCohere` is seconds of import on its own,
#   and `create_client(...)` opens the Supabase session.
#
# The client pings /health the moment someone signs in, precisely so that this
# happens while they are answering the scripted questions rather than while
# they are watching a spinner. That only works if the ping warms the parts that
# are slow — which a handler returning a dict does not.
#
# So the work happens here, on a daemon thread at startup: the port binds
# immediately (Render forwards the waiting request as soon as it does), and the
# heavy construction overlaps with the user's first few taps.

_warm = False


def _warm_up() -> None:
    """Build the expensive singletons. Every failure is survivable.

    A missing key or an unreachable Supabase must not stop the service booting
    — the routes that need them already report that honestly on their own, and
    they are the same objects, built the same way, just sooner.
    """
    global _warm
    started = time.perf_counter()

    try:
        from app.core.llm import get_model

        get_model()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Warm-up could not build the model: %s", exc)

    try:
        from app.core.config import SUPABASE_CONFIGURED
        from app.core.supabase_client import service_client

        if SUPABASE_CONFIGURED:
            service_client()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Warm-up could not build the Supabase client: %s", exc)

    _warm = True
    logger.info("Warm-up finished in %.0f ms", (time.perf_counter() - started) * 1000)


@app.middleware("http")
async def _log_duration(request: Request, call_next):
    """How long each request took, end to end.

    The stage timings inside a handler only add up to something if there is a
    total to compare them against — and the gap between this number and the sum
    of the stages is where the time nobody has instrumented is hiding.
    """
    started = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - started) * 1000
    if request.url.path != "/health":
        logger.info(
            "%s %s -> %d in %.0f ms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
    return response


@app.exception_handler(SupabaseError)
async def _supabase_error(request: Request, exc: SupabaseError) -> JSONResponse:
    """The backstop for a SupabaseError raised before an endpoint's own try
    block — chiefly authorize_chat, called first in every endpoint so the rest
    of the handler can assume the chat is real and is the caller's. A
    malformed chat_id fails there with no friendly message of its own to fall
    back on; without this, FastAPI's default handling would turn it into a
    bare 500 instead of the same honest "try again" every other Supabase
    failure here gets.
    """
    logger.exception("Unhandled SupabaseError for %s", request.url.path)
    return JSONResponse(
        status_code=503,
        content={"detail": "Something went wrong on our side. Please try again."},
    )

# Comma-separated, e.g. ALLOWED_ORIGINS="https://thoughtloom.app,http://localhost:8080".
# The client sends no cookies or auth headers, so credentials stay off — which is also
# what makes the "*" default legal (browsers reject "*" alongside credentials).
allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    # Authorization joins the list because every endpoint below /api carries the
    # caller's Supabase access token. Without it a browser build would fail CORS
    # preflight before the request left.
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(adaptive_router, prefix="/api", tags=["Conversation"])
app.include_router(recommendation_router, prefix="/api", tags=["Conversation"])
app.include_router(completion_router, prefix="/api", tags=["Conversation"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ThoughtLoom API is running",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "adaptive_question": "/api/adaptive-question",
            "recommendation": "/api/recommendation",
            "follow_up": "/api/follow-up",
            "complete_chat": "/api/complete-chat",
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    """Health check, and the thing the client pings to wake this service.

    `warm` is the useful field: the process answers this the moment it has
    booted, but "warm" means the model client and the Supabase session are
    built and the next real request will not pay for them. A client that wants
    to know the backend is genuinely ready waits for that, not for a 200.
    """
    return {"status": "healthy", "service": "ThoughtLoom", "warm": _warm}
