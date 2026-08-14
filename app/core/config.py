"""Environment configuration, read once at import."""

import os

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in."
        )
    return value


# --- Supabase -------------------------------------------------------------
#
# The service-role key bypasses Row Level Security, which is the whole point:
# this service reads a chat's context and writes the model's turns back, and it
# is not acting as any one user when it does. It must never reach the client —
# it is a master key to every row in the database.
#
# Ownership is therefore checked here rather than by the database. See
# app/core/auth.py: every endpoint takes the caller's Supabase JWT and proves
# the chat is theirs before this key touches anything.

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()

SUPABASE_CONFIGURED = bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)


def require_supabase() -> tuple[str, str]:
    """Fail loudly, and only when a route that needs Supabase is actually hit.

    Not at import: /health works without it, and a deployment that has not
    been given the keys yet should still boot far enough to say so.
    """
    return _require("SUPABASE_URL"), _require("SUPABASE_SERVICE_ROLE_KEY")


# --- Model ----------------------------------------------------------------

COHERE_MODEL = os.getenv("COHERE_MODEL", "command-r-plus-08-2024").strip()

# --- Web search -----------------------------------------------------------

WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", "true").strip().lower() not in (
    "false",
    "0",
    "no",
)

# How many results to pull per query. Enough to triangulate a fact, few enough
# that the snippets do not crowd out the user's own situation in the prompt.
WEB_SEARCH_RESULTS = int(os.getenv("WEB_SEARCH_RESULTS", "4"))

# --- Adaptive questioning -------------------------------------------------

# Hard ceiling on generated questions per chat. The model is asked to stop as
# soon as it has enough and usually does; this only catches a model that would
# happily interview someone forever.
MAX_ADAPTIVE_ROUNDS = int(os.getenv("MAX_ADAPTIVE_ROUNDS", "8"))

# --- Long-term memory -----------------------------------------------------

# Facts kept per user_memory row. Memory grows for the life of an account and is
# read in front of every generated question, so it needs a ceiling somewhere.
#
# The number is a backstop, not a target: the prompt's own bar — "would knowing
# this make a conversation next year better?" — is what actually keeps this
# small, and a person whose life needs thirty separate durable facts to describe
# is rare. This only catches a model that has decided to transcribe.
MEMORY_MAX_FACTS = int(os.getenv("MEMORY_MAX_FACTS", "30"))
