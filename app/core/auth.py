"""Who is calling, and may they touch this chat.

### Why this exists

Every endpoint here reads and writes with the service-role key, which bypasses
Row Level Security. In the Flutter client, RLS *is* the authorisation model —
the policies in schema.sql are what stop one user reading another's rows. The
moment this service uses the service-role key, that protection is gone, and
nothing replaces it unless something here does.

Without this module, `POST /api/adaptive-question {"chat_id": "<anyone's>"}`
would hand back a stranger's situation — the most private data in the product —
to any caller who could guess a uuid. So: the client sends its Supabase access
token, this proves the token is real, and the chat's `user_id` must match its
subject. Same rule the RLS policies enforce, restated where the database can no
longer enforce it.
"""

import logging
import os
import time

from fastapi import Header, HTTPException

from app.core.supabase_client import SupabaseError, fetch_chat, service_client

logger = logging.getLogger(__name__)

_UNAUTHORIZED = "Please sign in again."
_FORBIDDEN = "That conversation is not yours."

# How long a verified token is trusted without asking Supabase again.
#
# Verifying costs a network round trip to Supabase Auth, and it happens on
# *every* request — in front of the question the user is waiting on. The same
# token arrives a dozen times in a single conversation and the answer is the
# same every time.
#
# The trade is revocation latency: for up to this long, a user who signed out
# on another device can still reach this service. A minute is short against a
# session that lasts hours and long enough to take the round trip off nearly
# every request. Set AUTH_CACHE_SECONDS=0 to go back to verifying every time.
_CACHE_SECONDS = float(os.getenv("AUTH_CACHE_SECONDS", "60"))

# ponytail: a plain dict with a size cap, not an LRU. One process, one small
# user base; if this ever runs at a size where the cap starts evicting live
# sessions, swap it for cachetools.TTLCache rather than growing this.
_CACHE_MAX = 512
_verified: dict[str, tuple[float, str]] = {}


def _cached_user(token: str) -> str | None:
    entry = _verified.get(token)
    if entry is None:
        return None
    expires_at, user_id = entry
    if expires_at <= time.monotonic():
        _verified.pop(token, None)
        return None
    return user_id


def _remember(token: str, user_id: str) -> None:
    if _CACHE_SECONDS <= 0:
        return
    now = time.monotonic()
    if len(_verified) >= _CACHE_MAX:
        for stale, (expires_at, _) in list(_verified.items()):
            if expires_at <= now:
                _verified.pop(stale, None)
        if len(_verified) >= _CACHE_MAX:
            _verified.clear()
    _verified[token] = (now + _CACHE_SECONDS, user_id)


def forget_tokens() -> None:
    """Drop every cached verification. For tests."""
    _verified.clear()


def _bearer(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail=_UNAUTHORIZED)
    token = authorization[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail=_UNAUTHORIZED)
    return token


def current_user_id(authorization: str | None = Header(default=None)) -> str:
    """The signed-in user, from their Supabase access token.

    Verified by asking Supabase rather than by checking a signature locally.
    That needs no JWT secret in this service's environment and it honours
    revocation, where a locally-verified token would keep passing until it
    expired.

    The answer is cached for [_CACHE_SECONDS], because that round trip was
    being paid on every single request and the same token asks the same
    question all conversation long. Revocation is then honoured within a
    minute rather than instantly — see the note on the constant.
    """
    token = _bearer(authorization)

    cached = _cached_user(token)
    if cached is not None:
        return cached

    try:
        response = service_client().auth.get_user(token)
    except SupabaseError:
        raise
    except Exception as exc:  # noqa: BLE001 — an invalid token throws, variously
        logger.info("Rejected a token: %s", exc)
        raise HTTPException(status_code=401, detail=_UNAUTHORIZED) from exc

    user = getattr(response, "user", None)
    if user is None or not getattr(user, "id", None):
        raise HTTPException(status_code=401, detail=_UNAUTHORIZED)
    _remember(token, user.id)
    return user.id


def authorize_chat(chat_id: str, user_id: str) -> dict:
    """The chat, if it is this user's. Otherwise the request ends here.

    404 and 403 are deliberately not distinguished: telling an attacker that a
    chat id exists but belongs to someone else is a fact they should not get.
    """
    chat = fetch_chat(chat_id)
    if chat is None or chat.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail=_FORBIDDEN)
    return chat
