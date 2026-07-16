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

from fastapi import Header, HTTPException

from app.core.supabase_client import SupabaseError, fetch_chat, service_client

logger = logging.getLogger(__name__)

_UNAUTHORIZED = "Please sign in again."
_FORBIDDEN = "That conversation is not yours."


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
    That is a network round trip, but it needs no JWT secret in this service's
    environment and it honours revocation — a signed-out or deleted user stops
    working immediately, where a locally-verified token would keep passing
    until it expired.
    """
    token = _bearer(authorization)
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
