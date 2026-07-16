"""Which earlier conversations connect to this one.

Deliberately lightweight: no embeddings, no vector column, no second service.
Two signals, both of which a person would use —

  1. Is it the same part of their life? (category)
  2. Do the same words come up? (a keyword match against what was actually said)

— scored, ranked, and cut to the best three. That is enough to say "you were
weighing this same thing in March" without pretending to a similarity search
this product does not need and would have to keep in sync.

This runs in front of a user waiting on a question, so it is capped at two
queries and fails soft to nothing. Starting cold is the product working slightly
worse; a spinner that never resolves is the product broken.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime

from app.core.supabase_client import (
    fetch_chats_mentioning,
    fetch_past_chats,
    fetch_recommendations,
)

logger = logging.getLogger(__name__)

# How many past chats reach the prompt. Three is enough to establish "we have
# history"; more and the model starts writing about their archive rather than
# their question.
MAX_RELATED = 3

# Candidates considered before scoring. A user with hundreds of chats gets their
# most recent twenty looked at, which is where any live thread will be.
MAX_CANDIDATES = 20

# Keywords are the crude half of this, so they are held to a high bar: long
# enough to be a topic rather than a connective, and capped so one rambling
# description cannot turn the filter into "match anything".
MIN_KEYWORD_LENGTH = 5
MAX_KEYWORDS = 8

# Words that clear the length bar but say nothing about a topic. Everything a
# person writes about a hard decision is "really about" something, "thinking"
# something, "wanting" something — matching on those would connect every chat to
# every other chat.
_STOPWORDS = frozenset(
    """
    about above after again against because been before being below between both
    could didn doesn during each further having himself herself hasn haven itself
    might mustn myself other ought ourselves shan should shouldn since some such
    than that their theirs them themselves then there these they this those
    through under until very were what when where which while whom will with
    would your yours yourself yourselves
    actually always anyone anything basically better cannot
    completely constantly currently definitely doing does done enough especially
    every everyone everything feel feeling felt getting going gone honestly
    instead just keep keeps kind know known later lately like liked likely
    little maybe mean means meant much need needed needs never nothing often
    only over pretty probably quite rather really right same seem seems since
    something sometimes soon still stuck sure take taken taking talk talked
    tell telling thing things think thinking thought time times told trying
    understand until using usually want wanted wants whether whole
    """.split()
)


@dataclass(frozen=True)
class RelatedChat:
    """One past conversation, as the prompt will see it."""

    chat_id: str
    title: str
    category: str
    when: str
    gist: str

    def render(self) -> str:
        line = f'- "{self.title}" ({self.category}, {self.when})'
        if self.gist:
            line += f"\n  What you told them: {self.gist}"
        return line


def keywords_from(text: str) -> list[str]:
    """The words worth matching a past conversation on.

    Only [a-z]+ survives, which is also what makes these safe to interpolate
    into a PostgREST filter — see fetch_chats_mentioning.
    """
    words = re.findall(r"[a-zA-Z]+", text.lower())
    seen: set[str] = set()
    keywords: list[str] = []
    for word in words:
        if len(word) < MIN_KEYWORD_LENGTH or word in _STOPWORDS or word in seen:
            continue
        seen.add(word)
        keywords.append(word)
        if len(keywords) == MAX_KEYWORDS:
            break
    return keywords


def _when(chat: dict) -> str:
    """"July 2026". Coarse on purpose — the point is "a while back", not a
    timestamp, and a model given an exact date will quote it at the user."""
    raw = chat.get("created_at") or chat.get("updated_at") or ""
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).strftime(
            "%B %Y"
        )
    except ValueError:
        return "earlier"


def _gist(text: str, limit: int = 240) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def find_related(
    *,
    user_id: str,
    chat_id: str,
    category: str,
    keywords: list[str],
    limit: int = MAX_RELATED,
) -> list[RelatedChat]:
    """The past chats worth mentioning in this one. Empty for a new user."""
    candidates = fetch_past_chats(user_id, exclude_chat_id=chat_id, limit=MAX_CANDIDATES)
    if not candidates:
        return []

    ids = [c["id"] for c in candidates]
    mentioning = fetch_chats_mentioning(ids, keywords)

    scored: list[tuple[int, dict]] = []
    for chat in candidates:
        score = 0
        if chat.get("category") == category:
            score += 2
        if chat["id"] in mentioning:
            score += 2
        # A title is a summary of the whole chat, so a hit there is worth as
        # much as one in the body — and it costs no query, since the titles are
        # already in hand.
        title = (chat.get("title") or "").lower()
        if any(k in title for k in keywords):
            score += 2
        if score >= 2:
            scored.append((score, chat))

    # Ties break on recency: fetch_past_chats already ordered them, and sort is
    # stable, so the newer of two equally-connected chats stays in front.
    scored.sort(key=lambda pair: pair[0], reverse=True)
    best = [chat for _, chat in scored[:limit]]
    if not best:
        return []

    gists = fetch_recommendations([c["id"] for c in best])

    related = [
        RelatedChat(
            chat_id=chat["id"],
            # An untitled chat still gets recalled — its titling may simply not
            # have landed yet — but it has to be called something.
            title=(chat.get("title") or "").strip() or f"An earlier {chat.get('category', 'chat')} conversation",
            category=str(chat.get("category", "other")),
            when=_when(chat),
            gist=_gist(gists.get(chat["id"], "")),
        )
        for chat in best
    ]
    logger.info(
        "Chat %s: recalled %d of %d past chats", chat_id, len(related), len(candidates)
    )
    return related
