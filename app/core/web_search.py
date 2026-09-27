"""Web search, behind an interface.

An *optional* capability: the recommendation call decides for itself whether the
question turns on current real-world facts, and only searches if so. "Should I
tell my father I'm unhappy" needs no search. "Is the 2026 intake for X still
open in Pune" does.

DuckDuckGo is the default because it needs no API key and no billing account —
the same reasoning that put Supabase and Cohere in this stack. It is also the
least reliable link in it: no SLA, and it rate-limits. Everything here therefore
fails soft. A search that returns nothing produces advice grounded only in what
the user said, which is the product working slightly worse — not failing.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.core.config import WEB_SEARCH_RESULTS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    title: str
    snippet: str
    url: str


class WebSearch(ABC):
    @abstractmethod
    def search(self, query: str, *, limit: int = WEB_SEARCH_RESULTS) -> list[SearchResult]:
        """Best-effort. Returns [] rather than raising."""


class DuckDuckGoSearch(WebSearch):
    def search(self, query: str, *, limit: int = WEB_SEARCH_RESULTS) -> list[SearchResult]:
        # The whole thing is one fail-soft block, parsing included: a hit whose
        # shape has drifted (a library update, an ad slot with no "body") must
        # cost this one query, not turn a soft "search found nothing" into a
        # hard failure of the recommendation it was only ever grounding.
        try:
            # Imported lazily: the package is optional, and this service must
            # still boot and answer without it.
            from ddgs import DDGS

            with DDGS() as ddgs:
                hits = list(ddgs.text(query, max_results=limit))

            results = []
            for hit in hits:
                title = (hit.get("title") or "").strip()
                snippet = (hit.get("body") or "").strip()
                url = (hit.get("href") or hit.get("url") or "").strip()
                if snippet and url:
                    results.append(SearchResult(title=title, snippet=snippet, url=url))
            return results
        except Exception as exc:  # noqa: BLE001 — rate limits, network, import, shape drift
            logger.warning("Web search failed for %r: %s", query, exc)
            return []


class NullSearch(WebSearch):
    """Search turned off. What tests use, and what WEB_SEARCH_ENABLED=false gets."""

    def search(self, query: str, *, limit: int = WEB_SEARCH_RESULTS) -> list[SearchResult]:
        return []


_search: WebSearch | None = None


def get_search() -> WebSearch:
    global _search
    if _search is None:
        from app.core.config import WEB_SEARCH_ENABLED

        _search = DuckDuckGoSearch() if WEB_SEARCH_ENABLED else NullSearch()
    return _search


def set_search(search: WebSearch | None) -> None:
    """Swap the provider. For tests, and for moving off DuckDuckGo later."""
    global _search
    _search = search
