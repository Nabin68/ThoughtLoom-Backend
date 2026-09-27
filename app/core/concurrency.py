"""Run independent blocking calls at the same time.

Every slow endpoint here is mostly *waiting*: six Supabase round trips and up
to three web searches, each a few hundred milliseconds of network that the
process spends idle. Run one after another they add up to more than a second
before the model is even asked anything; run together they cost one round trip.

Threads rather than asyncio because the clients underneath are synchronous
(supabase-py, ddgs) and the callers already sit inside `run_in_threadpool` —
this is the same kind of waiting, one level down.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

T = TypeVar("T")


def gather(*calls: Callable[[], T]) -> list[T]:
    """Each of [calls] at once. Results come back in the order given.

    An exception in any one is raised here, exactly as it would have been had
    the call been made directly — every caller below has a single try block
    around the lot, so nothing changes about how failures are handled.

    A fresh pool per call rather than a shared one: thread creation is
    microseconds against the hundreds of milliseconds this saves, and a bounded
    shared pool would be one nested `gather` away from deadlocking itself.
    """
    if not calls:
        return []
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        return [future.result() for future in [pool.submit(c) for c in calls]]
