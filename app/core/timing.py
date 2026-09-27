"""Where the time actually goes.

"The AI feels slow" is one number covering four very different things: waking a
sleeping Render instance, a handful of Supabase round trips, up to three web
searches, and the model call itself. Which of those dominates is not guessable —
it depends on where the instance, the database and Cohere happen to be — so the
stages log their own durations and the answer is in `render logs`.

INFO level on purpose: this is meant to be readable in production without a
redeploy, and it is a few lines per request, not a trace.
"""

import logging
import time
from contextlib import contextmanager

logger = logging.getLogger("thoughtloom.timing")


@contextmanager
def timed(label: str):
    """Log how long the block took. Never changes what the block does."""
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.info("%s: %.0f ms", label, (time.perf_counter() - start) * 1000)
