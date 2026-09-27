"""The changes made for "it feels slow" and "the options feel off".

Both sets are easy to break silently — a parallel read that quietly serialises
again still returns the right answer, and a transcript that drops an option list
still produces a perfectly reasonable-looking question. So each one is pinned
here to the observable thing that would actually regress.
"""

import time

import pytest

from app.core import auth as auth_module
from app.core.concurrency import gather
from app.services.context import ChatContext

CHAT = {"id": "chat-1", "user_id": "user-1", "category": "relationship"}


def _context(messages):
    return ChatContext(chat=CHAT, profile={}, messages=messages)


class TestTranscriptCarriesWhatWasOffered:
    """What the model can see of its own earlier turns."""

    def test_a_multi_select_intake_answer_reads_as_several_things(self):
        # The bug this fixes: the scripted opening is multi-select too, and its
        # answers were being rendered as one joined sentence. That is most of
        # the context behind the *first* generated question, so a person who
        # ticked three separate complaints was read as one hedge at exactly the
        # point where there was least else to go on.
        transcript = _context(
            [
                {
                    "type": "intake",
                    "question_text": "What is wrong?",
                    "answer_text": "She doesn't give me time; I don't feel valued",
                    "metadata": {
                        "multi": True,
                        "selected": [
                            "She doesn't give me time",
                            "I don't feel valued",
                        ],
                    },
                }
            ]
        ).transcript()

        assert (
            "A (chose several): She doesn't give me time | I don't feel valued"
            in transcript
        )

    def test_one_tick_on_a_multi_intake_question_does_not_claim_several(self):
        transcript = _context(
            [
                {
                    "type": "intake",
                    "question_text": "What is wrong?",
                    "answer_text": "She doesn't give me time",
                    "metadata": {
                        "multi": True,
                        "selected": ["She doesn't give me time"],
                    },
                }
            ]
        ).transcript()

        assert "A: She doesn't give me time" in transcript
        assert "chose several" not in transcript

    def test_a_skipped_intake_answer_still_reads_as_skipped(self):
        transcript = _context(
            [
                {
                    "type": "intake",
                    "question_text": "What is wrong?",
                    "answer_text": None,
                    "metadata": {"skipped": True},
                }
            ]
        ).transcript()

        assert "A: (skipped)" in transcript

    def test_the_options_the_model_offered_reach_the_next_prompt(self):
        # Without this the model cannot see its own earlier choice sets, and
        # re-serves a near-identical one two rounds later.
        transcript = _context(
            [
                {
                    "type": "adaptive_question",
                    "question_text": "What changed?",
                    "answer_text": "She got a new job",
                    "metadata": {
                        "options": ["She got a new job", "I moved away"],
                        "multi": False,
                    },
                }
            ]
        ).transcript()

        assert "(options you gave: She got a new job | I moved away)" in transcript

    def test_a_question_with_no_options_renders_no_empty_line(self):
        transcript = _context(
            [
                {
                    "type": "adaptive_question",
                    "question_text": "What changed?",
                    "answer_text": "She got a new job",
                    "metadata": {},
                }
            ]
        ).transcript()

        assert "options you gave" not in transcript


class TestWhatTheyCameHereWith:
    def test_their_own_description_is_restated_in_the_prompt(self):
        summary = _context(
            [
                {"type": "free_text", "answer_text": "I want to leave my course."},
                {
                    "type": "intake",
                    "question_text": "Since when?",
                    "answer_text": "March",
                    "metadata": {},
                },
            ]
        ).summary()

        assert "WHAT THEY CAME HERE WITH" in summary
        # Once as the anchor, once in the transcript where it happened.
        assert summary.count("I want to leave my course.") == 2

    def test_a_later_reply_is_not_mistaken_for_the_description(self):
        # In a continued chat every user turn is free_text. The description is
        # the first one; the rest are answers, not the question they came with.
        context = _context(
            [
                {"type": "free_text", "answer_text": "I want to leave my course."},
                {"type": "recommendation", "answer_text": "Stay one more term."},
                {"type": "free_text", "answer_text": "But my father will not agree."},
            ]
        )

        assert context.stated_problem() == "I want to leave my course."

    def test_no_description_means_no_block_at_all(self):
        summary = _context(
            [
                {
                    "type": "intake",
                    "question_text": "Since when?",
                    "answer_text": "March",
                    "metadata": {},
                }
            ]
        ).summary()

        assert "WHAT THEY CAME HERE WITH" not in summary


class TestTheLastThingTheySaid:
    def test_it_is_the_most_recent_answered_turn(self):
        latest = _context(
            [
                {
                    "type": "intake",
                    "question_text": "Since when?",
                    "answer_text": "March",
                    "metadata": {},
                },
                {
                    "type": "adaptive_question",
                    "question_text": "Who else knows?",
                    "answer_text": "Nobody",
                    "metadata": {},
                },
            ]
        ).latest_answer()

        assert 'You asked: "Who else knows?"' in latest
        assert "They answered: Nobody" in latest

    def test_an_unanswered_question_is_not_the_last_thing_they_said(self):
        # next_question writes the question row before the user answers it, so
        # the newest row is routinely one nobody has said anything to yet.
        latest = _context(
            [
                {"type": "free_text", "answer_text": "I want to leave my course."},
                {
                    "type": "adaptive_question",
                    "question_text": "Who else knows?",
                    "answer_text": None,
                    "metadata": {},
                },
            ]
        ).latest_answer()

        assert 'They wrote: "I want to leave my course."' in latest

    def test_a_multi_select_answer_stays_in_parts(self):
        latest = _context(
            [
                {
                    "type": "adaptive_question",
                    "question_text": "What is wrong?",
                    "answer_text": "No time; No respect",
                    "metadata": {"multi": True, "selected": ["No time", "No respect"]},
                }
            ]
        ).latest_answer()

        assert "They answered: No time | No respect" in latest

    def test_the_first_question_of_a_chat_says_so(self):
        assert "nothing yet" in _context([]).latest_answer()


class TestGather:
    def test_results_come_back_in_the_order_given_not_finished(self):
        # The whole point of the flattening in _research and the unpacking in
        # load_context: position is what identifies a result, so a slow first
        # call must not end up second.
        assert gather(
            lambda: (time.sleep(0.05), "first")[1],
            lambda: "second",
        ) == ["first", "second"]

    def test_the_calls_really_do_overlap(self):
        started = time.perf_counter()
        gather(*[lambda: time.sleep(0.1) for _ in range(4)])
        elapsed = time.perf_counter() - started

        # Four 100ms sleeps in sequence is 400ms. Generous bound — this is
        # asserting "concurrent at all", not a performance target.
        assert elapsed < 0.3

    def test_a_failure_is_raised_to_the_caller(self):
        # Every call site has one try block around the lot and turns this into
        # a 503. Swallowing it here would turn a failed read into empty context
        # and advice grounded in nothing.
        def boom():
            raise ValueError("no")

        with pytest.raises(ValueError):
            gather(lambda: "fine", boom)

    def test_nothing_to_do_is_not_an_error(self):
        assert gather() == []


class TestTokenCache:
    """Verifying a token costs a round trip to Supabase on every request."""

    @pytest.fixture(autouse=True)
    def _clean(self):
        auth_module.forget_tokens()
        yield
        auth_module.forget_tokens()

    def _count_verifications(self, monkeypatch):
        calls = []

        class _Auth:
            def get_user(self, token):
                calls.append(token)
                return type("R", (), {"user": type("U", (), {"id": "user-1"})()})()

        monkeypatch.setattr(
            auth_module, "service_client", lambda: type("C", (), {"auth": _Auth()})()
        )
        return calls

    def test_the_same_token_is_only_verified_once(self, monkeypatch):
        calls = self._count_verifications(monkeypatch)

        for _ in range(5):
            assert auth_module.current_user_id("Bearer abc") == "user-1"

        assert len(calls) == 1

    def test_a_different_token_is_verified_on_its_own(self, monkeypatch):
        calls = self._count_verifications(monkeypatch)

        auth_module.current_user_id("Bearer abc")
        auth_module.current_user_id("Bearer xyz")

        assert calls == ["abc", "xyz"]

    def test_an_expired_entry_is_verified_again(self, monkeypatch):
        calls = self._count_verifications(monkeypatch)
        monkeypatch.setattr(auth_module, "_CACHE_SECONDS", 0.01)

        auth_module.current_user_id("Bearer abc")
        time.sleep(0.02)
        auth_module.current_user_id("Bearer abc")

        assert len(calls) == 2

    def test_caching_can_be_turned_off(self, monkeypatch):
        # AUTH_CACHE_SECONDS=0 restores verify-every-time, for a deployment
        # that would rather pay the round trip than let a signed-out session
        # linger for a minute.
        calls = self._count_verifications(monkeypatch)
        monkeypatch.setattr(auth_module, "_CACHE_SECONDS", 0)

        auth_module.current_user_id("Bearer abc")
        auth_module.current_user_id("Bearer abc")

        assert len(calls) == 2
