"""What the model is told, and — for a new user — what it is not told.

The cold-start tests are the point of this file. A first-time user has no memory
and no history, and the failure mode of a memory feature is a model that has
been handed an empty one and starts inventing a shared past to fill it. The rule
is that their prompt is the prompt this produced before any of it existed.
"""

from app.services.context import ChatContext
from app.services.recall import RelatedChat
from tests.conftest import unwrapped

CHAT = {"id": "chat-9", "user_id": "user-1", "category": "financial"}

SAID = [
    {
        "type": "free_text",
        "answer_text": "I keep lending my brother money and he never pays it back.",
        "metadata": {},
    }
]


def context_with(**kwargs):
    return ChatContext(chat=CHAT, profile={}, messages=SAID, **kwargs)


RELATED = RelatedChat(
    chat_id="chat-1",
    title="Whether to keep paying Ravi's fees",
    category="financial",
    when="March 2026",
    gist="Keep paying, but tell him it stops in June.",
)


class TestColdStart:
    def test_a_new_user_is_told_nothing_about_memory_at_all(self):
        summary = context_with().summary()

        # Not "(nothing yet)" — absent. A standing instruction to reference what
        # you remember, handed to a model with nothing to remember, is an
        # invitation to invent a history, which is the exact thing this feature
        # would be judged on.
        assert "WHAT YOU ALREADY KNOW" not in summary
        assert "EARLIER CONVERSATIONS" not in summary
        assert "nothing yet" not in summary.lower()

    def test_a_new_users_prompt_carries_no_trace_of_memory_or_recall(self):
        """The whole prompt, asserted exactly.

        This used to read "the prompt from before this feature existed", and
        checked that a cold user's prompt was byte-for-byte the pre-memory one.
        The prompt has since gained a block that restates the user's own
        description just before the transcript — for *everybody*, cold or not —
        so that string is no longer the right expectation.

        What the test is actually for survives unchanged: nothing about memory
        or past conversations may appear for someone who has neither, and no
        "(nothing yet)" placeholder may stand in for them. Asserting the whole
        string rather than a handful of `not in`s is still the point, because
        it is what catches a block nobody thought to check for.
        """
        summary = context_with().summary()

        assert summary == (
            "WHO THEY ARE:\n- (nothing recorded)\n\n"
            "TOPIC: financial\n\n"
            "WHAT THEY CAME HERE WITH, IN THEIR OWN WORDS\n"
            "This is the decision. Everything you ask must serve it. It is "
            "repeated here on\nits own because it is the one thing in this "
            "prompt that is easiest to drift away\nfrom once the profile and "
            "the scripted answers are in front of you.\n\n"
            '"I keep lending my brother money and he never pays it back."\n\n'
            "THE CONVERSATION SO FAR:\n"
            'They wrote: "I keep lending my brother money and he never pays it back."'
        )

    def test_a_user_with_only_empty_memory_rows_is_still_cold(self):
        # handle_new_user provisions a global memory row at sign-up, so "has a
        # row" and "has memory" are different things from the very first second
        # of an account's life.
        context = context_with(
            memories=[{"category": None, "summary": "", "facts": []}]
        )

        assert context.is_first_chat
        assert "WHAT YOU ALREADY KNOW" not in context.summary()


class TestMemoryInThePrompt:
    def test_facts_reach_the_model(self):
        context = context_with(
            memories=[{"category": None, "summary": "", "facts": ["Lives in Pune."]}]
        )

        summary = context.summary()

        assert "WHAT YOU ALREADY KNOW" in summary
        assert "- Lives in Pune." in summary
        assert not context.is_first_chat

    def test_another_categorys_memory_is_not_shown(self):
        context = context_with(
            memories=[
                {"category": None, "facts": ["Lives in Pune."], "summary": ""},
                {"category": "financial", "facts": ["Earns 32k."], "summary": ""},
                {"category": "relationship", "facts": ["Engaged to Meera."], "summary": ""},
            ]
        )

        summary = context.summary()

        # The whole reason user_memory has a category column. A conversation
        # about money has no business reciting someone's engagement back at
        # them.
        assert "Lives in Pune." in summary
        assert "Earns 32k." in summary
        assert "Meera" not in summary

    def test_the_model_is_told_the_user_is_right_when_memory_disagrees(self):
        context = context_with(
            memories=[{"category": None, "facts": ["Studying at Pune University."], "summary": ""}]
        )

        # People change their circumstances. Memory that argues with the person
        # in front of it is worse than no memory.
        assert "they are right and this is out of date" in unwrapped(context.summary())

    def test_the_model_is_told_to_reference_it_out_loud(self):
        context = context_with(
            memories=[{"category": None, "facts": ["Lives in Pune."], "summary": ""}]
        )

        # Being remembered is the feature. A model that silently uses memory and
        # never says so has built something the user cannot tell is there.
        assert "say so out loud" in unwrapped(context.summary())


class TestRelatedInThePrompt:
    def test_a_past_chat_reaches_the_model_with_what_we_told_them(self):
        summary = context_with(related=[RELATED]).summary()

        assert "EARLIER CONVERSATIONS" in summary
        assert "Whether to keep paying Ravi's fees" in summary
        assert "March 2026" in summary
        assert "Keep paying, but tell him it stops in June." in summary

    def test_related_chats_alone_are_enough_to_warm_the_prompt(self):
        # Memory merging can fail, or simply not have run yet. Past chats are a
        # second, independent route to "we have met before".
        context = context_with(related=[RELATED])

        assert not context.is_first_chat


class TestTranscript:
    """How an answer reads to the model — one thing, or several at once."""

    def _asked(self, **metadata):
        return ChatContext(
            chat=CHAT,
            profile={},
            messages=[
                {
                    "type": "adaptive_question",
                    "question_text": "Why does it feel like that?",
                    "answer_text": "I don't feel valued; She doesn't give me time",
                    "metadata": {"options": ["a", "b"], **metadata},
                }
            ],
        )

    def test_a_single_select_answer_is_never_reported_as_several(self):
        context = ChatContext(
            chat=CHAT,
            profile={},
            messages=[
                {
                    "type": "adaptive_question",
                    "question_text": "What is stopping you?",
                    "answer_text": "The money",
                    "metadata": {"options": ["The money"], "multi": False},
                }
            ],
        )
        transcript = context.transcript()

        # The rendering now carries the option list the model offered, so that
        # it can see its own earlier choice sets and stop re-serving them. The
        # question and the answer are therefore no longer adjacent lines, which
        # is what this used to assert.
        #
        # What must not change is the claim being made about the person: one
        # tick is one answer, and "chose several" over it would be a statement
        # about them that is not true.
        assert "Q (you asked): What is stopping you?" in transcript
        assert "A: The money" in transcript
        assert "chose several" not in transcript

    def test_a_multi_select_answer_reads_as_several_distinct_things(self):
        transcript = self._asked(
            multi=True,
            selected=["I don't feel valued", "She doesn't give me time"],
        ).transcript()

        # The point: these are two complaints that are both true, not one
        # sentence in which someone hedged.
        assert (
            "A (chose several): I don't feel valued | She doesn't give me time"
            in transcript
        )

    def test_selections_are_recovered_from_the_text_when_the_row_lacks_them(self):
        # `text` is the contract — the selections joined, in display order — so
        # a row written without the parts is not a row that lost them.
        transcript = self._asked(multi=True).transcript()

        assert (
            "A (chose several): I don't feel valued | She doesn't give me time"
            in transcript
        )

    def test_one_tick_on_a_multi_question_does_not_claim_several(self):
        context = ChatContext(
            chat=CHAT,
            profile={},
            messages=[
                {
                    "type": "adaptive_question",
                    "question_text": "Why does it feel like that?",
                    "answer_text": "I don't feel valued",
                    "metadata": {"multi": True, "selected": ["I don't feel valued"]},
                }
            ],
        )

        # "chose several" over a single tick is a claim about the person that
        # is not true.
        assert "A: I don't feel valued" in context.transcript()
        assert "chose several" not in context.transcript()

    def test_an_unanswered_multi_question_is_still_unanswered(self):
        context = ChatContext(
            chat=CHAT,
            profile={},
            messages=[
                {
                    "type": "adaptive_question",
                    "question_text": "Why does it feel like that?",
                    "answer_text": None,
                    "metadata": {"multi": True},
                }
            ],
        )

        assert "A: (not yet answered)" in context.transcript()


class TestKeywords:
    def test_they_come_from_what_the_user_said(self):
        assert "brother" in context_with().keywords()

    def test_the_questions_we_asked_are_not_keywords(self):
        context = ChatContext(
            chat=CHAT,
            profile={},
            messages=[
                {
                    "type": "intake",
                    "question_text": "How urgent is this scholarship decision?",
                    "answer_text": "Soon",
                    "metadata": {},
                }
            ],
        )

        # The scripted questions are ours and identical for everyone in this
        # category. Matching on them would connect every financial chat to every
        # other one.
        assert "scholarship" not in context.keywords()
