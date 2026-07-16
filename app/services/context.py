"""Everything known about one chat, rendered for a prompt.

One place assembles context, so the adaptive questions, the recommendation, and
the follow-up chat all reason over the same picture of a person. Three prompts
each hand-rolling this would drift, and the bug — advice grounded in a subset of
what the user said — would be invisible.

Since Prompt 6 that picture reaches past this chat: what we have learned about
them in every earlier one (`user_memory`), and the past conversations this one
connects to (`app/services/recall.py`). Both are additive and both are optional
— see [summary], where a user with no history renders exactly the block they
rendered before either existed.
"""

from dataclasses import dataclass, field

from app.core.supabase_client import fetch_memory_rows, fetch_messages, fetch_profile
from app.services.recall import RelatedChat, find_related, keywords_from

# The onboarding basic profile, in the order a human would want it, with the
# labels the app asked under. Keys must match onboarding_questions.dart.
_PROFILE_LABELS: list[tuple[str, str]] = [
    ("location", "Lives in"),
    ("age_range", "Age"),
    ("education_level", "Education"),
    ("field_of_study", "Field"),
    ("current_status", "Currently"),
    ("next_plan", "Hoping to do next"),
    ("time_horizon", "Timeline for that"),
    ("living_situation", "Lives with"),
    ("financial_context", "Money situation"),
    ("relationship_status", "Relationship"),
    ("decision_style", "Decides best by"),
    ("anything_else", "Also mentioned"),
]


# What the model is told about memory, carried in the block itself rather than
# in the system prompts.
#
# It has to live next to the facts because it only makes sense when there are
# facts. A system prompt saying "reference what you remember about them" is a
# standing instruction, and a standing instruction handed to a model with an
# empty memory is an invitation to invent a history — which is the exact failure
# this feature would be judged on. No memory, no instruction, no block.
_MEMORY_PREAMBLE = """WHAT YOU ALREADY KNOW ABOUT THEM
From your earlier conversations with this same person — not from this one.

Use it the way a friend would. Where it is genuinely relevant, say so out loud:
"you were weighing this same thing back in March" is the whole point of
remembering. Where it is not relevant, ignore it silently — do not list it back
at them, and never say anything like "according to my records".

If it contradicts what they are telling you now, they are right and this is out
of date. People change their minds and their circumstances; this is a note, not
a record."""

_RELATED_PREAMBLE = """EARLIER CONVERSATIONS THAT MAY CONNECT
Times you have talked with them before that look related to this one. Refer back
where it earns its place — if they are circling the same decision a third time,
that is worth naming plainly rather than answering as though it were the first.
Do not force it."""


@dataclass
class ChatContext:
    chat: dict
    profile: dict
    messages: list[dict] = field(default_factory=list)

    # Every user_memory row for this person: the global one, plus per-category.
    # Empty for a brand-new user, and for one whose memory could not be read.
    memories: list[dict] = field(default_factory=list)

    # Past chats this one appears to connect to. Empty on a first chat.
    related: list[RelatedChat] = field(default_factory=list)

    @property
    def category(self) -> str:
        return self.chat.get("category", "other")

    @property
    def is_first_chat(self) -> bool:
        """No memory and no history — this person is new. The prompts must read
        exactly as they did before any of this existed."""
        return not self.memory_lines() and not self.related

    @property
    def adaptive_rounds(self) -> int:
        """How many questions the model has already asked."""
        return sum(1 for m in self.messages if m.get("type") == "adaptive_question")

    @property
    def last_unanswered(self) -> dict | None:
        """The question waiting on an answer, if any.

        Lets a repeated call be idempotent: if the client asks for the next
        question while one is still unanswered — a retry after a dropped
        response, usually — it gets that same question back rather than a
        second one.
        """
        for message in reversed(self.messages):
            if message.get("type") == "adaptive_question" and not message.get(
                "answer_text"
            ):
                return message
        return None

    @property
    def has_recommendation(self) -> bool:
        return any(m.get("type") == "recommendation" for m in self.messages)

    def profile_lines(self) -> str:
        """The durable facts. Skipped answers are simply absent."""
        answers = self.profile.get("onboarding_answers") or {}
        lines = []
        for key, label in _PROFILE_LABELS:
            value = answers.get(key)
            if isinstance(value, str) and value.strip():
                lines.append(f"- {label}: {value.strip()}")
        name = (self.profile.get("display_name") or "").strip()
        if name:
            lines.insert(0, f"- Name: {name}")
        return "\n".join(lines) if lines else "- (nothing recorded)"

    def memory_lines(self) -> str:
        """The durable facts from every earlier chat, global ones first.

        The global row carries what is true of them whatever the topic; the
        category row carries what only matters here. Rows for *other* categories
        are deliberately not rendered — that split is the reason the table has a
        category column, and a financial chat has no business reciting their
        relationship history back at them.
        """
        lines: list[str] = []
        for row in self._relevant_memories():
            scope = row.get("category")
            for fact in row.get("facts") or []:
                if isinstance(fact, str) and fact.strip():
                    lines.append(f"- {fact.strip()}")
            summary = (row.get("summary") or "").strip()
            if summary:
                label = "In general" if scope is None else f"On {scope}"
                lines.append(f"- {label}: {summary}")
        return "\n".join(lines)

    def _relevant_memories(self) -> list[dict]:
        """The global row, then this chat's category row. Nothing else."""
        global_row = [m for m in self.memories if m.get("category") is None]
        category_row = [m for m in self.memories if m.get("category") == self.category]
        return global_row + category_row

    def keywords(self) -> list[str]:
        """What this chat is about, in words, for the recall lookup.

        Read from what the user actually said — their own description and their
        answers — rather than from the questions, which are ours and are the
        same for everyone in this category.
        """
        said = " ".join(
            (m.get("answer_text") or "")
            for m in self.messages
            if m.get("type") in ("free_text", "intake", "adaptive_question")
        )
        return keywords_from(said)

    def transcript(self) -> str:
        """The conversation so far, in the order it happened.

        Typed by role rather than dumped raw, because "we asked this" and "they
        volunteered this" carry different weight, and a model given an
        undifferentiated blob will treat a multiple-choice tap as though it
        were a confession.
        """
        lines = []
        for message in self.messages:
            kind = message.get("type")
            question = (message.get("question_text") or "").strip()
            answer = (message.get("answer_text") or "").strip()

            if kind == "intake":
                lines.append(f"Q (scripted): {question}")
                lines.append(f"A: {answer or '(skipped)'}")
            elif kind == "adaptive_question":
                lines.append(f"Q (you asked): {question}")
                lines.append(f"A: {answer or '(not yet answered)'}")
            elif kind == "free_text":
                how = (message.get("metadata") or {}).get("input_method")
                said = "said aloud" if how == "voice" else "wrote"
                lines.append(f"They {said}: \"{answer}\"")
            elif kind == "recommendation":
                lines.append(f"Your recommendation: {answer}")
            elif kind == "assistant_reply":
                lines.append(f"You replied: {answer}")
        return "\n".join(lines) if lines else "(nothing yet)"

    def summary(self) -> str:
        """Everything a prompt here is built on.

        Memory and past conversations are *omitted entirely* when there are
        none, headers and all — not rendered as "(nothing yet)". A first-time
        user's prompt is byte-for-byte the one this produced before Prompt 6,
        which is what keeps the feature from leaking into the experience of the
        person it cannot help yet.
        """
        blocks = [f"WHO THEY ARE:\n{self.profile_lines()}"]

        memory = self.memory_lines()
        if memory:
            blocks.append(f"{_MEMORY_PREAMBLE}\n\n{memory}")

        if self.related:
            recalled = "\n".join(r.render() for r in self.related)
            blocks.append(f"{_RELATED_PREAMBLE}\n\n{recalled}")

        blocks.append(f"TOPIC: {self.category}")
        blocks.append(f"THE CONVERSATION SO FAR:\n{self.transcript()}")
        return "\n\n".join(blocks)


def load_context(chat: dict, *, recall: bool = True) -> ChatContext:
    """Read a chat's full context. [chat] is already authorised by the caller.

    [recall] off skips the cross-chat reads. Titling and memory extraction reason
    about one conversation and nothing else, and would otherwise pay four extra
    queries to be told about chats they must not fold in.
    """
    user_id = chat["user_id"]
    context = ChatContext(
        chat=chat,
        profile=fetch_profile(user_id),
        messages=fetch_messages(chat["id"]),
    )
    if not recall:
        return context

    context.memories = fetch_memory_rows(user_id)
    context.related = find_related(
        user_id=user_id,
        chat_id=chat["id"],
        category=context.category,
        keywords=context.keywords(),
    )
    return context
