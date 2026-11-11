"""
Agent Brain — Conversation Manager
=======================================
Provides multi-turn conversation memory so follow-up questions
like "tell me more about that" or "what about the second one?"
have context from previous Q&A exchanges.

Design decisions:
  - Single global session (company brain is typically single-user via Hermes)
  - Last 10 turns kept (enough context without blowing token budget)
  - Auto-expires after 30 minutes of inactivity
  - In-memory only (no persistence needed — conversations are ephemeral)
  - Compact format to minimize token overhead (~50 tokens/turn)

Token budget:
  10 turns × ~50 tokens/turn = ~500 tokens total overhead per query
  This is negligible vs the 2,000-70,000 token agent payloads.

Usage:
    convo = ConversationManager()
    history = convo.get_context()        # "" if empty, or formatted history
    convo.add_turn("What ORM does Rideshare use?", "Dapper, not EF.")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)

MAX_TURNS = 5
INACTIVITY_TIMEOUT_SECONDS = 1800
MAX_ANSWER_CHARS_IN_HISTORY = 150


@dataclass
class ConversationTurn:
    """A single Q&A exchange."""
    question: str
    answer: str
    timestamp: float = field(default_factory=time.time)
    agents_used: list[str] = field(default_factory=list)


class ConversationManager:
    """
    In-memory conversation history for multi-turn context.

    The system injects conversation history into the synthesis prompt
    so the LLM can resolve pronouns and follow-up references:
      - "tell me more about that" → knows what "that" was
      - "what about the second one?" → knows the prior list
      - "and for project X?" → knows the prior question pattern
    """

    def __init__(self, max_turns: int = MAX_TURNS, timeout_seconds: int = INACTIVITY_TIMEOUT_SECONDS):
        self._turns: list[ConversationTurn] = []
        self._max_turns = max_turns
        self._timeout = timeout_seconds
        self._last_activity: float = 0.0

    def add_turn(self, question: str, answer: str, agents_used: list[str] = None):
        """
        Record a Q&A exchange.

        Args:
            question: The user's question
            answer: The system's answer
            agents_used: Which agents contributed
        """
        now = time.time()

        if self._last_activity > 0 and (now - self._last_activity) > self._timeout:
            logger.info("conversation.session_expired: idle_seconds=%s turns_cleared=%s", int(now - self._last_activity), len(self._turns))
            self._turns.clear()

        compact_answer = answer[:MAX_ANSWER_CHARS_IN_HISTORY]
        if len(answer) > MAX_ANSWER_CHARS_IN_HISTORY:
            compact_answer += "..."

        self._turns.append(ConversationTurn(
            question=question,
            answer=compact_answer,
            timestamp=now,
            agents_used=agents_used or [],
        ))

        if len(self._turns) > self._max_turns:
            removed = len(self._turns) - self._max_turns
            self._turns = self._turns[-self._max_turns:]
            logger.debug("conversation.trimmed: removed=%s", removed)

        self._last_activity = now
        logger.info("conversation.turn_added: turn_count=%s question=%s", len(self._turns), question[:60])

    def get_context(self) -> str:
        """
        Get formatted conversation history for injection into the synthesis prompt.

        Returns:
            Empty string if no history, otherwise a compact formatted block.
            Format designed to minimize tokens while maximizing LLM comprehension.
        """
        if self._last_activity > 0 and (time.time() - self._last_activity) > self._timeout:
            self._turns.clear()
            return ""

        if not self._turns:
            return ""

        lines = ["CONVERSATION HISTORY (most recent last):"]
        for i, turn in enumerate(self._turns, 1):
            lines.append(f"  Q{i}: {turn.question}")
            lines.append(f"  A{i}: {turn.answer}")
        lines.append("---")

        return "\n".join(lines)

    def get_last_question(self) -> Optional[str]:
        """Get the most recent question (for follow-up detection)."""
        if not self._turns:
            return None
        return self._turns[-1].question

    def get_last_answer(self) -> Optional[str]:
        """Get the most recent answer."""
        if not self._turns:
            return None
        return self._turns[-1].answer

    def is_follow_up(self, question: str) -> bool:
        """
        Detect if a question is likely a follow-up to the previous conversation.

        Heuristics:
          - Starts with a pronoun referencing prior context ("it", "that", "they")
          - Very short question (< 5 words) — implies context is assumed
          - Starts with "and", "but", "also", "what about"
        """
        if not self._turns:
            return False

        q = question.lower().strip()

        follow_up_starters = (
            "what about", "how about", "and ", "but ", "also ",
            "tell me more", "explain more", "more about",
            "can you elaborate", "go deeper", "expand on",
            "the second", "the first", "the third", "the last",
            "which one", "why is that", "why does",
        )
        if any(q.startswith(s) for s in follow_up_starters):
            return True

        pronoun_starters = ("it ", "that ", "they ", "this ", "those ", "these ", "its ")
        if any(q.startswith(p) for p in pronoun_starters):
            return True

        if len(q.split()) <= 3 and self._turns:
            return True

        return False

    def enrich_question(self, question: str) -> str:
        """
        If the question is a follow-up, enrich it with context from the last turn.
        This helps the Router make better routing decisions.

        Example:
            Last Q: "What ORM does Rideshare use?"
            Last A: "Rideshare uses Dapper..."
            Current Q: "And what about caching?"
            Enriched: "And what about caching? (Context: Previous question was about Rideshare's ORM - Dapper)"
        """
        if not self.is_follow_up(question):
            return question

        last_q = self.get_last_question()
        last_a = self.get_last_answer()

        if not last_q:
            return question

        context_parts = [f"Previous question: \"{last_q}\""]
        if last_a:
            first_sentence = last_a.split(".")[0].strip()
            if first_sentence and len(first_sentence) > 10:
                context_parts.append(f"Previous answer summary: \"{first_sentence}\"")

        context = " | ".join(context_parts)
        enriched = f"{question}\n[Follow-up context: {context}]"

        logger.info("conversation.question_enriched: original=%s context_chars=%s", question[:60], len(context))

        return enriched

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    def clear(self):
        """Clear all conversation history."""
        self._turns.clear()
        self._last_activity = 0.0
        logger.info("conversation.cleared")

    def get_stats(self) -> dict:
        """Return conversation statistics."""
        return {
            "turns": len(self._turns),
            "max_turns": self._max_turns,
            "timeout_seconds": self._timeout,
            "idle_seconds": int(time.time() - self._last_activity) if self._last_activity else 0,
            "is_active": bool(self._turns) and (time.time() - self._last_activity) < self._timeout,
        }
