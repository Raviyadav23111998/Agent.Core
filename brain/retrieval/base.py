from __future__ import annotations

"""
Agent Brain — Retrieval Agent Base
=====================================
Shared interface and data model for all retrieval agents.

Every agent (Graph, Search, Context) returns an AgentResult,
giving the Router and Synthesizer a uniform structure to work with.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentResult:
    """
    Standardized output from any retrieval agent.
    The Synthesizer consumes a list of these to build the final answer.
    """
    agent_name: str
    content: str
    sources: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    raw_data: Optional[dict] = None
    error: Optional[str] = None

    @property
    def has_content(self) -> bool:
        return bool(self.content and self.content.strip())

    @property
    def failed(self) -> bool:
        return self.error is not None


class BaseAgent(ABC):
    """Abstract base class for all retrieval agents."""

    AGENT_NAME: str = "base_agent"

    @abstractmethod
    async def query(self, question: str, **kwargs) -> AgentResult:
        """
        Execute a retrieval query and return results.

        Args:
            question: The user's question (or a sub-question from the Router)
            **kwargs: Agent-specific parameters

        Returns:
            AgentResult with findings, sources, and confidence
        """
        ...

    def _error_result(self, error: str) -> AgentResult:
        """Create a standardized error result."""
        return AgentResult(
            agent_name=self.AGENT_NAME,
            content="",
            confidence=0.0,
            error=error,
        )
