from __future__ import annotations

"""
Agent Brain — Base Parser
============================
Abstract interface that all format parsers implement.
Also defines ParsedDocument — the intermediate data model
that flows from Parsers into the Ingestion Pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from brain.models import DocumentType


@dataclass
class ParsedDocument:
    """
    Intermediate representation of a parsed document.
    Created by a Parser, consumed by the Ingestion Pipeline.

    This is NOT the same as core.models.Document (which is the
    stored version in PostgreSQL). This is the transient parsing output.
    """
    content: str
    source_path: str
    document_type: DocumentType
    title: str = ""
    metadata: dict = field(default_factory=dict)
    raw_size_bytes: int = 0

    @property
    def estimated_tokens(self) -> int:
        """Rough token estimate: ~4 chars per token for English."""
        return len(self.content) // 4

    @property
    def is_empty(self) -> bool:
        return len(self.content.strip()) == 0


class BaseParser(ABC):
    """
    Abstract base class for all document parsers.

    Each parser handles a specific format (PDF, email, code, etc.)
    and returns a ParsedDocument with clean text and extracted metadata.
    """

    SUPPORTED_EXTENSIONS: list[str] = []

    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocument:
        """
        Parse a file and return a ParsedDocument.

        Args:
            file_path: Absolute path to the file

        Returns:
            ParsedDocument with extracted text and metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
        """
        ...

    @abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the given file."""
        ...

    def _validate_file(self, file_path: Path) -> None:
        """Common validation for all parsers."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Not a file: {file_path}")
