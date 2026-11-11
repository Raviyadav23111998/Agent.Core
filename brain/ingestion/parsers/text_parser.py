from __future__ import annotations

"""
Agent Brain — Text Parser
============================
Handles plain text, Markdown, log files, and CSV files.
Detects encoding automatically via chardet.
Extracts title from first Markdown heading if present.
"""

import re
from pathlib import Path
import logging

from brain.models import DocumentType
from brain.ingestion.parsers.base import BaseParser, ParsedDocument

logger = logging.getLogger(__name__)


class TextParser(BaseParser):
    """
    Parser for plain text files.
    Handles: .txt, .md, .markdown, .log, .csv, .json, .yaml, .yml, .xml, .ini, .cfg, .conf
    """

    SUPPORTED_EXTENSIONS = [
        ".txt", ".md", ".markdown", ".log", ".csv",
        ".json", ".yaml", ".yml", ".xml",
        ".ini", ".cfg", ".conf", ".toml",
        ".rst", ".adoc",
    ]

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def parse(self, file_path: Path) -> ParsedDocument:
        self._validate_file(file_path)

        raw_bytes = file_path.read_bytes()
        raw_size = len(raw_bytes)

        content = self._decode_content(raw_bytes)

        suffix = file_path.suffix.lower()
        metadata = {
            "format": suffix.lstrip("."),
            "line_count": content.count("\n") + 1,
            "char_count": len(content),
        }

        title = self._extract_title(content, suffix, file_path)

        doc_type = self._classify_type(suffix)

        logger.info("text_parser.parsed: path=%s format=%s lines=%s", str(file_path), suffix, metadata["line_count"])

        return ParsedDocument(
            content=content,
            source_path=str(file_path),
            document_type=doc_type,
            title=title,
            metadata=metadata,
            raw_size_bytes=raw_size,
        )

    def _decode_content(self, raw_bytes: bytes) -> str:
        """Decode bytes to string, auto-detecting encoding."""
        try:
            return raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            pass

        try:
            import chardet
            detected = chardet.detect(raw_bytes)
            encoding = detected.get("encoding", "utf-8") or "utf-8"
            return raw_bytes.decode(encoding, errors="replace")
        except ImportError:
            return raw_bytes.decode("utf-8", errors="replace")

    def _extract_title(self, content: str, suffix: str, file_path: Path) -> str:
        """Extract a title from the content or fall back to filename."""
        if suffix in (".md", ".markdown", ".rst", ".adoc"):
            match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
            if match:
                return match.group(1).strip()

            lines = content.split("\n")
            for i, line in enumerate(lines):
                if i + 1 < len(lines) and re.match(r"^[=\-]{3,}$", lines[i + 1].strip()):
                    return line.strip()

        return file_path.stem.replace("_", " ").replace("-", " ").title()

    def _classify_type(self, suffix: str) -> DocumentType:
        """Classify the document type based on extension."""
        if suffix in (".md", ".markdown", ".rst", ".adoc", ".txt"):
            return DocumentType.DOCUMENT
        elif suffix in (".log",):
            return DocumentType.DOCUMENT
        elif suffix in (".json", ".yaml", ".yml", ".xml", ".toml", ".ini", ".cfg", ".conf"):
            return DocumentType.DOCUMENT
        elif suffix in (".csv",):
            return DocumentType.DOCUMENT
        return DocumentType.DOCUMENT
