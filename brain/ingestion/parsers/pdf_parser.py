from __future__ import annotations

"""
Agent Brain — PDF Parser
============================
Extracts text from PDF documents using PyPDF2.
Pulls metadata from PDF info dict (author, creation date, title).
"""

from pathlib import Path
import logging

from brain.models import DocumentType
from brain.ingestion.parsers.base import BaseParser, ParsedDocument

logger = logging.getLogger(__name__)


class PdfParser(BaseParser):
    """
    Parser for PDF documents.
    Handles: .pdf
    """

    SUPPORTED_EXTENSIONS = [".pdf"]

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def parse(self, file_path: Path) -> ParsedDocument:
        self._validate_file(file_path)

        raw_size = file_path.stat().st_size
        metadata = {"format": "pdf"}

        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(str(file_path))

            pages = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pages.append(f"--- Page {i + 1} ---\n{text}")

            content = "\n\n".join(pages)

            info = reader.metadata
            if info:
                if info.title:
                    metadata["pdf_title"] = info.title
                if info.author:
                    metadata["author"] = info.author
                if info.creator:
                    metadata["creator"] = info.creator
                if info.subject:
                    metadata["subject"] = info.subject
                if hasattr(info, "creation_date") and info.creation_date:
                    metadata["created_date"] = str(info.creation_date)

            metadata["page_count"] = len(reader.pages)
            metadata["char_count"] = len(content)

            title = metadata.get("pdf_title", "")
            if not title:
                title = file_path.stem.replace("_", " ").replace("-", " ").title()

            logger.info("pdf_parser.parsed: path=%s pages=%s chars=%s", str(file_path), metadata["page_count"], metadata["char_count"])

        except ImportError:
            logger.error("pdf_parser.pypdf2_not_installed")
            content = f"[PDF file: {file_path.name} — PyPDF2 not installed for parsing]"
            title = file_path.stem.replace("_", " ").title()
            metadata["error"] = "PyPDF2 not installed"

        except Exception as e:
            logger.error("pdf_parser.failed: path=%s error=%s", str(file_path), str(e))
            content = f"[PDF file: {file_path.name} — Failed to parse: {e}]"
            title = file_path.stem.replace("_", " ").title()
            metadata["error"] = str(e)

        return ParsedDocument(
            content=content,
            source_path=str(file_path),
            document_type=DocumentType.POLICY,
            title=title,
            metadata=metadata,
            raw_size_bytes=raw_size,
        )
