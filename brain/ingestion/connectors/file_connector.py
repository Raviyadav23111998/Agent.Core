from __future__ import annotations

"""
Agent Brain — File Connector
================================
Scans a local directory, discovers files, and routes each
to the correct parser based on file extension.

Uses a registry pattern: extension → parser mapping.
Supports glob filters and ignore patterns.
"""

from pathlib import Path
from typing import Optional
import logging

from brain.ingestion.parsers.base import BaseParser, ParsedDocument
from brain.ingestion.parsers.text_parser import TextParser
from brain.ingestion.parsers.pdf_parser import PdfParser
from brain.ingestion.parsers.code_parser import CodeParser
from brain.ingestion.parsers.email_parser import EmailParser

logger = logging.getLogger(__name__)


DEFAULT_IGNORE_DIRS = {
    ".git", ".svn", ".hg",
    "node_modules", "vendor", "venv", ".venv", "env",
    "__pycache__", ".pytest_cache", ".mypy_cache",
    "bin", "obj", "build", "dist", "target",
    ".vs", ".idea", ".vscode",
    "packages", ".nuget",
    ".angular", ".next", ".nuxt", ".svelte-kit", ".cache",
    ".ag_data", ".gemini",
}

DEFAULT_IGNORE_FILES = {
    ".DS_Store", "Thumbs.db",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "composer.lock", "Gemfile.lock", "Pipfile.lock",
}

BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".mp3", ".mp4", ".avi", ".mov", ".wav", ".flac",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".exe", ".dll", ".so", ".dylib",
    ".woff", ".woff2", ".ttf", ".eot",
    ".pyc", ".pyo", ".class",
    ".db", ".sqlite", ".sqlite3",
}


class FileConnector:
    """
    Scans a directory and produces ParsedDocument objects
    by routing files to the appropriate parser.
    """

    def __init__(
        self,
        ignore_dirs: Optional[set[str]] = None,
        ignore_files: Optional[set[str]] = None,
        max_file_size_mb: float = 10.0,
    ):
        self.ignore_dirs = ignore_dirs or DEFAULT_IGNORE_DIRS
        self.ignore_files = ignore_files or DEFAULT_IGNORE_FILES
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)

        self._parsers: list[BaseParser] = [
            EmailParser(),
            PdfParser(),
            CodeParser(),
            TextParser(),
        ]

    def scan_directory(
        self,
        directory: Path,
        recursive: bool = True,
        extensions_filter: Optional[set[str]] = None,
    ) -> list[ParsedDocument]:
        """
        Scan a directory and parse all supported files.

        Args:
            directory: Root directory to scan
            recursive: Whether to recurse into subdirectories
            extensions_filter: If set, only parse files with these extensions

        Returns:
            List of successfully parsed documents
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        documents = []
        skipped = 0
        errors = 0

        for file_path in self._discover_files(directory, recursive):
            if extensions_filter and file_path.suffix.lower() not in extensions_filter:
                continue

            try:
                doc = self.parse_file(file_path)
                if doc and not doc.is_empty:
                    documents.append(doc)
                else:
                    skipped += 1
            except Exception as e:
                logger.warning("file_connector.parse_error: path=%s error=%s", str(file_path), str(e))
                errors += 1

        logger.info("file_connector.scan_complete: directory=%s parsed=%s skipped=%s errors=%s", str(directory), len(documents), skipped, errors)

        return documents

    def parse_file(self, file_path: Path) -> Optional[ParsedDocument]:
        """Parse a single file using the appropriate parser."""
        file_path = Path(file_path)

        if not file_path.exists() or not file_path.is_file():
            return None

        for parser in self._parsers:
            if parser.can_parse(file_path):
                return parser.parse(file_path)

        logger.debug("file_connector.no_parser: path=%s", str(file_path))
        return None

    def _discover_files(self, directory: Path, recursive: bool) -> list[Path]:
        """Discover all parseable files in a directory."""
        files = []

        if recursive:
            iterator = directory.rglob("*")
        else:
            iterator = directory.glob("*")

        for path in sorted(iterator):
            if not path.is_file():
                continue

            if any(ignored in path.parts for ignored in self.ignore_dirs):
                continue

            if path.name in self.ignore_files:
                continue

            if path.suffix.lower() in BINARY_EXTENSIONS:
                continue

            try:
                if path.stat().st_size > self.max_file_size_bytes:
                    logger.debug("file_connector.file_too_large: path=%s", str(path))
                    continue
            except OSError:
                continue

            files.append(path)

        return files

    def get_supported_extensions(self) -> set[str]:
        """Get all file extensions supported by registered parsers."""
        extensions = set()
        for parser in self._parsers:
            extensions.update(parser.SUPPORTED_EXTENSIONS)
        return extensions
