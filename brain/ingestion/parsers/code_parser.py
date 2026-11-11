from __future__ import annotations

"""
Agent Brain — Code Parser
============================
Parses source code files while PRESERVING the code exactly as-is.
No stripping, no reformatting — the LLM needs to see the real code.

Adds metadata: language, line count, structural indicators
(has classes, has functions, has imports).
"""

import re
from pathlib import Path
import logging

from brain.models import DocumentType
from brain.ingestion.parsers.base import BaseParser, ParsedDocument

logger = logging.getLogger(__name__)


LANGUAGE_MAP = {
    ".cs": "csharp", ".c": "c", ".cpp": "cpp", ".h": "c_header",
    ".java": "java", ".kt": "kotlin", ".scala": "scala",
    ".js": "javascript", ".ts": "typescript", ".jsx": "jsx", ".tsx": "tsx",
    ".html": "html", ".css": "css", ".scss": "scss", ".less": "less",
    ".vue": "vue", ".svelte": "svelte",
    ".py": "python", ".pyw": "python", ".pyi": "python_stub",
    ".dart": "dart", ".swift": "swift", ".m": "objective_c",
    ".go": "go", ".rs": "rust", ".zig": "zig",
    ".rb": "ruby", ".php": "php", ".lua": "lua", ".pl": "perl",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".fish": "shell",
    ".ps1": "powershell", ".bat": "batch", ".cmd": "batch",
    ".sql": "sql", ".graphql": "graphql", ".proto": "protobuf",
    ".tex": "latex", ".r": "r", ".jl": "julia",
    ".tf": "terraform", ".dockerfile": "dockerfile",
}


class CodeParser(BaseParser):
    """
    Parser for source code files.
    Preserves code exactly as-is and adds structural metadata.
    """

    SUPPORTED_EXTENSIONS = list(LANGUAGE_MAP.keys())

    def can_parse(self, file_path: Path) -> bool:
        suffix = file_path.suffix.lower()
        name = file_path.name.lower()
        return suffix in LANGUAGE_MAP or name in ("dockerfile", "makefile", "rakefile", "gemfile")

    def parse(self, file_path: Path) -> ParsedDocument:
        self._validate_file(file_path)

        raw_bytes = file_path.read_bytes()
        raw_size = len(raw_bytes)

        try:
            content = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            content = raw_bytes.decode("utf-8", errors="replace")

        suffix = file_path.suffix.lower()
        name = file_path.name.lower()
        language = LANGUAGE_MAP.get(suffix, "unknown")
        if name == "dockerfile":
            language = "dockerfile"
        elif name == "makefile":
            language = "makefile"

        lines = content.split("\n")
        line_count = len(lines)

        metadata = {
            "format": "code",
            "language": language,
            "line_count": line_count,
            "char_count": len(content),
            "has_classes": self._has_pattern(content, language, "class"),
            "has_functions": self._has_pattern(content, language, "function"),
            "has_imports": self._has_pattern(content, language, "import"),
        }

        title = file_path.name

        logger.info("code_parser.parsed: path=%s language=%s lines=%s", str(file_path), language, line_count)

        return ParsedDocument(
            content=content,
            source_path=str(file_path),
            document_type=DocumentType.CODE,
            title=title,
            metadata=metadata,
            raw_size_bytes=raw_size,
        )

    def _has_pattern(self, content: str, language: str, pattern_type: str) -> bool:
        """Check if code contains structural patterns (classes, functions, imports)."""
        patterns = {
            "class": {
                "python": r"^\s*class\s+\w+",
                "csharp": r"\b(class|interface|struct|record)\s+\w+",
                "java": r"\b(class|interface|enum)\s+\w+",
                "typescript": r"\b(class|interface)\s+\w+",
                "javascript": r"\bclass\s+\w+",
                "dart": r"\bclass\s+\w+",
                "go": r"\btype\s+\w+\s+struct",
                "rust": r"\b(struct|enum|trait)\s+\w+",
            },
            "function": {
                "python": r"^\s*def\s+\w+",
                "csharp": r"\b(public|private|protected|internal|static)\s+\w+.*\(",
                "java": r"\b(public|private|protected|static)\s+\w+.*\(",
                "typescript": r"\b(function|const|let|var)\s+\w+.*[=(]",
                "javascript": r"\b(function|const|let|var)\s+\w+.*[=(]",
                "dart": r"\b\w+\s+\w+\s*\(",
                "go": r"\bfunc\s+",
                "rust": r"\bfn\s+\w+",
            },
            "import": {
                "python": r"^\s*(import|from)\s+",
                "csharp": r"^\s*using\s+",
                "java": r"^\s*import\s+",
                "typescript": r"^\s*(import|require)\s*",
                "javascript": r"^\s*(import|require)\s*",
                "dart": r"^\s*import\s+",
                "go": r"^\s*import\s+",
                "rust": r"^\s*use\s+",
            },
        }

        type_patterns = patterns.get(pattern_type, {})
        pattern = type_patterns.get(language)

        if pattern:
            return bool(re.search(pattern, content, re.MULTILINE))

        fallback = {
            "class": r"\bclass\s+\w+",
            "function": r"\b(function|def|func|fn)\s+\w+",
            "import": r"\b(import|using|require|include)\b",
        }
        fb_pattern = fallback.get(pattern_type)
        if fb_pattern:
            return bool(re.search(fb_pattern, content, re.MULTILINE))

        return False
