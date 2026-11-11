from __future__ import annotations

"""
Agent Brain — Git Connector
================================
Reads a Git repository (local path or clone URL) and produces
ParsedDocument objects for all tracked source files.

Preserves the full relative file path as metadata, so the LLM
knows that `src/auth/TokenService.cs` is in the auth module.
Extracts top-level commit metadata for the repository.
"""

from pathlib import Path
from typing import Optional
import logging

from brain.ingestion.connectors.file_connector import FileConnector
from brain.ingestion.parsers.base import ParsedDocument

logger = logging.getLogger(__name__)


class GitConnector:
    """
    Reads a Git repository and produces ParsedDocuments for all tracked files.
    Delegates individual file parsing to the FileConnector.
    """

    def __init__(
        self,
        max_file_size_mb: float = 5.0,
    ):
        self.file_connector = FileConnector(max_file_size_mb=max_file_size_mb)

    def ingest_repository(
        self,
        repo_path: str,
        clone_url: Optional[str] = None,
        branch: str = "main",
        extensions_filter: Optional[set[str]] = None,
    ) -> list[ParsedDocument]:
        """
        Ingest all tracked files from a Git repository.

        Args:
            repo_path: Local path to the repository (or target clone directory)
            clone_url: If provided, clone the repo to repo_path first
            branch: Branch to checkout (default: main)
            extensions_filter: Only parse files with these extensions

        Returns:
            List of ParsedDocument objects with git metadata
        """
        repo_path = Path(repo_path)

        if clone_url:
            repo_path = self._clone_repo(clone_url, repo_path, branch)

        if not repo_path.is_dir():
            raise NotADirectoryError(f"Repository path not found: {repo_path}")

        repo_metadata = self._extract_repo_metadata(repo_path)
        repo_name = repo_metadata.get("name", repo_path.name)

        documents = self.file_connector.scan_directory(
            repo_path,
            recursive=True,
            extensions_filter=extensions_filter,
        )

        for doc in documents:
            try:
                relative_path = Path(doc.source_path).relative_to(repo_path)
                doc.metadata["repo_relative_path"] = str(relative_path)
            except ValueError:
                doc.metadata["repo_relative_path"] = doc.source_path

            doc.metadata["repository"] = repo_name
            doc.metadata.update({
                k: v for k, v in repo_metadata.items()
                if k not in ("name",)
            })

            doc.source_path = f"repo://{repo_name}/{doc.metadata['repo_relative_path']}"

        logger.info("git_connector.ingested: repo=%s files=%s branch=%s", repo_name, len(documents), branch)

        return documents

    def _clone_repo(self, url: str, target: Path, branch: str) -> Path:
        """Clone a git repository."""
        try:
            import git
            if target.exists():
                logger.info("git_connector.repo_exists: path=%s", str(target))
                repo = git.Repo(target)
                repo.remotes.origin.pull()
            else:
                logger.info("git_connector.cloning: url=%s target=%s", url, str(target))
                git.Repo.clone_from(url, target, branch=branch)
            return target
        except ImportError:
            raise ImportError("gitpython is required for cloning. Install: pip install gitpython")

    def _extract_repo_metadata(self, repo_path: Path) -> dict:
        """Extract metadata about the repository."""
        metadata = {"name": repo_path.name}

        try:
            import git
            repo = git.Repo(repo_path)

            try:
                metadata["branch"] = str(repo.active_branch)
            except TypeError:
                metadata["branch"] = "detached"

            try:
                latest = repo.head.commit
                metadata["latest_commit"] = str(latest.hexsha)[:8]
                metadata["latest_commit_message"] = latest.message.strip()[:200]
                metadata["latest_commit_author"] = str(latest.author)
                metadata["latest_commit_date"] = str(latest.committed_datetime)
            except Exception:
                pass

            try:
                if repo.remotes:
                    metadata["remote_url"] = str(repo.remotes.origin.url)
            except Exception:
                pass

        except ImportError:
            logger.debug("git_connector.gitpython_not_available")
        except Exception as e:
            logger.debug("git_connector.metadata_extraction_failed: error=%s", str(e))

        return metadata
