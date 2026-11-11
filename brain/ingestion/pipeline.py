from __future__ import annotations

"""
Agent Brain — Ingestion Pipeline
====================================
The main orchestrator that connects:
  Data Connectors → Format Parsers → LLM Extractor → Storage Manager

This is the single entry point for ingesting ANY data into the brain.
It handles the full flow:
  1. Discover files (via connector)
  2. Parse them into clean text (via parsers)
  3. Optionally extract entities/relations (via LLM)
  4. Write to all storage pillars (via StorageManager)
"""

from pathlib import Path
from typing import Optional
import logging
import asyncio

from brain.models import DocumentType
from brain.exceptions import DuplicateDocumentError, ExtractionError
from brain.ingestion.parsers.base import ParsedDocument
from brain.ingestion.connectors.file_connector import FileConnector
from brain.ingestion.connectors.git_connector import GitConnector
from brain.storage.manager import StorageManager
from core.tenant import TENANT_SCOPE

logger = logging.getLogger(__name__)


GRAPH_EXTRACTABLE_TYPES = {
    DocumentType.EMAIL,
    DocumentType.POLICY,
    DocumentType.DOCUMENT,
    DocumentType.MEETING_NOTES,
    DocumentType.TICKET,
}


def _try_import(module_path: str, class_name: str):
    """Lazy-import a connector, returning None if it is unavailable.

    "Unavailable" must cover more than ImportError: a connector whose dependency
    chain fails (e.g. Microsoft connectors when `azure-identity` is absent) can
    leave a partially-initialised module in sys.modules and then surface as an
    AttributeError on the class lookup. An OPTIONAL connector going missing must
    never take down the whole brain bundle, so we degrade to None on any error.
    """
    try:
        mod = __import__(module_path, fromlist=[class_name])
        return getattr(mod, class_name)()
    except Exception as e:  # noqa: BLE001 — optional dependency, degrade gracefully
        logger.debug("Optional connector unavailable: %s.%s (%s)", module_path, class_name, e)
        return None


class IngestionPipeline:
    """
    Main ingestion orchestrator.
    Connects data sources to the storage layer through parsing and extraction.
    """

    def __init__(
        self,
        storage: StorageManager,
        llm_client = None,
        enable_graph_extraction: bool = True,
    ):
        """
        Args:
            storage: The StorageManager with all pillars initialized
            llm_client: LLM client for graph extraction (optional)
            enable_graph_extraction: Whether to run LLM extraction
        """
        self.storage = storage
        self.llm = llm_client
        self.enable_extraction = enable_graph_extraction and llm_client is not None

        self.file_connector = FileConnector()
        self.git_connector = GitConnector()

        self.jira_connector = _try_import("brain.ingestion.connectors.jira_connector", "JiraConnector")
        self.outlook_connector = _try_import("brain.ingestion.connectors.microsoft.outlook_connector", "OutlookConnector")
        self.teams_connector = _try_import("brain.ingestion.connectors.microsoft.teams_connector", "TeamsConnector")
        self.onedrive_connector = _try_import("brain.ingestion.connectors.microsoft.onedrive_connector", "OneDriveConnector")
        self.slack_connector = _try_import("brain.ingestion.connectors.slack_connector", "SlackConnector")
        self.confluence_connector = _try_import("brain.ingestion.connectors.confluence_connector", "ConfluenceConnector")

        self.extractor = None
        if self.enable_extraction and self.llm:
            try:
                from brain.extraction.graph_extractor import GraphExtractor
                self.extractor = GraphExtractor(self.llm)
            except ImportError:
                logger.debug("Graph extractor unavailable")


    async def ingest_directory(
        self,
        directory: str,
        recursive: bool = True,
        extensions_filter: Optional[set[str]] = None,
        bundle_name: Optional[str] = None,
        owner_id: str = TENANT_SCOPE,
    ) -> dict:
        """
        Ingest all supported files from a directory.

        Args:
            directory: Path to the directory
            recursive: Whether to recurse into subdirectories
            extensions_filter: Only parse files with these extensions
            bundle_name: Optional name for the auto-created context bundle
            owner_id: Data owner ('__tenant__' for shared, user_id for private)

        Returns:
            Summary dict with counts of ingested, skipped, and failed documents
        """
        logger.info("pipeline.ingest_directory.start: directory=%s owner=%s",
                    directory, owner_id)

        parsed_docs = self.file_connector.scan_directory(
            Path(directory),
            recursive=recursive,
            extensions_filter=extensions_filter,
        )

        result = await self._process_documents(parsed_docs, owner_id=owner_id)

        name = bundle_name or Path(directory).name
        self._create_bundle_from_results(name, f"Auto-bundle from directory: {directory}", result)

        return result

    async def ingest_git_repo(
        self,
        repo_path: str,
        clone_url: Optional[str] = None,
        branch: str = "main",
        extensions_filter: Optional[set[str]] = None,
        bundle_name: Optional[str] = None,
        owner_id: str = TENANT_SCOPE,
    ) -> dict:
        """
        Ingest all files from a Git repository.

        Args:
            repo_path: Local path to the repository
            clone_url: Optional URL to clone from
            branch: Branch to checkout
            extensions_filter: Only parse specific file types
            bundle_name: Optional name for the auto-created context bundle
            owner_id: Data owner

        Returns:
            Summary dict
        """
        logger.info("pipeline.ingest_repo.start: repo=%s owner=%s", repo_path, owner_id)

        parsed_docs = self.git_connector.ingest_repository(
            repo_path=repo_path,
            clone_url=clone_url,
            branch=branch,
            extensions_filter=extensions_filter,
        )

        result = await self._process_documents(parsed_docs, owner_id=owner_id)

        name = bundle_name or Path(repo_path).name
        self._create_bundle_from_results(name, f"Auto-bundle from repo: {clone_url or repo_path} ({branch})", result)

        return result

    async def ingest_jira(
        self,
        jql_query: str,
        max_results: int = 100,
        bundle_name: Optional[str] = None,
        force_full: bool = False,
    ) -> dict:
        """
        Ingest tickets from Jira based on a JQL query.
        Uses watermark-based incremental sync — appends 'AND updated > ...'
        to the JQL so only newly updated tickets are fetched.

        Args:
            jql_query: The JQL query (e.g. 'project = RIDESHARE')
            max_results: Maximum number of tickets to ingest
            bundle_name: Optional name for the auto-created context bundle
            force_full: If True, ignores watermark and fetches all
        """
        watermark_key = f"jira:{jql_query[:50]}"

        effective_jql = jql_query
        if not force_full:
            since = self.storage.documents.get_watermark(watermark_key)
            if since:
                jira_date = since[:16].replace("T", " ")
                effective_jql = f"({jql_query}) AND updated > '{jira_date}'"
                logger.info("pipeline.ingest_jira.incremental: since=%s", jira_date)

        logger.info("pipeline.ingest_jira.start: jql=%s", effective_jql)

        parsed_docs = await asyncio.to_thread(
            self.jira_connector.ingest_project,
            jql_query=effective_jql,
            max_results=max_results,
        )

        if not parsed_docs:
            if effective_jql != jql_query:
                return {"status": "ok", "ingested": 0, "message": "No updated tickets since last sync"}
            return {"status": "failed", "reason": "No documents returned from Jira connector"}

        result = await self._process_documents(parsed_docs, owner_id=TENANT_SCOPE)

        if result.get("ingested", 0) > 0:
            from datetime import datetime, timezone
            self.storage.documents.set_watermark(
                watermark_key,
                datetime.now(timezone.utc).isoformat(),
                result["ingested"],
            )

        name = bundle_name or "Jira_Tickets"
        self._create_bundle_from_results(name, f"Auto-bundle from Jira JQL: {jql_query}", result)

        return result


    async def ingest_outlook_folder(
        self,
        folder_name: str = "Inbox",
        max_results: int = 50,
        bundle_name: Optional[str] = None,
        force_full: bool = False,
    ) -> dict:
        """
        Ingest emails from an Outlook folder.
        Uses watermark-based incremental sync — only fetches emails
        received AFTER the last successful sync timestamp.

        Args:
            folder_name: Outlook folder to ingest (default: Inbox)
            max_results: Max emails per sync
            bundle_name: Optional context bundle name
            force_full: If True, ignores watermark and fetches all
        """
        watermark_key = f"outlook:{folder_name}"

        since = ""
        if not force_full:
            since = self.storage.documents.get_watermark(watermark_key) or ""

        logger.info("pipeline.ingest_outlook.start: folder=%s mode=%s since=%s", folder_name, "incremental" if since else "full", since or "never")

        parsed_docs = await asyncio.to_thread(
            self.outlook_connector.ingest_folder,
            folder_name=folder_name,
            max_results=max_results,
            since=since,
        )

        if not parsed_docs:
            if since:
                return {"status": "ok", "ingested": 0, "message": "No new emails since last sync"}
            return {"status": "failed", "reason": "No documents returned from Outlook connector"}

        result = await self._process_documents(parsed_docs, owner_id=TENANT_SCOPE)

        if result.get("ingested", 0) > 0:
            newest_date = ""
            for doc in parsed_docs:
                doc_date = doc.metadata.get("date", "")
                if doc_date > newest_date:
                    newest_date = doc_date
            if newest_date:
                self.storage.documents.set_watermark(
                    watermark_key, newest_date, result["ingested"]
                )

        if bundle_name:
            self._create_bundle_from_results(bundle_name, f"Outlook Folder: {folder_name}", result)

        return result

    async def ingest_teams_channel(
        self,
        team_id: str,
        channel_id: str,
        max_results: int = 50,
        force_full: bool = False,
    ) -> dict:
        """Ingest messages from a specific Microsoft Teams channel."""
        watermark_key = f"teams_channel:{team_id}:{channel_id}"
        since = "" if force_full else self.storage.documents.get_watermark(watermark_key) or ""

        logger.info("pipeline.ingest_teams.start: team=%s channel=%s mode=%s", team_id, channel_id, "incremental" if since else "full")
        
        if not self.teams_connector:
            return {"status": "failed", "reason": "Teams connector not initialized"}

        try:
            documents = await asyncio.to_thread(
                self.teams_connector.ingest_channel,
                team_id,
                channel_id,
                max_results,
                since,
            )
            if not documents:
                if since:
                    return {"status": "ok", "ingested": 0, "message": "No new messages since last sync"}
                return {"status": "failed", "reason": "No documents returned from Teams connector"}
                
            result = await self._process_documents(documents)
            
            if result.get("ingested", 0) > 0:
                newest_date = max((doc.metadata.get("date", "") for doc in documents), default="")
                if newest_date:
                    self.storage.documents.set_watermark(watermark_key, newest_date, result["ingested"])
                    
            return result
        except Exception as e:
            logger.error("pipeline.ingest_teams.error: error=%s", str(e))
            return {"status": "failed", "reason": str(e)}

    async def ingest_teams_chat(
        self,
        chat_id: str,
        max_results: int = 50,
        force_full: bool = False,
    ) -> dict:
        """Ingest messages from a specific Microsoft Teams personal or group chat."""
        watermark_key = f"teams_chat:{chat_id}"
        since = "" if force_full else self.storage.documents.get_watermark(watermark_key) or ""

        logger.info("pipeline.ingest_teams_chat.start: chat=%s mode=%s", chat_id, "incremental" if since else "full")
        
        if not self.teams_connector:
            return {"status": "failed", "reason": "Teams connector not initialized"}

        try:
            documents = await asyncio.to_thread(
                self.teams_connector.ingest_chat,
                chat_id,
                max_results,
                since,
            )
            if not documents:
                if since:
                    return {"status": "ok", "ingested": 0, "message": "No new messages since last sync"}
                return {"status": "failed", "reason": "No documents returned from Teams connector"}
                
            result = await self._process_documents(documents)

            if result.get("ingested", 0) > 0:
                newest_date = max((doc.metadata.get("date", "") for doc in documents), default="")
                if newest_date:
                    self.storage.documents.set_watermark(watermark_key, newest_date, result["ingested"])
            
            return result
        except Exception as e:
            logger.error("pipeline.ingest_teams_chat.error: error=%s", str(e))
            return {"status": "failed", "reason": str(e)}

    async def ingest_onedrive_folder(
        self,
        folder_path: str = "root/children",
        max_results: int = 50,
        bundle_name: Optional[str] = None,
        force_full: bool = False,
    ) -> dict:
        watermark_key = f"onedrive:{folder_path}"
        since = "" if force_full else self.storage.documents.get_watermark(watermark_key) or ""

        logger.info("pipeline.ingest_onedrive.start: folder=%s mode=%s", folder_path, "incremental" if since else "full")
        
        parsed_docs = await asyncio.to_thread(
            self.onedrive_connector.ingest_folder,
            folder_path=folder_path,
            max_results=max_results,
            since=since,
        )
        
        if not parsed_docs:
            if since:
                return {"status": "ok", "ingested": 0, "message": "No new files since last sync"}
            return {"status": "failed", "reason": "No documents returned from OneDrive connector"}

        result = await self._process_documents(parsed_docs, owner_id=TENANT_SCOPE)
        
        if result.get("ingested", 0) > 0:
            newest_date = max((doc.metadata.get("date", "") for doc in parsed_docs), default="")
            if newest_date:
                self.storage.documents.set_watermark(watermark_key, newest_date, result["ingested"])
        
        if bundle_name:
            self._create_bundle_from_results(bundle_name, f"OneDrive Folder: {folder_path}", result)
            
        return result


    async def ingest_slack_channel(
        self,
        channel_id: str,
        max_results: int = 100,
        include_threads: bool = True,
        bundle_name: Optional[str] = None,
    ) -> dict:
        """
        Ingest messages from a Slack channel.

        Args:
            channel_id: Slack channel ID (e.g. 'C0123456789').
            max_results: Maximum messages to fetch.
            include_threads: Whether to fetch thread replies.
            bundle_name: Optional name for the auto-created context bundle.

        Returns:
            Summary dict.
        """
        logger.info("pipeline.ingest_slack.start: channel=%s", channel_id)

        parsed_docs = self.slack_connector.ingest_channel(
            channel_id=channel_id,
            max_results=max_results,
            include_threads=include_threads,
        )

        if not parsed_docs:
            return {"status": "failed", "reason": "No documents returned from Slack connector"}

        result = await self._process_documents(parsed_docs, owner_id=TENANT_SCOPE)

        name = bundle_name or f"Slack_{channel_id}"
        self._create_bundle_from_results(name, f"Slack Channel: {channel_id}", result)

        return result


    async def ingest_confluence_space(
        self,
        space_key: str,
        max_results: int = 100,
        bundle_name: Optional[str] = None,
    ) -> dict:
        """
        Ingest wiki pages from a Confluence space.

        Args:
            space_key: The Confluence space key (e.g. 'ENG').
            max_results: Maximum pages to fetch.
            bundle_name: Optional name for the auto-created context bundle.

        Returns:
            Summary dict.
        """
        logger.info("pipeline.ingest_confluence.start: space=%s", space_key)

        parsed_docs = self.confluence_connector.ingest_space(
            space_key=space_key,
            max_results=max_results,
        )

        if not parsed_docs:
            return {"status": "failed", "reason": "No documents returned from Confluence connector"}

        result = await self._process_documents(parsed_docs, owner_id=TENANT_SCOPE)

        name = bundle_name or f"Confluence_{space_key}"
        self._create_bundle_from_results(name, f"Confluence Space: {space_key}", result)

        return result

    async def ingest_file(self, file_path: str, owner_id: str = TENANT_SCOPE,
                          app_id: int = 1) -> dict:
        """
        Ingest a single file.

        Args:
            file_path: Path to the file
            owner_id: Data owner ('__tenant__' for shared, user_id for private)
            app_id: Application this document belongs to

        Returns:
            Summary dict with the ingested document details
        """
        logger.info("pipeline.ingest_file.start: file=%s owner=%s app=%s", file_path, owner_id, app_id)

        parsed = self.file_connector.parse_file(Path(file_path))
        if not parsed or parsed.is_empty:
            return {"status": "skipped", "reason": "empty or unsupported"}

        return await self._process_single_document(parsed, owner_id=owner_id, app_id=app_id)

    async def ingest_text(
        self,
        content: str,
        title: str = "",
        document_type: DocumentType = DocumentType.DOCUMENT,
        source: str = "manual_input",
        metadata: Optional[dict] = None,
        owner_id: str = TENANT_SCOPE,
        app_id: int = 1,
    ) -> dict:
        """
        Ingest raw text directly (for API-submitted content).

        Args:
            content: Raw text content
            title: Human-readable title
            document_type: Type classification
            source: Source identifier (unique per document; the default
                    'manual_input' is auto-suffixed with a UUID to prevent
                    ON CONFLICT overwrites between separate ingestions)
            metadata: Additional metadata
            owner_id: Data owner

        Returns:
            Summary dict
        """
        import uuid
        from datetime import datetime, timezone
        if source == "manual_input":
            ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            source = f"manual_input/{ts}_{uuid.uuid4().hex[:8]}"

        parsed = ParsedDocument(
            content=content,
            source_path=source,
            document_type=document_type,
            title=title,
            metadata=metadata or {},
        )

        return await self._process_single_document(parsed, owner_id=owner_id, app_id=app_id)


    async def _process_documents(self, parsed_docs: list[ParsedDocument],
                                  owner_id: str = TENANT_SCOPE,
                                  app_id: int = 1) -> dict:
        """Process a batch of parsed documents through the full pipeline."""
        results = {
            "total": len(parsed_docs),
            "ingested": 0,
            "skipped_duplicates": 0,
            "extraction_errors": 0,
            "storage_errors": 0,
            "documents": [],
        }

        for parsed in parsed_docs:
            try:
                result = await self._process_single_document(parsed, owner_id=owner_id, app_id=app_id)
                if result.get("status") == "ingested":
                    results["ingested"] += 1
                elif result.get("status") == "duplicate":
                    results["skipped_duplicates"] += 1
                results["documents"].append(result)

            except Exception as e:
                results["storage_errors"] += 1
                logger.error("pipeline.document_failed: source=%s error=%s", parsed.source_path, str(e))
                results["documents"].append({
                    "status": "error",
                    "source": parsed.source_path,
                    "error": str(e),
                })

        logger.info("pipeline.batch_complete: total=%s ingested=%s duplicates=%s errors=%s", results["total"], results["ingested"], results["skipped_duplicates"], results["storage_errors"])

        return results

    async def _process_single_document(self, parsed: ParsedDocument,
                                         owner_id: str = TENANT_SCOPE,
                                         app_id: int = 1) -> dict:
        """Process a single parsed document through the full pipeline."""
        graph_extraction = None

        if self.enable_extraction and self.extractor:
            if parsed.document_type in GRAPH_EXTRACTABLE_TYPES:
                try:
                    graph_extraction = await self.extractor.extract(
                        document_id="pending",
                        content=parsed.content,
                        document_type=parsed.document_type,
                        language=parsed.metadata.get("language", ""),
                    )
                except ExtractionError as e:
                    logger.warning("pipeline.extraction_skipped: source=%s reason=%s", parsed.source_path, str(e))

        try:
            doc = self.storage.ingest_document(
                content=parsed.content,
                source_path=parsed.source_path,
                document_type=parsed.document_type,
                title=parsed.title,
                metadata=parsed.metadata,
                graph_extraction=graph_extraction,
                owner_id=owner_id,
                app_id=app_id,
            )

            return {
                "status": "ingested",
                "document_id": doc.id,
                "title": doc.title,
                "type": doc.document_type.value,
                "tokens": doc.token_count,
                "source": doc.source_path,
                "graph_entities": len(graph_extraction.entities) if graph_extraction else 0,
                "graph_relations": len(graph_extraction.relations) if graph_extraction else 0,
            }

        except DuplicateDocumentError as e:
            logger.info("pipeline.duplicate_skipped: source=%s", parsed.source_path)
            return {
                "status": "duplicate",
                "source": parsed.source_path,
                "existing_id": e.existing_id,
            }


    def _create_bundle_from_results(self, name: str, description: str, result: dict):
        """Auto-create a context bundle from ingestion results."""
        ingested_docs = [
            d for d in result.get("documents", [])
            if d.get("status") == "ingested"
        ]
        if not ingested_docs:
            return

        try:
            existing = self.storage.context.get_bundle_by_name(name)
            if existing:
                logger.info("pipeline.bundle_exists: name=%s id=%s", name, existing.id)
                return

            total_tokens = sum(d.get("tokens", 0) for d in ingested_docs)
            bundle = self.storage.context.create_bundle(
                name=name,
                description=description,
                max_tokens=max(total_tokens * 2, 1_000_000),
            )

            doc_ids = [d["document_id"] for d in ingested_docs]
            documents = self.storage.documents.get_documents_by_ids(doc_ids)
            if documents:
                self.storage.context.add_documents_to_bundle(bundle.id, documents)

            logger.info("pipeline.bundle_created: name=%s documents=%s tokens=%s", name, len(documents), total_tokens)
        except Exception as e:
            logger.error("pipeline.bundle_creation_failed: name=%s error=%s", name, str(e))
