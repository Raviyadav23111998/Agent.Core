"""
Agent Brain — Confluence Connector
=======================================
Ingests wiki pages from Atlassian Confluence using the REST API v2.
Supports space listing, page traversal, HTML-to-text conversion,
and page label/tag extraction.

Credentials resolved via CredentialRegistry (vault + bindings).
Fallback: CONFLUENCE_URL, CONFLUENCE_EMAIL, CONFLUENCE_API_TOKEN env vars.
"""

import os
import urllib.parse
from typing import Optional
import logging
import httpx
from bs4 import BeautifulSoup

from brain.models import DocumentType
from brain.ingestion.parsers.base import ParsedDocument

logger = logging.getLogger(__name__)


class ConfluenceConnector:
    """Connects to Atlassian Confluence to ingest wiki pages."""

    def __init__(self, credential_registry=None):
        self._registry = credential_registry
        self.base_url = ""
        self.email = ""
        self.api_token = ""
        self._load_credentials()

    def _load_credentials(self):
        """Load Confluence credentials from registry or env vars."""
        if self._registry:
            try:
                cred = self._registry.resolve("confluence")
                self.base_url = cred.config.get("url", "").rstrip("/")
                self.email = cred.config.get("email", "")
                self.api_token = cred.secret
                return
            except Exception as e:
                logger.debug("confluence._load_credentials registry failed: %s", e)
        
        self.base_url = os.getenv("CONFLUENCE_URL", "").rstrip("/")
        self.email = os.getenv("CONFLUENCE_EMAIL", "")
        self.api_token = os.getenv("CONFLUENCE_API_TOKEN", "")

    @property
    def is_configured(self) -> bool:
        return bool(self.base_url and self.email and self.api_token)

    def _client(self) -> httpx.Client:
        """Create an authenticated HTTP client."""
        return httpx.Client(
            base_url=f"{self.base_url}/wiki/api/v2" if "/wiki" not in self.base_url
            else f"{self.base_url}/api/v2",
            auth=(self.email, self.api_token),
            timeout=30.0,
        )


    def list_spaces(self) -> list[dict]:
        """List all available Confluence spaces."""
        if not self.is_configured:
            logger.warning("confluence_connector.not_configured")
            return []

        spaces = []
        try:
            with self._client() as client:
                resp = client.get("/spaces", params={"limit": 100})
                if resp.status_code != 200:
                    logger.error("confluence_connector.api_error: status=%s", resp.status_code)
                    return []

                for space in resp.json().get("results", []):
                    spaces.append({
                        "id": space.get("id"),
                        "key": space.get("key"),
                        "name": space.get("name"),
                        "type": space.get("type"),
                        "description": space.get("description", {}).get("plain", {}).get("value", ""),
                    })

            logger.info("confluence_connector.spaces_listed: count=%s", len(spaces))
        except Exception as e:
            logger.error("confluence_connector.list_spaces_failed: error=%s", str(e))

        return spaces


    def ingest_space(
        self,
        space_key: str,
        max_results: int = 100,
    ) -> list[ParsedDocument]:
        """
        Ingest all pages from a Confluence space.

        Args:
            space_key: The space key (e.g. 'ENG', 'DEVOPS').
            max_results: Maximum pages to fetch.

        Returns:
            List of ParsedDocument objects.
        """
        if not self.is_configured:
            logger.warning("confluence_connector.not_configured")
            return []

        documents = []
        try:
            with self._client() as client:
                params = {
                    "spaceKey": space_key,
                    "limit": min(max_results, 50),
                    "body-format": "storage",
                    "sort": "-modified-date",
                }

                cursor = None
                fetched = 0

                while fetched < max_results:
                    if cursor:
                        params["cursor"] = cursor

                    resp = client.get("/pages", params=params)
                    if resp.status_code != 200:
                        logger.error("confluence_connector.api_error: status=%s body=%s", resp.status_code, resp.text[:200])
                        break

                    data = resp.json()
                    pages = data.get("results", [])
                    if not pages:
                        break

                    for page in pages:
                        parsed = self._parse_page(page, space_key)
                        if parsed:
                            documents.append(parsed)
                            fetched += 1

                    links = data.get("_links", {})
                    next_link = links.get("next")
                    if not next_link:
                        break
                    parsed_url = urllib.parse.urlparse(next_link)
                    query_params = urllib.parse.parse_qs(parsed_url.query)
                    cursor = query_params.get("cursor", [None])[0]
                    if not cursor:
                        break

            logger.info("confluence_connector.ingested: space=%s count=%s", space_key, len(documents))
        except Exception as e:
            logger.error("confluence_connector.ingest_failed: space=%s error=%s", space_key, str(e))

        return documents

    def _parse_page(self, page: dict, space_key: str) -> Optional[ParsedDocument]:
        """Convert a Confluence page to a ParsedDocument."""
        page_id = page.get("id", "")
        title = page.get("title", "Untitled")
        status = page.get("status", "current")

        body_storage = page.get("body", {}).get("storage", {}).get("value", "")
        if not body_storage:
            return None

        soup = BeautifulSoup(body_storage, "html.parser")
        body_text = soup.get_text(separator="\n", strip=True)

        if not body_text.strip():
            return None

        content_parts = [
            f"Title: {title}",
            f"Space: {space_key}",
            f"Status: {status}",
            "\n--- Content ---",
            body_text,
        ]

        labels = [l.get("name", "") for l in page.get("labels", {}).get("results", [])]

        content = "\n".join(content_parts)

        metadata = {
            "format": "confluence_page",
            "page_id": page_id,
            "space_key": space_key,
            "status": status,
            "author": page.get("version", {}).get("authorId", ""),
            "version": page.get("version", {}).get("number", 1),
            "tags": labels,
            "url": f"{self.base_url}/wiki/spaces/{space_key}/pages/{page_id}",
        }

        parent_id = page.get("parentId")
        if parent_id:
            metadata["parent_page_id"] = parent_id

        return ParsedDocument(
            content=content,
            source_path=f"confluence://{space_key}/{page_id}",
            document_type=DocumentType.DOCUMENT,
            title=f"[{space_key}] {title}",
            metadata=metadata,
            raw_size_bytes=len(content.encode("utf-8")),
        )
