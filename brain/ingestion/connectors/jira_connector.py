"""
Agent Brain — Jira Connector (Enterprise)
=====================================================
Connects to Jira via REST API v3 for incremental sync.

Implements BaseConnector protocol:
- fetch_incremental(since) → SyncItems with summaries + full content
- fetch_item(uri) → single ticket on-demand

Architecture:
    Uses raw `requests` against Jira REST API v3 (no jira-python dependency).
    Credentials resolved via AppConfig (PostgreSQL-backed), matching the
    same pattern used by tools/jira_tools.py.

Requires: jira.server_url, jira.email, jira.api_token in ag_system.configs
"""

import logging
from typing import Optional

import requests

from brain.ingestion.connectors.base_connector import BaseConnector, SyncItem

logger = logging.getLogger(__name__)

# Servers already warned about a missing 'Browse projects' permission, so the
# every-30-min sync doesn't emit the same WARNING forever (see below).
_NO_PROJECT_WARNED: set = set()


class JiraConnector(BaseConnector):
    """Connects to Jira using REST API v3 for incremental sync.

    Credentials are resolved via AppConfig at call time, so the connector
    works correctly even when the Gateway starts before DB config is loaded.
    """

    def __init__(self, app_config=None):
        """Initialize the Jira connector.

        Args:
            app_config: AppConfig instance for resolving credentials from
                        PostgreSQL. If None, falls back to a lazy import
                        of the global singleton.
        """
        if app_config is None:
            from core.config import app_config as _cfg
            app_config = _cfg
        self._app_config = app_config
        self._project_cache: list[str] = []


    def _resolve_credentials(self) -> tuple:
        """Resolve Jira credentials from AppConfig (DB → env → default).

        Returns:
            (server_url, email, api_token) — any may be empty string.
        """
        server = self._app_config.resolve_env("JIRA_SERVER_URL", "").rstrip("/")
        email = self._app_config.resolve_env("JIRA_EMAIL", "")
        token = self._app_config.resolve_env("JIRA_API_TOKEN", "")
        return server, email, token

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        }


    def _discover_projects(self, server: str, auth: tuple, headers: dict) -> list[str]:
        """Discover accessible Jira project keys.

        Jira REST API v3 /search/jql requires bounded queries — unbounded
        JQL like "ORDER BY updated DESC" is rejected. We need to know
        which projects exist to build "project in (SCRUM, DEV, ...)" filters.

        Uses the paginated `/project/search` endpoint. The older `/project`
        endpoint is deprecated on Jira Cloud and frequently returns an empty
        array even for valid credentials — which previously surfaced as a silent
        `no_projects_found` every sync cycle. `/project/search` is the supported
        replacement and returns the projects the token is permitted to browse.

        Project keys are cached for the lifetime of the connector instance
        to avoid an extra API call on every sync cycle.
        """
        if self._project_cache:
            return self._project_cache

        try:
            projects: list[str] = []
            start_at, page_size = 0, 50
            last_status = None
            while True:
                url = f"{server}/rest/api/3/project/search"
                resp = requests.get(
                    url, headers=headers, auth=auth, timeout=15,
                    params={"startAt": start_at, "maxResults": page_size},
                )
                resp.encoding = "utf-8"
                last_status = resp.status_code
                resp.raise_for_status()

                payload = resp.json()
                values = payload.get("values", [])
                projects.extend(p["key"] for p in values)
                if payload.get("isLast", True) or not values:
                    break
                start_at += len(values)

            if projects:
                self._project_cache = projects
                _NO_PROJECT_WARNED.discard(server)  # recovered → allow future warns
                logger.info("jira_connector.projects_discovered: %s", ", ".join(projects))
            else:
                # Auth succeeded (2xx) but zero browsable projects → almost always
                # a permission scope issue on the API token, not a transient fault.
                # This is a stable config state re-checked every sync, so warn ONCE
                # per server and drop repeats to debug to avoid log spam.
                if server not in _NO_PROJECT_WARNED:
                    _NO_PROJECT_WARNED.add(server)
                    logger.warning(
                        "jira_connector.no_browsable_projects: server=%s http_status=%s — "
                        "the API token has no 'Browse projects' permission on any project "
                        "(suppressing repeats for this server)",
                        server, last_status,
                    )
                else:
                    logger.debug(
                        "jira_connector.no_browsable_projects (repeat): server=%s", server)
            return projects
        except Exception as e:
            logger.error("jira_connector.project_discovery_failed: %s", str(e))
            return []

    @property
    def source_type(self) -> str:
        return "jira"

    @property
    def is_configured(self) -> bool:
        server, email, token = self._resolve_credentials()
        return bool(server and email and token)

    def fetch_incremental(
        self,
        since: Optional[str] = None,
        max_results: int = 100,
    ) -> list[SyncItem]:
        """Fetch tickets updated since the given timestamp.

        Args:
            since: ISO timestamp (e.g., "2026-05-28T10:00:00Z").
                   If None, fetches the most recent max_results tickets.
            max_results: Max tickets per sync cycle.
        """
        server, email, token = self._resolve_credentials()
        if not server or not email or not token:
            logger.debug("jira_connector.not_configured")
            return []

        auth = (email, token)
        headers = self._headers()

        projects = self._discover_projects(server, auth, headers)
        if not projects:
            # _discover_projects already logged the specific reason (auth failure
            # vs. zero browsable projects); keep this path quiet to avoid a
            # duplicate warning every sync cycle.
            logger.debug("jira_connector.no_projects_found")
            return []

        project_filter = ", ".join(projects)
        jql_parts = [f"project in ({project_filter})"]
        if since:
            jql_date = since.replace("T", " ").replace("Z", "")[:16]
            jql_parts.append(f'updated >= "{jql_date}"')
        jql = " AND ".join(jql_parts) + " ORDER BY updated DESC"

        try:
            logger.info(
                "jira_connector.incremental_sync: jql=%s max=%d",
                jql[:120], max_results,
            )

            url = f"{server}/rest/api/3/search/jql"
            params = {
                "jql": jql,
                "maxResults": max_results,
                "fields": "summary,description,status,assignee,reporter,"
                          "issuetype,comment,updated,priority",
            }

            resp = requests.get(
                url,
                headers=headers,
                auth=auth,
                params=params,
                timeout=30,
            )
            resp.encoding = "utf-8"
            resp.raise_for_status()

            issues = resp.json().get("issues", [])

            items = []
            for issue in issues:
                item = self._issue_to_sync_item(issue, server)
                if item:
                    items.append(item)

            logger.info(
                "jira_connector.synced: count=%d since=%s",
                len(items), since or "full",
            )
            return items

        except requests.HTTPError as e:
            logger.error(
                "jira_connector.incremental_failed: status=%s body=%s",
                e.response.status_code, e.response.text[:300],
            )
            return []
        except Exception as e:
            logger.error("jira_connector.incremental_failed: %s", str(e))
            return []

    def fetch_item(self, source_uri: str) -> Optional[SyncItem]:
        """Fetch a single ticket by URI (e.g., 'jira://PROJ-1234')."""
        server, email, token = self._resolve_credentials()
        if not server or not email or not token:
            return None

        issue_key = source_uri.replace("jira://", "")

        try:
            url = f"{server}/rest/api/3/issue/{issue_key}"
            params = {
                "fields": "summary,description,status,assignee,reporter,"
                          "issuetype,comment,updated,priority",
            }

            resp = requests.get(
                url,
                headers=self._headers(),
                auth=(email, token),
                params=params,
                timeout=15,
            )
            resp.encoding = "utf-8"
            resp.raise_for_status()

            issue = resp.json()
            return self._issue_to_sync_item(issue, server)

        except Exception as e:
            logger.error(
                "jira_connector.fetch_item_failed: key=%s error=%s",
                issue_key, str(e),
            )
            return None


    def _issue_to_sync_item(self, issue: dict, server: str) -> Optional[SyncItem]:
        """Convert a Jira REST API issue JSON to a SyncItem."""
        try:
            key = issue["key"]
            fields = issue.get("fields", {})

            summary_text = fields.get("summary", "")

            description = self._extract_adf_text(fields.get("description"))

            assignee_obj = fields.get("assignee") or {}
            assignee = assignee_obj.get("displayName", "Unassigned")

            reporter_obj = fields.get("reporter") or {}
            reporter = reporter_obj.get("displayName", "Unknown")

            status = (fields.get("status") or {}).get("name", "Unknown")
            issue_type = (fields.get("issuetype") or {}).get("name", "Task")
            updated = fields.get("updated", "")
            priority = (fields.get("priority") or {}).get("name", "Medium")

            content_parts = [
                f"Jira Ticket: {key}",
                f"Title: {summary_text}",
                f"Type: {issue_type}",
                f"Status: {status}",
                f"Priority: {priority}",
                f"Assignee: {assignee}",
                f"Reporter: {reporter}",
                "\n--- Description ---",
                description,
            ]

            comment_people = []
            comment_block = fields.get("comment") or {}
            comments = comment_block.get("comments", [])
            if comments:
                content_parts.append("\n--- Comments ---")
                for comment in comments[-10:]:
                    author_obj = comment.get("author") or {}
                    author = author_obj.get("displayName", "Unknown")
                    comment_people.append(author)
                    comment_text = self._extract_adf_text(comment.get("body"))
                    created = comment.get("created", "")
                    content_parts.append(
                        f"[{created}] {author}: {comment_text}"
                    )

            full_content = "\n".join(content_parts)

            summary = self._extract_summary(
                f"{summary_text}. {description}. Status: {status}. "
                f"Assigned to {assignee}."
            )

            people = list({assignee, reporter} | set(comment_people))

            return SyncItem(
                source_uri=f"jira://{key}",
                source_type="jira",
                title=f"[{key}] {summary_text}",
                summary=summary,
                full_content=full_content,
                people=people,
                status=status.lower(),
                updated_at=updated,
                metadata={
                    "issue_key": key,
                    "issue_type": issue_type,
                    "priority": priority,
                    "assignee": assignee,
                    "reporter": reporter,
                    "url": f"{server}/browse/{key}",
                },
            )
        except Exception as e:
            logger.error("jira_connector.parse_failed: %s", str(e))
            return None

    @staticmethod
    def _extract_adf_text(adf_node) -> str:
        """Extract plain text from Atlassian Document Format (ADF) JSON.

        Jira REST API v3 returns description/comment bodies as ADF — a nested
        JSON tree of content nodes. This recursively walks the tree and
        concatenates all text nodes into a plain-text string.

        If the input is already a plain string (v2 compat), returns it as-is.
        If None/empty, returns empty string.
        """
        if adf_node is None:
            return ""
        if isinstance(adf_node, str):
            return adf_node

        if not isinstance(adf_node, dict):
            return str(adf_node)

        parts = []

        def _walk(node):
            if isinstance(node, dict):
                if node.get("type") == "text":
                    parts.append(node.get("text", ""))
                for child in node.get("content", []):
                    _walk(child)
                if node.get("type") in ("paragraph", "heading", "listItem",
                                        "blockquote", "codeBlock", "rule"):
                    parts.append("\n")

        _walk(adf_node)
        return "".join(parts).strip()
