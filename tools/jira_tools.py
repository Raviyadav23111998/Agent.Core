"""
Agent — Jira Action Tools
======================================
Jira ticket CRUD via REST API (no jira-python dependency).

Tools:
- jira_create_ticket:     Create a new issue
- jira_update_ticket:     Update fields (summary/description/priority/assignee/labels)
- jira_transition_ticket: Move an issue through its workflow (e.g. → Done)
- jira_add_comment:       Add a comment to an issue
- jira_add_worklog:       Log time spent against an issue (work log)
- jira_list_tickets:      Search with JQL

Credentials resolved via CredentialRegistry (vault + bindings).
"""

import requests
import logging

logger = logging.getLogger(__name__)

_STATUS_SYNONYMS = {
    "done": {"done", "resolve", "resolved", "close", "closed", "complete",
             "completed", "finish", "finished", "mark as done"},
    "in progress": {"in progress", "start", "start progress", "in-progress",
                    "begin", "doing", "work started"},
    "to do": {"to do", "todo", "open", "reopen", "backlog", "stop progress"},
    "in review": {"in review", "review", "code review", "ready for review",
                  "peer review"},
}


class JiraActionTools:
    """Jira ticket management via REST API.

    Credentials are resolved via CredentialRegistry at call time:
      - User context: uses the user's personal Jira credential
      - App context:  uses the app's bound Jira credential
      - System:       admin-configured global Jira (for cron jobs only)
    """

    def __init__(self, credential_registry=None, user_context_resolver=None, app_id: str = None):
        """
        Args:
            credential_registry:    CredentialRegistry instance
            user_context_resolver:  Callable returning current UserContext
            app_id:                 App ID for cron/automated context (optional)
        """
        self._registry = credential_registry
        self._user_context_resolver = user_context_resolver
        self._app_id = app_id
        self._configured = credential_registry is not None
        if not self._configured:
            logger.warning("JiraActionTools: no credential registry provided. Tools will NOT be registered.")

    def _resolve_credentials(self):
        """Resolve Jira credentials from the vault via registry.

        Resolution: user binding → (app binding → system) depending on context.
        """
        if not self._registry:
            return None, None, None

        user_id = None
        app_id = self._app_id
        if self._user_context_resolver:
            try:
                ctx = self._user_context_resolver()
                if ctx and hasattr(ctx, "user_id") and ctx.user_id:
                    uid_str = str(ctx.user_id).split("_")[-1]
                    user_id = int(uid_str) if uid_str.isdigit() else None
                if not app_id and ctx and getattr(ctx, "app_id", None):
                    app_id = ctx.app_id
            except Exception:
                pass

        try:
            cred = self._registry.resolve("jira", user_id=user_id, app_id=app_id)
            server = cred.config.get("server_url", "").rstrip("/")
            email = cred.config.get("email", "")
            logger.debug("Jira: resolved credential user_id=%s app_id=%s email=%s server=%s",
                         user_id, app_id, email, server)
            token = cred.secret
            return server, email, token
        except Exception as e:
            logger.error("Jira: credential resolution failed user_id=%s app_id=%s: %s",
                         user_id, app_id, str(e), exc_info=True)
            return None, None, None

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        }

    def _auth_or_error(self):
        """Resolve (auth, headers, server) or return (None, error_string).

        Centralises the credentials check so every tool reports a consistent,
        actionable message when Jira isn't connected for the current user.
        """
        server, email, token = self._resolve_credentials()
        if not server or not email or not token:
            return None, (
                "✗ Jira is not connected for you. Open Settings → Connections → "
                "Jira and add your Atlassian site URL, email, and API token."
            )
        return ((email, token), self._headers(), server), None

    def _account_id_for(self, server, auth, headers, identifier: str):
        """Resolve an assignee identifier (email or display name) to an accountId.

        Jira Cloud requires accountId (not name/email) for assignment. Returns
        the accountId string, or None if no match.
        """
        try:
            resp = requests.get(
                f"{server}/rest/api/2/user/search",
                headers=headers, auth=auth,
                params={"query": identifier, "maxResults": 5},
                timeout=10,
            )
            if resp.status_code == 200:
                users = resp.json()
                if users:
                    for u in users:
                        if u.get("emailAddress", "").lower() == identifier.lower():
                            return u.get("accountId")
                    return users[0].get("accountId")
        except Exception as e:
            logger.debug("Jira: assignee lookup failed for %r: %s", identifier, e)
        return None

    @staticmethod
    def _match_transition(status: str, transitions: list):
        """Find the best workflow transition for a requested status name.

        Matching tiers: exact name → synonym group → substring. Returns the
        transition dict or None.
        """
        want = status.strip().lower()

        for t in transitions:
            if t.get("name", "").lower() == want:
                return t
            if t.get("to", {}).get("name", "").lower() == want:
                return t

        synonyms = set()
        for canon, group in _STATUS_SYNONYMS.items():
            if want in group or want == canon:
                synonyms |= group
        if synonyms:
            for t in transitions:
                name = t.get("name", "").lower()
                to_name = t.get("to", {}).get("name", "").lower()
                if any(s in name or s in to_name for s in synonyms):
                    return t

        for t in transitions:
            name = t.get("name", "").lower()
            to_name = t.get("to", {}).get("name", "").lower()
            if want in name or name in want or want in to_name or to_name in want:
                return t
        return None

    def register_tools(self, mcp_server):
        if not self._configured:
            return

        instance = self

        @mcp_server.tool()
        def jira_create_ticket(
            project: str,
            summary: str,
            description: str = "",
            issue_type: str = "Task",
            priority: str = "Medium",
        ) -> str:
            """Create a new Jira ticket.

            Args:
                project: Project key (e.g., "SCRUM", "DEV")
                summary: Ticket title/summary
                description: Detailed description
                issue_type: Issue type (Task, Story, Epic, Bug). Defaults to Task.
                priority: Priority level (Highest, High, Medium, Low, Lowest)
            """
            try:
                server, email, token = instance._resolve_credentials()
                if not server or not email or not token:
                    logger.error(
                        "jira_create_ticket: credentials missing at call time "
                        "(server=%s, email=%s, token=%s)",
                        bool(server), bool(email), bool(token),
                    )
                    return (
                        "✗ Jira is not connected for you. Open Settings → "
                        "Connections → Jira and add your Atlassian site URL, "
                        "email, and API token."
                    )

                auth = (email, token)
                headers = instance._headers()
                project = project.strip().upper()

                valid_type = issue_type
                try:
                    proj_resp = requests.get(
                        f"{server}/rest/api/2/project/{project}",
                        headers=headers,
                        auth=auth,
                        timeout=10,
                    )
                    if proj_resp.status_code == 200:
                        available = [
                            t["name"] for t in proj_resp.json().get("issueTypes", [])
                            if not t.get("subtask", False)
                        ]
                        if available and valid_type not in available:
                            logger.warning(
                                "Issue type '%s' not in project %s (available: %s). Falling back to '%s'.",
                                valid_type, project, available, available[0],
                            )
                            valid_type = "Task" if "Task" in available else available[0]
                    elif proj_resp.status_code in (400, 404):
                        logger.warning(
                            "jira_create_ticket: project '%s' not found (HTTP %s). Auto-discovering...",
                            project, proj_resp.status_code,
                        )
                        all_proj_resp = requests.get(
                            f"{server}/rest/api/2/project",
                            headers=headers,
                            auth=auth,
                            timeout=10,
                        )
                        if all_proj_resp.status_code == 200:
                            projects = all_proj_resp.json()
                            if projects:
                                old_project = project
                                project = projects[0]["key"]
                                logger.warning(
                                    "jira_create_ticket: auto-corrected project '%s' → '%s' (available: %s)",
                                    old_project, project,
                                    ", ".join(p["key"] for p in projects),
                                )
                                proj2_resp = requests.get(
                                    f"{server}/rest/api/2/project/{project}",
                                    headers=headers, auth=auth, timeout=10,
                                )
                                if proj2_resp.status_code == 200:
                                    available = [
                                        t["name"] for t in proj2_resp.json().get("issueTypes", [])
                                        if not t.get("subtask", False)
                                    ]
                                    if available and valid_type not in available:
                                        valid_type = "Task" if "Task" in available else available[0]
                except Exception:
                    pass

                url = f"{server}/rest/api/2/issue"
                payload = {
                    "fields": {
                        "project": {"key": project},
                        "summary": summary,
                        "description": description,
                        "issuetype": {"name": valid_type},
                        "priority": {"name": priority},
                    }
                }
                logger.info(
                    "jira_create_ticket: POST %s project=%s type=%s auth_email=%s",
                    url, project, valid_type, email,
                )
                resp = requests.post(
                    url,
                    headers=headers,
                    auth=auth,
                    json=payload,
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                key = data.get("key", "?")
                return (
                    f"✓ Created {key}: \"{summary}\"\n"
                    f"  URL: {server}/browse/{key}"
                )
            except requests.HTTPError as e:
                logger.error(
                    "jira_create_ticket failed: %s — %s",
                    e.response.status_code, e.response.text[:300],
                )
                return f"✗ Create failed: {e.response.status_code} — {e.response.text[:200]}"
            except Exception as e:
                logger.error("jira_create_ticket error: %s", e)
                return f"✗ Create error: {str(e)}"

        @mcp_server.tool()
        def jira_update_ticket(
            key: str,
            comment: str = "",
            status: str = "",
            summary: str = "",
            description: str = "",
            priority: str = "",
            assignee: str = "",
            labels: str = "",
        ) -> str:
            """Update a Jira ticket: edit fields, add a comment, and/or transition status.

            For a pure workflow move (e.g. → Done) prefer jira_transition_ticket.
            For a pure comment prefer jira_add_comment. This tool handles the
            general case and any combination of changes in one call.

            Args:
                key: Ticket key (e.g., "SCRUM-51")
                comment: Comment text to add (optional)
                status: Target workflow status to transition to, e.g. "Done",
                        "In Progress" (optional)
                summary: New summary/title (optional)
                description: New description (optional)
                priority: New priority — Highest, High, Medium, Low, Lowest (optional)
                assignee: New assignee email or display name (optional)
                labels: Comma-separated labels to set, e.g. "backend,urgent" (optional)
            """
            results = []
            try:
                ctx, err = instance._auth_or_error()
                if err:
                    return err
                auth, headers, server = ctx

                fields = {}
                if summary:
                    fields["summary"] = summary
                if description:
                    fields["description"] = description
                if priority:
                    fields["priority"] = {"name": priority}
                if labels:
                    fields["labels"] = [l.strip() for l in labels.split(",") if l.strip()]
                if assignee:
                    account_id = instance._account_id_for(server, auth, headers, assignee)
                    if account_id:
                        fields["assignee"] = {"accountId": account_id}
                    else:
                        results.append(f"⚠ Could not find a Jira user matching '{assignee}'")
                if fields:
                    resp = requests.put(
                        f"{server}/rest/api/2/issue/{key}",
                        headers=headers, auth=auth,
                        json={"fields": fields}, timeout=15,
                    )
                    resp.raise_for_status()
                    results.append(f"✓ Updated {', '.join(fields.keys())} on {key}")

                if comment:
                    resp = requests.post(
                        f"{server}/rest/api/2/issue/{key}/comment",
                        headers=headers, auth=auth,
                        json={"body": comment}, timeout=10,
                    )
                    resp.raise_for_status()
                    results.append(f"✓ Comment added to {key}")

                if status:
                    results.append(instance._do_transition(server, auth, headers, key, status))

                if results:
                    return "\n".join(results)
                return (
                    f"⚠ No change specified for {key}. Provide at least one of: "
                    f"status (to move workflow), comment, summary, description, "
                    f"priority, assignee, or labels. To mark it Done, call "
                    f"jira_transition_ticket(key=\"{key}\", status=\"Done\")."
                )
            except requests.HTTPError as e:
                body = e.response.text if e.response is not None else ""
                logger.error("jira_update_ticket failed for %s: %s — %s",
                             key, getattr(e.response, "status_code", "?"), body[:300])
                return f"✗ Update failed for {key}: {getattr(e.response, 'status_code', '?')} — {body[:200]}"
            except Exception as e:
                logger.error("jira_update_ticket error for %s: %s", key, e, exc_info=True)
                return f"✗ Update error: {str(e)}"

        @mcp_server.tool()
        def jira_transition_ticket(key: str, status: str) -> str:
            """Move a Jira ticket through its workflow to a new status.

            This is the right tool for "mark SCRUM-51 as Done", "move to In
            Progress", "close this ticket", etc. It looks up the project's
            available transitions and matches your requested status by name or
            synonym (e.g. "Done" matches a "Resolve Issue" transition).

            Args:
                key: Ticket key (e.g., "SCRUM-51")
                status: Target status, e.g. "Done", "In Progress", "To Do", "Closed"
            """
            try:
                ctx, err = instance._auth_or_error()
                if err:
                    return err
                auth, headers, server = ctx
                return instance._do_transition(server, auth, headers, key, status)
            except requests.HTTPError as e:
                body = e.response.text if e.response is not None else ""
                logger.error("jira_transition_ticket failed for %s: %s — %s",
                             key, getattr(e.response, "status_code", "?"), body[:300])
                return f"✗ Transition failed for {key}: {getattr(e.response, 'status_code', '?')} — {body[:200]}"
            except Exception as e:
                logger.error("jira_transition_ticket error for %s: %s", key, e, exc_info=True)
                return f"✗ Transition error: {str(e)}"

        @mcp_server.tool()
        def jira_add_comment(key: str, comment: str) -> str:
            """Add a comment to a Jira ticket.

            Args:
                key: Ticket key (e.g., "SCRUM-51")
                comment: The comment text to add
            """
            try:
                ctx, err = instance._auth_or_error()
                if err:
                    return err
                auth, headers, server = ctx
                resp = requests.post(
                    f"{server}/rest/api/2/issue/{key}/comment",
                    headers=headers, auth=auth,
                    json={"body": comment}, timeout=10,
                )
                resp.raise_for_status()
                return f"✓ Comment added to {key}"
            except requests.HTTPError as e:
                body = e.response.text if e.response is not None else ""
                logger.error("jira_add_comment failed for %s: %s", key, body[:300])
                return f"✗ Comment failed for {key}: {getattr(e.response, 'status_code', '?')} — {body[:200]}"
            except Exception as e:
                logger.error("jira_add_comment error for %s: %s", key, e, exc_info=True)
                return f"✗ Comment error: {str(e)}"

        @mcp_server.tool()
        def jira_add_worklog(
            key: str,
            time_spent: str,
            comment: str = "",
            started: str = "",
        ) -> str:
            """Log time worked (a work log) against a Jira ticket.

            Use this to record effort, e.g. "log 2h on SCRUM-51 for the auth fix".
            The time is added to the issue's tracked work and remaining estimate
            is recalculated by Jira automatically.

            Args:
                key: Ticket key (e.g., "SCRUM-51")
                time_spent: Jira duration string, e.g. "2h", "30m", "1d 4h", "3w 2d"
                comment: Optional note describing what was done
                started: Optional ISO-8601 start time
                         (e.g. "2026-06-21T10:00:00.000+0000"). Defaults to now.
            """
            try:
                ctx, err = instance._auth_or_error()
                if err:
                    return err
                auth, headers, server = ctx

                payload = {"timeSpent": time_spent.strip()}
                if comment:
                    payload["comment"] = comment
                if started:
                    payload["started"] = started

                resp = requests.post(
                    f"{server}/rest/api/2/issue/{key}/worklog",
                    headers=headers, auth=auth,
                    json=payload, timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                logged = data.get("timeSpent", time_spent)
                return (
                    f"✓ Logged {logged} on {key}"
                    + (f" — \"{comment}\"" if comment else "")
                )
            except requests.HTTPError as e:
                body = e.response.text if e.response is not None else ""
                logger.error("jira_add_worklog failed for %s: %s — %s",
                             key, getattr(e.response, "status_code", "?"), body[:300])
                code = getattr(e.response, "status_code", "?")
                hint = ""
                if code == 400:
                    hint = " (check time_spent format, e.g. \"2h 30m\")"
                return f"✗ Worklog failed for {key}: {code} — {body[:200]}{hint}"
            except Exception as e:
                logger.error("jira_add_worklog error for %s: %s", key, e, exc_info=True)
                return f"✗ Worklog error: {str(e)}"

        @mcp_server.tool()
        def jira_list_tickets(
            jql: str = "",
            project: str = "",
            status: str = "",
            limit: int = 10,
        ) -> str:
            """Search Jira tickets using JQL or filters.

            Args:
                jql: Raw JQL query (takes priority if provided)
                project: Filter by project key (e.g., "SUP")
                status: Filter by status (e.g., "Open", "In Progress")
                limit: Maximum results (default: 10)
            """
            try:
                server, email, token = instance._resolve_credentials()
                if not server or not email or not token:
                    return "✗ Jira credentials not configured."
                auth = (email, token)
                headers = instance._headers()

                if not jql:
                    parts = []
                    if project:
                        parts.append(f"project = {project}")
                    if status:
                        parts.append(f'status = "{status}"')
                    if not parts:
                        try:
                            proj_resp = requests.get(
                                f"{server}/rest/api/3/project",
                                headers=headers, auth=auth, timeout=10,
                            )
                            if proj_resp.status_code == 200:
                                keys = [p["key"] for p in proj_resp.json()]
                                if keys:
                                    parts.append(
                                        f"project in ({', '.join(keys)})"
                                    )
                        except Exception:
                            pass
                    if not parts:
                        return "✗ No projects found or accessible."
                    jql = " AND ".join(parts) + " ORDER BY updated DESC"

                url = f"{server}/rest/api/3/search/jql"
                payload = {
                    "jql": jql,
                    "maxResults": limit,
                    "fields": ["summary", "status", "assignee", "priority", "updated"],
                }
                resp = requests.post(
                    url,
                    headers=headers,
                    auth=auth,
                    json=payload,
                    timeout=15,
                )
                resp.raise_for_status()
                issues = resp.json().get("issues", [])

                if not issues:
                    return f"No tickets found for: {jql}"

                lines = [f"Tickets ({len(issues)}):"]
                for issue in issues:
                    key = issue["key"]
                    fields = issue["fields"]
                    summary = fields.get("summary", "?")
                    status_name = fields.get("status", {}).get("name", "?")
                    priority_name = fields.get("priority", {}).get("name", "?")
                    assignee = (fields.get("assignee") or {}).get("displayName", "Unassigned")

                    lines.append(
                        f"  [{key}] {summary}\n"
                        f"    Status: {status_name} | Priority: {priority_name} | Assignee: {assignee}"
                    )

                return "\n".join(lines)
            except requests.HTTPError as e:
                return f"✗ Search failed: {e.response.status_code} — {e.response.text[:200]}"
            except Exception as e:
                return f"✗ Search error: {str(e)}"

    def _do_transition(self, server, auth, headers, key, status) -> str:
        """Execute a workflow transition. Returns a result line (✓/⚠/✗).

        Shared by jira_transition_ticket and jira_update_ticket so both paths
        behave identically.
        """
        trans_url = f"{server}/rest/api/2/issue/{key}/transitions"
        resp = requests.get(trans_url, headers=headers, auth=auth, timeout=10)
        resp.raise_for_status()
        transitions = resp.json().get("transitions", [])
        if not transitions:
            return (f"⚠ {key} has no available transitions for your account "
                    f"(check permissions or current status).")

        target = self._match_transition(status, transitions)
        if not target:
            available = ", ".join(t["name"] for t in transitions)
            return f"⚠ Can't move {key} to '{status}'. Available transitions: {available}"

        resp = requests.post(
            trans_url, headers=headers, auth=auth,
            json={"transition": {"id": target["id"]}}, timeout=10,
        )
        resp.raise_for_status()
        to_name = target.get("to", {}).get("name", target.get("name", status))
        return f"✓ {key} → {to_name}"
