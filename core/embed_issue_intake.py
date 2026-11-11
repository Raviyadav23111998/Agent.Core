"""
Website issue intake: the only Jira access a public chatbot gets.
=================================================================
An embedded chatbot runs for anonymous visitors, so the full Jira toolset
(list, update, transition, comment, worklog) is never exposed to it; see
core/embed_sandbox.py. Some sites still want visitors to be able to report a
problem. This module adds a single tool, `report_issue`, when an admin opts in
on the bot:

    ag_bot.bots.guardrail_policy = {
      ...,
      "jira_intake": {
        "enabled": true,
        "project": "SCRUM",            # fixed; the model cannot choose
        "issue_type": "Task",          # fixed
        "labels": ["website-report"],  # always added
        "max_per_session": 3,          # per visitor conversation
        "max_per_hour": 20             # per bot, all visitors together
      }
    }

Guardrails:
- Create only. The tool never calls a Jira read endpoint and returns only the
  new issue key, so a visitor cannot see, search or change existing issues.
- Project, issue type, priority and labels come from the config, never from
  the model or the visitor. Assignee, reporter and other fields cannot be set.
- Visitor text is length-capped, stripped of control characters and placed in
  a {noformat} block, so it cannot @mention people, add links or inject Jira
  markup.
- Rate limits per visitor and per bot, plus duplicate suppression.
- Errors are generic; Jira's response body is logged, never returned.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

TOOL_NAME = "report_issue"

_SUMMARY_MAX = 150
_FIELD_MAX = 3000
_EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,190}\.[A-Za-z]{2,24}$")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PROJECT_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,19}$")

_lock = threading.Lock()
_per_session: Dict[str, int] = defaultdict(int)
_per_bot: Dict[str, deque] = defaultdict(deque)
_recent_summaries: Dict[str, set] = defaultdict(set)


def _clean(text: Any, limit: int) -> str:
    s = _CTRL_RE.sub("", str(text or "")).strip()
    return s[:limit]


def _noformat(text: str) -> str:
    # A visitor cannot close the block early: drop any {noformat} they typed.
    return "{noformat}\n" + re.sub(r"(?i)\{noformat\}", "", text) + "\n{noformat}"


def _check_limits(bot_key: str, session_key: str, summary: str,
                  max_session: int, max_hour: int) -> Optional[str]:
    now = time.time()
    with _lock:
        window = _per_bot[bot_key]
        while window and now - window[0] > 3600:
            window.popleft()
        if len(window) >= max_hour:
            return "rate"
        if _per_session[session_key] >= max_session:
            return "session"
        fp = summary.lower()
        if fp in _recent_summaries[session_key]:
            return "duplicate"
        window.append(now)
        _per_session[session_key] += 1
        _recent_summaries[session_key].add(fp)
    return None


def intake_config(guardrail_policy: Optional[dict]) -> Optional[dict]:
    """Validated intake config, or None when disabled or misconfigured."""
    cfg = (guardrail_policy or {}).get("jira_intake") or {}
    if not cfg.get("enabled"):
        return None
    project = str(cfg.get("project") or "").strip().upper()
    if not _PROJECT_RE.match(project):
        logger.warning("jira_intake: invalid or missing project key; tool not offered")
        return None
    labels = [re.sub(r"[^A-Za-z0-9_-]", "", str(l))[:40] for l in (cfg.get("labels") or ["website-report"])]
    return {
        "project": project,
        "issue_type": str(cfg.get("issue_type") or "Task").strip()[:40] or "Task",
        "labels": [l for l in labels if l][:5] or ["website-report"],
        "max_per_session": max(1, min(int(cfg.get("max_per_session") or 3), 10)),
        "max_per_hour": max(1, min(int(cfg.get("max_per_hour") or 20), 200)),
    }


def register_issue_intake(registry, connections, cfg: dict, *, app_id: Any,
                          integration_id: Any, session_key: str,
                          site: str = "") -> None:
    """Add `report_issue` to an embed session's (already scoped) registry."""
    bot_key = f"bot:{integration_id}"
    sess_key = f"{bot_key}:{session_key or 'anon'}"
    source = site or "the website"

    def report_issue(summary: str, description: str,
                     steps_to_reproduce: str = "", contact_email: str = "") -> str:
        import requests

        summary_c = _clean(summary, _SUMMARY_MAX)
        desc_c = _clean(description, _FIELD_MAX)
        steps_c = _clean(steps_to_reproduce, _FIELD_MAX)
        email_c = _clean(contact_email, 254)
        if len(summary_c) < 5 or len(desc_c) < 10:
            return ("Not filed: ask the visitor for a short title and a description "
                    "of what went wrong, then try again.")
        if email_c and not _EMAIL_RE.match(email_c):
            email_c = ""

        blocked = _check_limits(bot_key, sess_key, summary_c,
                                cfg["max_per_session"], cfg["max_per_hour"])
        if blocked == "duplicate":
            return "Already filed: this report was submitted earlier in this conversation."
        if blocked:
            logger.warning("jira_intake: %s limit reached (integration=%s)", blocked, integration_id)
            return ("Not filed: the reporting limit has been reached. Ask the visitor to "
                    "try again later or use the site's contact options.")

        try:
            cred = connections.resolve("jira", user_id=None, app_id=app_id)
            server = (cred.config.get("server_url") or "").rstrip("/")
            email, token = cred.config.get("email") or "", cred.secret
            if not (server and email and token):
                raise RuntimeError("jira connection incomplete")
        except Exception as e:
            logger.error("jira_intake: no Jira connection for app %s: %s", app_id, e)
            return "Not filed: issue reporting is unavailable right now."

        parts = [f"Reported by a visitor through the {source} chat assistant.",
                 f"Contact: {email_c or 'not provided'}", "", "*Description*", _noformat(desc_c)]
        if steps_c:
            parts += ["", "*Steps to reproduce*", _noformat(steps_c)]
        payload = {"fields": {
            "project": {"key": cfg["project"]},
            "issuetype": {"name": cfg["issue_type"]},
            "summary": summary_c,
            "description": "\n".join(parts),
            "labels": cfg["labels"],
        }}
        try:
            resp = requests.post(f"{server}/rest/api/2/issue", json=payload,
                                 auth=(email, token), timeout=15,
                                 headers={"Accept": "application/json",
                                          "Content-Type": "application/json"})
            if resp.status_code >= 300:
                logger.error("jira_intake: create failed %s: %s", resp.status_code, resp.text[:300])
                return "Not filed: the issue tracker rejected the report. Please try again later."
            key = (resp.json() or {}).get("key", "")
        except Exception as e:
            logger.error("jira_intake: create error: %s", e)
            return "Not filed: the issue tracker could not be reached. Please try again later."

        logger.info("jira_intake: created %s (integration=%s)", key, integration_id)
        return (f"Filed as {key}. Give the visitor this reference number. You cannot look "
                f"up, change or discuss this or any other issue.")

    registry.register(
        TOOL_NAME,
        ("File a problem report for the team. Use only when a visitor describes a bug or "
         "problem with the site and agrees to send a report. Summarise it in your own words. "
         "This tool can only create a new report; it cannot read, search, update or discuss "
         "existing issues, so never claim to know the status of any issue."),
        {"summary": "string", "description": "string",
         "steps_to_reproduce?": "string", "contact_email?": "string"},
        report_issue,
        param_descriptions={
            "summary": "One-line title, under 150 characters.",
            "description": "What happened and what the visitor expected.",
            "steps_to_reproduce": "Optional numbered steps.",
            "contact_email": "Optional; only if the visitor offered it.",
        },
        side_effect="write",
    )
