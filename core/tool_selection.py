"""
Dynamic tool selection — send only the tools a turn actually needs.
═══════════════════════════════════════════════════════════════════

Measured on UAT: the OpenAI `tools` array is ~5.5k tokens (58 tools) and is
re-sent on EVERY LLM call — ~89% of a typical request payload, paid even for a
greeting. Most turns touch one domain (Jira, or email, or files).

This selects a small ALWAYS-ON core plus the domain groups whose trigger words
appear in the turn. Selection happens ONCE per turn (stable across the turn's
iterations) so the cached tools prefix isn't invalidated mid-turn.

Conservative by design: when in doubt it includes more, and the agent loop
falls back to the full toolset if selection would yield nothing useful. Toggle
with `agent.dynamic_tools` (default on).
"""

from __future__ import annotations

from typing import Iterable, Set

CORE_TOOLS: Set[str] = {
    "view_file", "list_dir", "grep_search", "glob_files",
    "write_to_file", "replace_file_content",
    "run_command", "ask_question", "search_web",
    "brain_ask", "brain_search",
    "brain_ingest_file", "brain_ingest_text",
    "add_memory", "memory_store",
    "memory_recall", "get_memories",
    "delegate_task", "spawn_agent", "todo_write",
    # Procedural memory (skills) — engine infrastructure like memory:
    # always eligible for the turn's tool subset when present.
    "skills_list", "skill_view", "skill_manage",
}

GROUPS = {
    "jira": {
        "tools": {"jira_create_ticket", "jira_list_tickets", "jira_update_ticket",
                  "jira_transition_ticket", "jira_add_comment", "jira_add_worklog"},
        "keywords": {"jira", "ticket", "scrum", "sprint", "backlog", "issue",
                     "epic", "story", "kanban", "worklog", "log time", "log work",
                     "transition", "mark as done", "move to"},
    },
    "email": {
        "tools": {"outlook_forward", "outlook_reply", "outlook_send",
                  "fetch_emails_imap", "fetch_recent_emails", "search_emails",
                  "send_email_smtp", "mark_email_processed"},
        "keywords": {"email", "mail", "inbox", "outlook", "reply", "forward",
                     "smtp", "compose", "imap"},
    },
    "teams": {
        "tools": {"teams_list_channels", "teams_post_channel", "teams_send_message",
                  "teams_notify_me"},
        "keywords": {"teams", "channel", "post to", "message the team",
                     "dm me", "notify me", "ping me", "message me"},
    },
    "git": {
        "tools": {"git_branch", "git_commit", "git_diff", "git_status"},
        "keywords": {"git", "commit", "branch", "diff", "repo", "stage", "pull request"},
    },
    "brain_admin": {
        "tools": {"brain_ingest_file", "brain_ingest_directory", "brain_ingest_text",
                  "brain_ingest_current_chat", "brain_sync", "brain_sync_status",
                  "brain_stats", "brain_fetch", "query_neo4j", "search_elasticsearch"},
        "keywords": {"ingest", "sync", "knowledge base", "index", "neo4j",
                     "elasticsearch", "brain stat", "import document"},
    },
    "memory_write": {
        "tools": {"add_memory", "memory_store", "memory_company_store",
                  "memory_entity_link", "memory_forget", "memory_stats",
                  "memory_user_context"},
        "keywords": {"remember", "memorize", "memorise", "forget", "note that",
                     "store this", "keep in mind"},
    },
    "file_edit": {
        "tools": {"multi_replace_file_content"},
        "keywords": {"refactor", "edit", "modify", "replace in", "patch", "rewrite"},
    },
    "admin": {
        "tools": {"agent_create", "agent_delete", "agent_list", "agent_logs",
                  "agent_run", "cron_create", "cron_list", "cron_remove",
                  "schedule", "manage_task"},
        "keywords": {"cron", "schedule", "agent", "spawn", "task list",
                     "recurring", "automation", "every day", "every hour"},
    },
    "page": {
        "tools": {"page_observe", "page_act", "page_navigate", "page_read", "page_wait"},
        "keywords": {"page", "browser", "navigate", "click", "web page", "scrape", "website"},
    },
}


def select_tool_names(user_input: str, available: Iterable[str], *,
                      dynamic_tools: Iterable[str] = (),
                      dynamic_picker=None) -> Set[str]:
    """Return the subset of `available` tools relevant to this turn.

    Always includes the core set; adds a domain group when any of its keywords
    appears in the user input. Tools not in any group (persona-custom, MCP,
    etc.) are kept by default so nothing the persona deliberately exposed
    silently disappears.

    Dynamic API-integration tools are the exception to that default-keep: a
    tenant may have hundreds of them, so keeping all ungrouped ops would refill
    `tools[]` and defeat selection. Pass `dynamic_tools` (every `{slug}__op`
    name) to EXCLUDE them from the default-keep, and `dynamic_picker(text,
    available) -> set` to add back only the turn-relevant few (embedding tool-RAG;
    see core/integrations + plan §5.5a). Both default to inert, so existing
    callers are unaffected.
    """
    available = set(available)
    text = (user_input or "").lower()
    dynamic_tools = set(dynamic_tools)

    grouped: Set[str] = set()
    for g in GROUPS.values():
        grouped |= g["tools"]
    grouped |= dynamic_tools

    selected = {t for t in available if t in CORE_TOOLS}
    for g in GROUPS.values():
        if any(kw in text for kw in g["keywords"]):
            selected |= (g["tools"] & available)

    selected |= {t for t in available if t not in grouped}

    if dynamic_picker is not None and dynamic_tools:
        try:
            picked = dynamic_picker(text, available)
            if picked:
                selected |= (set(picked) & available)
        except Exception:
            pass

    return selected
