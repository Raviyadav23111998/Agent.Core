"""
Tool Policy — per-tool execution metadata.
═══════════════════════════════════════════

Classifies every tool by its SIDE EFFECT and execution traits so the rest of
the system can make smart decisions:

  - side_effect: read | write | irreversible
        read         → safe, repeatable, results CACHEABLE
        write        → mutates state we own (files, DB, memory) — recoverable
        irreversible → external / unrecoverable (send email, shell, push) —
                       the seam for Phase-2 human-in-the-loop approval gates
  - idempotent: calling twice with same args has the same effect
  - cacheable:  pure read whose result can be memoized for a short TTL
  - timeout_s:  soft execution budget
  - retries:    how many times a transient failure may be retried

Tools can declare a policy at registration; otherwise a name-pattern default
is inferred (so all ~50 existing tools get a sensible policy with no rewrite).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


READ = "read"
WRITE = "write"
IRREVERSIBLE = "irreversible"


@dataclass(frozen=True)
class ToolPolicy:
    side_effect: str = WRITE
    idempotent: bool = False
    cacheable: bool = False
    timeout_s: float = 60.0
    retries: int = 0
    dedup_guard: bool = True

    @property
    def requires_approval(self) -> bool:
        """Irreversible/external actions are the HITL approval candidates."""
        return self.side_effect == IRREVERSIBLE


_IRREVERSIBLE_PATTERNS = (
    r".*_send$", r"^outlook_send", r"^outlook_reply", r"^outlook_forward",
    r"^smtp_", r"^teams_send", r"^teams_post", r"^send_",
    r"^jira_create", r"^jira_add_", r"^jira_transition",
    r".*_delete$", r".*_remove$", r"^git_push",
    r".*_push$", r"^cron_create$", r"^cron_remove$",
)
# Shell commands CAN have external side effects, so they stay approval-gated —
# but re-running the identical command is NORMAL agent behavior (edit → retest),
# so the double-send dedup guard must never block them.
_SHELL_PATTERNS = (r"^run_command$", r"^shell")
_READ_PATTERNS = (
    r"^view_file$", r"^list_dir$", r"^grep_search$", r"^read_",
    r".*_fetch.*", r".*_search$", r".*_list$", r".*_get$", r"^get_",
    r"^fetch_", r"^search_", r"^jira_list", r".*_list_.*",
    # brain_ingest_* / brain_sync are writes — only the query side is read.
    r"^brain_(ask|search|stats|fetch|sync_status)",
    r"^memory_recall$", r".*_query$",
)
_WRITE_PATTERNS = (
    r"^write_to_file$", r"^replace_file_content$", r"^multi_replace",
    r".*_create$", r".*_update$", r"^memory_store$", r"^remember$",
)


def _matches(name: str, patterns) -> bool:
    return any(re.match(p, name) for p in patterns)


def policy_for_kind(side_effect: str, *, shell: bool = False) -> ToolPolicy:
    """The canonical policy for a declared KIND. This is the ONE place that
    owns the mechanics (cacheability, timeouts, retries, dedup) so a tool only
    ever declares *what it is* (`side_effect="read"`), never the numbers —
    change the caching/retry behaviour of every read in one edit here.

    - read         → repeatable + result-cacheable + one retry
    - write        → mutates our own state; not cached; no retry (safe default)
    - irreversible → external/unrecoverable; HITL-approval seam; `shell=True`
                     drops the double-send guard (re-running a command is
                     normal edit→retest behaviour)
    """
    if side_effect == READ:
        return ToolPolicy(side_effect=READ, idempotent=True, cacheable=True,
                          timeout_s=60.0, retries=1)
    if side_effect == IRREVERSIBLE:
        return ToolPolicy(side_effect=IRREVERSIBLE, idempotent=False,
                          cacheable=False, timeout_s=120.0, retries=0,
                          dedup_guard=not shell)
    return ToolPolicy(side_effect=WRITE, idempotent=False, cacheable=False,
                      timeout_s=60.0, retries=0)


def default_policy_for(name: str) -> ToolPolicy:
    """LEGACY FALLBACK: infer a tool's KIND from its name, then map it through
    policy_for_kind. Used only when a tool did not declare its own kind at
    registration (the preferred, scalable path). Conservative where ambiguous
    → defaults to WRITE (never caches something it shouldn't)."""
    if _matches(name, _SHELL_PATTERNS):
        return policy_for_kind(IRREVERSIBLE, shell=True)
    if _matches(name, _IRREVERSIBLE_PATTERNS):
        return policy_for_kind(IRREVERSIBLE)
    if _matches(name, _READ_PATTERNS):
        return policy_for_kind(READ)
    if _matches(name, _WRITE_PATTERNS):
        return policy_for_kind(WRITE)
    return policy_for_kind(WRITE)
