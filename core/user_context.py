"""
Agent — User Context
=================================
Central identity object that flows through the entire system.

Every session (CLI, Gateway, Cron, Agent) has a UserContext that determines:
  - Which memories are accessible
  - Which brain documents are visible
  - Where checkpoints/logs/summaries are stored
  - Which agents are visible

Identity hierarchy (Microsoft Teams model):
  Tenant → Users → Groups → Agents

Data visibility model (two-layer):
  Layer 1: Physical isolation between tenants (separate DB files)
  Layer 2: Logical isolation within a tenant:
    - __tenant__ → org-wide data (Jira, Confluence, Slack)
    - user_{id}  → user's private data (Outlook, uploads)
    - group_{id} → group-specific data (team docs, group chat context)

  Each user sees: own data + groups they belong to + __tenant__ data

Usage:
    from core.user_context import UserContext, CLI_USER, SYSTEM_USER
    from core.tenant import TENANT_SCOPE

    ctx = UserContext("user_12345", "John", "telegram", group_ids=("group_abc",))
    ctx.visible_scopes  # ["user_12345", "__tenant__", "group_abc"]
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from contextvars import ContextVar

from core.tenant import TENANT_SCOPE


@dataclass(frozen=True)
class UserContext:
    """Identity and configuration for the current session's user.

    This object flows from the entry point (CLI/Gateway/Cron) through
    the agent loop, into tools, and down to storage layers.

    Frozen (immutable) to prevent accidental identity mutation mid-session.

    Attributes:
        user_id:        Unique identifier ("user_12345" for Telegram, "__cli__" for CLI)
        user_name:      Display name for logging/UI
        platform:       Origin platform ("telegram", "cli", "cron", "agent")
        share_ingested: If True, files ingested by this user go to __tenant__ scope.
                        If False, private (owner_id = user_id).
        parent_user_id: For agent contexts: the user_id of who spawned them.
        group_ids:      Tuple of group_ids this user belongs to (frozen for immutability).
                        Populated from MembershipResolver at session creation.
    """

    user_id: str
    user_name: str = ""
    platform: str = "cli"
    share_ingested: bool = False
    parent_user_id: Optional[str] = None
    group_ids: Tuple[str, ...] = ()
    app_id: Optional[str] = None
    role: str = ""

    @property
    def is_system(self) -> bool:
        """System users: __cli__, __cron__, __system__, __agent__*."""
        return self.user_id.startswith("__")

    @property
    def data_dir_name(self) -> str:
        """Filesystem-safe directory name for per-user storage.

        Sanitizes user_id to prevent path traversal or invalid chars.
        """
        return (
            self.user_id
            .replace("/", "_")
            .replace("\\", "_")
            .replace("..", "_")
            .replace(" ", "_")
            .lower()
        )

    @property
    def data_scope_id(self) -> str:
        """The user_id to use for data queries (search, retrieval).

        For regular users, this is their own user_id.
        For spawned agents, this is the parent's user_id so agents
        can see the spawning user's private documents.
        """
        return self.parent_user_id or self.user_id

    @property
    def visible_scopes(self) -> list:
        """All owner_ids this user can read from.

        Returns a list suitable for SQL IN (...) clauses:
          - data_scope_id (own user_id, or parent's for agents)
          - TENANT_SCOPE (__tenant__ — org-wide data)
          - All group_ids the user belongs to

        Example:
            ['user_12345', '__tenant__', 'group_-1001234', 'group_eng']
        """
        scopes = [self.data_scope_id, TENANT_SCOPE]
        scopes.extend(self.group_ids)
        return scopes

    @property
    def owner_id_for_ingestion(self) -> str:
        """Determine owner_id when this user ingests data.

        If share_ingested=True, ingested files go to __tenant__ (visible to all).
        Otherwise, they go to the user's own scope (private).
        """
        if self.share_ingested:
            return TENANT_SCOPE
        return self.data_scope_id

    def __str__(self) -> str:
        name = self.user_name or self.user_id
        groups = f" groups={len(self.group_ids)}" if self.group_ids else ""
        return f"{name}@{self.platform}{groups}"

    def __repr__(self) -> str:
        return (
            f"UserContext(user_id={self.user_id!r}, user_name={self.user_name!r}, "
            f"platform={self.platform!r}, share_ingested={self.share_ingested}, "
            f"groups={len(self.group_ids)})"
        )


COMMON_SCOPE = TENANT_SCOPE


current_user_context: ContextVar[Optional[UserContext]] = ContextVar("current_user_context", default=None)


CLI_USER = UserContext(
    user_id="__cli__",
    user_name="CLI User",
    platform="cli",
    share_ingested=True,
)

SYSTEM_USER = UserContext(
    user_id="__system__",
    user_name="System",
    platform="system",
    share_ingested=True,
)

CRON_USER = UserContext(
    user_id="__cron__",
    user_name="Cron Scheduler",
    platform="cron",
    share_ingested=True,
)


def create_gateway_user(
    user_id: str,
    user_name: str = "",
    platform: str = "telegram",
    share_ingested: bool = False,
    group_ids: Tuple[str, ...] = (),
) -> UserContext:
    """Factory for gateway-originated user contexts.

    Args:
        user_id: Platform-specific user ID (e.g., Telegram user ID)
        user_name: Display name
        platform: Platform name
        share_ingested: Whether user's ingested files are shared
        group_ids: Groups this user belongs to (from MembershipResolver)
    """
    return UserContext(
        user_id=str(user_id),
        user_name=user_name or f"User-{user_id}",
        platform=platform,
        share_ingested=share_ingested,
        group_ids=group_ids,
    )


def create_agent_user(
    agent_name: str,
    parent_context: Optional['UserContext'] = None,
    app_id: Optional[str] = None,
) -> UserContext:
    """Factory for spawned agent contexts.

    Agents inherit the parent user's data scope for document access,
    but get a distinct user_id for memory isolation.

    The parent_user_id is stored so data queries resolve to the parent's
    scope. Agents also inherit the parent's group memberships.

    Args:
        agent_name: Name of the agent being spawned
        parent_context: The user context of the session that spawned the agent
        app_id: Owning application id. When set, credential resolution (Jira,
                M365, SMTP…) can reach that app's connections — this is what
                lets an app-owned cron/agent authenticate as its app even
                though its user_id is the synthetic ``__agent__…`` id. Falls
                back to the parent's app_id.
    """
    parent_id = parent_context.user_id if parent_context else "__cli__"
    parent_groups = parent_context.group_ids if parent_context else ()
    resolved_app = app_id or (parent_context.app_id if parent_context else None)
    return UserContext(
        user_id=f"__agent__{agent_name}__",
        user_name=f"Agent:{agent_name}",
        platform="agent",
        share_ingested=parent_context.share_ingested if parent_context else False,
        parent_user_id=parent_id,
        group_ids=parent_groups,
        app_id=str(resolved_app) if resolved_app else None,
    )


def create_cron_user(app_id: Optional[str] = None) -> UserContext:
    """Factory for a cron run's context, scoped to the owning application.

    A cron job is an unattended automation owned by an APP, so it should
    authenticate with that app's connections. Carrying the job's app_id here
    is what makes app-scoped credentials (Jira, M365, …) resolve for cron —
    without it, resolution can only reach org-scope connections.
    """
    return UserContext(
        user_id="__cron__",
        user_name="Cron Scheduler",
        platform="cron",
        share_ingested=True,
        app_id=str(app_id) if app_id else None,
    )
