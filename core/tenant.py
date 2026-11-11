"""
Agent — Multi-Tenant Identity Model
=================================================
Follows Microsoft Teams hierarchy: Tenant → Users → Groups → Agents

Architecture (two-layer isolation):
  Layer 1: PHYSICAL isolation between tenants (separate PostgreSQL schemas per org)
  Layer 2: LOGICAL isolation within a tenant (owner_id / user_id rows)

Tenant = Organization (e.g. CompanyX, CompanyY)
  - Each tenant gets its own PostgreSQL database via TenantDB
  - Tenants CANNOT see each other's data (physical separation)

User   = Person within a tenant
  - Has private data (owner_id = user_id) within the tenant's DB
  - Sees org-wide data (owner_id = '__tenant__')
  - Sees group data for groups they belong to

Group  = Subset of users with shared context
  - Telegram group chats, Teams channels, project teams
  - Members see group-scoped data (owner_id = 'group_{id}')

Agent  = Service identity acting on behalf of a user or the tenant
  - Inherits parent user's data scope
  - Has its own memory namespace

Usage:
    from core.tenant import TenantConfig, TenantRegistry, MembershipResolver, TENANT_SCOPE

    registry = TenantRegistry('.ag_data')
    tenant = registry.get_or_create('companyX', name='Company X')
    membership = MembershipResolver(tenant.memberships_path)

    scopes = membership.get_visible_scopes('user_12345')
    # → ['user_12345', 'group_-1001234', '__tenant__']
"""

import json
import os
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


TENANT_SCOPE = "__tenant__"

DEFAULT_TENANT_ID = "default"


@dataclass
class TenantConfig:
    """Configuration for a single tenant (organization).

    All persistent data is stored in PostgreSQL via TenantDB.
    This config holds filesystem paths for local state only
    (context cache, agent configs, membership files).

    Attributes:
        tenant_id:   Unique identifier for this tenant
        name:        Human-readable name
        data_dir:    Absolute path to tenant's local state directory
    """

    tenant_id: str = DEFAULT_TENANT_ID
    name: str = "Default Organization"
    data_dir: str = ""


    @property
    def context_cache_dir(self) -> str:
        return os.path.join(self.data_dir, "context_cache")

    @property
    def sync_state_path(self) -> str:
        return os.path.join(self.data_dir, "sync_state.json")

    @property
    def agents_dir(self) -> str:
        return os.path.join(self.data_dir, "agents")

    @property
    def cron_dir(self) -> str:
        return os.path.join(self.data_dir, "cron")

    @property
    def memberships_path(self) -> str:
        return os.path.join(self.data_dir, "memberships.json")

    @property
    def config_path(self) -> str:
        return os.path.join(self.data_dir, "tenant.json")


    def user_dir(self, user_id: str) -> str:
        """Per-user state directory (logs, checkpoints, summaries)."""
        safe_id = _sanitize_id(user_id)
        return os.path.join(self.data_dir, "users", safe_id)

    def group_dir(self, group_id: str) -> str:
        """Per-group state directory (logs, summaries)."""
        safe_id = _sanitize_id(group_id)
        return os.path.join(self.data_dir, "groups", safe_id)


def _sanitize_id(raw_id: str) -> str:
    """Make an ID filesystem-safe."""
    return (
        raw_id
        .replace("/", "_")
        .replace("\\", "_")
        .replace("..", "_")
        .replace(" ", "_")
        .lower()
    )


class TenantRegistry:
    """Manages tenant lifecycle: create, resolve, list, delete.

    Backed by a tenants.json file in the root .ag_data directory.
    Each tenant gets a subdirectory: .ag_data/tenants/{tenant_id}/

    For single-org deployments, a 'default' tenant is auto-created
    pointing to the existing .ag_data/ directory (backward compatible).

    Usage:
        registry = TenantRegistry('/path/to/.ag_data')
        tenant = registry.get_or_create('companyX', name='Company X')
        # → TenantConfig with data_dir = '/path/to/.ag_data/tenants/companyX'
    """

    def __init__(self, ag_data_dir: str):
        self._ag_data = os.path.abspath(ag_data_dir)
        self._tenants_dir = os.path.join(self._ag_data, "tenants")
        self._registry_path = os.path.join(self._ag_data, "tenants.json")
        self._cache: Dict[str, TenantConfig] = {}

    def get_or_create(
        self,
        tenant_id: str = DEFAULT_TENANT_ID,
        name: str = "",
    ) -> TenantConfig:
        """Get existing tenant or create a new one.

        Auto-creates tenant directory and initializes config.

        For backward compatibility: the 'default' tenant uses
        the existing .ag_data/ directory directly (no subdirectory).
        """
        if tenant_id in self._cache:
            return self._cache[tenant_id]

        registry = self._load_registry()
        if tenant_id in registry:
            config = self._config_from_dict(tenant_id, registry[tenant_id])
            self._cache[tenant_id] = config
            return config

        if tenant_id == DEFAULT_TENANT_ID:
            data_dir = self._ag_data
        else:
            data_dir = os.path.join(self._tenants_dir, _sanitize_id(tenant_id))

        config = TenantConfig(
            tenant_id=tenant_id,
            name=name or tenant_id,
            data_dir=data_dir,
        )

        self._ensure_dirs(config)

        registry[tenant_id] = {
            "name": config.name,
            "data_dir": config.data_dir,
            "created_at": _now_iso(),
        }
        self._save_registry(registry)

        self._cache[tenant_id] = config
        logger.info("tenant.created: id=%s dir=%s", tenant_id, data_dir)
        return config

    def get(self, tenant_id: str) -> Optional[TenantConfig]:
        """Get tenant config without creating."""
        if tenant_id in self._cache:
            return self._cache[tenant_id]
        registry = self._load_registry()
        if tenant_id in registry:
            config = self._config_from_dict(tenant_id, registry[tenant_id])
            self._cache[tenant_id] = config
            return config
        return None

    def list_tenants(self) -> List[TenantConfig]:
        """List all registered tenants."""
        registry = self._load_registry()
        return [
            self._config_from_dict(tid, data)
            for tid, data in registry.items()
        ]

    def delete_tenant(self, tenant_id: str, purge_data: bool = False) -> bool:
        """Remove a tenant from the registry.

        Args:
            purge_data: If True, also delete the tenant's data directory.
                        Use with extreme caution — this is irreversible.
        """
        if tenant_id == DEFAULT_TENANT_ID:
            logger.warning("tenant.delete_refused: cannot delete default tenant")
            return False

        registry = self._load_registry()
        if tenant_id not in registry:
            return False

        data_dir = registry[tenant_id].get("data_dir", "")

        del registry[tenant_id]
        self._save_registry(registry)
        self._cache.pop(tenant_id, None)

        if purge_data and data_dir and os.path.isdir(data_dir):
            import shutil
            shutil.rmtree(data_dir, ignore_errors=True)
            logger.warning("tenant.purged: id=%s dir=%s", tenant_id, data_dir)
        else:
            logger.info("tenant.deleted: id=%s (data preserved at %s)", tenant_id, data_dir)

        return True


    def _ensure_dirs(self, config: TenantConfig):
        """Create all required directories for a tenant."""
        os.makedirs(config.data_dir, exist_ok=True)
        os.makedirs(os.path.join(config.data_dir, "users"), exist_ok=True)
        os.makedirs(os.path.join(config.data_dir, "groups"), exist_ok=True)
        os.makedirs(config.context_cache_dir, exist_ok=True)
        os.makedirs(config.agents_dir, exist_ok=True)

    def _config_from_dict(self, tenant_id: str, data: dict) -> TenantConfig:
        return TenantConfig(
            tenant_id=tenant_id,
            name=data.get("name", tenant_id),
            data_dir=data.get("data_dir", ""),
        )

    def _load_registry(self) -> dict:
        if not os.path.exists(self._registry_path):
            return {}
        try:
            with open(self._registry_path, "r", encoding="utf-8") as f:
                return json.load(f).get("tenants", {})
        except (json.JSONDecodeError, IOError):
            return {}

    def _save_registry(self, registry: dict):
        os.makedirs(os.path.dirname(self._registry_path), exist_ok=True)
        data = {"tenants": registry, "updated_at": _now_iso()}
        tmp = self._registry_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._registry_path)


class MembershipResolver:
    """Resolves which data scopes (owner_ids) a user can access.

    This is the ACCESS CONTROL layer — equivalent to Azure AD group membership.
    PostgreSQL-backed (ag_auth.memberships).

    For a given user, returns the list of owner_ids they can read from:
      - Their own user_id (private data)
      - All group_ids they belong to (group data)
      - '__tenant__' (org-wide data visible to everyone)
    """

    def __init__(self, db_or_path):
        """Accept either a TenantDB instance or a legacy path (ignored).
        
        Args:
            db_or_path: TenantDB instance for PostgreSQL-backed membership
        """
        self._db = db_or_path

    def get_visible_scopes(self, user_id: str) -> List[str]:
        """Return all owner_ids this user can read from."""
        scopes = [user_id, TENANT_SCOPE]
        scopes.extend(self.get_user_groups(user_id))
        return scopes

    def get_user_groups(self, user_id: str) -> List[str]:
        """Return group_ids this user belongs to."""
        rows = self._db.fetch_all(
            "SELECT group_ext_id FROM ag_auth.memberships WHERE user_ext_id = %s",
            (user_id,)
        )
        return [r["group_ext_id"] for r in rows]

    def get_group_members(self, group_id: str) -> List[str]:
        """Return user_ids in a group."""
        rows = self._db.fetch_all(
            "SELECT user_ext_id FROM ag_auth.memberships WHERE group_ext_id = %s",
            (group_id,)
        )
        return [r["user_ext_id"] for r in rows]

    def add_to_group(self, user_id: str, group_id: str,
                     group_name: str = "", user_display_name: str = ""):
        """Add a user to a group. Idempotent."""
        self._db.execute("""
            INSERT INTO ag_auth.memberships (user_ext_id, display_name, group_ext_id, group_name)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (user_ext_id, group_ext_id) DO UPDATE SET
                display_name = COALESCE(NULLIF(EXCLUDED.display_name, ''), ag_auth.memberships.display_name),
                group_name = COALESCE(NULLIF(EXCLUDED.group_name, ''), ag_auth.memberships.group_name)
        """, (user_id, user_display_name, group_id, group_name or group_id))
        logger.debug("membership.added: user=%s group=%s", user_id, group_id)

    def remove_from_group(self, user_id: str, group_id: str):
        """Remove a user from a group."""
        self._db.execute(
            "DELETE FROM ag_auth.memberships WHERE user_ext_id = %s AND group_ext_id = %s",
            (user_id, group_id)
        )

    def resolve_chat_context(self, chat_id: str, user_id: str,
                              user_name: str = "") -> str:
        """Resolve owner_id from a chat context and auto-register membership."""
        if str(chat_id) == str(user_id):
            return user_id

        group_id = f"group_{chat_id}"
        self.add_to_group(user_id, group_id, user_display_name=user_name)
        return group_id

    def get_stats(self) -> dict:
        """Return membership statistics."""
        total_users = self._db.fetch_scalar(
            "SELECT COUNT(DISTINCT user_ext_id) FROM ag_auth.memberships"
        ) or 0
        total_groups = self._db.fetch_scalar(
            "SELECT COUNT(DISTINCT group_ext_id) FROM ag_auth.memberships"
        ) or 0
        return {
            "total_users": total_users,
            "total_groups": total_groups,
        }


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
