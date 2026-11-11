"""
Agent — Centralized Configuration Provider
========================================================
Single source of truth for ALL application configuration.

Architecture:
    1. Bootstrap: ONE env var (AG_DATABASE_URL) to connect to PostgreSQL
    2. All other configs loaded from ag_system.configs table
    3. Secrets marked is_secret=TRUE (never logged)
    4. Prefix-based grouping: llm.*, neo4j.*, gateway.*, etc.
    5. Fallback: env vars → DB → hardcoded defaults (in that priority)

Usage:
    from core.config import app_config

    # Get a single value
    model = app_config.get("llm.model")
    token = app_config.get("gateway.telegram.bot_token")

    # Get a group
    llm = app_config.group("llm")
    # → {"provider": "openrouter", "model": "...", "api_key": "...", ...}

    # Get with env-var fallback
    key = app_config.get("llm.api_key", env="OPENROUTER_API_KEY")

Why not just use os.environ?
    - Configs are tenant-specific (different tenants have different keys)
    - Secrets stored with is_secret flag for audit
    - Centralized: one DB query loads everything
    - No .env file parsing, no scattered os.environ.get() calls
"""

import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)


ENV_TO_CONFIG = {
    "OPENROUTER_API_KEY": "llm.api_key",
    "LLM_MODEL": "llm.model",
    "LLM_PROVIDER": "llm.provider",
    "LLM_BASE_URL": "llm.base_url",
    "LLM_PROVIDER_ORDER": "llm.provider_order",
    "LLM_MAX_TOKENS": "llm.max_tokens",
    "LLM_TEMPERATURE": "llm.temperature",

    "TELEGRAM_BOT_TOKEN": "gateway.telegram.bot_token",
    "TELEGRAM_ALLOWED_USERS": "gateway.telegram.allowed_users",
    "TELEGRAM_HOME_CHANNEL": "gateway.telegram.home_channel",

    "JIRA_SERVER_URL": "jira.server_url",
    "JIRA_EMAIL": "jira.email",
    "JIRA_API_TOKEN": "jira.api_token",


    "NEO4J_URI": "neo4j.uri",
    "NEO4J_USER": "neo4j.user",
    "NEO4J_PASSWORD": "neo4j.password",

    "ELASTICSEARCH_URL": "elasticsearch.url",
    "ELASTICSEARCH_INDEX": "elasticsearch.index",

    "REDIS_URL": "redis.url",

    "AG_HISTORY_TOKEN_BUDGET": "agent.history_token_budget",
    "AG_DATABASE_URL": "db.url",

    "SECRET_ENCRYPTION_KEY": "secret.encryption_key",
    "SECRET_PRIVATE_KEY": "secret.private_key",
    "SECRET_PUBLIC_KEY": "secret.public_key",
    "AG_FIELD_ENCRYPTION_KEY": "secret.field_encryption_key",
}

CONFIG_TO_ENV = {v: k for k, v in ENV_TO_CONFIG.items()}

DEFAULTS = {
    "llm.provider": "openrouter",
    "llm.base_url": "https://openrouter.ai/api/v1/chat/completions",
    "llm.provider_order": "DeepInfra",
    "llm.model": "deepseek/deepseek-v4-flash",
    "llm.max_tokens": "16000",
    "llm.temperature": "0.7",
    "neo4j.uri": "bolt://localhost:7687",
    "neo4j.user": "neo4j",
    "elasticsearch.url": "http://localhost:9200",
    "elasticsearch.index": "brain",
    "agent.data_dir": ".ag_data",
    "agent.history_token_budget": "120000",
    "agent.max_tools": "20",
    "agent.timeout_seconds": "300",
    "memory.decay_days": "7",
    "memory.session_limit": "50",
    "brain.auto_sync": "false",
    "brain.sync_interval": "30m",
    "gateway.platform": "telegram",
    "db.port": "5432",
    "db.name": "agent_uat",
    # Media/attachment storage — mirrors SMSystems ImageBasePath/DOCBasePath/BasePath.
    # Deploy-set in prod; the default is a local root used for dev.
    "media.base_path": "/var/agent-media/",
    "media.image_base_path": "/var/agent-media/",
    "media.doc_base_path": "/var/agent-media/Document/",
    "media.audio_base_path": "/var/agent-media/Audio/",
    "media.video_base_path": "/var/agent-media/Video/",
    "media.max_upload_mb": "20",
    "media.thumbnail_px": "500",
    # Long-edge cap for the image variant sent to the model, so the base64 stays
    # under provider per-image limits (Anthropic ~5MB/image) and token cost is
    # bounded. The original is kept for display.
    "media.vision_px": "1568",
    # Media GENERATION (Part 2 — model → user). Provider tools are only exposed
    # when a provider + key are configured; otherwise the model has no way to
    # generate and the UI reports output support = false.
    "media.image.provider": "",            # replicate | fal | openai
    "media.image.model": "black-forest-labs/flux-schnell",
    "media.image.size": "1024x1024",
    "media.gen.api_key": "",               # provider API token
    "media.gen.timeout_s": "90",
}


class AppConfig:
    """Centralized config provider backed by ag_system.configs.

    Resolution order (first wins):
        1. os.environ (for the mapped env var name)
        2. ag_system.configs (from PostgreSQL)
        3. DEFAULTS dict (hardcoded fallbacks)

    This means env vars always override DB values — useful for
    local dev overrides without touching the DB.
    """

    def __init__(self):
        self._cache: Dict[str, str] = {}
        self._secrets: set = set()
        self._loaded = False
        self._db = None
        self._tenant_caches: Dict[str, Dict[str, str]] = {}
        self._grouter = None

    def load(self, database_url: str = None):
        """Load all configs from the database.

        Args:
            database_url: PostgreSQL URL. If None, uses AG_DATABASE_URL env var.
        """
        if self._loaded:
            return

        from core.tenant_db import TenantDB

        url = database_url or os.environ.get("AG_DATABASE_URL")
        if not url:
            logger.warning(
                "AppConfig: No AG_DATABASE_URL set. Using defaults + env vars only."
            )
            self._loaded = True
            return

        try:
            self._db = TenantDB(url)
            self._db.connect()

            rows = self._db.fetch_all("""
                SELECT config_key, config_value, is_secret
                FROM ag_system.configs
                WHERE is_active = TRUE
                ORDER BY config_key
            """)

            for row in rows:
                self._cache[row['config_key']] = row['config_value'] or ""
                if row['is_secret']:
                    self._secrets.add(row['config_key'])

            self._loaded = True
            logger.info("AppConfig loaded %d configs from DB (%d secrets)",
                        len(self._cache), len(self._secrets))
        except Exception as e:
            logger.warning("AppConfig: DB load failed (%s). Using defaults.", e)
            self._loaded = True

    @property
    def db(self):
        """Access the underlying TenantDB (for direct queries)."""
        return self._db

    def _tenant_cache(self) -> Optional[Dict[str, str]]:
        """Return the current tenant's config overlay (lazily loaded), or None
        for the default/single-tenant context. Best-effort: any failure returns
        None so config resolution silently falls back to the global cache —
        config reads must never throw or block a request."""
        try:
            from core.tenant_context import current_tenant_id, DEFAULT_TENANT_ID
            tid = current_tenant_id()
            if not tid or tid == DEFAULT_TENANT_ID:
                return None
            cached = self._tenant_caches.get(tid)
            if cached is not None:
                return cached

            from core.tenant_resolver import get_global_resolver
            from core.tenant_context import current_tenant
            resolver = get_global_resolver()
            if resolver is None:
                return None
            db = resolver.for_tenant(current_tenant()).agent_db
            rows = db.fetch_all(
                "SELECT config_key, config_value FROM ag_system.configs "
                "WHERE is_active = TRUE")
            overlay = {r["config_key"]: (r["config_value"] or "") for r in rows}
            self._tenant_caches[tid] = overlay
            logger.info("AppConfig: loaded %d tenant configs (tenant=%s)",
                        len(overlay), tid)
            return overlay
        except Exception as e:
            logger.debug("AppConfig: tenant overlay unavailable (%s)", e)
            return None

    def get(self, key: str, default: str = None, env: str = None) -> Optional[str]:
        """Get a config value.

        Resolution: tenant DB overlay → control DB cache → env var → DEFAULTS
        → default param. The tenant overlay is the current tenant's own
        ag_system.configs (multi-tenant); absent keys fall back to the shared
        control-plane config, so a tenant only overrides what it customises.

        Args:
            key: Config key (e.g. "llm.model", "jira.server_url")
            default: Fallback if not found anywhere
            env: Explicit env var name to check (overrides auto-mapping)
        """
        from core.tenant_context import current_tenant_id, DEFAULT_TENANT_ID
        tid = current_tenant_id()
        if tid and tid != DEFAULT_TENANT_ID:
            overlay = self._tenant_cache() or {}
            val = overlay.get(key)
            if val:
                return val
            return DEFAULTS.get(key, default)

        if key in self._cache and self._cache[key]:
            return self._cache[key]
        env_var = env or CONFIG_TO_ENV.get(key)
        if env_var:
            env_val = os.environ.get(env_var)
            if env_val:
                return env_val
        if key in DEFAULTS:
            return DEFAULTS[key]
        return default

    def get_global(self, key: str, default: str = None, env: str = None) -> Optional[str]:
        """GLOBAL PLANE — infrastructure + platform crypto, tenant-independent.

        Single source: deploy-time ENV → hard DEFAULTS. It does NOT consult the
        control DB (agent_prod) or any tenant DB, so a global value can never come
        from agent_prod — the acid test is that deleting agent_prod must not break
        a tenant. Therefore infra (Redis, Postgres, ES, Neo4j, embedding endpoint)
        and platform crypto (JWT/encryption/field keys) MUST be provided via ENV.
        Safe from background threads with no tenant context.
        See MULTITENANT_CONFIG_ARCHITECTURE.md §2 (Plane A) and core/infra_config.py.
        """
        env_var = env or CONFIG_TO_ENV.get(key)
        if env_var:
            env_val = os.environ.get(env_var)
            if env_val:
                return env_val
        val = self._agent_config().get(key)
        if val:
            return val
        if key in DEFAULTS:
            return DEFAULTS[key]
        return default

    def _agent_config(self) -> Dict[str, str]:
        """GLOBAL plane store — tenant_mgmt.agent_config in MSystem (cached, lazy).
        Best-effort: returns {} if MSystem is unreachable so get_global() falls
        back to DEFAULTS. Never touches agent_prod or any tenant DB."""
        try:
            router = self._global_router()
            return router.agent_config() if router else {}
        except Exception:
            return {}

    def _global_router(self):
        """Resolve a MSystem router for agent_config reads: reuse the process
        resolver's router if registered, else construct a standalone one from
        AG_MSYSTEM_URL (env)."""
        if self._grouter is None:
            try:
                from core.tenant_resolver import get_global_resolver
                resolver = get_global_resolver()
                self._grouter = getattr(resolver, "_router", None) if resolver else None
            except Exception:
                self._grouter = None
            if self._grouter is None:
                try:
                    from core.msystem_router import MSystemRouter
                    self._grouter = MSystemRouter()
                except Exception:
                    self._grouter = None
        return self._grouter

    def get_int(self, key: str, default: int = 0) -> int:
        """Get a config value as integer."""
        val = self.get(key)
        try:
            return int(val) if val else default
        except (ValueError, TypeError):
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get a config value as float."""
        val = self.get(key)
        try:
            return float(val) if val else default
        except (ValueError, TypeError):
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get a config value as boolean."""
        val = self.get(key)
        if val is None:
            return default
        return val.lower() in ("true", "1", "yes", "on")

    def get_list(self, key: str, separator: str = ",") -> list:
        """Get a config value as a list (comma-separated by default)."""
        val = self.get(key, "")
        if not val:
            return []
        return [v.strip() for v in val.split(separator) if v.strip()]

    def group(self, prefix: str) -> Dict[str, str]:
        """Get all configs under a prefix as a flat dict.

        Strips the prefix from keys:
            app_config.group("llm")
            → {"provider": "openrouter", "model": "...", "api_key": "..."}
        """
        result = {}
        prefix_dot = prefix if prefix.endswith(".") else f"{prefix}."

        from core.tenant_context import current_tenant_id, DEFAULT_TENANT_ID
        tid = current_tenant_id()
        if tid and tid != DEFAULT_TENANT_ID:
            for k, v in DEFAULTS.items():
                if k.startswith(prefix_dot):
                    result[k[len(prefix_dot):]] = v
            overlay = self._tenant_cache() or {}
            for k, v in overlay.items():
                if v and k.startswith(prefix_dot):
                    result[k[len(prefix_dot):]] = v
            return result

        for k, v in self._cache.items():
            if k.startswith(prefix_dot):
                result[k[len(prefix_dot):]] = v
        for k, v in DEFAULTS.items():
            if k.startswith(prefix_dot):
                short_key = k[len(prefix_dot):]
                if short_key not in result:
                    result[short_key] = v
        for env_var, config_key in ENV_TO_CONFIG.items():
            if config_key.startswith(prefix_dot):
                env_val = os.environ.get(env_var)
                if env_val:
                    result[config_key[len(prefix_dot):]] = env_val

        return result

    def set(self, key: str, value: str, is_secret: bool = False,
            description: str = None):
        """Set a config value (writes to DB + updates cache + notifies others).

        TENANT-FIRST (mirrors get()): when the request is bound to a tenant, the
        write goes to THAT tenant's ag_system.configs and its overlay cache — so
        editing config from the web admin actually updates the tenant whose value
        is read back. Without this, writes hit the control DB while reads resolve
        the tenant DB first, so web edits silently had no effect."""
        try:
            from core.tenant_context import (
                current_tenant_id, current_tenant, DEFAULT_TENANT_ID)
            from core.tenant_resolver import get_global_resolver
            tid = current_tenant_id()
            resolver = get_global_resolver()
            if tid and tid != DEFAULT_TENANT_ID and resolver is not None:
                db = resolver.for_tenant(current_tenant()).agent_db
                db.set_config(key, value, is_secret, description)
                overlay = self._tenant_caches.get(tid)
                if overlay is None:
                    overlay = {}
                    self._tenant_caches[tid] = overlay
                overlay[key] = value
                try:
                    from core.redis_client import publish_config_reload
                    publish_config_reload(key)
                except Exception:
                    pass
                return
        except Exception as e:
            logger.error("AppConfig.set (tenant) failed for %s: %s — falling back to control DB", key, e)

        self._cache[key] = value
        if is_secret:
            self._secrets.add(key)

        if self._db:
            try:
                self._db.set_config(key, value, is_secret, description)
            except Exception as e:
                logger.error("AppConfig.set failed for %s: %s", key, e)

        try:
            from core.redis_client import publish_config_reload
            publish_config_reload(key)
        except Exception:
            pass

    def is_secret(self, key: str) -> bool:
        """Check if a key is marked as secret."""
        return key in self._secrets

    def resolve_env(self, env_var: str, default: str = "") -> str:
        """Resolve a legacy env var name through the DB.

        Usage (in existing code that uses os.environ.get):
            # Before:
            token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
            # After:
            token = app_config.resolve_env("TELEGRAM_BOT_TOKEN", "")
        """
        config_key = ENV_TO_CONFIG.get(env_var)
        if config_key and config_key in self._cache and self._cache[config_key]:
            return self._cache[config_key]

        env_val = os.environ.get(env_var)
        if env_val:
            return env_val

        return default

    def dump(self, show_secrets: bool = False) -> Dict[str, str]:
        """Dump all resolved configs (for debugging). Masks secrets by default."""
        result = {}
        all_keys = set(list(self._cache.keys()) + list(DEFAULTS.keys()))
        for key in sorted(all_keys):
            val = self.get(key)
            if val and self.is_secret(key) and not show_secrets:
                result[key] = "***"
            else:
                result[key] = val or ""
        return result

    def reload(self):
        """Hot-reload all configs from the database.

        Called by the Redis pub/sub subscriber when another process
        updates a config. Re-reads the entire configs table.
        """
        if not self._db:
            return
        try:
            rows = self._db.fetch_all("""
                SELECT config_key, config_value, is_secret
                FROM ag_system.configs
                WHERE is_active = TRUE
                ORDER BY config_key
            """)
            new_cache = {}
            new_secrets = set()
            for row in rows:
                new_cache[row['config_key']] = row['config_value'] or ""
                if row['is_secret']:
                    new_secrets.add(row['config_key'])
            self._cache = new_cache
            self._secrets = new_secrets
            self._tenant_caches.clear()
            try:
                if self._grouter is not None:
                    self._grouter.invalidate()
            except Exception:
                pass
            logger.info("AppConfig reloaded %d configs from DB", len(new_cache))
        except Exception as e:
            logger.error("AppConfig.reload failed: %s", e)

    def close(self):
        """Close the DB connection."""
        if self._db:
            self._db.close()


app_config = AppConfig()
