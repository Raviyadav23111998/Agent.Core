"""
Connections — the credential system, rebuilt from scratch.
═══════════════════════════════════════════════════════════════════════════
One concept: a Connection. A single row in ag_cred.connections says
"this principal (org / user / app) can use this provider". There is no vault,
no bindings table, no active-pointer indirection — uniqueness on
(provider_key, scope, owner_ref) IS the active credential.

Two provider kinds drive resolution (ag_cred.providers.kind):
  • direct      — each connection is self-contained (endpoint + secret).
  • shared_app  — the ORG connection holds the app registration + secret + a
                  default identity; user/app connections hold ONLY identity
                  (e.g. a mailbox). resolve() merges them.

Resolution (one function, total ordering, no per-call-site surprises):
  shared_app:  org app registration  +  (user → app → org-default) identity,
               secret ALWAYS the org's.
  direct:      user → app → org   (org = an explicit admin-published default).

App principals are keyed by app_uuid, never a numeric id, so an app can never
collide with / shadow a user id.

See CREDENTIAL_SYSTEM_REDESIGN.md.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.security import encrypt_secret, decrypt_secret

logger = logging.getLogger(__name__)


# Provider definitions change rarely (only when an integration is connected or
# removed) but are read on every credential resolution. The cache is PROCESS-WIDE
# (not per-Connections-instance — the API builds a fresh Connections per request)
# and keyed by tenant. Writers publish a Redis 'providers' signal so the gateway /
# other workers, which hold a long-lived Connections, drop their copy immediately
# instead of waiting out the TTL. See invalidate_provider_cache().
_PROVIDER_CACHE: Dict[str, tuple] = {}


def invalidate_provider_cache(tenant_id: Optional[str] = None) -> None:
    """Clear the process-wide provider cache for one tenant, or all tenants.

    Called locally right after a provider write, and by the Redis config-reload
    subscriber (api.main / gateway.runner) when another process signals a
    'providers' change — so a newly connected or deleted integration is visible
    everywhere at once rather than up to _PROVIDER_TTL seconds later.
    """
    if tenant_id is None:
        _PROVIDER_CACHE.clear()
    else:
        _PROVIDER_CACHE.pop(tenant_id, None)


@dataclass(frozen=True)
class ResolvedCredential:
    """Immutable, ready-to-use credential returned by resolve()."""
    connection_id: int
    provider_key: str
    config: Dict[str, Any]
    secret: str
    scope: str = "org"
    owner_ref: str = ""
    label: str = "Default"
    status: str = "active"

    def config_get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)


@dataclass
class ConnectionSummary:
    """Listing view — never carries the secret value."""
    connection_id: int
    provider_key: str
    scope: str
    owner_ref: str
    label: str
    status: str
    config_keys: List[str]
    has_secret: bool
    last_verified_at: Optional[str] = None
    last_used_at: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class ConnectionProvider:
    """Provider definition from ag_cred.providers."""
    provider_key: str
    display_name: str
    category: str
    kind: str
    fields: List[Dict[str, Any]]
    supports_user: bool
    supports_verify: bool
    is_active: bool
    sort_order: int
    icon_url: Optional[str] = None
    docs_url: Optional[str] = None


class ConnectionNotConfigured(Exception):
    """resolve() found no usable connection."""
    def __init__(self, provider: str, reason: str = ""):
        self.provider = provider
        self.reason = reason
        msg = f"{provider} is not configured"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)


class ConnectionValidationError(Exception):
    """A connection's fields violate the provider schema (HTTP 400)."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class Connections:
    """The single interface tools, services and the API use for credentials.

    Tools never touch the DB, never call os.getenv(), never decrypt by hand.
    """

    _PROVIDER_TTL = 300

    def __init__(self, db):
        self._db = db
        # Process-wide, tenant-keyed (see module-level _PROVIDER_CACHE).
        self._provider_cache = _PROVIDER_CACHE

    @staticmethod
    def _tid() -> str:
        try:
            from core.tenant_context import current_tenant_id
            return current_tenant_id()
        except Exception:
            return "default"

    def get_providers(self, include_inactive: bool = False) -> List[ConnectionProvider]:
        now = time.time()
        tid = self._tid()
        cached = self._provider_cache.get(tid)
        if cached and not include_inactive and now - cached[0] < self._PROVIDER_TTL:
            return list(cached[1].values())

        where = "" if include_inactive else "WHERE is_active = TRUE"
        rows = self._db.fetch_all(
            f"SELECT * FROM ag_cred.providers {where} ORDER BY sort_order, provider_key"
        )
        out, cache = [], {}
        for r in rows:
            fields = r.get("fields") or []
            if isinstance(fields, str):
                fields = json.loads(fields)
            p = ConnectionProvider(
                provider_key=r["provider_key"], display_name=r["display_name"],
                category=r["category"], kind=r["kind"], fields=fields,
                supports_user=r["supports_user"], supports_verify=r["supports_verify"],
                is_active=r["is_active"], sort_order=r["sort_order"],
                icon_url=r.get("icon_url"), docs_url=r.get("docs_url"),
            )
            out.append(p)
            if p.is_active:
                cache[p.provider_key] = p
        if not include_inactive:
            self._provider_cache[tid] = (now, cache)
        return out

    def get_provider(self, provider_key: str) -> Optional[ConnectionProvider]:
        cached = self._provider_cache.get(self._tid())
        if not cached:
            self.get_providers()
            cached = self._provider_cache.get(self._tid())
        return (cached[1] if cached else {}).get(provider_key)

    def upsert_provider(self, provider_key: str, display_name: str, *,
                        kind: str = "direct", category: str = "api",
                        fields: Optional[List[Dict[str, Any]]] = None,
                        supports_user: bool = True, supports_verify: bool = True,
                        sort_order: int = 500, icon_url: Optional[str] = None,
                        docs_url: Optional[str] = None) -> None:
        """Create or update a provider definition (the write side of the registry).

        Dynamic API integrations call this to materialize an 'api_{slug}' provider
        from a spec's security scheme, so the existing Connections form +
        _split/_sanitize/_validate + resolve() machinery work unchanged. `fields`
        is the standard schema: [{key,label,type,required,secret,scope,help}].
        Idempotent; busts the per-tenant provider cache so the new provider is
        visible immediately (the cache is keyed by tenant — see __init__).
        """
        if kind not in ("direct", "shared_app"):
            raise ConnectionValidationError(f"Invalid provider kind: {kind}")
        self._db.execute(
            """INSERT INTO ag_cred.providers
                   (provider_key, display_name, category, kind, fields,
                    supports_user, supports_verify, is_active, sort_order,
                    icon_url, docs_url)
               VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, TRUE, %s, %s, %s)
               ON CONFLICT (provider_key) DO UPDATE SET
                   display_name = EXCLUDED.display_name,
                   category = EXCLUDED.category,
                   kind = EXCLUDED.kind,
                   fields = EXCLUDED.fields,
                   supports_user = EXCLUDED.supports_user,
                   supports_verify = EXCLUDED.supports_verify,
                   is_active = TRUE,
                   sort_order = EXCLUDED.sort_order,
                   icon_url = EXCLUDED.icon_url,
                   docs_url = EXCLUDED.docs_url""",
            (provider_key, display_name, category, kind, json.dumps(fields or []),
             supports_user, supports_verify, sort_order, icon_url, docs_url),
        )
        self._invalidate_providers()
        logger.info("connections.upsert_provider: %s (kind=%s, %d fields)",
                    provider_key, kind, len(fields or []))

    def resolve(self, provider: str, *, user_id: Optional[int] = None,
                app_id: Optional[str] = None) -> ResolvedCredential:
        """Resolve a usable credential for a provider. See module docstring.

        `app_id` may be a numeric application id (translated to its app_uuid) or
        an app_uuid directly. Raises ConnectionNotConfigured if nothing resolves.
        """
        p = self.get_provider(provider)
        if not p:
            raise ConnectionNotConfigured(provider, "unknown provider")

        app_uuid = self._app_uuid(app_id) if app_id else None

        if p.kind == "shared_app":
            org = self._load(provider, "org", "")
            if not org:
                raise ConnectionNotConfigured(provider, "organization app not configured")
            identity = None
            if user_id is not None:
                identity = self._load(provider, "user", str(user_id))
            if identity is None and app_uuid:
                identity = self._load(provider, "app", app_uuid)
            chosen = identity or org
            merged = dict(org.config)
            if identity:
                merged.update(identity.config)
            self._touch(chosen.connection_id)
            return ResolvedCredential(
                connection_id=chosen.connection_id, provider_key=provider,
                config=merged, secret=org.secret, scope=chosen.scope,
                owner_ref=chosen.owner_ref, label=chosen.label, status=chosen.status,
            )

        chain = []
        if user_id is not None:
            chain.append(("user", str(user_id)))
        if app_uuid:
            chain.append(("app", app_uuid))
        chain.append(("org", ""))
        for scope, ref in chain:
            row = self._load(provider, scope, ref)
            if row:
                self._touch(row.connection_id)
                return row
        raise ConnectionNotConfigured(provider)

    def resolve_or_none(self, provider: str, *, user_id: Optional[int] = None,
                        app_id: Optional[str] = None) -> Optional[ResolvedCredential]:
        try:
            return self.resolve(provider, user_id=user_id, app_id=app_id)
        except ConnectionNotConfigured:
            return None

    def list_connections(self, *, provider: Optional[str] = None,
                         scope: Optional[str] = None,
                         owner_ref: Optional[str] = None) -> List[ConnectionSummary]:
        conds, params = ["1=1"], []
        if provider:
            conds.append("provider_key = %s"); params.append(provider)
        if scope:
            conds.append("scope = %s"); params.append(scope)
        if owner_ref is not None:
            conds.append("owner_ref = %s"); params.append(owner_ref)
        rows = self._db.fetch_all(
            f"""SELECT connection_id, provider_key, scope, owner_ref, label, status,
                       config, (secret_enc IS NOT NULL) AS has_secret,
                       last_verified_at, last_used_at, created_at
                FROM ag_cred.connections
                WHERE {' AND '.join(conds)}
                ORDER BY provider_key, scope, owner_ref""",
            tuple(params),
        )
        out = []
        for r in rows:
            cfg = r.get("config") or {}
            if isinstance(cfg, str):
                cfg = json.loads(cfg)
            out.append(ConnectionSummary(
                connection_id=r["connection_id"], provider_key=r["provider_key"],
                scope=r["scope"], owner_ref=r["owner_ref"], label=r["label"],
                status=r["status"], config_keys=list(cfg.keys()),
                has_secret=r["has_secret"],
                last_verified_at=str(r["last_verified_at"]) if r.get("last_verified_at") else None,
                last_used_at=str(r["last_used_at"]) if r.get("last_used_at") else None,
                created_at=str(r["created_at"]) if r.get("created_at") else None,
            ))
        return out

    def get_masked(self, connection_id: int) -> Optional[dict]:
        """Connection detail with secret fields masked, for the edit form."""
        r = self._db.fetch_one(
            """SELECT connection_id, provider_key, scope, owner_ref, label, status,
                      config, last_used_at, created_at
               FROM ag_cred.connections WHERE connection_id = %s""",
            (connection_id,),
        )
        if not r:
            return None
        cfg = r.get("config") or {}
        if isinstance(cfg, str):
            cfg = json.loads(cfg)
        secret_keys = self._secret_keys(r["provider_key"])
        masked = {k: ("••••••" if k in secret_keys else v) for k, v in cfg.items()}
        return {
            "connection_id": r["connection_id"], "provider_key": r["provider_key"],
            "scope": r["scope"], "owner_ref": r["owner_ref"], "label": r["label"],
            "status": r["status"], "config": masked,
            "last_used_at": str(r["last_used_at"]) if r.get("last_used_at") else None,
            "created_at": str(r["created_at"]) if r.get("created_at") else None,
        }

    def upsert(self, provider: str, scope: str, owner_ref: str,
               fields: Dict[str, Any], label: str = "Default",
               created_by: Optional[int] = None,
               status: str = "active") -> int:
        """Create or replace the connection for (provider, scope, owner_ref).

        `fields` is the raw form payload; the server splits secrets, drops
        out-of-scope fields and enforces required ones (the ONE place pollution
        is prevented). Raises ConnectionValidationError on a bad payload.
        """
        if not self.get_provider(provider):
            raise ConnectionValidationError(f"Unknown provider: {provider}")
        if scope not in ("org", "user", "app"):
            raise ConnectionValidationError(f"Invalid scope: {scope}")
        owner_ref = "" if scope == "org" else (owner_ref or "")
        if scope != "org" and not owner_ref:
            raise ConnectionValidationError(f"{scope} connections require an owner.")

        config, secret = self._split(provider, fields)
        config = self._sanitize(provider, config, scope)
        self._validate(provider, config, secret, scope)
        removed = self._removed_keys(provider, fields, scope)

        secret_enc = encrypt_secret(secret) if secret else None
        keep_secret_clause = "secret_enc = EXCLUDED.secret_enc" if secret_enc else \
            "secret_enc = ag_cred.connections.secret_enc"

        base_cfg = "COALESCE(ag_cred.connections.config,'{}'::jsonb) || EXCLUDED.config"
        if removed:
            config_clause = f"config = ({base_cfg}) - %s::text[]"
            extra_params = [removed]
        else:
            config_clause = f"config = {base_cfg}"
            extra_params = []

        row = self._db.execute_returning(
            f"""INSERT INTO ag_cred.connections
                   (provider_key, scope, owner_ref, label, config, secret_enc,
                    status, created_by)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s)
                ON CONFLICT (provider_key, scope, owner_ref) DO UPDATE SET
                   label = EXCLUDED.label,
                   {config_clause},
                   {keep_secret_clause},
                   status = EXCLUDED.status,
                   updated_at = NOW()
                RETURNING connection_id""",
            (provider, scope, owner_ref, label, json.dumps(config),
             secret_enc, status, created_by, *extra_params),
        )
        cid = row["connection_id"]
        logger.info("connections.upsert: %s %s:%s -> id=%s", provider, scope, owner_ref or "·", cid)
        return cid

    def update(self, connection_id: int, *, fields: Optional[Dict[str, Any]] = None,
               label: Optional[str] = None, status: Optional[str] = None) -> None:
        meta = self._db.fetch_one(
            "SELECT provider_key, scope FROM ag_cred.connections WHERE connection_id = %s",
            (connection_id,),
        )
        if not meta:
            raise ConnectionValidationError("Connection not found.")
        sets, params = [], []
        if fields is not None:
            config, secret = self._split(meta["provider_key"], fields)
            config = self._sanitize(meta["provider_key"], config, meta["scope"])
            removed = self._removed_keys(meta["provider_key"], fields, meta["scope"])
            if removed:
                sets.append("config = (COALESCE(config,'{}'::jsonb) || %s::jsonb) - %s::text[]")
                params.append(json.dumps(config)); params.append(removed)
            else:
                sets.append("config = COALESCE(config,'{}'::jsonb) || %s::jsonb")
                params.append(json.dumps(config))
            if secret:
                sets.append("secret_enc = %s"); params.append(encrypt_secret(secret))
        if label is not None:
            sets.append("label = %s"); params.append(label)
        if status is not None:
            sets.append("status = %s"); params.append(status)
        if not sets:
            return
        sets.append("updated_at = NOW()")
        params.append(connection_id)
        self._db.execute(
            f"UPDATE ag_cred.connections SET {', '.join(sets)} WHERE connection_id = %s",
            tuple(params),
        )

    def delete(self, connection_id: int) -> None:
        self._db.execute(
            "DELETE FROM ag_cred.connections WHERE connection_id = %s", (connection_id,)
        )
        logger.info("connections.delete: id=%s", connection_id)

    def delete_provider(self, provider_key: str) -> None:
        """Remove a provider and ALL its stored connections.

        Used when an integration that AUTO-CREATED its provider (`api_{slug}`) is
        deleted, so a deleted integration never leaves an orphaned provider +
        secret behind. Caller must ensure the provider is exclusively owned by
        that integration (api_* keys are 1:1 with an integration slug). Built-in
        providers (jira/microsoft365/…) must NOT be passed here.
        """
        self._db.execute(
            "DELETE FROM ag_cred.connections WHERE provider_key = %s", (provider_key,))
        self._db.execute(
            "DELETE FROM ag_cred.providers WHERE provider_key = %s", (provider_key,))
        self._invalidate_providers()
        logger.info("connections.delete_provider: %s (provider + connections removed)",
                    provider_key)

    def _invalidate_providers(self) -> None:
        """Drop this process's cached providers for the current tenant and tell
        other workers to do the same (best-effort Redis publish)."""
        invalidate_provider_cache(self._tid())
        try:
            from core.redis_client import publish_config_reload
            publish_config_reload("providers")
        except Exception as e:
            logger.debug("connections: provider-cache reload publish failed: %s", e)

    def verify(self, connection_id: int) -> tuple:
        """Live-test a stored connection. shared-app identity rows are merged
        with the org registration first, so the test uses exactly what a tool
        would resolve. Returns (ok, message)."""
        r = self._db.fetch_one(
            """SELECT provider_key, scope, owner_ref, config, secret_enc
               FROM ag_cred.connections WHERE connection_id = %s""",
            (connection_id,),
        )
        if not r:
            return False, "Connection not found."
        provider = r["provider_key"]
        cfg = r.get("config") or {}
        if isinstance(cfg, str):
            cfg = json.loads(cfg)
        secret = decrypt_secret(r["secret_enc"]) if r.get("secret_enc") else ""

        p = self.get_provider(provider)
        if p and p.kind == "shared_app" and r["scope"] != "org":
            org = self._load(provider, "org", "")
            if org:
                cfg = {**org.config, **cfg}
                secret = org.secret or secret

        from core.credential_verify import verify_credential
        ok, msg = verify_credential(provider, cfg, secret, connections=self)
        try:
            self._db.execute(
                "UPDATE ag_cred.connections SET last_verified_at = NOW(), "
                "status = %s WHERE connection_id = %s",
                ("active" if ok else "error", connection_id),
            )
        except Exception:
            pass
        return ok, msg

    def _fields(self, provider: str) -> List[Dict[str, Any]]:
        p = self.get_provider(provider)
        return [f for f in (p.fields if p else []) if isinstance(f, dict) and f.get("key")]

    def _secret_keys(self, provider: str) -> set:
        return {f["key"] for f in self._fields(provider) if f.get("secret")}

    @staticmethod
    def _allowed_scopes(scope: str) -> set:
        if scope == "org":
            return {"org", "both", None}
        return {"identity", "both", None}

    def _split(self, provider: str, fields: Dict[str, Any]) -> tuple:
        """Split a raw form payload into (config, secret) using the schema.
        One secret → plain string; multiple → JSON string. Masked/empty dropped.
        """
        defs = {f["key"]: f for f in self._fields(provider)}
        config: Dict[str, Any] = {}
        secrets: Dict[str, str] = {}
        for key, val in (fields or {}).items():
            d = defs.get(key)
            is_secret = bool(d and d.get("secret"))
            if is_secret:
                if val not in (None, "", "••••••"):
                    secrets[key] = val
            elif val not in (None, ""):
                config[key] = val
        if len(secrets) == 1:
            secret = next(iter(secrets.values()))
        elif len(secrets) > 1:
            secret = json.dumps(secrets)
        else:
            secret = ""
        return config, secret

    def _sanitize(self, provider: str, config: Dict[str, Any], scope: str) -> Dict[str, Any]:
        """Drop fields that don't belong at this scope, and any stray secret."""
        defs = {f["key"]: f for f in self._fields(provider)}
        allowed = self._allowed_scopes(scope)
        clean = {}
        for key, val in (config or {}).items():
            d = defs.get(key)
            if d is None:
                clean[key] = val
                continue
            if d.get("secret"):
                continue
            if d.get("scope") in allowed:
                clean[key] = val
        return clean

    def _removed_keys(self, provider: str, fields: Dict[str, Any], scope: str) -> List[str]:
        """Non-secret schema fields the form submitted EMPTY at this scope.

        These are explicit clears: config keeps merge semantics (so a partial
        update — e.g. rotating only the secret — never wipes other keys), but a
        field the user blanked in the edit form is actually removed instead of
        silently retaining its old value.
        """
        defs = {f["key"]: f for f in self._fields(provider)}
        allowed = self._allowed_scopes(scope)
        removed = []
        for key, val in (fields or {}).items():
            d = defs.get(key)
            if not d or d.get("secret") or d.get("scope") not in allowed:
                continue
            if val is None or (isinstance(val, str) and not val.strip()):
                removed.append(key)
        return removed

    def _validate(self, provider: str, config: Dict[str, Any], secret: str, scope: str) -> None:
        """Enforce required fields this scope is responsible for."""
        allowed = self._allowed_scopes(scope)
        cfg = config or {}
        has_secret = bool(secret)
        for f in self._fields(provider):
            if not f.get("required") or f.get("scope") not in allowed:
                continue
            if f.get("secret"):
                continue
            val = cfg.get(f["key"])
            if val is None or (isinstance(val, str) and not val.strip()):
                raise ConnectionValidationError(f"{f.get('label', f['key'])} is required.")

    def _load(self, provider: str, scope: str, owner_ref: str) -> Optional[ResolvedCredential]:
        r = self._db.fetch_one(
            """SELECT connection_id, provider_key, scope, owner_ref, label, status,
                      config, secret_enc
               FROM ag_cred.connections
               WHERE provider_key = %s AND scope = %s AND owner_ref = %s
                 AND status <> 'disabled'""",
            (provider, scope, owner_ref),
        )
        if not r:
            return None
        cfg = r.get("config") or {}
        if isinstance(cfg, str):
            cfg = json.loads(cfg)
        secret = decrypt_secret(r["secret_enc"]) if r.get("secret_enc") else ""
        return ResolvedCredential(
            connection_id=r["connection_id"], provider_key=r["provider_key"],
            config=cfg, secret=secret, scope=r["scope"], owner_ref=r["owner_ref"],
            label=r["label"], status=r["status"],
        )

    def _app_uuid(self, app_id: str) -> Optional[str]:
        """Translate a numeric application id to its app_uuid (passthrough if it
        already looks like a uuid). NOT cached: a shared Connections instance
        serves every tenant via the DB proxy, so app_id 1 maps to a different
        uuid per tenant — the lookup must hit the (tenant-routed) DB each time.
        """
        key = str(app_id)
        if "-" in key:
            return key
        try:
            row = self._db.fetch_one(
                "SELECT app_uuid FROM ag_system.applications WHERE app_id::text = %s", (key,)
            )
            return str(row["app_uuid"]) if row and row.get("app_uuid") else None
        except Exception as e:
            logger.debug("connections._app_uuid lookup failed for %s: %s", key, e)
            return None

    def _touch(self, connection_id: int) -> None:
        try:
            self._db.execute(
                "UPDATE ag_cred.connections SET last_used_at = NOW() WHERE connection_id = %s",
                (connection_id,),
            )
        except Exception:
            pass
