"""
Agent — Tenant Database Connector
===============================================
PostgreSQL connection management for the agent_uat database.

Provides TenantDB — a connection pool that resolves tenant-scoped
PostgreSQL connections, matching the SMSystems DatabaseContext pattern:

    SMSystems (C#):
        _headerData.CurrentTenant → DbEndpoint, DbName, DbUser, DbPassword
        → NpgsqlConnectionStringBuilder → NpgsqlConnection

    Agent (Python):
        TenantDB(database_url) → connection_pool
        tenant_db.get_connection() → psycopg connection
        tenant_db.execute(sql, params) → result

Usage:
    from core.tenant_db import TenantDB

    db = TenantDB("postgresql://user:pass@host:5432/agent_uat")

    # Query with schema-aware SQL
    rows = db.fetch_all(\"\"\"
        SELECT * FROM ag_memory.memories
        WHERE user_id = ANY(%s)
    \"\"\", ([visible_scopes],))

    # Insert
    db.execute(\"\"\"
        INSERT INTO ag_system.audit_log (action, entity_type, user_id)
        VALUES (%s, %s, %s)
    \"\"\", ("brain.ingest", "document", user_id))
"""

import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

DEFAULT_DB_URL = os.environ.get(
    "AG_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/agent_uat"
)


class TenantDB:
    """PostgreSQL connection manager for Agent tenant database.

    Uses psycopg (v3) with connection pooling.
    Falls back to psycopg2 if psycopg3 is not available.

    Architecture:
        - Single database per tenant: agent_uat
        - 6 schemas: ag_auth, ag_brain, ag_memory, ag_agents, ag_system, ag_common
        - All queries are schema-qualified (no search_path tricks)
        - Connection pooling with min=2, max=10 connections
    """

    def __init__(self, database_url: str = None):
        self._url = database_url or DEFAULT_DB_URL
        self._pool = None
        self._use_psycopg3 = False
        self._connected = False

    def connect(self):
        """Initialize the connection pool. Idempotent."""
        if self._connected:
            return

        if (self._url == DEFAULT_DB_URL
                and not os.environ.get("AG_DATABASE_URL")):
            try:
                from core.tenant_context import multi_tenant_enabled
                if multi_tenant_enabled():
                    raise RuntimeError(
                        "AG_DATABASE_URL is not set but AG_MULTI_TENANT is enabled. "
                        "Refusing to fall back to the localhost dev database. "
                        "Set AG_DATABASE_URL to the control-plane DSN."
                    )
            except ImportError:
                pass

        import os as _os
        try:
            pool_min = int(_os.environ.get("AG_DB_POOL_MIN", "2"))
            pool_max = int(_os.environ.get("AG_DB_POOL_MAX", "10"))
        except ValueError:
            pool_min, pool_max = 2, 10

        try:
            import psycopg_pool

            self._pool = psycopg_pool.ConnectionPool(
                self._url,
                min_size=pool_min,
                max_size=pool_max,
                open=True,
                kwargs={"autocommit": False}
            )
            self._use_psycopg3 = True
            self._connected = True
            logger.info("TenantDB connected (psycopg3 pool) → %s",
                        self._url.split("@")[-1] if "@" in self._url else "***")
        except ImportError:
            try:
                import psycopg2
                import psycopg2.pool

                self._pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=pool_min,
                    maxconn=pool_max,
                    dsn=self._url,
                )
                self._use_psycopg3 = False
                self._connected = True
                logger.info("TenantDB connected (psycopg2 pool) → %s",
                            self._url.split("@")[-1] if "@" in self._url else "***")
            except ImportError:
                logger.error(
                    "Neither psycopg nor psycopg2 installed. "
                    "Install with: pip install 'psycopg[binary]' psycopg_pool"
                )
                raise ImportError(
                    "PostgreSQL driver required. Install: pip install 'psycopg[binary]' psycopg_pool"
                )

    @property
    def is_connected(self) -> bool:
        """Check if the pool is initialized."""
        return self._connected

    @contextmanager
    def connection(self):
        """Get a connection from the pool (context manager).

        Usage:
            with db.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT ...")
        """
        self.connect()

        if self._use_psycopg3:
            with self._pool.connection() as conn:
                yield conn
        else:
            conn = self._pool.getconn()
            try:
                yield conn
            finally:
                self._pool.putconn(conn)

    @contextmanager
    def cursor(self, commit: bool = True):
        """Get a cursor from the pool (context manager with auto-commit).

        On exception, the transaction is rolled back automatically.

        Usage:
            with db.cursor() as cur:
                cur.execute("INSERT INTO ag_memory.memories ...")
        """
        with self.connection() as conn:
            if self._use_psycopg3:
                with conn.cursor() as cur:
                    try:
                        yield cur
                        if commit:
                            conn.commit()
                    except Exception:
                        conn.rollback()
                        raise
            else:
                cur = conn.cursor()
                try:
                    yield cur
                    if commit:
                        conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    cur.close()

    @contextmanager
    def transaction(self):
        """Explicit transaction context — multiple statements, single commit.

        Usage:
            with db.transaction() as cur:
                cur.execute("INSERT INTO ...")
                cur.execute("UPDATE ...")
                # auto-commits on exit, rolls back on exception
        """
        with self.cursor(commit=True) as cur:
            yield cur


    def execute(self, sql: str, params: Union[tuple, list, None] = None) -> int:
        """Execute a SQL statement and return the rowcount.

        Args:
            sql: Schema-qualified SQL (e.g. "INSERT INTO ag_memory.memories ...")
            params: Query parameters (tuple or list)

        Returns:
            Number of affected rows
        """
        with self.cursor() as cur:
            cur.execute(sql, params)
            return cur.rowcount

    def execute_returning(self, sql: str, params: Union[tuple, list, None] = None) -> Optional[Dict[str, Any]]:
        """Execute a SQL statement with RETURNING and return the row.

        Useful for INSERT ... RETURNING id, INSERT ... RETURNING *.

        Args:
            sql: SQL with RETURNING clause
            params: Query parameters

        Returns:
            Dict of returned columns, or None
        """
        with self.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            if row is None:
                return None
            cols = [desc[0] for desc in cur.description]
            return dict(zip(cols, row))

    def execute_returning_all(self, sql: str, params: Union[tuple, list, None] = None) -> List[Dict[str, Any]]:
        """Execute a mutating statement with RETURNING and return ALL rows.

        Commits (unlike fetch_all) — required for atomic claim patterns like
        UPDATE … WHERE id IN (SELECT … FOR UPDATE SKIP LOCKED) RETURNING *.
        """
        with self.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            if not rows:
                return []
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in rows]

    def fetch_one(self, sql: str, params: Union[tuple, list, None] = None) -> Optional[Dict[str, Any]]:
        """Execute SQL and return one row as a dict, or None."""
        with self.cursor(commit=False) as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            if row is None:
                return None
            cols = [desc[0] for desc in cur.description]
            return dict(zip(cols, row))

    def fetch_all(self, sql: str, params: Union[tuple, list, None] = None) -> List[Dict[str, Any]]:
        """Execute SQL and return all rows as list of dicts."""
        with self.cursor(commit=False) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            if not rows:
                return []
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in rows]

    def fetch_scalar(self, sql: str, params: Union[tuple, list, None] = None) -> Any:
        """Execute SQL and return a single scalar value."""
        with self.cursor(commit=False) as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return row[0] if row else None

    def fetch_count(self, table: str, where: str = "", params: Union[tuple, list, None] = None) -> int:
        """Quick count query on a schema-qualified table.

        Args:
            table: e.g. "ag_brain.documents"
            where: Optional WHERE clause (without the WHERE keyword)
            params: Query parameters for the WHERE clause
        """
        sql = f"SELECT COUNT(*) FROM {table}"
        if where:
            sql += f" WHERE {where}"
        return self.fetch_scalar(sql, params) or 0

    def exists(self, sql: str, params: Union[tuple, list, None] = None) -> bool:
        """Check if a query returns any results."""
        with self.cursor(commit=False) as cur:
            cur.execute(sql, params)
            return cur.fetchone() is not None


    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Look up a tenant user by internal ID."""
        return self.fetch_one("""
            SELECT u.*, r.role_key, r.role_name
            FROM ag_auth.users u
            JOIN ag_auth.roles r ON u.role_id = r.role_id
            WHERE u.user_id = %s
        """, (user_id,))

    def get_user_by_external_id(self, external_id: str) -> Optional[Dict]:
        """Look up a tenant user by their external platform ID (Telegram, etc.)."""
        return self.fetch_one("""
            SELECT u.*, r.role_key, r.role_name
            FROM ag_auth.users u
            JOIN ag_auth.roles r ON u.role_id = r.role_id
            WHERE u.external_id = %s AND u.is_active = TRUE
        """, (external_id,))

    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Look up a tenant user by email."""
        return self.fetch_one("""
            SELECT u.*, r.role_key, r.role_name
            FROM ag_auth.users u
            JOIN ag_auth.roles r ON u.role_id = r.role_id
            WHERE LOWER(u.email) = LOWER(%s) AND u.is_active = TRUE
        """, (email,))

    def upsert_user(self, external_id: str, full_name: str = "",
                    email: str = None, platform: str = "web",
                    role_key: str = "member", metadata: dict = None) -> Dict:
        """Create or update a user by external_id. Returns the user dict.

        This is the main entry point for user auto-provisioning (e.g., first
        Telegram message auto-creates the user).
        """
        existing = self.get_user_by_external_id(external_id)
        if existing:
            self.execute("""
                UPDATE ag_auth.users SET last_login_at = NOW(), 
                    full_name = COALESCE(NULLIF(%s, ''), full_name),
                    platform = %s
                WHERE user_id = %s
            """, (full_name, platform, existing['user_id']))
            existing['last_login_at'] = 'now'
            return existing

        role_id = self.fetch_scalar(
            "SELECT role_id FROM ag_auth.roles WHERE role_key = %s",
            (role_key,)
        ) or 3

        return self.execute_returning("""
            INSERT INTO ag_auth.users
                (external_id, full_name, email, role_id, platform, 
                 metadata_json, last_login_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            RETURNING *, 
                (SELECT role_key FROM ag_auth.roles WHERE role_id = ag_auth.users.role_id) AS role_key,
                (SELECT role_name FROM ag_auth.roles WHERE role_id = ag_auth.users.role_id) AS role_name
        """, (external_id, full_name or f"User-{external_id}",
              email, role_id, platform,
              json.dumps(metadata or {})))

    def update_last_login(self, user_id: int):
        """Update the last login timestamp for a user."""
        self.execute(
            "UPDATE ag_auth.users SET last_login_at = NOW() WHERE user_id = %s",
            (user_id,)
        )


    def get_user_permissions(self, user_id: int) -> List[Dict]:
        """Get effective permissions for a user (role + overrides merged).

        Uses the effective_permissions view which merges role_permissions
        with user_permission_overrides (override wins).
        """
        return self.fetch_all("""
            SELECT module_key, can_create, can_read, can_update, can_delete, is_overridden
            FROM ag_auth.effective_permissions
            WHERE user_id = %s
            ORDER BY module_key
        """, (user_id,))

    def check_permission(self, user_id: int, module_key: str, action: str = "read") -> bool:
        """Check if a user has a specific permission on a module.

        Uses the effective_permissions view (merges role + overrides).

        Args:
            user_id: Tenant user ID
            module_key: e.g. 'brain_ingest', 'agent_mgmt'
            action: 'create', 'read', 'update', or 'delete'
        """
        valid_actions = {"create", "read", "update", "delete"}
        if action not in valid_actions:
            raise ValueError(f"Invalid action '{action}'. Must be one of: {valid_actions}")

        col = f"can_{action}"
        result = self.fetch_scalar(f"""
            SELECT {col}
            FROM ag_auth.effective_permissions
            WHERE user_id = %s AND module_key = %s
        """, (user_id, module_key))
        return bool(result)


    def get_config(self, key: str, default: str = None) -> Optional[str]:
        """Get a tenant configuration value."""
        result = self.fetch_scalar("""
            SELECT config_value FROM ag_system.configs
            WHERE config_key = %s AND is_active = TRUE
        """, (key,))
        return result if result is not None else default

    def get_configs(self, prefix: str = "") -> Dict[str, str]:
        """Get all configs matching a prefix as a dict.

        Usage:
            llm_config = db.get_configs("llm.")
            # → {"llm.provider": "openrouter", "llm.model": "...", ...}
        """
        rows = self.fetch_all("""
            SELECT config_key, config_value FROM ag_system.configs
            WHERE config_key LIKE %s AND is_active = TRUE
            ORDER BY config_key
        """, (f"{prefix}%",))
        return {r['config_key']: r['config_value'] for r in rows}

    def set_config(self, key: str, value: str, is_secret: bool = False,
                   description: str = None):
        """Set a tenant configuration value (upsert)."""
        self.execute("""
            INSERT INTO ag_system.configs (config_key, config_value, is_secret, description)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (config_key)
            DO UPDATE SET config_value = EXCLUDED.config_value,
                          is_secret = EXCLUDED.is_secret,
                          description = COALESCE(EXCLUDED.description, ag_system.configs.description),
                          updated_at = NOW()
        """, (key, value, is_secret, description))


    def audit(self, action: str, entity_type: str = None,
              entity_id: str = None, user_id: int = None,
              details: dict = None, ip_address: str = None):
        """Log an audit entry. Fire-and-forget (no return value)."""
        self.execute("""
            INSERT INTO ag_system.audit_log
                (action, entity_type, entity_id, user_id, details_json, ip_address)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s)
        """, (action, entity_type, entity_id, user_id,
              json.dumps(details or {}), ip_address))


    def get_tenant_stats(self) -> Dict[str, Any]:
        """Get dashboard stats for this tenant."""
        return self.fetch_one("SELECT * FROM ag_system.tenant_stats") or {}


    def get_integration(self, service_key: str) -> Optional[Dict]:
        """Get integration config by service key (jira, outlook, slack, etc.)."""
        return self.fetch_one("""
            SELECT * FROM ag_system.integrations WHERE service_key = %s
        """, (service_key,))

    def upsert_integration(self, service_key: str, display_name: str,
                           config: dict = None, credentials_enc: str = None,
                           is_enabled: bool = False):
        """Create or update an integration."""
        self.execute("""
            INSERT INTO ag_system.integrations
                (service_key, display_name, config_json, credentials_enc, is_enabled, status)
            VALUES (%s, %s, %s::jsonb, %s, %s, CASE WHEN %s THEN 'active' ELSE 'unconfigured' END)
            ON CONFLICT (service_key)
            DO UPDATE SET
                display_name = EXCLUDED.display_name,
                config_json = COALESCE(EXCLUDED.config_json, ag_system.integrations.config_json),
                credentials_enc = COALESCE(EXCLUDED.credentials_enc, ag_system.integrations.credentials_enc),
                is_enabled = EXCLUDED.is_enabled,
                status = CASE WHEN EXCLUDED.is_enabled THEN 'active' ELSE ag_system.integrations.status END
        """, (service_key, display_name, json.dumps(config or {}),
              credentials_enc, is_enabled, is_enabled))


    def close(self):
        """Close all connections in the pool."""
        if self._pool and self._connected:
            try:
                if self._use_psycopg3:
                    self._pool.close()
                else:
                    self._pool.closeall()
            except Exception as e:
                logger.warning("Error closing TenantDB pool: %s", e)
            finally:
                self._connected = False
                self._pool = None
                logger.info("TenantDB connection pool closed")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        driver = "psycopg3" if self._use_psycopg3 else "psycopg2"
        db = self._url.split("/")[-1].split("?")[0] if self._url else "?"
        return f"TenantDB({db}, {driver}, {status})"
