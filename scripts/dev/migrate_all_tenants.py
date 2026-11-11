#!/usr/bin/env python3
"""
Per-tenant migration runner — apply an ag_* migration to EVERY agent tenant DB.

Mirrors MSystem's migration model (ops.migration_scripts → ops.tenant_migrations):
iterate every provisioned agent tenant (tenant_mgmt.app_databases where app='agent'),
and apply the migration unless its version is already recorded in that tenant's
ag_system.migration_history. Idempotent + per-tenant status.

Schema changes to ag_* MUST go through this so all tenant DBs stay in lockstep
(the single-DB dev flow keeps using the same V-scripts).

Usage:
  AG_MSYSTEM_URL=postgresql://postgres:postgres@localhost:5432/msystems_prod \\
  python3 migrate_all_tenants.py --migration ../../agent.api/migrations/V0XX_foo.sql
  # add --include-default to also apply to the control DB (AG_DATABASE_URL)
"""
import argparse
import os
import re
import sys

import psycopg


def version_of(path: str) -> str:
    m = re.match(r"(V\d+)", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def apply_to(dsn: str, label: str, version: str, script_name: str, sql: str) -> str:
    try:
        with psycopg.connect(dsn, autocommit=False) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 FROM ag_system.migration_history WHERE version=%s", (version,))
            if cur.fetchone():
                return f"  {label:28s} → already applied ({version})"
            cur.execute(sql)
            cur.execute(
                "INSERT INTO ag_system.migration_history (version, script_name, description, applied_by) "
                "VALUES (%s,%s,%s,'migrate_all_tenants') ON CONFLICT DO NOTHING",
                (version, script_name, f"per-tenant runner: {script_name}"))
            conn.commit()
            return f"  {label:28s} → APPLIED {version}"
    except Exception as e:
        return f"  {label:28s} → ERROR: {str(e)[:80]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--migration", required=True, help="path to a V0xx_*.sql file")
    ap.add_argument("--include-default", action="store_true",
                    help="also apply to the control DB (AG_DATABASE_URL)")
    ap.add_argument("--db-su", default="postgres")
    ap.add_argument("--db-su-pass", default=os.environ.get("PGPASSWORD", "postgres"))
    args = ap.parse_args()

    path = os.path.abspath(args.migration)
    sql = open(path, "r", encoding="utf-8").read()
    version, script_name = version_of(path), os.path.basename(path)
    print(f"Applying {script_name} ({version}) to all agent tenants…")

    results = []
    if args.include_default and os.environ.get("AG_DATABASE_URL"):
        results.append(apply_to(os.environ["AG_DATABASE_URL"], "default(control)", version, script_name, sql))

    with psycopg.connect(os.environ["AG_MSYSTEM_URL"]) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT t.tenant_key, ad.db_name, ad.db_endpoint, ad.db_port, ad.db_user, ad.db_password
            FROM tenant_mgmt.app_databases ad
            JOIN tenant_mgmt.apps a ON a.id=ad.app_id AND a.app_key='agent'
            JOIN tenant_mgmt.tenants t ON t.tenant_id=ad.tenant_id
            ORDER BY t.tenant_key
        """)
        tenants = cur.fetchall()

    for tkey, db_name, host, port, user, pw in tenants:
        u = user or args.db_su
        p = pw or args.db_su_pass
        dsn = f"postgresql://{u}:{p}@{host}:{port}/{db_name}"
        results.append(apply_to(dsn, tkey, version, script_name, sql))

    print("\n".join(results))
    print(f"\nDone — {len(tenants)} agent tenant(s).")
    if any("ERROR" in r for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
