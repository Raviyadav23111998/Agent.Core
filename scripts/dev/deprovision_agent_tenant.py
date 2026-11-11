#!/usr/bin/env python3
"""
Deprovision (offboard) an agent tenant — full lifecycle teardown.

Mirrors MSystem's DatabaseProvisioner (DROP DATABASE IF EXISTS / DROP ROLE) and
extends it across ALL of the agent tenant's separate stores:

  1. Neo4j   — graph.delete_tenant()  (Mode C subgraph: MATCH {tenant_id} DETACH DELETE)
  2. Elastic — DELETE index brain_<suffix8>
  3. Postgres— DROP DATABASE agent_tenant_<suffix8>  (+ DROP ROLE agent_app_<suffix8>)
  4. MSystem — remove tenant_mgmt.app_databases + agent subscription
               (+ tenant/domain/profile only with --purge-tenant)

Safety: requires --confirm. Reads store coordinates from the tenant DB BEFORE
dropping it. Best-effort per store so one failure doesn't strand the others.

Usage:
  AG_MSYSTEM_URL=postgresql://postgres:postgres@localhost:5432/msystems_prod \\
  python3 deprovision_agent_tenant.py --tenant-key agent-pilot2 --confirm \\
      --pg-bin /opt/homebrew/opt/postgresql@17/bin
"""
import argparse
import json
import os
import subprocess
import sys

import psycopg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant-key", required=True)
    ap.add_argument("--confirm", action="store_true", help="required — actually tear down")
    ap.add_argument("--purge-tenant", action="store_true",
                    help="also delete the tenant/domain/profile rows (full offboard)")
    ap.add_argument("--pg-bin", default="/opt/homebrew/opt/postgresql@17/bin")
    ap.add_argument("--db-host", default="127.0.0.1")
    ap.add_argument("--db-port", type=int, default=5432)
    ap.add_argument("--db-su", default="postgres")
    ap.add_argument("--db-su-pass", default=os.environ.get("PGPASSWORD", "postgres"))
    args = ap.parse_args()
    msys_url = os.environ["AG_MSYSTEM_URL"]
    env = dict(os.environ, PGPASSWORD=args.db_su_pass)
    psql = os.path.join(args.pg_bin, "psql")
    base = [psql, "-h", args.db_host, "-p", str(args.db_port), "-U", args.db_su]

    with psycopg.connect(msys_url) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT t.tenant_id::text, right(replace(t.tenant_id::text,'-',''),8),
                   ad.db_name, ad.db_user
            FROM tenant_mgmt.tenants t
            LEFT JOIN tenant_mgmt.app_databases ad ON ad.tenant_id=t.tenant_id
              AND ad.app_id=(SELECT id FROM tenant_mgmt.apps WHERE app_key='agent')
            WHERE t.tenant_key=%s
        """, (args.tenant_key,))
        row = cur.fetchone()
    if not row:
        sys.exit(f"tenant_key '{args.tenant_key}' not found")
    tenant_id, sfx, db_name, db_user = row
    db_name = db_name or f"agent_tenant_{sfx}"
    print(f"offboarding tenant={args.tenant_key} id={tenant_id} suffix={sfx} db={db_name}")
    if not args.confirm:
        sys.exit("refusing to proceed without --confirm")

    es_index = neo_uri = neo_user = neo_pw = None
    try:
        with psycopg.connect(f"postgresql://{args.db_su}:{args.db_su_pass}@{args.db_host}:{args.db_port}/{db_name}") as c, c.cursor() as cur:
            cur.execute("SELECT service_key, config_json FROM ag_system.service_connections")
            for k, cfg in cur.fetchall():
                cfg = cfg if isinstance(cfg, dict) else json.loads(cfg)
                if k == "elastic":
                    es_index = cfg.get("index")
                elif k == "neo4j":
                    neo_uri, neo_user, neo_pw = cfg.get("uri"), cfg.get("user"), cfg.get("password")
    except Exception as e:
        print(f"  (could not read service_connections: {e})")

    if neo_uri:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from brain.storage.graph_store import GraphStore
            g = GraphStore(neo_uri, neo_user or "neo4j", neo_pw or "", tenant_id=tenant_id)
            g.connect(); g.delete_tenant(); g.close()
            print(f"  ✓ neo4j: deleted subgraph tenant_id={tenant_id[:8]}…")
        except Exception as e:
            print(f"  ✗ neo4j delete failed: {e}")

    if es_index:
        try:
            import urllib.request
            req = urllib.request.Request(f"http://localhost:9200/{es_index}", method="DELETE")
            urllib.request.urlopen(req, timeout=10)
            print(f"  ✓ elastic: deleted index {es_index}")
        except Exception as e:
            print(f"  ✗ elastic delete failed: {e}")

    subprocess.run(base + ["-d", "postgres", "-q", "-c",
                   f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='{db_name}' AND pid<>pg_backend_pid();"],
                   env=env, capture_output=True)
    r = subprocess.run(base + ["-d", "postgres", "-c", f'DROP DATABASE IF EXISTS "{db_name}";'],
                       env=env, capture_output=True, text=True)
    print(f"  {'✓' if r.returncode==0 else '✗'} postgres: dropped database {db_name} {r.stderr.strip()[:80]}")
    if db_user and db_user not in ("postgres",):
        subprocess.run(base + ["-d", "postgres", "-c", f'DROP ROLE IF EXISTS "{db_user}";'],
                       env=env, capture_output=True)
        print(f"  ✓ postgres: dropped role {db_user}")

    with psycopg.connect(msys_url) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM tenant_mgmt.app_databases WHERE tenant_id=%s::uuid AND app_id=(SELECT id FROM tenant_mgmt.apps WHERE app_key='agent')", (tenant_id,))
        cur.execute("DELETE FROM tenant_mgmt.subscriptions WHERE tenant_id=%s::uuid AND app_id=(SELECT id FROM tenant_mgmt.apps WHERE app_key='agent')", (tenant_id,))
        if args.purge_tenant:
            cur.execute("DELETE FROM tenant_mgmt.tenant_domains WHERE tenant_id=%s::uuid", (tenant_id,))
            cur.execute("DELETE FROM tenant_mgmt.tenant_profile WHERE tenant_id=%s::uuid", (tenant_id,))
            cur.execute("DELETE FROM tenant_mgmt.tenants WHERE tenant_id=%s::uuid", (tenant_id,))
        conn.commit()
    print(f"  ✓ msystem: removed agent app_databases + subscription" + (" + tenant/domain/profile" if args.purge_tenant else ""))
    print(f"\nDONE → offboarded {args.tenant_key}")


if __name__ == "__main__":
    main()
