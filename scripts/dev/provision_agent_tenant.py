#!/usr/bin/env python3
"""
Provision a NEW agent tenant — MSystem registration + agent DB.

Mirrors MSystems OnboardService.OnboardTenantAsync (the C# onboarding flow):
  1. validate 'agent' app + tenant_key uniqueness
  2. resolvedDomain = "{tenant_key}.{baseDomain}"
  3. INSERT tenant_mgmt.tenants            (TenantId = gen_random_uuid())
  4. INSERT tenant_mgmt.tenant_profile     (branding)
  5. INSERT tenant_mgmt.tenant_domains     (primary domain)
  6. INSERT tenant_mgmt.subscriptions      (agent app, active)
  7. INSERT tenant_mgmt.app_databases      (agent_tenant_<suffix8> routing)
  8. CREATE DATABASE agent_tenant_<suffix8> + clone ag_* schema + seed service_connections

The agent is a SEPARATE product → a SEPARATE tenant registration (new GUID →
new suffix8 → fresh DB). It does NOT reuse any SMSystem (school) tenant.

This is the minimal seed of N9 (Phase 5 tenant_provisioner). Idempotent on tenant_key.

Usage:
  AG_MSYSTEM_URL=postgresql://postgres:postgres@localhost:5432/msystems_prod \\
  python3 provision_agent_tenant.py --tenant-key agent-pilot \\
      --org "Agent Pilot" --base-domain storeglance.com --instance-id 4 \\
      --template-db agent_uat --pg-bin /opt/homebrew/opt/postgresql@17/bin
"""
import argparse
import os
import subprocess
import sys

import psycopg


def suffix8(tenant_id: str) -> str:
    return tenant_id.replace("-", "").lower()[-8:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant-key", required=True)
    ap.add_argument("--org", default=None)
    ap.add_argument("--base-domain", default="storeglance.com")
    ap.add_argument("--domain", default=None, help="override full domain")
    ap.add_argument("--instance-id", type=int, default=4)
    ap.add_argument("--template-db", default="agent_uat",
                    help="fallback schema source to clone (used only if --bootstrap-sql is missing)")
    ap.add_argument("--bootstrap-sql",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "..", "..", "agent.api", "migrations",
                                         "bootstrap", "agent_bootstrap.sql"),
                    help="canonical bootstrap SQL (schema V034 + curated platform seed, no creds). "
                         "This is the SAME artifact MSystem runs in prod "
                         "(ops.migration_scripts → agent_bootstrap). Preferred over --template-db.")
    ap.add_argument("--pg-bin", default="/opt/homebrew/opt/postgresql@17/bin")
    ap.add_argument("--db-host", default="127.0.0.1")
    ap.add_argument("--db-port", type=int, default=5432)
    ap.add_argument("--db-su", default="postgres")
    ap.add_argument("--db-su-pass", default=os.environ.get("PGPASSWORD", "postgres"))
    args = ap.parse_args()

    org = args.org or args.tenant_key
    domain = args.domain or f"{args.tenant_key}.{args.base_domain}"
    msys_url = os.environ["AG_MSYSTEM_URL"]

    with psycopg.connect(msys_url, autocommit=False) as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM tenant_mgmt.apps WHERE app_key='agent'")
        row = cur.fetchone()
        if not row:
            sys.exit("ERROR: 'agent' app not registered (run MSYS_V001).")
        app_id = row[0]

        cur.execute("SELECT tenant_id FROM tenant_mgmt.tenants WHERE tenant_key=%s", (args.tenant_key,))
        existing = cur.fetchone()
        if existing:
            tenant_id = str(existing[0])
            print(f"tenant_key '{args.tenant_key}' already exists → {tenant_id} (idempotent: reusing)")
        else:
            cur.execute(
                """
                INSERT INTO tenant_mgmt.tenants
                    (tenant_key, display_name, owner_account_id, primary_instance_id,
                     provision_state, is_active, primary_domain)
                VALUES (%s, %s, gen_random_uuid(), %s, 'active', TRUE, %s)
                RETURNING tenant_id::text
                """,
                (args.tenant_key, org, args.instance_id, domain),
            )
            tenant_id = cur.fetchone()[0]
            print(f"created tenant {tenant_id} (key={args.tenant_key})")

            cur.execute(
                """
                INSERT INTO tenant_mgmt.tenant_profile (tenant_id, org_name, theme_color, timezone)
                VALUES (%s, %s, '#1976d2', 'Asia/Kolkata')
                ON CONFLICT DO NOTHING
                """, (tenant_id, org))

            cur.execute(
                """
                INSERT INTO tenant_mgmt.tenant_domains (tenant_id, domain, is_primary, verified)
                VALUES (%s, %s, TRUE, TRUE) ON CONFLICT DO NOTHING
                """, (tenant_id, domain))

            cur.execute(
                """
                INSERT INTO tenant_mgmt.subscriptions
                    (tenant_id, app_id, billing_cycle_id, status, current_period_start, current_period_end, auto_renew)
                SELECT %s, %s, bc.id, 'active', NOW(), NOW() + INTERVAL '365 days', TRUE
                FROM tenant_mgmt.app_billing_cycles bc
                WHERE bc.app_id=%s AND bc.cycle_key='monthly'
                  AND NOT EXISTS (SELECT 1 FROM tenant_mgmt.subscriptions s
                                  WHERE s.tenant_id=%s AND s.app_id=%s)
                """, (tenant_id, app_id, app_id, tenant_id, app_id))

        sfx = suffix8(tenant_id)
        db_name = f"agent_tenant_{sfx}"
        db_user = f"agent_app_{sfx}"

        cur.execute(
            """
            INSERT INTO tenant_mgmt.app_databases
                (tenant_id, app_id, instance_id, db_name, db_endpoint, db_port, db_user, db_password, connection_is_secret)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, FALSE)
            ON CONFLICT (tenant_id, app_id) DO UPDATE SET db_name=EXCLUDED.db_name
            """,
            (tenant_id, app_id, args.instance_id, db_name, args.db_host, args.db_port,
             args.db_su, args.db_su_pass),
        )
        conn.commit()
        print(f"registered app_databases → {db_name} (user={db_user} [dev:{args.db_su}])")

    env = dict(os.environ, PGPASSWORD=args.db_su_pass)
    psql = os.path.join(args.pg_bin, "psql")
    pg_dump = os.path.join(args.pg_bin, "pg_dump")
    base = [psql, "-h", args.db_host, "-p", str(args.db_port), "-U", args.db_su]

    exists = subprocess.run(base + ["-d", "postgres", "-tAc",
                            f"SELECT 1 FROM pg_database WHERE datname='{db_name}'"],
                            capture_output=True, text=True, env=env).stdout.strip()
    if exists != "1":
        subprocess.run(base + ["-d", "postgres", "-c", f'CREATE DATABASE {db_name};'],
                       check=True, env=env)
        print(f"created database {db_name}")

        bootstrap_sql = os.path.abspath(args.bootstrap_sql) if args.bootstrap_sql else None
        if bootstrap_sql and os.path.exists(bootstrap_sql):
            with open(bootstrap_sql, "r", encoding="utf-8") as f:
                sql = f.read()
            subprocess.run(base + ["-d", db_name, "-q", "-v", "ON_ERROR_STOP=1"],
                           input=sql, text=True, env=env, check=True)
            print(f"applied bootstrap {os.path.basename(bootstrap_sql)} (schema V034 + platform seed)")
        else:
            print(f"WARNING: bootstrap SQL not found ({bootstrap_sql}); "
                  f"falling back to schema clone from {args.template_db}")
            dump = subprocess.run([pg_dump, "-h", args.db_host, "-p", str(args.db_port),
                                   "-U", args.db_su, "--schema-only", args.template_db],
                                  capture_output=True, text=True, env=env)
            subprocess.run(base + ["-d", db_name, "-q"], input=dump.stdout, text=True, env=env)
            print(f"cloned ag_* schema from {args.template_db}")

            seed_tables = ["ag_auth.roles", "ag_system.applications", "ag_system.modules"]
            seed = subprocess.run([pg_dump, "-h", args.db_host, "-p", str(args.db_port),
                                   "-U", args.db_su, "--data-only",
                                   *sum([["-t", t] for t in seed_tables], []), args.template_db],
                                  capture_output=True, text=True, env=env)
            subprocess.run(base + ["-d", db_name, "-q"], input=seed.stdout, text=True, env=env)
            print(f"seeded reference data: {', '.join(seed_tables)}")
    else:
        print(f"database {db_name} already exists")

    seed = f"""
    INSERT INTO ag_system.service_connections (service_key, config_json, is_enabled, description) VALUES
     ('elastic', '{{"url":"http://localhost:9200","index":"brain_{sfx}"}}'::jsonb, TRUE, 'lexical index'),
     ('neo4j',   '{{"uri":"bolt://localhost:7687","db":"neo4j","user":"neo4j","scheme":"shared"}}'::jsonb, TRUE, 'shared graph + tenant_id'),
     ('redis',   '{{"url":"redis://localhost:6379/0","namespace":"ag:{sfx}"}}'::jsonb, TRUE, 'redis namespace'),
     ('embeddings','{{"provider":"ollama","model":"nomic-embed-text","endpoint":"http://localhost:11434"}}'::jsonb, TRUE, 'shared embeddings')
    ON CONFLICT (service_key) DO UPDATE SET config_json=EXCLUDED.config_json;
    """
    subprocess.run(base + ["-d", db_name, "-q"], input=seed, text=True, env=env, check=True)
    print(f"seeded service_connections (index brain_{sfx}, ns ag:{sfx})")
    print(f"\nDONE → tenant={args.tenant_key} id={tenant_id} suffix8={sfx} db={db_name} domain={domain}")


if __name__ == "__main__":
    main()
