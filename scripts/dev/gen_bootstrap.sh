#!/usr/bin/env bash
# Generate the agent_bootstrap.sql from the golden tenant prod_agent_tenant_2a5fe12b.
# Schema (all ag_* + public extension, minus *_bak_v029) + curated platform-default
# seed (NO credentials, NO tenant-private data, NO secret configs).
set -euo pipefail

SRC=prod_agent_tenant_2a5fe12b
OUT=/tmp/agent_bootstrap_new.sql
PGD="sudo -n -u postgres pg_dump"
PSQL="sudo -n -u postgres psql"

# Seed tables (platform defaults only). FK order handled by pg_dump.
SEED_TABLES=(
  ag_auth.roles
  ag_auth.role_permissions
  ag_system.applications
  ag_system.modules
  ag_system.code_types
  ag_system.codes
  ag_system.agent_personas
  ag_cred.providers
  ag_system.credential_providers
)

{
  echo "--"
  echo "-- Storeglance Agent tenant BOOTSTRAP (no creds, no tenant data)"
  echo "-- Source of truth: golden tenant ${SRC}"
  echo "-- Contents: full ag_* schema (V034) + curated platform-default seed."
  echo "-- Excludes: all credentials/secrets, all tenant-private runtime data,"
  echo "--           *_bak_v029 backup tables, secret/env-specific configs."
  echo "-- Regenerate with scripts/dev/gen_bootstrap.sh"
  echo "--"
  echo ""

  echo "-- ============================ SCHEMA ============================"
  $PGD --schema-only --no-owner --no-privileges \
       -T 'ag_system.*bak_v029' "$SRC"

  echo ""
  echo "-- ===================== PLATFORM SEED DATA ======================"
  TARGS=()
  for t in "${SEED_TABLES[@]}"; do TARGS+=(-t "$t"); done
  $PGD --data-only --no-owner --no-privileges "${TARGS[@]}" "$SRC"

  echo ""
  echo "-- ag_system.configs — non-secret, non-env-endpoint defaults only"
  COLS=$($PSQL -d "$SRC" -tAc "SELECT string_agg(column_name, ',' ORDER BY ordinal_position) FROM information_schema.columns WHERE table_schema='ag_system' AND table_name='configs'")
  echo "COPY ag_system.configs (${COLS}) FROM stdin;"
  $PSQL -d "$SRC" -tAc "COPY (SELECT ${COLS} FROM ag_system.configs WHERE is_secret = false AND config_key NOT IN ('llm.api_key','llm.base_url','llm.embedding_base_url') ORDER BY config_key) TO STDOUT"
  echo "\\."
  echo ""
} > "$OUT"

echo "WROTE $OUT"
wc -l "$OUT"
echo "--- schemas created ---"; grep -oE 'CREATE SCHEMA [a-z_]+' "$OUT" | sort -u | tr '\n' ' '; echo
echo "--- has ag_api / ag_cred ? ---"; grep -cE 'CREATE SCHEMA ag_api|CREATE SCHEMA ag_cred' "$OUT"
echo "--- seed COPY blocks ---"; grep -oE '^COPY [a-z_]+\.[a-z_]+ ' "$OUT" | sort | uniq -c
echo "--- bak tables leaked? (want 0) ---"; grep -c 'bak_v029' "$OUT" || true
echo "--- llm.api_key value leaked? (want 0) ---"; grep -c 'llm.api_key' "$OUT" || true
