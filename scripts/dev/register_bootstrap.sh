#!/usr/bin/env bash
# Register the new golden bootstrap into ops.migration_scripts on msystems_prod (UPDATE)
# and msystems_uat (INSERT). Content loaded server-side via pg_read_file (no escaping).
set -uo pipefail
P="sudo -n -u postgres psql"
SRC=/tmp/agent_bootstrap_new.sql
DESC='Agent tenant bootstrap regenerated from golden tenant prod_agent_tenant_2a5fe12b (schema V034 incl ag_api + ag_cred). Curated platform-default seed only; ALL credentials/secrets, tenant-private runtime data, *_bak_v029 tables, and is_secret/env configs EXCLUDED.'

chmod 644 "$SRC"

echo "=== 0. backup current prod bootstrap content ==="
$P -At -d msystems_prod -c "SELECT sql_content FROM ops.migration_scripts WHERE app_id='agent' AND version='V004_bootstrap_no_creds'" > /tmp/agent_bootstrap.prev.sql
echo "backed up $(wc -l < /tmp/agent_bootstrap.prev.sql) lines -> /tmp/agent_bootstrap.prev.sql"

echo "=== 1. PROD: update in place ==="
$P -v ON_ERROR_STOP=1 -d msystems_prod <<SQL
UPDATE ops.migration_scripts
SET sql_content = pg_read_file('$SRC'),
    checksum    = md5(pg_read_file('$SRC')),
    description = \$\$${DESC}\$\$,
    is_published = TRUE,
    is_active    = TRUE,
    updated_at   = now()
WHERE app_id='agent' AND version='V004_bootstrap_no_creds';
SQL

echo "=== 2. UAT: insert (mirror prod metadata) ==="
$P -v ON_ERROR_STOP=1 -d msystems_uat <<SQL
INSERT INTO ops.migration_scripts
    (app_id, script_name, version, script_type, description,
     sql_content, checksum, is_published, is_active, created_by)
SELECT 'agent', 'agent_bootstrap', 'V004_bootstrap_no_creds', 'bootstrap',
       \$\$${DESC}\$\$,
       pg_read_file('$SRC'), md5(pg_read_file('$SRC')), TRUE, TRUE, 'platform-provisioning'
WHERE NOT EXISTS (
    SELECT 1 FROM ops.migration_scripts WHERE app_id='agent' AND version='V004_bootstrap_no_creds');
-- if a row already exists, update it instead
UPDATE ops.migration_scripts
SET sql_content = pg_read_file('$SRC'), checksum = md5(pg_read_file('$SRC')),
    description = \$\$${DESC}\$\$, is_published=TRUE, is_active=TRUE, updated_at=now()
WHERE app_id='agent' AND version='V004_bootstrap_no_creds';
SQL

echo "=== 3. verify both masters ==="
for M in msystems_prod msystems_uat; do
  echo "--- $M ---"
  $P -d $M -tAc "SELECT version, script_type, is_published, length(sql_content) AS len, checksum, (checksum = md5(sql_content)) AS checksum_ok, (position('ag_api' in sql_content) > 0) AS has_ag_api, (position('ag_cred' in sql_content) > 0) AS has_ag_cred FROM ops.migration_scripts WHERE app_id='agent'"
done
