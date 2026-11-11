#!/usr/bin/env bash
set -uo pipefail
SCRATCH=agent_bootstrap_verify_tmp
P="sudo -n -u postgres psql"
$P -d postgres -c "DROP DATABASE IF EXISTS $SCRATCH" >/dev/null 2>&1
$P -d postgres -c "CREATE DATABASE $SCRATCH" >/dev/null 2>&1

echo "=== applying bootstrap to fresh DB (errors only) ==="
$P -v ON_ERROR_STOP=0 -d $SCRATCH -f /tmp/agent_bootstrap_new.sql 2>&1 | grep -iE "ERROR|FATAL" | head -30
echo "(no ERROR lines above = clean apply)"

echo "=== table count (expect 66 = golden 68 - 2 bak) ==="
$P -d $SCRATCH -tAc "SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE c.relkind='r' AND n.nspname LIKE 'ag_%'"

echo "=== seed row counts (should be populated) ==="
$P -d $SCRATCH -tAc "SELECT 'roles', count(*) FROM ag_auth.roles
UNION ALL SELECT 'role_permissions', count(*) FROM ag_auth.role_permissions
UNION ALL SELECT 'applications', count(*) FROM ag_system.applications
UNION ALL SELECT 'modules', count(*) FROM ag_system.modules
UNION ALL SELECT 'code_types', count(*) FROM ag_system.code_types
UNION ALL SELECT 'codes', count(*) FROM ag_system.codes
UNION ALL SELECT 'agent_personas', count(*) FROM ag_system.agent_personas
UNION ALL SELECT 'cred.providers', count(*) FROM ag_cred.providers
UNION ALL SELECT 'configs', count(*) FROM ag_system.configs ORDER BY 1"

echo "=== privacy gate: these MUST all be 0 ==="
$P -d $SCRATCH -tAc "SELECT 'users', count(*) FROM ag_auth.users
UNION ALL SELECT 'connections', count(*) FROM ag_cred.connections
UNION ALL SELECT 'conversations', count(*) FROM ag_system.conversations
UNION ALL SELECT 'messages', count(*) FROM ag_system.messages
UNION ALL SELECT 'documents', count(*) FROM ag_brain.documents
UNION ALL SELECT 'memories', count(*) FROM ag_memory.memories
UNION ALL SELECT 'secret_configs', count(*) FROM ag_system.configs WHERE is_secret
UNION ALL SELECT 'api_integrations', count(*) FROM ag_api.integrations
UNION ALL SELECT 'bots', count(*) FROM ag_bot.bots ORDER BY 1"

$P -d postgres -c "DROP DATABASE $SCRATCH" >/dev/null 2>&1
echo "scratch dropped"
