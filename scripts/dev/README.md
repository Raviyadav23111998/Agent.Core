# Dev/CI Scaffold — NOT the production provisioning path

These scripts stand up agent tenants **locally** so the agent runtime can be
developed and tested end-to-end without running the C# MSystem Orchestrator.

**Production provisioning is owned by MSystem** (`MSystems.API / MSystem.Orchestrator`):
DB + role + grants + ag_* bootstrap + migrations + `app_databases` are created by
`CreateDatabaseHandler` from the `ops.provision_queue`; offboarding by
`DropAgentDatabaseHandler`. See `MULTITENANT_PROVISIONING_PRODUCTION_PLAN.md`.

Do **not** run these against production. The agent process holds no DB-creation
privileges in prod.

- `provision_agent_tenant.py`   — dev stand-in for MSystem CreateDatabaseHandler
- `deprovision_agent_tenant.py` — dev stand-in for MSystem DropAgentDatabaseHandler
- `migrate_all_tenants.py`       — dev stand-in for the MSystem MigrationWorker
