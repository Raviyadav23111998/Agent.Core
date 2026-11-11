"""
Agent — Agent Registry (PostgreSQL-backed)
=======================================================
Persistent storage for dynamic agent definitions.
Migrated from JSON files to ag_agents.definitions table.

An agent definition contains:
- name, description, system_prompt
- tools (whitelist of allowed MCP tools)
- schedule (cron/interval)
- state (enabled, last_run, run_count)

Storage: ag_agents.definitions (PostgreSQL)
Logs:    ag_agents.runs (PostgreSQL)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List

from core.tenant import TENANT_SCOPE

logger = logging.getLogger(__name__)


@dataclass
class AgentDefinition:
    """A dynamic agent configuration — everything needed to spawn it."""

    name: str
    description: str
    system_prompt: str = ""
    tools: list = field(default_factory=list)
    model: str = ""
    schedule: str = ""
    enabled: bool = True
    workspace_path: str = ""
    deliver: str = "local"
    owner_id: str = TENANT_SCOPE
    visibility: str = "global"
    created_at: str = ""
    last_run_at: str = ""
    last_status: str = ""
    last_error: str = ""
    run_count: int = 0
    agent_id: int = 0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


class AgentRegistry:
    """Persistent storage for agent definitions — PostgreSQL-backed.

    Usage:
        registry = AgentRegistry(db)
        registry.save(AgentDefinition(name="bot", description="..."))
        agent = registry.load("bot")
        all_agents = registry.list_all()
    """

    def __init__(self, db):
        """
        Args:
            db: TenantDB instance
        """
        self._db = db

    @staticmethod
    def _safe_name(name: str) -> str:
        """Sanitize agent name."""
        return name.replace("/", "_").replace(" ", "_").replace("..", "_").lower()

    def _row_to_definition(self, row: dict) -> AgentDefinition:
        """Convert a DB row to an AgentDefinition."""
        tools = row.get("tools", [])
        if isinstance(tools, str):
            tools = json.loads(tools)
        return AgentDefinition(
            name=row["name"],
            description=row.get("description", ""),
            system_prompt=row.get("system_prompt", ""),
            tools=tools,
            model=row.get("model", ""),
            schedule=row.get("schedule", ""),
            enabled=row.get("enabled", True),
            workspace_path=row.get("workspace_path", ""),
            deliver=row.get("deliver", "local"),
            owner_id=row.get("owner_id", TENANT_SCOPE),
            visibility=row.get("visibility", "global"),
            created_at=str(row.get("created_at", "")),
            last_run_at=str(row["last_run_at"]) if row.get("last_run_at") else "",
            last_status=row.get("last_status", ""),
            last_error=row.get("last_error", ""),
            run_count=row.get("run_count", 0),
            agent_id=row.get("agent_id", 0),
        )


    def save(self, agent: AgentDefinition):
        """Save or update an agent definition (upsert by name)."""
        tools_json = json.dumps(agent.tools)
        self._db.execute("""
            INSERT INTO ag_agents.definitions (
                name, description, system_prompt, tools, model, schedule,
                enabled, workspace_path, deliver, owner_id, visibility
            ) VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (name) DO UPDATE SET
                description = EXCLUDED.description,
                system_prompt = EXCLUDED.system_prompt,
                tools = EXCLUDED.tools,
                model = EXCLUDED.model,
                schedule = EXCLUDED.schedule,
                enabled = EXCLUDED.enabled,
                workspace_path = EXCLUDED.workspace_path,
                deliver = EXCLUDED.deliver,
                owner_id = EXCLUDED.owner_id,
                visibility = EXCLUDED.visibility,
                updated_at = NOW()
        """, (
            agent.name, agent.description, agent.system_prompt,
            tools_json, agent.model, agent.schedule,
            agent.enabled, agent.workspace_path, agent.deliver,
            agent.owner_id, agent.visibility,
        ))
        logger.info("agent_registry.saved: name=%s tools=%d",
                    agent.name, len(agent.tools))

    def load(self, name: str) -> Optional[AgentDefinition]:
        """Load an agent definition by name."""
        row = self._db.fetch_one(
            "SELECT * FROM ag_agents.definitions WHERE name = %s", (name,)
        )
        if not row:
            return None
        return self._row_to_definition(row)

    def list_all(self) -> List[AgentDefinition]:
        """List all registered agents (unfiltered)."""
        rows = self._db.fetch_all(
            "SELECT * FROM ag_agents.definitions ORDER BY name"
        )
        return [self._row_to_definition(r) for r in rows]

    def list_for_user(self, user_id: str) -> List[AgentDefinition]:
        """List agents visible to a specific user."""
        rows = self._db.fetch_all("""
            SELECT * FROM ag_agents.definitions
            WHERE visibility = 'global'
               OR owner_id = %s
               OR owner_id = %s
            ORDER BY name
        """, (user_id, TENANT_SCOPE))
        return [self._row_to_definition(r) for r in rows]

    def delete(self, name: str) -> bool:
        """Delete an agent definition."""
        count = self._db.execute(
            "DELETE FROM ag_agents.definitions WHERE name = %s", (name,)
        )
        if count > 0:
            logger.info("agent_registry.deleted: name=%s", name)
            return True
        return False

    def exists(self, name: str) -> bool:
        """Check if an agent exists."""
        return self._db.exists(
            "SELECT 1 FROM ag_agents.definitions WHERE name = %s", (name,)
        )

    def update_run_state(self, name: str, status: str, error: str = ""):
        """Update an agent's last run state."""
        self._db.execute("""
            UPDATE ag_agents.definitions SET
                last_run_at = NOW(),
                last_status = %s,
                last_error = %s,
                run_count = run_count + 1,
                updated_at = NOW()
            WHERE name = %s
        """, (status, error, name))


    def save_run_log(self, name: str, output: str):
        """Save agent run output to ag_agents.runs table."""
        agent = self.load(name)
        if not agent or not agent.agent_id:
            return
        self._db.execute("""
            INSERT INTO ag_agents.runs (agent_id, status, output_text, completed_at)
            VALUES (%s, %s, %s, NOW())
        """, (agent.agent_id, agent.last_status or "ok", output))

    def get_logs(self, name: str, last_n: int = 5) -> List[str]:
        """Get recent run logs for an agent."""
        agent = self.load(name)
        if not agent or not agent.agent_id:
            return []
        rows = self._db.fetch_all("""
            SELECT output_text, started_at FROM ag_agents.runs
            WHERE agent_id = %s
            ORDER BY started_at DESC
            LIMIT %s
        """, (agent.agent_id, last_n))
        results = []
        for row in rows:
            ts = str(row.get("started_at", "?"))[:19]
            content = row.get("output_text", "")
            if len(content) > 1000:
                content = content[:1000] + f"\n... (truncated, full: {len(content)} chars)"
            results.append(f"[{ts}]\n{content}")
        return results
