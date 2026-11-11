"""
Agent — Memory MCP Tools
======================================
7 MCP tools exposing the persistent memory system to the agent.

Tools:
  memory_store         — Store a fact, preference, or observation
  memory_recall        — FTS5-ranked search across all memories
  memory_user_context  — Get all memories for the current user
  memory_entity_link   — Create/query entity relationships
  memory_forget        — Delete a specific memory by key
  memory_stats         — System diagnostics
  memory_company_store — Store a company-wide fact
"""

import json
import time
import logging
from .native_mcp import NativeMCP
from core.tenant import TENANT_SCOPE

logger = logging.getLogger(__name__)


class MemoryMCPTools:
    """MCP tool wrappers for the persistent memory system."""

    def __init__(self, memory_store, user_id_resolver=None):
        """
        Args:
            memory_store: MemoryStore instance
            user_id_resolver: Optional callable that returns current user_id.
                              If None, uses '__cli__'.
        """
        self._store = memory_store
        self._resolve_user = user_id_resolver or (lambda: "__cli__")
        self._user_context_resolver = None

    def set_user_context_resolver(self, resolver):
        """Set resolver that returns UserContext (for visible_scopes)."""
        self._user_context_resolver = resolver

    def _resolve_visible_scopes(self):
        """Get visible scopes from UserContext if available."""
        if self._user_context_resolver:
            ctx = self._user_context_resolver()
            return ctx.visible_scopes if ctx else None
        return None

    def _resolve_app_id(self):
        """The application this turn belongs to, from the ambient UserContext.

        The same shape skill_service._ambient_scope uses, and deliberately so —
        two ways of answering "which app is this?" is how one store ends up
        scoped and the other does not.

        Read from the contextvar rather than the injected resolver because the
        contextvar is what every entry point already sets (chat, service chat,
        embed, cron, spawned agents), while the resolver was only ever wired
        for the CLI.
        """
        try:
            from core.user_context import current_user_context
            ctx = current_user_context.get()
            if ctx is None and self._user_context_resolver:
                ctx = self._user_context_resolver()
            app_id = getattr(ctx, "app_id", None) if ctx else None
            return int(app_id) if app_id is not None else None
        except (TypeError, ValueError, ImportError):
            return None

    def register_tools(self, mcp_server: NativeMCP):
        """Register all 7 memory MCP tools."""

        @mcp_server.tool()
        def memory_store(key: str, value: str, category: str = None, tags: str = None) -> str:
            """Store a fact, preference, or observation in persistent memory.
            
            The memory persists across sessions and can be recalled later.
            Use clear, descriptive keys like 'pref.dark_mode', 'fact.api_uses_jwt'.
            
            Args:
                key: Unique identifier (e.g. 'pref.dark_mode', 'fact.api_auth_method')
                value: The content to remember. Should be clear and self-contained.
                category: Optional: preference, fact, pattern, correction (auto-detected if omitted)
                tags: Optional comma-separated tags for grouping (e.g. 'ui,theme')
            """
            tag_list = [t.strip() for t in tags.split(",")] if tags else None
            user_id = self._resolve_user()
            last_err = None
            for attempt in range(3):
                try:
                    self._store.remember(
                        key=key, value=value, user_id=user_id,
                        category=category, tags=tag_list,
                        source="memory_store_tool",
                        app_id=self._resolve_app_id(),
                    )
                    stats = self._store.get_stats()
                    return f"Stored: [{category or 'auto'}] {key} = {value[:100]}. Total memories: {stats['total_memories']}"
                except Exception as e:
                    last_err = e
                    if "locked" in str(e).lower() and attempt < 2:
                        logger.warning("memory_store: DB locked, retry %d/3", attempt + 1)
                        time.sleep(1 * (attempt + 1))
                    else:
                        raise
            return f"✗ Memory store failed after retries: {last_err}"

        @mcp_server.tool()
        def memory_recall(query: str, category: str = None, limit: str = "10") -> str:
            """Search persistent memory using full-text search.
            
            Returns ranked results matching the query. Searches across all
            user memories and company-wide knowledge.
            
            Args:
                query: What to search for (keywords or natural language)
                category: Optional filter: preference, fact, pattern, correction
                limit: Max results to return (default 10)
            """
            user_id = self._resolve_user()
            scopes = self._resolve_visible_scopes()
            results = self._store.recall(
                query=query, user_id=user_id,
                category=category, limit=int(limit),
                visible_scopes=scopes,
                app_id=self._resolve_app_id(),
            )

            if not results:
                return f"No memories matching '{query}'."

            lines = []
            for r in results:
                cat = r.get("category", "?")
                key = r.get("key", "")
                val = r.get("value", "")[:200]
                conf = r.get("confidence", 0)
                lines.append(f"[{cat}|{conf:.1f}] {key}: {val}")

            return "\n".join(lines)

        @mcp_server.tool()
        def memory_user_context() -> str:
            """Get all memories for the current user as context.
            
            Returns a formatted list of all stored memories (preferences, facts,
            patterns, corrections) for the active user plus company-wide knowledge.
            Useful for understanding user context at session start.
            """
            user_id = self._resolve_user()
            scopes = self._resolve_visible_scopes()
            context = self._store.get_user_context(user_id=user_id,
                                                    visible_scopes=scopes,
                                                    app_id=self._resolve_app_id())
            if not context:
                return "No memories stored for this user."
            return context

        @mcp_server.tool()
        def memory_entity_link(entity: str, entity_type: str,
                               related_to: str, related_type: str,
                               relation: str = "RELATED_TO") -> str:
            """Create a relationship between two entities in the knowledge graph.
            
            This builds a lightweight entity graph that can be queried for
            connections, dependencies, and organizational structure.
            
            Args:
                entity: Name of the source entity (e.g. 'Auth Service')
                entity_type: Type of source (e.g. 'service', 'person', 'project')
                related_to: Name of the target entity (e.g. 'JWT Library')
                related_type: Type of target (e.g. 'library', 'person', 'project')
                relation: Relationship type (e.g. 'DEPENDS_ON', 'WORKS_ON', 'AUTHORED')
            """
            user_id = self._resolve_user()
            success = self._store.add_relation(
                source_name=entity, source_type=entity_type,
                target_name=related_to, target_type=related_type,
                relation_type=relation, user_id=user_id,
            )
            if success:
                return f"Linked: ({entity}:{entity_type}) —[{relation}]→ ({related_to}:{related_type})"
            return f"Failed to create link."

        @mcp_server.tool()
        def memory_forget(key: str) -> str:
            """Delete a specific memory by its key.
            
            Use this to remove outdated or incorrect memories.
            
            Args:
                key: The exact key of the memory to delete
            """
            user_id = self._resolve_user()
            deleted = self._store.forget(key=key, user_id=user_id)
            if deleted:
                return f"Forgot: {key}"
            return f"No memory found with key '{key}'."

        @mcp_server.tool()
        def memory_stats() -> str:
            """Get memory system diagnostics: counts by layer, category, disk usage.
            
            Returns a summary of the memory system's current state.
            """
            stats = self._store.get_stats()
            lines = [
                f"Total memories: {stats['total_memories']}",
                f"Entities: {stats['entities']} | Relations: {stats['relations']}",
                f"DB size: {stats['db_size_human']}",
                f"By layer: {json.dumps(stats['by_layer'])}",
                f"By category: {json.dumps(stats['by_category'])}",
            ]
            return "\n".join(lines)

        @mcp_server.tool()
        def memory_company_store(key: str, value: str, category: str = None) -> str:
            """Store a company-wide fact that all users can access.
            
            Use for organizational knowledge: API standards, team structure,
            project conventions, deployment procedures, etc.
            
            Args:
                key: Unique key (e.g. 'company.api_standard', 'team.backend_lead')
                value: The company-wide fact to remember
                category: Optional: fact, pattern (default: fact)
            """
            # TENANT_SCOPE is the USER dimension — "everyone in this org" —
            # and it is orthogonal to the app. A company fact from one product
            # is still that product's; without app_id it would be readable by
            # every other application in the tenant, which is the leak this
            # whole change exists to close, just via a different door.
            self._store.remember(
                key=key, value=value,
                user_id=TENANT_SCOPE,
                category=category or "fact",
                layer="company",
                source="memory_company_store_tool",
                app_id=self._resolve_app_id(),
            )
            return f"Company memory stored: {key} = {value[:100]}"
