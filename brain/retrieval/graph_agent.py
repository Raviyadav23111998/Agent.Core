from __future__ import annotations

"""
Agent Brain — Graph Agent
============================
Retrieval agent for the Knowledge Graph (Neo4j).

Flow:
  1. User question → LLM generates a Cypher query
  2. Cypher executes against Neo4j → raw graph results
  3. LLM summarizes graph results into human-readable text

This is the agent for RELATIONAL queries:
  "Who emailed whom about the auth bug?"
  "What technologies does Project Phoenix depend on?"
  "Which policies reference the authentication module?"
"""

import logging

from brain.storage.graph_store import GraphStore
from brain.retrieval.base import BaseAgent, AgentResult

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "the", "and", "for", "what", "who", "whom", "whose", "which", "where", "when",
    "how", "why", "did", "does", "are", "was", "were", "with", "about", "from",
    "into", "that", "this", "they", "them", "their", "all", "any", "can", "could",
    "would", "should", "tell", "show", "find", "list", "give", "get", "between",
}


CYPHER_GENERATION_PROMPT = """Convert a natural language question to a Neo4j Cypher query.

SCHEMA:
- Nodes: :Entity {name, entity_type, properties, source_document_id, confidence}
- entity_type: person, organization, project, technology, file, module, policy, concept, event, location
- Type labels: :Person, :Organization, :Project, :Technology, :File, :Module, :Policy, :Concept, :Event, :Location
- Edges: AUTHORED, EMAILED, MENTIONED_IN, DEPENDS_ON, PART_OF, WORKS_ON, REPORTED, APPROVED, MODIFIED, REFERENCES, RELATED_TO

RULES: Use toLower() for name matching. RETURN name+entity_type, not raw nodes. LIMIT 25. Return ONLY Cypher, no markdown."""


GRAPH_SUMMARY_PROMPT = """You are summarizing Knowledge Graph query results for a user.

The user asked: "{question}"

The Cypher query returned these results:
{results}

RULES:
1. Summarize the findings in clear, natural language.
2. Mention specific entity names and relationships found.
3. If no results were found, say so clearly and suggest what might help.
4. Be concise but thorough. Cite entity names exactly as found.
5. Do NOT make up information not present in the results."""


class GraphAgent(BaseAgent):
    """
    Retrieval agent that queries the Knowledge Graph.
    Uses LLM to generate Cypher, executes it, then summarizes results.
    """

    AGENT_NAME = "graph_agent"

    def __init__(self, graph_store: GraphStore, llm_client):
        self.graph = graph_store
        self.llm = llm_client

    async def query(self, question: str, **kwargs) -> AgentResult:
        """
        Query the Knowledge Graph with a natural language question.

        Args:
            question: Natural language question
            **kwargs:
                direct_cypher: Optional pre-written Cypher query (skips LLM generation)

        Returns:
            AgentResult with graph findings
        """
        try:
            if not self.graph._driver:
                return AgentResult(
                    agent_name=self.AGENT_NAME,
                    content="Knowledge Graph is currently unavailable (Neo4j not connected). Try again after running `docker-compose up -d`.",
                    confidence=0.0,
                )

            if getattr(self.graph, "multi_tenant", False):
                return await self._safe_tenant_query(question)

            cypher = kwargs.get("direct_cypher")
            if not cypher:
                cypher = await self._generate_cypher(question)

            if not cypher:
                return self._error_result("Failed to generate Cypher query")

            logger.info("graph_agent.cypher_generated: cypher=%s", cypher[:100])

            raw_results = self.graph.run_cypher(cypher)

            if not raw_results:
                return AgentResult(
                    agent_name=self.AGENT_NAME,
                    content=f"No results found in the Knowledge Graph for: '{question}'",
                    sources=[{"cypher": cypher, "result_count": 0}],
                    confidence=0.1,
                    raw_data={"cypher": cypher, "results": []},
                )

            summary = self._format_results(question, raw_results)

            sources = [
                {"cypher": cypher, "result_count": len(raw_results)},
            ]
            for row in raw_results[:5]:
                sources.append({"graph_row": {k: str(v) for k, v in row.items()}})

            return AgentResult(
                agent_name=self.AGENT_NAME,
                content=summary,
                sources=sources,
                confidence=min(0.3 + (len(raw_results) * 0.1), 0.95),
                raw_data={"cypher": cypher, "results": raw_results},
            )

        except Exception as e:
            logger.error("graph_agent.query_failed: question=%s error=%s", question[:50], str(e))
            return self._error_result(str(e))

    async def _safe_tenant_query(self, question: str) -> AgentResult:
        """Mode C graph retrieval WITHOUT raw Cypher — tenant-scoped templates only.

        Finds entities matching salient question terms (every query injects
        tenant_id inside GraphStore) and explores the top hit's neighborhood.
        Degraded vs. free-form Cypher, but structurally leak-proof.
        """
        import re
        terms = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-]{2,}", question or "")
                 if w.lower() not in _STOPWORDS]
        found: list[dict] = []
        seen = set()
        for term in terms[:6]:
            for ent in self.graph.find_entity(term):
                key = (ent.get("name"), ent.get("type"))
                if key not in seen:
                    seen.add(key)
                    found.append(ent)
        if not found:
            return AgentResult(
                agent_name=self.AGENT_NAME,
                content=f"No related entities found in the Knowledge Graph for: '{question}'",
                confidence=0.1,
            )
        neighborhood = []
        try:
            neighborhood = self.graph.get_entity_neighborhood(found[0]["name"], depth=2)
        except Exception:
            pass
        lines = [f"Knowledge Graph (entities) for: '{question}'"]
        for i, e in enumerate(found[:10], 1):
            lines.append(f"  {i}. {e.get('name')} [{e.get('type')}]")
        for row in neighborhood[:10]:
            lines.append(f"  ({row.get('from_name','?')}) —[{row.get('relation','?')}]→ ({row.get('to_name','?')})")
        sources = [{"graph_entity": {"name": e.get("name"), "type": e.get("type")}} for e in found[:5]]
        return AgentResult(
            agent_name=self.AGENT_NAME,
            content="\n".join(lines),
            sources=sources,
            confidence=min(0.3 + len(found) * 0.1, 0.9),
            raw_data={"entities": found, "neighborhood": neighborhood},
        )

    async def _generate_cypher(self, question: str) -> str:
        """Use the LLM to convert natural language to Cypher."""
        response = await self.llm.chat(
            messages=[
                {"role": "system", "content": CYPHER_GENERATION_PROMPT},
                {"role": "user", "content": question},
            ],
            purpose="routing",
            temperature=0.0,
            max_tokens=256,
        )

        cypher = response.strip()
        if cypher.startswith("```"):
            lines = cypher.split("\n")
            cypher = "\n".join(lines[1:-1])

        return cypher.strip()

    @staticmethod
    def _format_results(question: str, results: list[dict]) -> str:
        """Format raw graph query results as structured text (no LLM call needed)."""
        lines = [f"Knowledge Graph results for: '{question}'"]
        for i, row in enumerate(results[:10], 1):
            row_str = ", ".join(f"{k}: {v}" for k, v in row.items())
            lines.append(f"  {i}. {row_str}")
        if len(results) > 10:
            lines.append(f"  ...and {len(results) - 10} more results")
        return "\n".join(lines)

    async def explore_entity(self, entity_name: str, depth: int = 2) -> AgentResult:
        """
        Explore the neighborhood of an entity in the graph.
        Useful for discovery: "Tell me everything about John Smith."
        """
        try:
            results = self.graph.get_entity_neighborhood(entity_name, depth=depth)

            if not results:
                return AgentResult(
                    agent_name=self.AGENT_NAME,
                    content=f"Entity '{entity_name}' not found in the Knowledge Graph.",
                    confidence=0.1,
                )

            connections = []
            for row in results:
                connections.append(
                    f"  ({row.get('from_name', '?')}) —[{row.get('relation', '?')}]→ ({row.get('to_name', '?')})"
                )

            content = f"Knowledge Graph connections for '{entity_name}':\n" + "\n".join(connections)

            return AgentResult(
                agent_name=self.AGENT_NAME,
                content=content,
                sources=[{"entity": entity_name, "depth": depth, "connections": len(results)}],
                confidence=min(0.3 + (len(results) * 0.05), 0.9),
                raw_data={"entity": entity_name, "connections": results},
            )

        except Exception as e:
            return self._error_result(str(e))
