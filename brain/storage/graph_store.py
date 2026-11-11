from __future__ import annotations

"""
Agent Brain — Knowledge Graph Store (Neo4j)
=============================================
PILLAR 1: The Relationship Engine

This is where the LLM's extracted entities and relations live.
Neo4j stores nodes (Person, Project, Policy, etc.) and edges
(EMAILED, AUTHORED, DEPENDS_ON, etc.) forming a rich knowledge graph.

Why Neo4j over vectors?
- Vectors find "similar" text. Graphs find "connected" knowledge.
- "Who emailed whom about the authentication bug?" is a graph traversal,
  not a similarity search.
- Multi-hop reasoning: Person → authored → Policy → references → Module → has_bug

The LLM writes TO the graph (during extraction).
The Graph Agent reads FROM the graph (during retrieval via Cypher queries).
"""

import json
from typing import Optional
import logging
try:
    from neo4j import GraphDatabase, Driver, ManagedTransaction
except ImportError:
    GraphDatabase = None
    Driver = None
    ManagedTransaction = None

from brain.models import Entity, EntityType, GraphExtractionResult
from brain.exceptions import GraphConnectionError

logger = logging.getLogger(__name__)


class GraphStore:
    """
    Neo4j-backed Knowledge Graph.
    Provides methods to:
    - Store entities (nodes) and relations (edges)
    - Query the graph with Cypher
    - Get entity neighborhoods for context building
    """

    def __init__(self, uri: str, user: str, password: str,
                 database: str = "neo4j", tenant_id: Optional[str] = None):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database or "neo4j"
        self.tenant_id = tenant_id
        self._driver: Optional[Driver] = None

    @property
    def multi_tenant(self) -> bool:
        return self.tenant_id is not None


    def connect(self):
        """Establish connection to Neo4j."""
        try:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self._driver.verify_connectivity()
            self._init_constraints()
            logger.info("graph_store.connected: uri=%s", self.uri)
        except Exception as e:
            raise GraphConnectionError(f"Failed to connect to Neo4j at {self.uri}: {e}")

    def close(self):
        """Close the Neo4j driver."""
        if self._driver:
            self._driver.close()
            logger.info("graph_store.closed")

    @property
    def driver(self) -> Driver:
        if not self._driver:
            raise GraphConnectionError("Not connected. Call connect() first.")
        return self._driver

    def _session(self):
        """Session bound to the configured database (Neo4j Community → 'neo4j')."""
        return self.driver.session(database=self.database)

    def _init_constraints(self):
        """Create uniqueness constraints and indexes for performance.

        Mode C: uniqueness is composite (tenant_id, id) so two tenants can share
        the same entity id without collision; index on (tenant_id, name).

        Constraint/index creation is BEST-EFFORT — a failure (e.g. the legacy
        global id-unique constraint can't be created because Mode C data already
        exists on a shared graph) is logged but never breaks connect().
        """
        def _try(session, stmt):
            try:
                session.run(stmt)
            except Exception as e:
                logger.warning("graph_store.constraint_skipped: %s — %s", stmt.split(" IF ")[0], e)

        with self._session() as session:
            if self.multi_tenant:
                _try(session, "DROP CONSTRAINT entity_id_unique IF EXISTS")
                _try(session, "CREATE CONSTRAINT entity_tenant_id_unique IF NOT EXISTS "
                              "FOR (e:Entity) REQUIRE (e.tenant_id, e.id) IS UNIQUE")
                _try(session, "CREATE INDEX entity_tenant_name IF NOT EXISTS "
                              "FOR (e:Entity) ON (e.tenant_id, e.name)")
            else:
                _try(session, "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
                              "FOR (e:Entity) REQUIRE e.id IS UNIQUE")
                _try(session, "CREATE INDEX entity_name_index IF NOT EXISTS "
                              "FOR (e:Entity) ON (e.name)")
            _try(session, "CREATE INDEX entity_type_index IF NOT EXISTS "
                          "FOR (e:Entity) ON (e.entity_type)")
            logger.info("graph_store.constraints_initialized: tenant=%s", self.tenant_id or "-")


    def store_extraction(self, result: GraphExtractionResult):
        """
        Atomically store all entities and relations from a single document extraction.
        Uses a transaction to ensure consistency. In Mode C every node/edge is
        keyed by self.tenant_id so tenants sharing the graph never cross-merge.
        """
        with self._session() as session:
            session.execute_write(self._write_extraction_tx, result)
            logger.info("graph_store.extraction_stored: tenant=%s document_id=%s entities=%s relations=%s",
                        self.tenant_id or "-", result.document_id, len(result.entities), len(result.relations))

    def _write_extraction_tx(self, tx: ManagedTransaction, result: GraphExtractionResult):
        """Transaction function: create all nodes and edges (tenant-keyed in Mode C)."""
        tid = self.tenant_id
        ekey = "{tenant_id: $tid, name: $name, entity_type: $entity_type}" if tid \
            else "{name: $name, entity_type: $entity_type}"

        for entity in result.entities:
            params = dict(
                id=entity.id, name=entity.name.lower().strip(),
                entity_type=entity.entity_type.value,
                properties=json.dumps(entity.properties),
                source_doc=entity.source_document_id, confidence=entity.confidence,
            )
            if tid:
                params["tid"] = tid
            tx.run(
                f"""
                MERGE (e:Entity {ekey})
                ON CREATE SET
                    e.id = $id, e.properties = $properties,
                    e.source_document_id = $source_doc, e.confidence = $confidence,
                    e.created_at = datetime(){', e.tenant_id = $tid' if tid else ''}
                ON MATCH SET
                    e.properties = e.properties + $properties, e.updated_at = datetime()
                """,
                **params,
            )
            label = entity.entity_type.value.capitalize()
            lparams = {"name": entity.name.lower().strip(),
                       "entity_type": entity.entity_type.value}
            if tid:
                lparams["tid"] = tid
            tx.run(f"MATCH (e:Entity {ekey}) SET e:{label}", **lparams)

        for relation in result.relations:
            source_entity = next((e for e in result.entities if e.id == relation.source_entity_id), None)
            target_entity = next((e for e in result.entities if e.id == relation.target_entity_id), None)
            if not source_entity or not target_entity:
                continue
            akey = "{tenant_id: $tid, name: $source_name, entity_type: $source_type}" if tid \
                else "{name: $source_name, entity_type: $source_type}"
            bkey = "{tenant_id: $tid, name: $target_name, entity_type: $target_type}" if tid \
                else "{name: $target_name, entity_type: $target_type}"
            rparams = dict(
                id=relation.id,
                source_name=source_entity.name.lower().strip(),
                source_type=source_entity.entity_type.value,
                target_name=target_entity.name.lower().strip(),
                target_type=target_entity.entity_type.value,
                properties=json.dumps(relation.properties),
                source_doc=relation.source_document_id, confidence=relation.confidence,
            )
            if tid:
                rparams["tid"] = tid
            tx.run(
                f"""
                MATCH (a:Entity {akey})
                MATCH (b:Entity {bkey})
                MERGE (a)-[r:{relation.relation_type.value}]->(b)
                ON CREATE SET
                    r.id = $id, r.properties = $properties,
                    r.source_document_id = $source_doc, r.confidence = $confidence,
                    r.created_at = datetime(){', r.tenant_id = $tid' if tid else ''}
                """,
                **rparams,
            )

    def add_entity(self, entity: Entity):
        """Add a single entity to the graph."""
        with self.driver.session() as session:
            tid = self.tenant_id
            ekey = "{tenant_id: $tid, name: $name, entity_type: $entity_type}" if tid \
                else "{name: $name, entity_type: $entity_type}"
            params = dict(
                id=entity.id, name=entity.name.lower().strip(),
                entity_type=entity.entity_type.value,
                properties=json.dumps(entity.properties),
                source_doc=entity.source_document_id, confidence=entity.confidence,
            )
            if tid:
                params["tid"] = tid
            session.run(
                f"""
                MERGE (e:Entity {ekey})
                ON CREATE SET
                    e.id = $id, e.properties = $properties,
                    e.source_document_id = $source_doc, e.confidence = $confidence,
                    e.created_at = datetime(){', e.tenant_id = $tid' if tid else ''}
                """,
                **params,
            )


    def _run_cypher_internal(self, query: str, parameters: dict | None = None) -> list[dict]:
        """Internal Cypher executor used by the tenant-scoped methods below.
        NOT exposed to LLM-generated queries — those go through run_cypher()."""
        parameters = parameters or {}
        with self._session() as session:
            result = session.run(query, parameters)
            return [dict(record) for record in result]

    def run_cypher(self, query: str, parameters: dict | None = None) -> list[dict]:
        """Execute raw Cypher (e.g. LLM-generated by the Graph Agent).

        FIREWALL (Mode C): on a SHARED tenant-keyed graph, arbitrary Cypher is a
        cross-tenant read primitive (the LLM cannot be trusted to scope tenant_id).
        So in multi-tenant mode raw Cypher is DISABLED — callers must use the
        parameterized, tenant-scoped methods (find_entity / get_entity_neighborhood).
        See MULTITENANT_ELASTIC_NEO4J_DESIGN §3.7.3.
        """
        if self.multi_tenant:
            raise PermissionError(
                "raw Cypher is disabled in multi-tenant mode (shared graph). "
                "Use the tenant-scoped graph methods instead."
            )
        parameters = parameters or {}
        with self._session() as session:
            result = session.run(query, parameters)
            records = [dict(record) for record in result]
            logger.info("graph_store.cypher_executed: query=%s results=%s", query[:100], len(records))
            return records

    def _tfilter(self, alias: str = "e") -> str:
        """Tenant WHERE-fragment (Mode C) — '' in legacy single-tenant mode."""
        return f" AND {alias}.tenant_id = $tid" if self.multi_tenant else ""

    def find_entity(self, name: str, entity_type: Optional[EntityType] = None) -> list[dict]:
        """Find entities by name (case-insensitive fuzzy match), tenant-scoped."""
        params = {"name": name}
        if self.multi_tenant:
            params["tid"] = self.tenant_id
        type_clause = " AND e.entity_type = $type" if entity_type else ""
        if entity_type:
            params["type"] = entity_type.value
        query = f"""
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($name){self._tfilter('e')}{type_clause}
            RETURN e.id AS id, e.name AS name, e.entity_type AS type,
                   e.properties AS properties, e.confidence AS confidence
            LIMIT 20
        """
        return self._run_cypher_internal(query, params)

    def get_entity_neighborhood(self, entity_name: str, depth: int = 2, limit: int = 50) -> list[dict]:
        """Entities/relations within N hops of an entity — confined to this tenant.

        Mode C confines the WHOLE path to the tenant (all path nodes tenant_id=$tid)
        so a traversal can never cross into another tenant's subgraph.
        """
        depth = max(1, min(int(depth), 5))
        params = {"name": entity_name, "limit": limit}
        path_guard = ""
        if self.multi_tenant:
            params["tid"] = self.tenant_id
            path_guard = " AND start.tenant_id = $tid AND all(n IN nodes(path) WHERE n.tenant_id = $tid)"
        query = f"""
            MATCH path = (start:Entity)-[*1..{depth}]-(connected:Entity)
            WHERE toLower(start.name) = toLower($name){path_guard}
            UNWIND relationships(path) AS rel
            WITH startNode(rel) AS from_node, rel, endNode(rel) AS to_node
            RETURN DISTINCT
                from_node.name AS from_name, from_node.entity_type AS from_type,
                type(rel) AS relation,
                to_node.name AS to_name, to_node.entity_type AS to_type
            LIMIT $limit
        """
        return self._run_cypher_internal(query, params)

    def get_all_entity_types(self) -> dict:
        """Count of entities grouped by type (tenant-scoped)."""
        params = {"tid": self.tenant_id} if self.multi_tenant else {}
        where = "WHERE e.tenant_id = $tid" if self.multi_tenant else ""
        query = f"""
            MATCH (e:Entity) {where}
            RETURN e.entity_type AS type, COUNT(e) AS count ORDER BY count DESC
        """
        return {r["type"]: r["count"] for r in self._run_cypher_internal(query, params)}

    def get_all_relation_types(self) -> dict:
        """Count of relations grouped by type (tenant-scoped)."""
        params = {"tid": self.tenant_id} if self.multi_tenant else {}
        where = "WHERE r.tenant_id = $tid" if self.multi_tenant else ""
        query = f"""
            MATCH ()-[r]->() {where}
            RETURN type(r) AS type, COUNT(r) AS count ORDER BY count DESC
        """
        return {r["type"]: r["count"] for r in self._run_cypher_internal(query, params)}

    def get_stats(self) -> dict:
        """Graph statistics — tenant-scoped in Mode C."""
        with self._session() as session:
            if self.multi_tenant:
                node_count = session.run(
                    "MATCH (e:Entity) WHERE e.tenant_id=$tid RETURN COUNT(e) AS c",
                    tid=self.tenant_id).single()["c"]
                edge_count = session.run(
                    "MATCH ()-[r]->() WHERE r.tenant_id=$tid RETURN COUNT(r) AS c",
                    tid=self.tenant_id).single()["c"]
            else:
                node_count = session.run("MATCH (n) RETURN COUNT(n) AS c").single()["c"]
                edge_count = session.run("MATCH ()-[r]->() RETURN COUNT(r) AS c").single()["c"]
            return {
                "total_nodes": node_count,
                "total_edges": edge_count,
                "entity_types": self.get_all_entity_types(),
                "relation_types": self.get_all_relation_types(),
            }


    def delete_by_document(self, document_id: str):
        """Remove entities sourced from a document (tenant-scoped in Mode C)."""
        params = {"doc_id": document_id}
        match = "{source_document_id: $doc_id}"
        if self.multi_tenant:
            params["tid"] = self.tenant_id
            match = "{source_document_id: $doc_id, tenant_id: $tid}"
        with self._session() as session:
            session.run(f"MATCH (e:Entity {match}) DETACH DELETE e", **params)
            logger.info("graph_store.document_purged: tenant=%s document_id=%s",
                        self.tenant_id or "-", document_id)

    def delete_tenant(self):
        """Offboarding: delete this tenant's entire subgraph (Mode C only)."""
        if not self.multi_tenant:
            raise PermissionError("delete_tenant requires Mode C (tenant_id set).")
        with self._session() as session:
            session.run("MATCH (e:Entity {tenant_id: $tid}) DETACH DELETE e", tid=self.tenant_id)
            logger.warning("graph_store.tenant_deleted: tenant=%s", self.tenant_id)

    def clear_all(self):
        """⚠️ Nuclear option: delete everything in the graph."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.warning("graph_store.cleared_all")
