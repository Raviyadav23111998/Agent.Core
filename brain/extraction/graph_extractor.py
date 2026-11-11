from __future__ import annotations

"""
Agent Brain — Graph Extractor
=================================
Calls the LLM with document-type-specific prompts to extract
entities and relations, then parses the JSON response into
our Dataclass models (Entity, Relation, GraphExtractionResult).

This is the bridge between raw documents and the Knowledge Graph.
"""

import logging

from brain.models import (
    Entity, Relation, EntityType, RelationType,
    GraphExtractionResult, DocumentType,
)
from brain.exceptions import ExtractionError
from brain.extraction.prompts import EXTRACTION_SYSTEM_PROMPT, get_extraction_prompt

logger = logging.getLogger(__name__)

ENTITY_TYPE_MAP = {
    "person": EntityType.PERSON,
    "organization": EntityType.ORGANIZATION,
    "project": EntityType.PROJECT,
    "technology": EntityType.TECHNOLOGY,
    "file": EntityType.FILE,
    "module": EntityType.MODULE,
    "policy": EntityType.POLICY,
    "concept": EntityType.CONCEPT,
    "event": EntityType.EVENT,
    "location": EntityType.LOCATION,
}

RELATION_TYPE_MAP = {
    "AUTHORED": RelationType.AUTHORED,
    "EMAILED": RelationType.EMAILED,
    "MENTIONED_IN": RelationType.MENTIONED_IN,
    "DEPENDS_ON": RelationType.DEPENDS_ON,
    "PART_OF": RelationType.PART_OF,
    "WORKS_ON": RelationType.WORKS_ON,
    "REPORTED": RelationType.REPORTED,
    "APPROVED": RelationType.APPROVED,
    "MODIFIED": RelationType.MODIFIED,
    "REFERENCES": RelationType.REFERENCES,
    "RELATED_TO": RelationType.RELATED_TO,
}


class GraphExtractor:
    """
    Uses the LLM to extract entities and relations from documents.
    Produces GraphExtractionResult objects ready to store in Neo4j.
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    async def extract(
        self,
        document_id: str,
        content: str,
        document_type: DocumentType,
        language: str = "",
        max_content_chars: int = 20_000,
    ) -> GraphExtractionResult:
        """
        Extract entities and relations from a document.

        Args:
            document_id: ID of the source document
            content: Raw text content
            document_type: Type of document (determines which prompt to use)
            language: Programming language (for code files)
            max_content_chars: Truncate content if too long (for cheap extraction)

        Returns:
            GraphExtractionResult with extracted entities and relations

        Raises:
            ExtractionError: If the LLM fails to extract or returns invalid data
        """
        truncated = content[:max_content_chars]
        if len(content) > max_content_chars:
            truncated += f"\n\n[... truncated, {len(content) - max_content_chars} chars remaining ...]"

        prompt_kwargs = {}
        if document_type == DocumentType.CODE:
            prompt_kwargs["language"] = language or "unknown"

        user_prompt = get_extraction_prompt(
            document_type=document_type.value,
            content=truncated,
            **prompt_kwargs,
        )

        try:
            response = await self.llm.extract_json(
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                use_extraction_model=True,
            )

            if not response:
                logger.warning("graph_extractor.empty_response: document_id=%s", document_id)
                return GraphExtractionResult(document_id=document_id)

            result = self._parse_response(document_id, response)

            logger.info("graph_extractor.extracted: document_id=%s entities=%s relations=%s", document_id, len(result.entities), len(result.relations))

            return result

        except Exception as e:
            logger.error("graph_extractor.failed: document_id=%s error=%s", document_id, str(e))
            raise ExtractionError(document_id, str(e))

    def _parse_response(self, document_id: str, response: dict) -> GraphExtractionResult:
        """Parse the LLM's JSON response into Entity and Relation models."""
        entities = []
        entity_name_map = {}

        raw_entities = response.get("entities", [])
        for raw in raw_entities:
            if not isinstance(raw, dict):
                continue

            name = raw.get("name", "").strip().lower()
            if not name:
                continue

            type_str = raw.get("type", "concept").lower()
            entity_type = ENTITY_TYPE_MAP.get(type_str, EntityType.CONCEPT)

            entity = Entity(
                name=name,
                entity_type=entity_type,
                properties=raw.get("properties", {}),
                source_document_id=document_id,
                confidence=min(max(float(raw.get("confidence", 0.8)), 0.0), 1.0),
            )
            entities.append(entity)
            entity_name_map[name] = entity

        relations = []
        raw_relations = response.get("relations", [])
        for raw in raw_relations:
            if not isinstance(raw, dict):
                continue

            source_name = raw.get("source", "").strip().lower()
            target_name = raw.get("target", "").strip().lower()

            if not source_name or not target_name:
                continue

            source_entity = entity_name_map.get(source_name)
            target_entity = entity_name_map.get(target_name)

            if not source_entity or not target_entity:
                if not source_entity:
                    source_entity = Entity(
                        name=source_name,
                        entity_type=EntityType.CONCEPT,
                        properties={},
                        source_document_id=document_id,
                        confidence=0.5,
                    )
                    entities.append(source_entity)
                    entity_name_map[source_name] = source_entity

                if not target_entity:
                    target_entity = Entity(
                        name=target_name,
                        entity_type=EntityType.CONCEPT,
                        properties={},
                        source_document_id=document_id,
                        confidence=0.5,
                    )
                    entities.append(target_entity)
                    entity_name_map[target_name] = target_entity

            type_str = raw.get("type", "RELATED_TO").upper()
            relation_type = RELATION_TYPE_MAP.get(type_str, RelationType.RELATED_TO)

            relation = Relation(
                source_entity_id=source_entity.id,
                target_entity_id=target_entity.id,
                relation_type=relation_type,
                properties=raw.get("properties", {}),
                source_document_id=document_id,
                confidence=min(max(float(raw.get("confidence", 0.8)), 0.0), 1.0),
            )
            relations.append(relation)

        return GraphExtractionResult(
            document_id=document_id,
            entities=entities,
            relations=relations,
        )
