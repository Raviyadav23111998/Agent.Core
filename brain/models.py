"""
Agent Brain — Data Models
========================================
All core data structures as stdlib dataclasses (zero dependencies).
These flow across storage, ingestion, extraction, and retrieval.
"""

from __future__ import annotations
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import uuid


class DocumentType(str, Enum):
    EMAIL = "email"
    POLICY = "policy"
    CODE = "code"
    DOCUMENT = "document"
    CHAT_MESSAGE = "chat_message"
    MEETING_NOTES = "meeting_notes"
    TICKET = "ticket"


class EntityType(str, Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    PROJECT = "project"
    TECHNOLOGY = "technology"
    FILE = "file"
    MODULE = "module"
    POLICY = "policy"
    CONCEPT = "concept"
    EVENT = "event"
    LOCATION = "location"


class RelationType(str, Enum):
    AUTHORED = "AUTHORED"
    EMAILED = "EMAILED"
    MENTIONED_IN = "MENTIONED_IN"
    DEPENDS_ON = "DEPENDS_ON"
    PART_OF = "PART_OF"
    WORKS_ON = "WORKS_ON"
    REPORTED = "REPORTED"
    APPROVED = "APPROVED"
    MODIFIED = "MODIFIED"
    REFERENCES = "REFERENCES"
    RELATED_TO = "RELATED_TO"


class IngestionStatus(str, Enum):
    PENDING = "pending"
    PARSING = "parsing"
    EXTRACTING = "extracting"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


class QueryIntent(str, Enum):
    GRAPH = "graph"
    SEARCH = "search"
    CONTEXT = "context"
    HYBRID = "hybrid"


@dataclass
class Document:
    source_path: str
    document_type: DocumentType
    content: str
    title: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict = field(default_factory=dict)
    content_hash: str = ""
    token_count: int = 0
    status: IngestionStatus = IngestionStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Entity:
    name: str
    entity_type: EntityType
    source_document_id: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    properties: dict = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class Relation:
    source_entity_id: str
    target_entity_id: str
    relation_type: RelationType
    source_document_id: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    properties: dict = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class GraphExtractionResult:
    document_id: str
    entities: list = field(default_factory=list)
    relations: list = field(default_factory=list)


@dataclass
class SearchDocument:
    document_id: str
    title: str
    content: str
    document_type: str
    source_path: str
    author: Optional[str] = None
    tags: list = field(default_factory=list)
    created_at: str = ""
    owner_id: str = "__tenant__"
    app_id: int = 1


@dataclass
class SearchResult:
    document_id: str
    title: str
    content_snippet: str
    score: float
    document_type: str
    source_path: str


@dataclass
class ContextBundle:
    name: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    document_ids: list = field(default_factory=list)
    total_tokens: int = 0
    max_tokens: int = 1_000_000
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class QueryPlan:
    original_query: str
    intents: list = field(default_factory=list)
    graph_query: Optional[str] = None
    search_query: Optional[str] = None
    semantic_query: Optional[str] = None
    context_query: Optional[str] = None
    context_bundle_id: Optional[str] = None


@dataclass
class BrainResponse:
    answer: str
    sources: list = field(default_factory=list)
    confidence: float = 0.0
    agents_used: list = field(default_factory=list)
    query_plan: Optional[QueryPlan] = None
