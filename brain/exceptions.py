"""
Agent Brain — Custom Exceptions
Clean error hierarchy for all storage and retrieval operations.
"""


class BrainError(Exception):
    """Base exception for the entire brain system."""
    pass


class StorageError(BrainError):
    """Base error for all storage operations."""
    pass


class DocumentNotFoundError(StorageError):
    """Raised when a document ID doesn't exist in the Document Store."""
    def __init__(self, document_id: str):
        self.document_id = document_id
        super().__init__(f"Document not found: {document_id}")


class DuplicateDocumentError(StorageError):
    """Raised when attempting to ingest a document that already exists (same content hash)."""
    def __init__(self, content_hash: str, existing_id: str):
        self.content_hash = content_hash
        self.existing_id = existing_id
        super().__init__(f"Duplicate document (hash={content_hash[:16]}...), existing ID: {existing_id}")


class GraphConnectionError(StorageError):
    """Raised when Neo4j is unreachable."""
    pass


class SearchConnectionError(StorageError):
    """Raised when Elasticsearch is unreachable."""
    pass


class LLMError(BrainError):
    """Base error for LLM operations."""
    pass


class ExtractionError(LLMError):
    """Raised when the LLM fails to extract entities/relations from a document."""
    def __init__(self, document_id: str, reason: str):
        self.document_id = document_id
        super().__init__(f"Extraction failed for document {document_id}: {reason}")


class RoutingError(LLMError):
    """Raised when the LLM router fails to classify a query."""
    pass


class ContextBudgetExceededError(BrainError):
    """Raised when a context bundle exceeds its maximum token budget."""
    def __init__(self, bundle_id: str, current_tokens: int, max_tokens: int):
        self.bundle_id = bundle_id
        super().__init__(
            f"Context bundle {bundle_id} exceeds budget: "
            f"{current_tokens:,} tokens > {max_tokens:,} max"
        )
