"""
Agent Brain — Context Cache Manager (PostgreSQL)
=============================================================
PILLAR 3: The Long-Context Engine

Instead of chunking documents into vectors, we keep FULL documents
and load them directly into an LLM's massive context window (1M-2M tokens).

Storage: PostgreSQL ag_brain.context_bundles (via TenantDB)
Migrated from .ag_data/context_cache/ (JSON + text files) to PostgreSQL.
"""

import json
import logging
from typing import Optional

from brain.models import ContextBundle, Document
from brain.exceptions import ContextBudgetExceededError, DocumentNotFoundError

logger = logging.getLogger(__name__)


class ContextCache:
    """Manages Context Bundles — collections of documents pre-assembled
    for loading into LLM context windows.
    
    PostgreSQL-backed via TenantDB (ag_brain.context_bundles).
    """

    def __init__(self, db=None, cache_dir: str = None, **kwargs):
        """
        Args:
            db: TenantDB instance
            cache_dir: Legacy param (ignored when db is provided)
        """
        self._db = db
        self._bundles: dict[str, ContextBundle] = {}

        if self._db:
            self._load_bundles()


    def create_bundle(
        self,
        name: str,
        description: str = "",
        max_tokens: int = 1_000_000,
    ) -> ContextBundle:
        """Create a new empty context bundle."""
        bundle = ContextBundle(
            name=name,
            description=description,
            max_tokens=max_tokens,
        )

        if self._db:
            self._db.execute("""
                INSERT INTO ag_brain.context_bundles
                    (name, description, max_tokens)
                VALUES (%s, %s, %s)
                ON CONFLICT (name) DO NOTHING
            """, (name, description, max_tokens))

        self._bundles[bundle.name] = bundle
        logger.info("context_cache.bundle_created: id=%s name=%s", bundle.id, name)
        return bundle

    def get_bundle(self, bundle_id: str) -> ContextBundle:
        """Get a bundle by ID or name."""
        bundle = self._bundles.get(bundle_id)
        if not bundle:
            for b in self._bundles.values():
                if b.id == bundle_id:
                    return b
            raise DocumentNotFoundError(bundle_id)
        return bundle

    def get_bundle_by_name(self, name: str) -> Optional[ContextBundle]:
        """Find a bundle by its human-readable name."""
        return self._bundles.get(name)

    def list_bundles(self) -> list[ContextBundle]:
        """List all existing context bundles."""
        return list(self._bundles.values())

    def delete_bundle(self, bundle_id: str) -> bool:
        """Delete a bundle and its cached payload."""
        bundle = None
        key = None
        if bundle_id in self._bundles:
            key = bundle_id
            bundle = self._bundles[bundle_id]
        else:
            for k, b in self._bundles.items():
                if b.id == bundle_id:
                    key, bundle = k, b
                    break
        if not bundle:
            return False

        if self._db:
            self._db.execute(
                "DELETE FROM ag_brain.context_bundles WHERE name = %s",
                (bundle.name,)
            )

        del self._bundles[key]
        logger.info("context_cache.bundle_deleted: name=%s", bundle.name)
        return True


    def add_documents_to_bundle(
        self,
        bundle_id: str,
        documents: list[Document],
    ) -> ContextBundle:
        """Add documents to a context bundle."""
        bundle = self.get_bundle(bundle_id)

        new_tokens = sum(doc.token_count for doc in documents)
        projected_total = bundle.total_tokens + new_tokens

        if projected_total > bundle.max_tokens:
            raise ContextBudgetExceededError(bundle_id, projected_total, bundle.max_tokens)

        for doc in documents:
            if doc.id not in bundle.document_ids:
                bundle.document_ids.append(doc.id)

        bundle.total_tokens = projected_total
        self._bundles[bundle.name] = bundle
        self._save_bundle(bundle)
        self._invalidate_cache(bundle)

        logger.info("context_cache.documents_added: bundle=%s added=%d total_tokens=%d",
                    bundle_id, len(documents), projected_total)
        return bundle

    def remove_document_from_bundle(self, bundle_id: str, document_id: str) -> ContextBundle:
        """Remove a specific document from a bundle."""
        bundle = self.get_bundle(bundle_id)
        if document_id in bundle.document_ids:
            bundle.document_ids.remove(document_id)
            self._bundles[bundle.name] = bundle
            self._save_bundle(bundle)
            self._invalidate_cache(bundle)
        return bundle


    def build_context_payload(self, bundle_id: str, documents: list[Document]) -> str:
        """Build the full text payload for an LLM context window."""
        bundle = self.get_bundle(bundle_id)

        cached = self._load_cached_payload(bundle)
        if cached:
            logger.info("context_cache.payload_from_cache: bundle_id=%s", bundle_id)
            return cached

        parts = []
        parts.append("=" * 80)
        parts.append(f"CONTEXT BUNDLE: {bundle.name}")
        parts.append(f"Description: {bundle.description}")
        parts.append(f"Total Documents: {len(documents)}")
        parts.append(f"Total Tokens: ~{bundle.total_tokens:,}")
        parts.append("=" * 80)
        parts.append("")

        for i, doc in enumerate(documents, 1):
            parts.append(f"{'─' * 60}")
            parts.append(f"📄 DOCUMENT {i}/{len(documents)}")
            parts.append(f"   Title: {doc.title}")
            parts.append(f"   Type: {doc.document_type.value}")
            parts.append(f"   Source: {doc.source_path}")
            if doc.metadata:
                for key, value in doc.metadata.items():
                    parts.append(f"   {key}: {value}")
            parts.append(f"{'─' * 60}")
            parts.append("")
            parts.append(doc.content)
            parts.append("")

        payload = "\n".join(parts)

        self._save_cached_payload(bundle, payload)
        logger.info("context_cache.payload_built: bundle=%s documents=%d chars=%d",
                    bundle_id, len(documents), len(payload))

        return payload

    def build_file_index(self, bundle_id: str, documents: list[Document]) -> str:
        """Build a lightweight file index for two-stage retrieval."""
        lines = [
            f"FILE INDEX for bundle '{self.get_bundle(bundle_id).name}' ({len(documents)} files)",
            "=" * 60,
            "",
        ]

        for i, doc in enumerate(documents, 1):
            preview = doc.content[:200].replace('\n', ' ').replace('\r', '').strip()
            if len(doc.content) > 200:
                preview += "..."
            lines.append(
                f"{i}. [{doc.document_type.value}] {doc.source_path}\n"
                f"   Preview: {preview}\n"
            )

        return "\n".join(lines)

    def build_selective_payload(
        self, bundle_id: str, documents: list[Document], selected_indices: list[int]
    ) -> str:
        """Build a context payload with ONLY the selected documents."""
        bundle = self.get_bundle(bundle_id)

        selected_docs = []
        for idx in selected_indices:
            if 1 <= idx <= len(documents):
                selected_docs.append(documents[idx - 1])

        if not selected_docs:
            logger.warning("context_cache.selection_empty_fallback: bundle_id=%s", bundle_id)
            selected_docs = documents[:10]

        parts = []
        parts.append("=" * 60)
        parts.append(f"SELECTED CONTEXT from '{bundle.name}' ({len(selected_docs)}/{len(documents)} files)")
        parts.append("=" * 60)
        parts.append("")

        for i, doc in enumerate(selected_docs, 1):
            parts.append(f"{'─' * 40}")
            parts.append(f"📄 FILE {i}/{len(selected_docs)}: {doc.source_path}")
            parts.append(f"   Type: {doc.document_type.value}")
            parts.append(f"{'─' * 40}")
            parts.append("")
            parts.append(doc.content)
            parts.append("")

        return "\n".join(parts)

    def get_bundle_token_usage(self, bundle_id: str) -> dict:
        """Get token usage statistics for a bundle."""
        bundle = self.get_bundle(bundle_id)
        return {
            "bundle_name": bundle.name,
            "total_tokens": bundle.total_tokens,
            "max_tokens": bundle.max_tokens,
            "usage_percent": round((bundle.total_tokens / bundle.max_tokens) * 100, 1) if bundle.max_tokens > 0 else 0,
            "remaining_tokens": bundle.max_tokens - bundle.total_tokens,
            "document_count": len(bundle.document_ids),
        }


    def _save_bundle(self, bundle: ContextBundle):
        """Persist bundle metadata to PostgreSQL."""
        if not self._db:
            return
        self._db.execute("""
            INSERT INTO ag_brain.context_bundles
                (name, description, max_tokens, total_tokens, document_count, doc_ids)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (name) DO UPDATE SET
                description = EXCLUDED.description,
                total_tokens = EXCLUDED.total_tokens,
                document_count = EXCLUDED.document_count,
                doc_ids = EXCLUDED.doc_ids,
                updated_at = NOW()
        """, (
            bundle.name, bundle.description, bundle.max_tokens,
            bundle.total_tokens, len(bundle.document_ids),
            json.dumps(bundle.document_ids),
        ))

    def _save_cached_payload(self, bundle: ContextBundle, payload: str):
        """Save assembled payload to PostgreSQL."""
        if not self._db:
            return
        self._db.execute("""
            UPDATE ag_brain.context_bundles SET
                cached_payload = %s,
                payload_built_at = NOW()
            WHERE name = %s
        """, (payload, bundle.name))

    def _load_cached_payload(self, bundle: ContextBundle) -> Optional[str]:
        """Load cached payload from PostgreSQL."""
        if not self._db:
            return None
        row = self._db.fetch_one(
            "SELECT cached_payload FROM ag_brain.context_bundles WHERE name = %s AND cached_payload IS NOT NULL",
            (bundle.name,)
        )
        return row["cached_payload"] if row else None

    def _invalidate_cache(self, bundle: ContextBundle):
        """Invalidate cached payload when bundle contents change."""
        if not self._db:
            return
        self._db.execute(
            "UPDATE ag_brain.context_bundles SET cached_payload = NULL, payload_built_at = NULL WHERE name = %s",
            (bundle.name,)
        )
        logger.info("context_cache.cache_invalidated: bundle=%s", bundle.name)

    def _load_bundles(self):
        """Load bundle metadata from PostgreSQL."""
        if not self._db:
            return
        try:
            rows = self._db.fetch_all("SELECT * FROM ag_brain.context_bundles")
            for row in rows:
                doc_ids = row.get("doc_ids") or []
                if isinstance(doc_ids, str):
                    doc_ids = json.loads(doc_ids)
                bundle = ContextBundle(
                    name=row["name"],
                    description=row.get("description", ""),
                    document_ids=doc_ids,
                    total_tokens=row.get("total_tokens", 0),
                    max_tokens=row.get("max_tokens", 1_000_000),
                )
                self._bundles[bundle.name] = bundle
            logger.info("context_cache.bundles_loaded: count=%d", len(self._bundles))
        except Exception as e:
            logger.error("context_cache.load_failed: %s", e)

    def get_stats(self) -> dict:
        """Get overall context cache statistics."""
        total_tokens = sum(b.total_tokens for b in self._bundles.values())
        total_docs = sum(len(b.document_ids) for b in self._bundles.values())

        return {
            "total_bundles": len(self._bundles),
            "total_documents_in_bundles": total_docs,
            "total_tokens_across_bundles": total_tokens,
            "bundles": [
                {
                    "id": b.id,
                    "name": b.name,
                    "documents": len(b.document_ids),
                    "tokens": b.total_tokens,
                    "usage_percent": round((b.total_tokens / b.max_tokens) * 100, 1) if b.max_tokens > 0 else 0,
                }
                for b in self._bundles.values()
            ],
        }
