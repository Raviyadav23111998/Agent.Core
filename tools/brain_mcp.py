"""
Agent — Brain MCP Tools
=====================================
MCP tools exposing the brain's ingestion and query capabilities to the agent.

Tools:
  brain_ask             — Ask the brain a question (routed through Brain.ask())
  brain_ingest_file     — Ingest a single file into the knowledge base
  brain_ingest_directory — Ingest an entire directory
  brain_ingest_text     — Ingest raw text content
  brain_stats           — Get system stats (documents, graph, memory)
  brain_search          — Direct keyword search (bypasses routing)
"""

import json
import asyncio
import threading
import logging
from core.tenant import TENANT_SCOPE
from core.user_context import current_user_context

logger = logging.getLogger(__name__)


class _BundleProxy:
    """Resolves to the active tenant's (storage|brain|pipeline) per access.

    idx: 0=storage, 1=brain, 2=pipeline. Falsy when the target is None so the
    existing `if not storage:` / `if storage.lexical:` guards keep working.
    """
    __slots__ = ("_owner", "_idx")

    def __init__(self, owner, idx):
        self._owner = owner
        self._idx = idx

    def _target(self):
        return self._owner._active_bundle()[self._idx]

    def __getattr__(self, name):
        target = self._target()
        if target is None:
            raise AttributeError(
                f"brain store unavailable for the active tenant (not resolved): .{name}")
        return getattr(target, name)

    def __bool__(self):
        return self._target() is not None


def _run_async(coro):
    """Run an async coroutine safely from sync context.

    Handles both cases:
      - No event loop running → create one and run (CLI mode)
      - Event loop already running → spawn in a background thread (gateway mode)

    This avoids the 'This event loop is already running' RuntimeError
    that occurs when MCP tool functions are called from async gateway handlers.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        return asyncio.run(coro)
    else:
        result = [None]
        error = [None]

        def _thread_target():
            try:
                result[0] = asyncio.run(coro)
            except Exception as e:
                error[0] = e

        t = threading.Thread(target=_thread_target, daemon=True)
        t.start()
        t.join(timeout=120)

        if error[0]:
            raise error[0]
        if t.is_alive():
            raise TimeoutError("Brain operation timed out after 120s")
        return result[0]


class BrainMCPTools:
    """MCP tool wrappers for the brain ingestion and retrieval system."""

    def __init__(self, pipeline=None, brain=None, storage=None, sync_worker=None,
                 user_context_resolver=None, chat_history_resolver=None):
        """
        Args:
            pipeline: IngestionPipeline instance (for ingest tools)
            brain: Brain retrieval instance (for ask tool)
            storage: StorageManager instance (for stats/search tools)
            sync_worker: SyncWorker instance (for sync tools)
            user_context_resolver: Callable that returns the current UserContext.
                                   Used to determine owner_id for data isolation.
            chat_history_resolver: Callable that returns the current chat messages list.
        """
        self._pipeline = pipeline
        self._brain = brain
        self._storage = storage
        self._sync_worker = sync_worker
        self._resolve_user = user_context_resolver
        self._resolve_chat = chat_history_resolver
        self._last_sources: list = []

    _MAX_CITATIONS = 8

    def _active_bundle(self):
        """Return (storage, brain, pipeline) for the ACTIVE tenant.

        Single-tenant / flag-off / default → the shared singletons (unchanged).
        Real tenant → a lazily-built, cached bundle whose StorageManager uses the
        tenant's own agent DB + ES index (brain_<id>), reusing the shared LLM.
        """
        try:
            from core.tenant_context import current_tenant, multi_tenant_enabled
            from core.tenant_resolver import get_global_resolver
        except Exception:
            return self._storage, self._brain, self._pipeline
        ctx = current_tenant()
        resolver = get_global_resolver()
        if not multi_tenant_enabled() or ctx.is_default or resolver is None:
            return self._storage, self._brain, self._pipeline

        # The bundle is owned and cached by the tenant's connection object, so it
        # shares one lifecycle with the storage handles it wraps. We deliberately
        # keep NO independent cache here — that second cache was how brain_mcp
        # ended up serving a Brain backed by an already-closed pool after a config
        # refresh invalidated the tenant.
        try:
            conns = resolver.for_tenant(ctx)
            llm = getattr(self._brain, "llm", None) or getattr(self._pipeline, "llm", None)
            return conns.brain_bundle(
                llm_client=llm,
                want_brain=self._brain is not None,
                want_pipeline=self._pipeline is not None,
            )
        except Exception as e:
            logger.error("brain_mcp.tenant_bundle_failed: tenant=%s err=%s — failing closed (no default fallback)",
                         ctx.tenant_key, e)
            return None, None, None

    def _record_sources(self, sources):
        """Accumulate structured sources for the next pop_sources() (deduped)."""
        if not sources:
            return
        seen = {(s.get("document_id"), s.get("title")) for s in self._last_sources}
        for s in sources:
            if not isinstance(s, dict):
                continue
            key = (s.get("document_id"), s.get("title"))
            if key in seen:
                continue
            seen.add(key)
            self._last_sources.append({
                "title": s.get("title") or s.get("source_path") or "source",
                "documentId": s.get("document_id"),
                "type": s.get("type"),
                "sourcePath": s.get("source_path"),
                "snippet": (s.get("snippet") or "")[:280],
            })
            if len(self._last_sources) >= self._MAX_CITATIONS:
                break

    def pop_sources(self) -> list:
        """Return and clear the accumulated citation sources."""
        out = self._last_sources
        self._last_sources = []
        return out

    def register_tools(self, mcp_server):
        """Register all brain MCP tools."""

        instance = self
        pipeline = _BundleProxy(self, 2)
        brain = _BundleProxy(self, 1)
        storage = _BundleProxy(self, 0)
        sync_worker = self._sync_worker
        resolve_user = self._resolve_user
        resolve_chat = self._resolve_chat

        def _get_owner_id() -> str:
            """Get the owner_id for ingestion (where new data goes)."""
            if resolve_user:
                ctx = resolve_user()
                return ctx.owner_id_for_ingestion if ctx else TENANT_SCOPE
            return TENANT_SCOPE

        def _get_visible_scopes() -> list:
            """Get all owner_ids the current user can see (Teams model).

            Returns list like ['user_12345', 'group_-1001234', '__tenant__']
            or None for unscoped access (CLI/admin).
            """
            if resolve_user:
                ctx = resolve_user()
                return ctx.visible_scopes if ctx else None
            return None

        def _get_app_id():
            """Get app_id from current UserContext for knowledge scoping.

            Returns int app_id or None (unscoped — CLI/admin mode).

            Priority:
              1. current_user_context ContextVar (set by AgentLoop per-request)
              2. resolve_user callback (fallback for non-AgentLoop callers)
            """
            ctx = current_user_context.get(None)
            if ctx and ctx.app_id:
                logger.debug("_get_app_id: from ContextVar → app_id=%s", ctx.app_id)
                return int(ctx.app_id)

            if resolve_user:
                ctx = resolve_user()
                if ctx and ctx.app_id:
                    logger.debug("_get_app_id: from resolver → app_id=%s", ctx.app_id)
                    return int(ctx.app_id)

            logger.debug("_get_app_id: no app_id found (unscoped mode)")
            return None

        def _get_user_id_for_search() -> str:
            """Legacy: get single user_id for backward compat.

            Uses data_scope_id so spawned agents inherit parent's data scope.
            Prefer _get_visible_scopes() for new code.
            """
            if resolve_user:
                ctx = resolve_user()
                return ctx.data_scope_id if ctx else None
            return None

        @mcp_server.tool()
        def brain_ingest_file(file_path: str) -> str:
            """Ingest a single file into the knowledge base.

            Parses the file (code, text, PDF, email), stores it in the document
            store, indexes it in Elasticsearch for search, and optionally extracts
            entities/relations into the knowledge graph.

            The file is owned by the current user (private) unless
            the user has configured share_ingested=True.

            Args:
                file_path: Absolute path to the file to ingest
            """
            if not pipeline:
                return "Brain ingestion pipeline not initialized."
            try:
                owner = _get_owner_id()
                result = _run_async(pipeline.ingest_file(file_path, owner_id=owner))
                status = result.get("status", "unknown")
                if status == "ingested":
                    return (
                        f"✓ Ingested: {result.get('title', file_path)} "
                        f"({result.get('type', '?')}, {result.get('tokens', 0)} tokens, "
                        f"{result.get('graph_entities', 0)} entities, "
                        f"{result.get('graph_relations', 0)} relations, "
                        f"owner={owner})"
                    )
                elif status == "duplicate":
                    return f"⊘ Duplicate: {file_path} (already in knowledge base as {result.get('existing_id', '?')})"
                else:
                    return f"⚠ Skipped: {file_path} — {result.get('reason', 'unknown')}"
            except Exception as e:
                return f"✗ Ingest failed: {str(e)}"

        @mcp_server.tool()
        def brain_ingest_directory(directory: str, recursive: str = "true") -> str:
            """Ingest all supported files from a directory into the knowledge base.

            Scans for code, text, PDF, and email files, skipping binary files
            and common ignore patterns (node_modules, .git, __pycache__, etc.)

            Args:
                directory: Absolute path to the directory to ingest
                recursive: Whether to recurse into subdirectories (default: true)
            """
            if not pipeline:
                return "Brain ingestion pipeline not initialized."
            try:
                is_recursive = recursive.lower() in ("true", "yes", "1")
                owner = _get_owner_id()
                result = _run_async(
                    pipeline.ingest_directory(directory, recursive=is_recursive,
                                              owner_id=owner)
                )
                return (
                    f"Directory ingestion complete (owner={owner}):\n"
                    f"  Total files: {result.get('total', 0)}\n"
                    f"  Ingested: {result.get('ingested', 0)}\n"
                    f"  Duplicates skipped: {result.get('skipped_duplicates', 0)}\n"
                    f"  Errors: {result.get('storage_errors', 0)}"
                )
            except Exception as e:
                return f"✗ Directory ingest failed: {str(e)}"

        @mcp_server.tool()
        def brain_ingest_text(content: str, title: str = "", document_type: str = "document") -> str:
            """Ingest raw text content directly into the knowledge base.

            Use this for content that doesn't come from a file — API responses,
            pasted documents, meeting notes, etc.

            Args:
                content: The text content to ingest
                title: Optional human-readable title
                document_type: Type: email, policy, code, document, meeting_notes, ticket
            """
            if not pipeline:
                return "Brain ingestion pipeline not initialized."
            try:
                from brain.models import DocumentType
                doc_type = DocumentType(document_type) if document_type else DocumentType.DOCUMENT
                owner = _get_owner_id()
                result = _run_async(
                    pipeline.ingest_text(content=content, title=title,
                                         document_type=doc_type, owner_id=owner)
                )
                status = result.get("status", "unknown")
                if status == "ingested":
                    return f"✓ Ingested: {title or 'manual text'} ({result.get('tokens', 0)} tokens, owner={owner})"
                return f"⚠ {status}: {result.get('reason', 'unknown')}"
            except Exception as e:
                return f"✗ Text ingest failed: {str(e)}"

        @mcp_server.tool()
        def brain_index_health() -> str:
            """Report ingestion coverage across the knowledge-base pillars.

            For each pillar — full-text (Elasticsearch), graph (Neo4j), vector
            (embeddings) — shows how many documents are indexed (ok), still
            pending, failed, or skipped (pillar not configured). The reconciler
            heals pending/failed automatically; a stuck 'failed' count points at
            a real problem (e.g. an unreachable service or bad credential). Use
            this to verify ingestion is actually working end-to-end.
            """
            if not storage:
                return "Brain storage not initialized."
            try:
                h = storage.index_health()
            except Exception as e:
                return f"✗ index health unavailable: {e}"
            cfg = h.get("configured", {})
            lines = ["Brain index health (documents per pillar):"]
            for pillar in ("lexical", "graph", "vector"):
                states = h.get("pillars", {}).get(pillar, {})
                wired = "configured" if cfg.get(pillar) else "NOT configured"
                parts = ", ".join(f"{k}={v}" for k, v in sorted(states.items())) or "no rows"
                lines.append(f"  • {pillar} ({wired}): {parts}")
            return "\n".join(lines)

        @mcp_server.tool()
        def brain_ingest_current_chat() -> str:
            """Ingest the full history of the current chat session into the knowledge base.
            
            This allows the agent to take all previous conversational context and permanently
            save it as a document so it can be recalled in future sessions.
            """
            if not pipeline:
                return "Brain ingestion pipeline not initialized."
            if not self._resolve_chat:
                return "Chat history resolver not available in this context."
                
            try:
                chat_messages = self._resolve_chat()
                if not chat_messages:
                    return "No chat history found for the current session."
                
                transcript = ["# Conversation Transcript\n"]
                for msg in chat_messages:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if content and isinstance(content, str):
                        transcript.append(f"**{role.upper()}**:\n{content}\n")
                    elif content and isinstance(content, list):
                        text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                        transcript.append(f"**{role.upper()}**:\n{''.join(text_parts)}\n")
                        
                full_text = "\n".join(transcript)
                
                owner = _get_owner_id()
                from brain.models import DocumentType
                
                result = _run_async(
                    pipeline.ingest_text(
                        content=full_text,
                        title="Conversation Transcript",
                        document_type=DocumentType.TICKET,
                        owner_id=owner
                    )
                )
                
                status = result.get("status", "unknown")
                if status == "ingested":
                    return f"✓ Ingested conversation history ({len(chat_messages)} messages, {result.get('tokens', 0)} tokens, owner={owner})"
                return f"⚠ {status}: {result.get('reason', 'unknown')}"
            except Exception as e:
                return f"✗ Chat ingest failed: {str(e)}"

        @mcp_server.tool()
        def brain_ask(question: str) -> str:
            """Ask the brain a question.

            Routes the question through the intelligent retrieval pipeline:
            1. Query Router classifies intent (graph/search/context/hybrid)
            2. Appropriate agents execute (Neo4j, Elasticsearch, long-context)
            3. Synthesizer merges results into a coherent answer

            Results are scoped to the current user's documents + shared data.

            Args:
                question: The question to ask
            """
            if not brain:
                return "Brain retrieval system not initialized."
            try:
                scopes = _get_visible_scopes()
                search_user = _get_user_id_for_search()
                app_id = _get_app_id()
                response = _run_async(brain.ask(question, owner_id=search_user,
                                                visible_scopes=scopes,
                                                app_id=app_id))
                result = f"{response.answer}\n\n"
                if response.sources:
                    instance._record_sources(response.sources)
                    result += "Sources:\n"
                    for s in response.sources[:5]:
                        result += f"  - {json.dumps(s)}\n"
                result += f"Confidence: {response.confidence:.1%} | Agents: {', '.join(response.agents_used)}"
                return result
            except Exception as e:
                return f"✗ Brain query failed: {str(e)}"

        @mcp_server.tool()
        def brain_search(query: str, limit: str = "5") -> str:
            """Direct keyword search across the document store.

            Uses Elasticsearch when available, falls back to PostgreSQL FTS.
            Returns the current user's documents + common (shared) documents.

            Args:
                query: Search keywords
                limit: Max results to return
            """
            if not storage:
                return "Brain storage not initialized."
            try:
                max_results = int(limit)
                scopes = _get_visible_scopes()
                app_id = _get_app_id()

                if storage.lexical:
                    try:
                        results = storage.lexical.search(
                            query=query, limit=max_results,
                            visible_scopes=scopes,
                            app_id=app_id,
                        )
                        if results:
                            instance._record_sources([{
                                "document_id": getattr(h, "document_id", None),
                                "title": h.title,
                                "type": str(getattr(h, "document_type", "")),
                                "source_path": getattr(h, "source_path", None),
                                "snippet": getattr(h, "content_snippet", "") or "",
                            } for h in results])
                            lines = [f"Found {len(results)} result(s):"]
                            for i, hit in enumerate(results, 1):
                                lines.append(f"  {i}. {hit.title} (score: {hit.score:.2f}, type: {hit.document_type})")
                                if hit.content_snippet:
                                    lines.append(f"     {hit.content_snippet[:150]}...")
                            return "\n".join(lines)
                    except Exception:
                        logger.debug("brain_search: ES unavailable, falling back to FTS5")

                docs = storage.documents.search_fts(query, limit=max_results,
                                                     visible_scopes=scopes,
                                                     app_id=app_id)
                if not docs:
                    return f"No documents found matching: '{query}'"
                instance._record_sources([{
                    "document_id": getattr(d, "id", None),
                    "title": d.title,
                    "type": d.document_type.value if hasattr(d.document_type, "value") else str(d.document_type),
                    "source_path": getattr(d, "source_path", None),
                    "snippet": (d.content or "")[:200].replace("\n", " "),
                } for d in docs])
                lines = [f"Found {len(docs)} result(s):"]
                for i, doc in enumerate(docs, 1):
                    lines.append(f"  {i}. {doc.title} ({doc.document_type.value}, {doc.token_count} tokens)")
                    snippet = doc.content[:150].replace('\n', ' ')
                    lines.append(f"     {snippet}...")
                return "\n".join(lines)
            except Exception as e:
                return f"✗ Search failed: {str(e)}"

        @mcp_server.tool()
        def brain_stats() -> str:
            """Get brain system statistics.

            Shows document count (scoped to your docs + shared), graph size,
            storage sizes, and index status.
            """
            if not storage:
                return "Brain storage not initialized."
            try:
                search_user = _get_user_id_for_search()
                scopes = _get_visible_scopes()
                app_id = _get_app_id()
                doc_stats = storage.documents.count_documents(visible_scopes=scopes, app_id=app_id)
                global_stats = storage.documents.count_documents()
                stats = storage.get_system_stats()
                lines = ["Brain Statistics:"]

                lines.append(f"  Your documents: {doc_stats.get('total_documents', '?')}")
                lines.append(f"  Your total tokens: {doc_stats.get('total_tokens', 0):,}")
                by_type = doc_stats.get('by_type', {})
                if by_type:
                    lines.append(f"  By type: {', '.join(f'{k}={v}' for k,v in by_type.items())}")

                lines.append(f"  Total documents (all users): {global_stats.get('total_documents', '?')}")
                by_owner = global_stats.get('by_owner', {})
                if by_owner:
                    lines.append(f"  By owner: {', '.join(f'{k}={v}' for k,v in by_owner.items())}")

                graph = stats.get("graph_store", {})
                lines.append(f"  Graph: {graph.get('status', 'unknown')}")

                lexical = stats.get("lexical_store", {})
                lines.append(f"  Search: {lexical.get('status', 'unknown')}")

                ctx = stats.get("context_cache", {})
                lines.append(f"  Context bundles: {ctx.get('total_bundles', 0)}")

                cc = stats.get("content_cache", {})
                lines.append(f"  Content cache: {cc.get('total_items', 0)} items, "
                           f"{cc.get('total_mb', 0)} MB, "
                           f"hit rate: {cc.get('hit_rate_percent', 0)}%")

                ei = stats.get("entity_index", {})
                lines.append(f"  Entities: {ei.get('total_entities', 0)}, "
                           f"Relations: {ei.get('total_relations', 0)}")

                return "\n".join(lines)
            except Exception as e:
                return f"✗ Stats failed: {str(e)}"


        @mcp_server.tool()
        def brain_sync(source: str = "all") -> str:
            """Trigger incremental sync for data sources (Outlook, Jira, Slack).

            Fetches only new/updated items since last sync. Stores summaries
            permanently and caches full content for 24 hours.

            Args:
                source: Which source to sync: 'all', 'outlook', 'jira', or 'slack'
            """
            if not sync_worker:
                return "Sync worker not initialized."
            try:
                if source == "all":
                    reports = _run_async(sync_worker.sync_all())
                else:
                    report = _run_async(sync_worker.sync_source(source))
                    reports = [report] if report else []

                if not reports:
                    return f"No configured sources found for '{source}'."

                lines = ["Sync complete:"]
                for r in reports:
                    status = "✓" if not r.errors else "⚠"
                    lines.append(
                        f"  {status} {r.source_type}: "
                        f"{r.items_new} new, {r.items_updated} updated "
                        f"({r.duration_ms}ms)"
                    )
                    for err in r.errors[:3]:
                        lines.append(f"    ✗ {err}")
                return "\n".join(lines)
            except Exception as e:
                return f"✗ Sync failed: {str(e)}"

        @mcp_server.tool()
        def brain_sync_status() -> str:
            """Show sync status for all data sources.

            Displays last sync time, items synced, and errors per source.
            Also shows content cache and entity index statistics.
            """
            if not sync_worker:
                return "Sync worker not initialized."
            try:
                status = sync_worker.get_sync_status()
                lines = [f"Configured sources: {', '.join(status['configured_sources']) or 'none'}"]

                for src, info in status.get('sources', {}).items():
                    configured = '✓' if info['configured'] else '✗'
                    lines.append(f"\n  {configured} {src}:")
                    if info['configured']:
                        lines.append(f"    Last sync: {info['last_sync_at'] or 'never'}")
                        lines.append(f"    Total items: {info['total_items_synced']}")
                        lines.append(f"    Last batch: {info['last_sync_items']} items, "
                                   f"{info['last_sync_duration_ms']}ms")
                        if info['last_sync_errors']:
                            lines.append(f"    Errors: {info['last_sync_errors']}")

                cc = status.get('content_cache', {})
                lines.append(f"\n  Content cache: {cc.get('total_items', 0)} items, "
                           f"{cc.get('total_mb', 0)} MB")
                lines.append(f"    Hit rate: {cc.get('hit_rate_percent', 0)}% "
                           f"({cc.get('hits', 0)} hits / {cc.get('misses', 0)} misses)")

                ei = status.get('entity_index', {})
                lines.append(f"\n  Entity index: {ei.get('total_entities', 0)} entities, "
                           f"{ei.get('total_relations', 0)} relations")

                return "\n".join(lines)
            except Exception as e:
                return f"✗ Status failed: {str(e)}"

        @mcp_server.tool()
        def brain_fetch(source_uri: str) -> str:
            """Fetch full content for a specific source item on-demand.

            Checks content cache first. On cache miss, fetches from source API.

            Args:
                source_uri: Source URI (e.g., 'jira://PROJ-1234', 'outlook://msgid')
            """
            if not sync_worker:
                return "Sync worker not initialized."
            try:
                content = _run_async(sync_worker.fetch_full_content(source_uri))
                if content:
                    if len(content) > 2000:
                        return content[:2000] + f"\n\n... (truncated, full size: {len(content)} chars)"
                    return content
                return f"Could not fetch content for '{source_uri}'."
            except Exception as e:
                return f"✗ Fetch failed: {str(e)}"
