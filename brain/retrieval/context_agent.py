from __future__ import annotations

"""
Agent Brain — Context Agent (Two-Stage Retrieval + Gemini Caching)
=====================================================================
Retrieval agent for the Long-Context Cache.

BEFORE (v1): Load ALL 71 documents (70K tokens) → send everything → wait 14s
AFTER  (v2): Stage 1 picks relevant files (2K tokens, 1s)
             Stage 2 analyzes only those files (5-10K tokens, 3s)
             Total: ~4s instead of ~14s — a 3.5x speedup

v3 UPGRADE — Gemini Context Caching:
  Large payloads (>4K tokens) are uploaded to Google's server-side cache
  on first query. Subsequent queries reuse the cache at ~75% lower cost.
  Cache TTL: 1 hour. Auto-refreshed on expiry.

Fallback: If file selection fails or question is too broad,
falls back to the full bundle (v1 behavior).
"""

import os
import re
from typing import Optional
import logging

from brain.storage.document_store import DocumentStore
from brain.storage.context_cache import ContextCache
from state.memory_store import MemoryStore
from brain.storage.answer_cache import AnswerCache
try:
    from brain.storage.gemini_cache import GeminiCacheManager
except ImportError:
    GeminiCacheManager = None
from brain.retrieval.base import BaseAgent, AgentResult

try:
    from code_review_graph.graph import GraphStore
    _CRG_AVAILABLE = True
except ImportError:
    GraphStore = None  # type: ignore
    _CRG_AVAILABLE = False

logger = logging.getLogger(__name__)


def _ambient_identity():
    """(user_id, app_id) for the turn in flight, or (CLI_USER, None).

    The same contextvar memory_mcp._resolve_app_id and
    skill_service._ambient_scope read. Kept here rather than threaded through
    the constructor because this agent is built once and reused across turns
    belonging to different applications — a constructor argument would pin it
    to whichever turn happened to create it.
    """
    try:
        from core.user_context import current_user_context
        from state.memory_store import CLI_USER
        ctx = current_user_context.get()
        if ctx is None:
            return CLI_USER, None
        app_id = getattr(ctx, "app_id", None)
        return (getattr(ctx, "user_id", CLI_USER) or CLI_USER,
                int(app_id) if app_id is not None else None)
    except (TypeError, ValueError, ImportError):
        from state.memory_store import CLI_USER
        return CLI_USER, None


FILE_SELECTION_PROMPT = """You are a file selection assistant. Given a question and a file index, identify which files are most relevant to answering the question.

RULES:
1. Return ONLY the file numbers (comma-separated). Nothing else.
2. Select 3-10 files maximum. Prefer fewer, more relevant files.
3. If the question is very broad (e.g., "what does this project do?"), return "ALL".
4. Focus on files whose path or preview content relates to the question.

Examples:
  Question: "How does authentication work?"
  Answer: 3, 7, 12, 15

  Question: "What is this entire codebase about?"
  Answer: ALL"""

CONTEXT_ANALYSIS_PROMPT = """You are a senior software engineer and knowledge analyst with access to a company's internal documents and codebase.

You have been given a CONTEXT BUNDLE containing the full text of relevant documents. Your job is to analyze this context and answer the user's question with precision.

RULES:
1. Base your answer ONLY on the provided context. Do NOT use external knowledge.
2. Cite specific files, documents, line references, or entities when possible.
3. If the context doesn't contain enough information, say so explicitly.
4. For code analysis: mention specific file names, class names, function names, and line-level details.
5. For policy questions: quote relevant policy sections directly.
6. For email/communication questions: reference specific senders, dates, and subjects.
7. Be thorough but concise. Prioritize the most relevant findings.
8. Structure your answer with clear sections if the question has multiple parts."""


MAX_SELECTED_FILES = 10

SKIP_SELECTION_THRESHOLD = 8


class ContextAgent(BaseAgent):
    """
    Retrieval agent with Two-Stage Retrieval.

    Stage 1: Send lightweight file index (~2K tokens) to Flash model
             → LLM returns which files are relevant
    Stage 2: Load only those files → send to Pro model for deep analysis

    Falls back to full-bundle loading for very broad questions or small bundles.
    """

    AGENT_NAME = "context_agent"

    def __init__(
        self,
        context_cache: ContextCache,
        document_store: DocumentStore,
        llm_client = None,
        code_graph: Optional["GraphStore"] = None,
        memory_store: Optional[MemoryStore] = None,
        answer_cache: Optional[AnswerCache] = None,
        credential_registry = None,
    ):
        self.cache = context_cache
        self.doc_store = document_store
        self.llm = llm_client
        self.code_graph = code_graph
        self.memory_store = memory_store
        self.answer_cache = answer_cache

        self._gemini_cache = None
        google_api_key = ""
        use_gemini_cache = False
        self._gemini_cache_ttl = 3600
        
        self._credential_cache_model = None

        if credential_registry:
            try:
                cred = credential_registry.resolve("google_gemini")
                google_api_key = cred.secret
                self._gemini_cache_ttl = int(cred.config.get("cache_ttl", 3600))
                self._credential_cache_model = cred.config.get("cache_model")
                use_gemini_cache = bool(google_api_key)
            except Exception:
                pass

        if not google_api_key:
            google_api_key = os.environ.get("GOOGLE_API_KEY", "")
            use_gemini_cache = os.environ.get("USE_GEMINI_CACHE", "").lower() == "true"
            self._gemini_cache_ttl = int(os.environ.get("GEMINI_CACHE_TTL", "3600"))
            self._credential_cache_model = os.environ.get("MODEL_CONTEXT")
            
    @property
    def cache_model(self):
        from core.config import app_config
        return self._credential_cache_model or app_config.get("llm.model")

    async def query(
        self,
        question: str,
        **kwargs,
    ) -> AgentResult:
        """
        Two-Stage Retrieval query.

        Stage 1: File selection (fast, ~1s with Flash)
        Stage 2: Deep analysis (focused, ~3s with Pro on fewer tokens)
        Fallback: Full bundle if selection fails or question is too broad
        """
        bundle_id = kwargs.get("bundle_id")
        bundle_name = kwargs.get("bundle_name")
        max_tokens = kwargs.get("max_tokens", 4096)

        try:
            bundle = None
            if bundle_id:
                bundle = self.cache.get_bundle(bundle_id)
            elif bundle_name:
                bundle = self.cache.get_bundle_by_name(bundle_name)

            if not bundle:
                return self._error_result(
                    f"No context bundle found. Specify bundle_id or bundle_name. "
                    f"Available bundles: {[b.name for b in self.cache.list_bundles()]}"
                )

            if not bundle.document_ids:
                return AgentResult(
                    agent_name=self.AGENT_NAME,
                    content="The context bundle is empty — no documents loaded.",
                    confidence=0.1,
                )

            documents = self.doc_store.get_documents_by_ids(bundle.document_ids)
            if not documents:
                return self._error_result("Failed to load documents from the bundle")

            memory_context = ""
            if self.memory_store:
                # This class holds no identity of its own, so it took whatever
                # recall() defaults to — CLI_USER, every application. The result
                # is pasted into the prompt as "REMEMBERED FACTS", which makes
                # an unscoped read here indistinguishable from a deliberate one.
                #
                # The ambient contextvar is the same source memory_mcp and
                # skill_service read; two answers to "whose turn is this?" is
                # how one path ends up scoped and its neighbour does not.
                _uid, _app = _ambient_identity()
                memories = self.memory_store.recall(
                    question, limit=5, user_id=_uid, app_id=_app)
                if memories:
                    memory_lines = ["REMEMBERED FACTS:"]
                    for mem in memories:
                        memory_lines.append(f"• {mem.key}: {mem.value}")
                    memory_context = "\n".join(memory_lines)
                    logger.info("context_agent.memory_recall: memories_found=%s", len(memories))

            if len(documents) <= SKIP_SELECTION_THRESHOLD:
                logger.info("context_agent.small_bundle_direct: bundle=%s documents=%s", bundle.name, len(documents))
                return await self._full_bundle_query(
                    question, bundle, documents, max_tokens
                )

            selected_indices = None
            if self.code_graph:
                selected_indices = self._graph_select_files(question, documents)

            if selected_indices is None:
                selected_indices = await self._select_relevant_files(
                    question, bundle, documents
                )

            if selected_indices is None:
                logger.info("context_agent.fallback_full_bundle: bundle=%s", bundle.name)
                return await self._full_bundle_query(
                    question, bundle, documents, max_tokens
                )

            return await self._focused_query(
                question, bundle, documents, selected_indices, max_tokens,
                memory_context=memory_context,
            )

        except Exception as e:
            logger.error("context_agent.query_failed: question=%s error=%s", question[:50], str(e))
            return self._error_result(str(e))


    def _graph_select_files(
        self,
        question: str,
        documents: list,
    ) -> Optional[list[int]]:
        """
        Use the vendored code-review-graph's GraphStore to find relevant files.
        This is FREE — pure graph query, no LLM call needed.

        Uses search_nodes() to find symbols matching the question keywords,
        then maps the resulting file paths back to bundle document indices.

        Returns:
            List of 1-indexed file numbers, or None if no graph hits.
        """
        try:
            graph_results = self.code_graph.search_nodes(question, limit=30)

            if not graph_results:
                keywords = [w for w in question.split() if len(w) > 3]
                for kw in keywords[:5]:
                    graph_results.extend(self.code_graph.search_nodes(kw, limit=10))

            if not graph_results:
                logger.info("context_agent.graph_miss: question=%s", question[:60])
                return None

            matching_paths = list(set(r.file_path for r in graph_results))

            indices = []
            for i, doc in enumerate(documents, 1):
                doc_path = getattr(doc, 'source_path', '') or ''
                doc_title = getattr(doc, 'title', '') or ''
                for path in matching_paths:
                    if doc_path and (
                        doc_path in path or path.endswith(doc_path)
                        or doc_title in path
                    ):
                        if i not in indices:
                            indices.append(i)
                        break

            if not indices:
                logger.info("context_agent.graph_no_bundle_match",
                            graph_files=len(matching_paths), bundle_docs=len(documents))
                return None

            indices = indices[:MAX_SELECTED_FILES]

            logger.info("context_agent.graph_hit: files=%s count=%s total=%s graph_nodes=%s cost=%s", indices, len(indices), len(documents), len(graph_results), "$0.00")
            return indices

        except Exception as e:
            logger.warning("context_agent.graph_query_failed: error=%s", str(e))
            return None


    async def _select_relevant_files(
        self,
        question: str,
        bundle,
        documents: list,
    ) -> Optional[list[int]]:
        """
        Stage 1: Ask a fast model which files are relevant.

        Returns:
            List of 1-indexed file numbers, or None if "ALL" / selection failed
        """
        file_index = self.cache.build_file_index(bundle.id, documents)

        logger.info("context_agent.stage1_file_selection: bundle=%s index_chars=%s index_tokens_est=%s", bundle.name, len(file_index), len(file_index) // 4)

        try:
            response = await self.llm.generate_text(
                system_prompt=FILE_SELECTION_PROMPT,
                user_prompt=f"Question: {question}\n\nFile Index:\n{file_index}",
                temperature=0.0,
                max_tokens=256,
                purpose="routing",
            )

            return self._parse_file_selection(response, len(documents))

        except Exception as e:
            logger.warning("context_agent.stage1_failed: error=%s", str(e))
            return None

    def _parse_file_selection(self, response: str, total_files: int) -> Optional[list[int]]:
        """Parse the LLM's file selection response into a list of indices."""
        cleaned = response.strip().upper()

        if "ALL" in cleaned:
            logger.info("context_agent.stage1_selected_all")
            return None

        numbers = re.findall(r'\d+', cleaned)

        if not numbers:
            logger.warning("context_agent.stage1_no_numbers: response=%s", response[:100])
            return None

        indices = []
        for n in numbers:
            idx = int(n)
            if 1 <= idx <= total_files and idx not in indices:
                indices.append(idx)

        indices = indices[:MAX_SELECTED_FILES]

        if not indices:
            return None

        logger.info("context_agent.stage1_selected: files=%s count=%s total=%s", indices, len(indices), total_files)

        return indices


    async def _focused_query(
        self,
        question: str,
        bundle,
        documents: list,
        selected_indices: list[int],
        max_tokens: int,
        memory_context: str = "",
    ) -> AgentResult:
        """
        Stage 2: Load only the selected files and do deep analysis.

        This sends ~5-10K tokens instead of 70K — a 7-14x reduction.
        If Gemini Caching is enabled, the payload is cached server-side
        for ~75% additional cost reduction on subsequent queries.
        """
        if self.answer_cache:
            cached_answer = self.answer_cache.lookup(question, bundle_name=bundle.name)
            if cached_answer:
                logger.info("context_agent.answer_cache_hit: question=%s bundle=%s", question[:60], bundle.name)
                selected_docs = [documents[i - 1] for i in selected_indices if 1 <= i <= len(documents)]
                sources = [{"bundle_name": bundle.name, "retrieval": "answer_cache_hit"}]
                for doc in selected_docs:
                    sources.append({"document_id": doc.id, "title": doc.title, "type": doc.document_type.value, "source_path": doc.source_path})
                return AgentResult(agent_name=self.AGENT_NAME, content=cached_answer, sources=sources, confidence=0.90)

        focused_payload = self.cache.build_selective_payload(
            bundle.id, documents, selected_indices
        )

        from core.injection_guard import guard_retrieved_content
        focused_payload = guard_retrieved_content("context_bundle", focused_payload)

        full_context = focused_payload
        if memory_context:
            full_context = f"{memory_context}\n\n{focused_payload}"

        payload_tokens = len(full_context) // 4

        logger.info("context_agent.stage2_focused_analysis: bundle=%s selected_files=%s payload_chars=%s payload_tokens_est=%s full_bundle_tokens=%s reduction=%s", bundle.name, len(selected_indices), len(full_context), payload_tokens, bundle.total_tokens, f"{bundle.total_tokens // max(1, payload_tokens)}x")

        answer = await self._generate_with_optional_cache(
            cache_key=f"{bundle.id}_focused",
            context=full_context,
            question=question,
            max_tokens=max_tokens,
        )

        if self.answer_cache and answer:
            self.answer_cache.store(
                question=question,
                answer=answer,
                bundle_name=bundle.name,
                input_tokens_saved=payload_tokens,
            )

        selected_docs = [documents[i - 1] for i in selected_indices if 1 <= i <= len(documents)]
        sources = [
            {
                "bundle_name": bundle.name,
                "bundle_id": bundle.id,
                "retrieval": "two_stage_cached",
                "selected_files": len(selected_docs),
                "total_files": len(documents),
                "payload_tokens": payload_tokens,
                "full_bundle_tokens": bundle.total_tokens,
            },
        ]
        for doc in selected_docs:
            sources.append({
                "document_id": doc.id,
                "title": doc.title,
                "type": doc.document_type.value,
                "source_path": doc.source_path,
            })

        return AgentResult(
            agent_name=self.AGENT_NAME,
            content=answer,
            sources=sources,
            confidence=0.85,
            raw_data={
                "retrieval": "two_stage_cached",
                "bundle": bundle.name,
                "selected": len(selected_docs),
                "total": len(documents),
                "payload_chars": len(full_context),
                "full_bundle_chars": bundle.total_tokens * 4,
            },
        )


    async def _full_bundle_query(
        self,
        question: str,
        bundle,
        documents: list,
        max_tokens: int,
    ) -> AgentResult:
        """
        Full bundle query — loads ALL documents into context.
        Used when the question is too broad or the bundle is small.

        v3: Uses Gemini Context Caching when available. The full bundle
        is uploaded to Google's cache on the first query and reused for
        subsequent queries at ~75% lower cost.
        """
        context_payload = self.cache.build_context_payload(bundle.id, documents)

        logger.info("context_agent.full_bundle_query: bundle=%s documents=%s tokens=%s payload_chars=%s", bundle.name, len(documents), bundle.total_tokens, len(context_payload))

        answer = await self._generate_with_optional_cache(
            cache_key=f"{bundle.id}_full",
            context=context_payload,
            question=question,
            max_tokens=max_tokens,
        )

        sources = [
            {
                "bundle_name": bundle.name,
                "bundle_id": bundle.id,
                "retrieval": "full_bundle_cached",
                "documents_loaded": len(documents),
                "total_tokens": bundle.total_tokens,
            },
        ]
        for doc in documents[:10]:
            sources.append({
                "document_id": doc.id,
                "title": doc.title,
                "type": doc.document_type.value,
                "source_path": doc.source_path,
            })

        return AgentResult(
            agent_name=self.AGENT_NAME,
            content=answer,
            sources=sources,
            confidence=0.85,
            raw_data={
                "retrieval": "full_bundle_cached",
                "bundle": bundle.name,
                "documents": len(documents),
                "context_chars": len(context_payload),
                "tokens_used": bundle.total_tokens,
            },
        )


    async def _generate_with_optional_cache(
        self,
        cache_key: str,
        context: str,
        question: str,
        max_tokens: int,
    ) -> str:
        """
        Generate a response, using Gemini's server-side cache if available.

        Flow:
          1. If GeminiCacheManager is active, try to get/create a cache for this payload
          2. If cache exists, use generate_with_cache() → ~75% cheaper input tokens
          3. If caching fails or is unavailable, fall back to standard LLM call
        """
        if self._gemini_cache:
            try:
                cache_model = self.cache_model

                cache_name = self._gemini_cache.get_or_create_cache(
                    bundle_id=cache_key,
                    content=context,
                    model=cache_model,
                    system_instruction=CONTEXT_ANALYSIS_PROMPT,
                    ttl_seconds=self._gemini_cache_ttl,
                )

                if cache_name:
                    logger.info("context_agent.using_gemini_cache: cache_key=%s cache_name=%s", cache_key, cache_name)
                    answer = self._gemini_cache.generate_with_cache(
                        cache_name=cache_name,
                        model=cache_model,
                        question=question,
                        system_prompt=CONTEXT_ANALYSIS_PROMPT,
                        max_tokens=max_tokens,
                    )
                    return answer

            except Exception as e:
                logger.warning("context_agent.gemini_cache_fallback: cache_key=%s error=%s", cache_key, str(e))

        return await self.llm.generate_text(
            system_prompt=CONTEXT_ANALYSIS_PROMPT,
            user_prompt=question,
            context=context,
            temperature=0.2,
            max_tokens=max_tokens,
            purpose="context",
        )

    async def quick_scan(
        self,
        question: str,
        document_ids: list[str],
        max_tokens: int = 2048,
    ) -> AgentResult:
        """
        Quick context scan — load specific documents (not a full bundle)
        into the LLM and ask a question. Cheaper than a full bundle query.
        """
        try:
            documents = self.doc_store.get_documents_by_ids(document_ids)
            if not documents:
                return self._error_result("No documents found for the given IDs")

            parts = []
            for doc in documents:
                parts.append(f"--- {doc.title} ({doc.document_type.value}) ---")
                parts.append(doc.content)
                parts.append("")

            from core.injection_guard import guard_retrieved_content
            context = guard_retrieved_content("quick_scan", "\n".join(parts))

            answer = await self.llm.generate_text(
                system_prompt=CONTEXT_ANALYSIS_PROMPT,
                user_prompt=question,
                context=context,
                temperature=0.2,
                max_tokens=max_tokens,
                purpose="context",
            )

            return AgentResult(
                agent_name=self.AGENT_NAME,
                content=answer,
                sources=[{"document_id": d.id, "title": d.title} for d in documents],
                confidence=0.75,
                raw_data={"documents": len(documents), "context_chars": len(context)},
            )

        except Exception as e:
            return self._error_result(str(e))
