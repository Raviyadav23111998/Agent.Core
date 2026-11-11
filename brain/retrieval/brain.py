from __future__ import annotations

"""
Agent Brain — Brain Orchestrator
====================================
The single entry point for asking the brain a question.
Wires: Router → Agents → Synthesizer into one ask() call.

Usage:
    brain = Brain(storage, llm)
    response = await brain.ask("Who emailed Sarah about the auth bug?")
"""

import os
import asyncio
import hashlib
from typing import Optional
import logging

from brain.models import BrainResponse, QueryIntent
from brain.storage.manager import StorageManager
from state.memory_store import MemoryStore
from core.tenant import TENANT_SCOPE
from brain.retrieval.router import QueryRouter
from brain.retrieval.graph_agent import GraphAgent
from brain.retrieval.search_agent import SearchAgent
from brain.retrieval.context_agent import ContextAgent
from brain.retrieval.synthesizer import Synthesizer
from brain.retrieval.base import AgentResult
from brain.storage.answer_cache import AnswerCache
from brain.retrieval.conversation import ConversationManager

try:
    from code_review_graph.graph import GraphStore
    _CRG_AVAILABLE = True
except ImportError:
    GraphStore = None
    _CRG_AVAILABLE = False

logger = logging.getLogger(__name__)


class Brain:
    """
    The Centralized Company Brain.
    Single ask() method — question in, answer out.
    """

    def __init__(self, storage: StorageManager, llm_client, db=None):
        self.storage = storage
        self.llm = llm_client

        _db = db or getattr(storage.documents, '_db', None)

        try:
            self.answer_cache = AnswerCache(db=_db)
            cache_stats = self.answer_cache.get_stats()
            logger.info("brain.answer_cache_loaded: entries=%s total_hits=%s", cache_stats["total_entries"], cache_stats["total_cache_hits"])
        except Exception as e:
            logger.warning("brain.answer_cache_failed: error=%s", str(e))
            self.answer_cache = None

        code_graph = None
        crg_db_path = os.environ.get("CRG_DB_PATH", ".ag_data/crg.db")
        if _CRG_AVAILABLE and os.path.exists(crg_db_path):
            try:
                code_graph = GraphStore(crg_db_path)
                stats = code_graph.get_stats()
                logger.info("brain.code_graph_loaded: nodes=%s edges=%s files=%s", stats.total_nodes, stats.total_edges, stats.files_count)
            except Exception as e:
                logger.warning("brain.code_graph_failed: error=%s", str(e))
                code_graph = None
        else:
            logger.info("brain.code_graph_unavailable: path=%s", crg_db_path)

        self.memory_store = None
        try:
            self.memory_store = MemoryStore(db=_db)
            mem_stats = self.memory_store.get_stats()
            logger.info("brain.memory_loaded: memories=%s", mem_stats["total_memories"])
        except Exception as e:
            logger.warning("brain.memory_failed: error=%s", str(e))

        self.router = QueryRouter(llm_client=llm_client, context_cache=storage.context)
        self.graph_agent = GraphAgent(graph_store=storage.graph, llm_client=llm_client)
        self.search_agent = SearchAgent(
            lexical_store=storage.lexical, document_store=storage.documents,
            embeddings=getattr(storage, "embeddings", None))
        self.context_agent = ContextAgent(
            context_cache=storage.context,
            document_store=storage.documents,
            llm_client=llm_client,
            code_graph=code_graph,
            memory_store=self.memory_store,
            answer_cache=self.answer_cache,
        )
        self.synthesizer = Synthesizer(llm_client=llm_client)

        self.conversation = ConversationManager()

    _SIMPLE_QUERY_PATTERNS = (
        "find ", "search ", "who ", "list ", "show ", "get ",
        "where is", "what is", "latest ", "last ",
    )

    def _is_simple_query(self, question: str) -> bool:
        """Detect questions that don't need LLM routing — keyword heuristics suffice."""
        q = question.lower().strip()
        if len(q.split()) <= 8 and any(q.startswith(p) for p in self._SIMPLE_QUERY_PATTERNS):
            return True
        return False

    async def ask(
        self,
        question: str,
        default_bundle: Optional[str] = None,
        use_smart_routing: bool = True,
        owner_id: Optional[str] = None,
        visible_scopes: list = None,
        app_id: int = None,
    ) -> BrainResponse:
        """
        Ask the brain a question. This is the main API.

        Args:
            question: Natural language question
            default_bundle: Default context bundle for the Context Agent
            use_smart_routing: If True, use LLM routing. If False, use keyword heuristics.
            owner_id: Legacy single owner scope. Prefer visible_scopes.
            visible_scopes: List of owner_ids the user can see (Teams model).
                            e.g. ['user_12345', 'group_-1001234', '__tenant__']
                            None = no scoping (sees all docs).

        Returns:
            BrainResponse with synthesized answer, sources, and confidence
        """
        logger.info("brain.ask: question=%s", question[:80])

        if self.answer_cache:
            cached_answer = self.answer_cache.lookup(
                question, bundle_name=default_bundle or "",
                visible_scopes=visible_scopes,
            )
            if cached_answer:
                logger.info("brain.cache_hit: question=%s", question[:50])
                self.conversation.add_turn(question, cached_answer, agents_used=["answer_cache"])
                return BrainResponse(
                    answer=cached_answer,
                    sources=[{"cache": "answer_cache", "type": "exact_or_fuzzy"}],
                    confidence=0.85,
                    agents_used=["answer_cache"],
                    query_plan=None,
                )

        effective_question = self.conversation.enrich_question(question)
        if effective_question != question:
            logger.info("brain.follow_up_detected: original=%s enriched_len=%s", question[:50], len(effective_question))

        conversation_context = self.conversation.get_context()

        plan = self.router.route_simple(effective_question)
        if use_smart_routing and QueryIntent.HYBRID in plan.intents:
            plan = await self.router.route(effective_question, default_bundle=default_bundle)

        logger.info("brain.plan: intents=%s", [i.value for i in plan.intents])

        core_intents = {i for i in plan.intents if i != QueryIntent.HYBRID}

        if QueryIntent.CONTEXT in core_intents:
            has_bundle = bool(plan.context_bundle_id or default_bundle)
            if not has_bundle:
                logger.warning("brain.context_fallback: no bundle available, switching to search")
                core_intents.discard(QueryIntent.CONTEXT)
                core_intents.add(QueryIntent.SEARCH)
                if not plan.search_query and plan.context_query:
                    plan.search_query = plan.context_query

        tasks = []
        task_names = []

        if QueryIntent.GRAPH in core_intents:
            tasks.append(self.graph_agent.query(plan.graph_query or question))
            task_names.append("graph")

        if QueryIntent.SEARCH in core_intents:
            tasks.append(self.search_agent.query(
                plan.search_query or question, owner_id=owner_id,
                visible_scopes=visible_scopes, app_id=app_id,
                semantic_query=plan.semantic_query or question,
            ))
            task_names.append("search")

        if QueryIntent.CONTEXT in core_intents:
            kwargs = {}
            if plan.context_bundle_id:
                kwargs["bundle_name"] = plan.context_bundle_id
            elif default_bundle:
                kwargs["bundle_name"] = default_bundle
            kwargs["visible_scopes"] = visible_scopes
            tasks.append(self.context_agent.query(plan.context_query or question, **kwargs))
            task_names.append("context")

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        agent_results: list[AgentResult] = []
        agent_errors: list[str] = []
        for name, result in zip(task_names, raw_results):
            if isinstance(result, Exception):
                logger.error("brain.agent_failed: agent=%s error=%s", name, str(result))
                agent_errors.append(f"{name}: {str(result)[:100]}")
            else:
                agent_results.append(result)

        if not agent_results and agent_errors:
            error_summary = "; ".join(agent_errors)
            agent_results.append(AgentResult(
                agent_name="error_reporter",
                content=f"All retrieval agents encountered errors: {error_summary}",
                confidence=0.0,
                error=error_summary,
            ))

        response = await self.synthesizer.synthesize(
            question=question,
            agent_results=agent_results,
            query_plan=plan,
            conversation_context=conversation_context,
        )

        if self.answer_cache and response.confidence >= 0.1:
            try:
                est_input_tokens = len(question) // 4 + sum(
                    len(r.content) // 4 for r in agent_results if r.has_content
                )
                ttl = 7200 if response.confidence >= 0.5 else 1800
                self.answer_cache.store(
                    question=question,
                    answer=response.answer,
                    bundle_name=default_bundle or "",
                    model=getattr(self.llm, '_client', {}).model if hasattr(getattr(self.llm, '_client', None) or object(), 'model') else "unknown",
                    input_tokens_saved=est_input_tokens,
                    ttl_seconds=ttl,
                    visible_scopes=visible_scopes,
                )
            except Exception as e:
                logger.warning("brain.cache_store_failed: error=%s", str(e))

        if self.memory_store and response.confidence >= 0.5:
            try:
                self._extract_and_remember(question, response, owner_id=owner_id)
            except Exception as e:
                logger.debug("brain.memory_store_failed: error=%s", str(e))

        logger.info("brain.response: agents=%s confidence=%s sources=%s", response.agents_used, round(response.confidence, 2), len(response.sources))

        self.conversation.add_turn(
            question=question,
            answer=response.answer,
            agents_used=response.agents_used,
        )

        return response

    def _extract_and_remember(self, question: str, response: BrainResponse,
                               owner_id: str = None):
        """Extract key facts from a successful answer and store in persistent memory."""
        if not self.memory_store or not response.agents_used:
            return

        if response.confidence >= 0.7 and len(response.answer) > 50:
            q_key = question.lower().strip()[:80]
            q_hash = hashlib.md5(question.encode()).hexdigest()[:8]
            fact_key = f"qa.{q_hash}"

            compact_answer = response.answer[:200]
            if len(response.answer) > 200:
                compact_answer += "..."

            self.memory_store.remember(
                category="qa_fact",
                key=fact_key,
                value=compact_answer,
                source=f"brain.ask: {q_key}",
                confidence=response.confidence,
                tags=["auto_learned"] + response.agents_used[:2],
                user_id=owner_id or TENANT_SCOPE,
            )
