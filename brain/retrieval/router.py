from __future__ import annotations

"""
Agent Brain — Query Router
===============================
The brain's traffic controller. Takes a user's natural language
question and decides which retrieval agents to invoke.

Uses a single cheap LLM call to classify intent and produce a
QueryPlan. The plan specifies:
  - Which agents to call (graph, search, context, or hybrid)
  - Optimized sub-queries tailored for each agent

Flow:
  User Question → Router (cheap LLM) → QueryPlan → Agent Executor
"""

from typing import Optional
import logging

from brain.models import QueryPlan, QueryIntent
from brain.storage.context_cache import ContextCache
from brain import NLP_STOP_WORDS

logger = logging.getLogger(__name__)


ROUTER_SYSTEM_PROMPT = """Route a user question to the correct retrieval agent(s). Pick MINIMUM agents needed.

AGENTS:
- graph: Relational queries (who→whom, dependencies, team structure)
- search: Finding docs, error codes, terms, tech stack questions. DEFAULT if unsure.
- context: Deep code/doc analysis. ONLY if a matching context bundle is listed below. If no bundles → use search.

RULES:
- search_query MUST be short keywords, NOT full sentences (e.g. "auth JWT PostgreSQL" not "What auth does it use?")
- semantic_query MUST be ONE declarative sentence stating the topic/ideal answer (for embedding search), e.g. "The application authenticates users with JWT tokens backed by PostgreSQL."
- Return ONLY valid JSON:

{"intents":["search"],"search_query":"keywords","semantic_query":"one declarative sentence","graph_query":null,"context_query":null,"context_bundle":null,"reasoning":"one sentence"}"""


class QueryRouter:
    """
    Routes user questions to the appropriate retrieval agents.
    Uses a cheap LLM to classify intent and produce a QueryPlan.
    """

    def __init__(
        self,
        llm_client,
        context_cache: Optional[ContextCache] = None,
    ):
        self.llm = llm_client
        self.context_cache = context_cache

    async def route(self, question: str, default_bundle: Optional[str] = None) -> QueryPlan:
        """
        Analyze a user question and produce a routing plan.

        Args:
            question: The user's natural language question
            default_bundle: Default context bundle name to use if context agent is needed

        Returns:
            QueryPlan specifying which agents to call and with what queries
        """
        try:
            bundle_context = ""
            if self.context_cache:
                bundles = self.context_cache.list_bundles()
                if bundles:
                    bundle_names = [f"'{b.name}' ({b.description})" for b in bundles]
                    bundle_context = f"\n\nAVAILABLE CONTEXT BUNDLES:\n" + "\n".join(f"  - {n}" for n in bundle_names)

            user_prompt = question
            if bundle_context:
                user_prompt += bundle_context

            response = await self.llm.extract_json(
                system_prompt=ROUTER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                use_extraction_model=True,
            )

            if not response:
                logger.warning("router.empty_response: question=%s", question[:50])
                return self._fallback_plan(question, default_bundle)

            plan = self._parse_response(question, response, default_bundle)

            logger.info("router.plan_created: question=%s intents=%s reasoning=%s", question[:50], [i.value for i in plan.intents], response.get("reasoning", ""))

            return plan

        except Exception as e:
            logger.error("router.failed: question=%s error=%s", question[:50], str(e))
            return self._fallback_plan(question, default_bundle)

    def _parse_response(
        self,
        question: str,
        response: dict,
        default_bundle: Optional[str],
    ) -> QueryPlan:
        """Parse the LLM's JSON response into a QueryPlan."""
        raw_intents = response.get("intents", [])
        intents = []
        for intent_str in raw_intents:
            intent_str = intent_str.lower().strip()
            if intent_str in ("graph", "search", "context"):
                intents.append(QueryIntent(intent_str))

        if not intents:
            intents = [QueryIntent.SEARCH]

        if len(intents) > 1:
            intents.append(QueryIntent.HYBRID)

        context_bundle = response.get("context_bundle") or default_bundle

        return QueryPlan(
            original_query=question,
            intents=list(set(intents)),
            graph_query=response.get("graph_query"),
            search_query=response.get("search_query"),
            semantic_query=response.get("semantic_query"),
            context_query=response.get("context_query"),
            context_bundle_id=context_bundle,
        )

    def _fallback_plan(self, question: str, default_bundle: Optional[str] = None) -> QueryPlan:
        """
        Fallback plan when the LLM fails to classify.
        Strategy: call both search and graph (cheap), skip context (expensive).
        """
        logger.info("router.using_fallback: question=%s", question[:50])

        keywords = self._extract_keywords(question)

        return QueryPlan(
            original_query=question,
            intents=[QueryIntent.SEARCH, QueryIntent.GRAPH, QueryIntent.HYBRID],
            graph_query=question,
            search_query=keywords,
            context_query=None,
            context_bundle_id=default_bundle,
        )

    @staticmethod
    def _extract_keywords(question: str) -> str:
        """Strip stop words to produce BM25-friendly keyword queries."""
        words = question.rstrip("?!.").split()
        keywords = [w for w in words if w.lower() not in NLP_STOP_WORDS and len(w) > 1]
        return " ".join(keywords) if keywords else question

    def route_simple(self, question: str) -> QueryPlan:
        """
        Rule-based routing without LLM (zero cost, instant).
        Uses keyword heuristics as a fast alternative.
        Good for high-volume or latency-sensitive scenarios.
        """
        q = question.lower()

        intents = []
        graph_query = None
        search_query = None
        context_query = None

        relational_keywords = [
            "who ", "whom", "emailed", "authored", "approved", "reported",
            "works on", "depends on", "connected to", "related to",
            "relationship", "team", "manager", "responsible",
        ]
        if any(kw in q for kw in relational_keywords):
            intents.append(QueryIntent.GRAPH)
            graph_query = question

        search_keywords = [
            "find", "search", "where is", "error", "log", "err-",
            "documents about", "files containing", "look for",
            "specific", "exact",
        ]
        if any(kw in q for kw in search_keywords):
            intents.append(QueryIntent.SEARCH)
            search_query = question

        context_keywords = [
            "explain", "how does", "analyze", "summarize", "review",
            "bug in", "what does", "walk through", "code",
            "architecture", "design", "implementation",
        ]
        if any(kw in q for kw in context_keywords):
            intents.append(QueryIntent.CONTEXT)
            context_query = question

        if not intents:
            intents.append(QueryIntent.SEARCH)
            search_query = question

        if len(intents) > 1:
            intents.append(QueryIntent.HYBRID)

        return QueryPlan(
            original_query=question,
            intents=list(set(intents)),
            graph_query=graph_query,
            search_query=search_query,
            context_query=context_query,
        )
