from __future__ import annotations

"""
Agent Brain — Answer Synthesizer
====================================
The final stage of the brain. Takes results from multiple
retrieval agents, ranks them by confidence, and uses the LLM
to produce a single coherent, cited answer.

Flow:
  AgentResult[] → rank by confidence → build context → LLM → BrainResponse
"""

import logging

from brain.models import BrainResponse, QueryPlan
from brain.retrieval.base import AgentResult

logger = logging.getLogger(__name__)


SYNTHESIS_SYSTEM_PROMPT = """Synthesize findings from corporate knowledge agents into ONE coherent answer.
Rules: Cite sources (agent name, doc title, file path). When a claim comes from a specific document excerpt, quote the key phrase ("...") so the reader can verify it. Note conflicts. Don't hallucinate — only state what the findings support. Be concise. Structure multi-part answers with sections."""


SYNTHESIS_USER_TEMPLATE = """ORIGINAL QUESTION:
{question}

{conversation_block}AGENT FINDINGS:
{agent_findings}

Synthesize these findings into a single, comprehensive answer with citations."""


class Synthesizer:
    """
    Merges results from multiple retrieval agents into a final BrainResponse.
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    async def synthesize(
        self,
        question: str,
        agent_results: list[AgentResult],
        query_plan: QueryPlan | None = None,
        conversation_context: str = "",
    ) -> BrainResponse:
        """
        Synthesize multiple agent results into a final answer.

        Args:
            question: The original user question
            agent_results: Results from all invoked agents
            query_plan: The routing plan (for metadata)
            conversation_context: Formatted conversation history for multi-turn context

        Returns:
            BrainResponse with synthesized answer, sources, and confidence
        """
        valid_results = [r for r in agent_results if not r.failed and r.has_content]
        valid_results.sort(key=lambda r: r.confidence, reverse=True)

        if not valid_results:
            return self._empty_response(question, agent_results, query_plan)

        if len(valid_results) == 1:
            return self._single_agent_response(valid_results[0], agent_results, query_plan)

        return await self._multi_agent_synthesis(question, valid_results, agent_results, query_plan, conversation_context)

    async def _multi_agent_synthesis(
        self,
        question: str,
        valid_results: list[AgentResult],
        all_results: list[AgentResult],
        query_plan: QueryPlan | None,
        conversation_context: str = "",
    ) -> BrainResponse:
        """Use the LLM to synthesize findings from multiple agents."""
        try:
            from core.injection_guard import guard_retrieved_content
            findings_parts = []
            for result in valid_results[:2]:
                findings_parts.append(f"--- {result.agent_name.upper()} (confidence: {result.confidence:.0%}) ---")
                content = result.content
                if len(content) > 800:
                    content = content[:800] + "\n[...truncated]"
                content = guard_retrieved_content(result.agent_name, content)
                findings_parts.append(content)
                findings_parts.append("")

            agent_findings = "\n".join(findings_parts)

            conversation_block = ""
            if conversation_context:
                conversation_block = f"{conversation_context}\n\n"

            user_prompt = SYNTHESIS_USER_TEMPLATE.format(
                question=question,
                conversation_block=conversation_block,
                agent_findings=agent_findings,
            )

            answer = await self.llm.generate_text(
                system_prompt=SYNTHESIS_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=2048,
            )

            sources = self._collect_sources(valid_results)

            confidence = self._calculate_confidence(valid_results)

            logger.info("synthesizer.multi_agent: agents=%s confidence=%s", [r.agent_name for r in valid_results], round(confidence, 2))

            return BrainResponse(
                answer=answer,
                sources=sources,
                confidence=confidence,
                agents_used=[r.agent_name for r in valid_results],
                query_plan=query_plan,
            )

        except Exception as e:
            logger.error("synthesizer.failed: error=%s", str(e))
            return self._fallback_response(question, valid_results, all_results, query_plan)

    def _single_agent_response(
        self,
        result: AgentResult,
        all_results: list[AgentResult],
        query_plan: QueryPlan | None,
    ) -> BrainResponse:
        """When only one agent returned data, use it directly (no synthesis needed)."""
        failed = [r for r in all_results if r.failed]
        suffix = ""
        if failed:
            suffix = f"\n\n_Note: {len(failed)} agent(s) were unavailable ({', '.join(r.agent_name for r in failed)})._"

        return BrainResponse(
            answer=result.content + suffix,
            sources=result.sources,
            confidence=result.confidence,
            agents_used=[result.agent_name],
            query_plan=query_plan,
        )

    def _empty_response(
        self,
        question: str,
        all_results: list[AgentResult],
        query_plan: QueryPlan | None,
    ) -> BrainResponse:
        """When no agents returned useful data."""
        errors = [r.error for r in all_results if r.failed and r.error]
        error_detail = ""
        if errors:
            error_detail = " Errors encountered: " + "; ".join(errors)

        return BrainResponse(
            answer=f"I could not find relevant information to answer: \"{question}\".{error_detail}\n\n"
                   f"Suggestions:\n"
                   f"- Try rephrasing your question with more specific terms\n"
                   f"- Ensure relevant documents have been ingested into the system\n"
                   f"- Check if the required services (Neo4j, Elasticsearch) are running",
            sources=[],
            confidence=0.0,
            agents_used=[],
            query_plan=query_plan,
        )

    def _fallback_response(
        self,
        question: str,
        valid_results: list[AgentResult],
        all_results: list[AgentResult],
        query_plan: QueryPlan | None,
    ) -> BrainResponse:
        """Fallback when the synthesis LLM call fails — concatenate results."""
        parts = [f"**Results for:** {question}\n"]
        for result in valid_results:
            parts.append(f"### From {result.agent_name} (confidence: {result.confidence:.0%})")
            parts.append(result.content)
            parts.append("")

        return BrainResponse(
            answer="\n".join(parts),
            sources=self._collect_sources(valid_results),
            confidence=self._calculate_confidence(valid_results),
            agents_used=[r.agent_name for r in valid_results],
            query_plan=query_plan,
        )

    @staticmethod
    def _collect_sources(results: list[AgentResult]) -> list[dict]:
        """Aggregate and deduplicate sources from multiple agents."""
        sources = []
        seen_ids = set()
        for result in results:
            for source in result.sources:
                doc_id = source.get("document_id", "")
                if doc_id and doc_id in seen_ids:
                    continue
                if doc_id:
                    seen_ids.add(doc_id)
                source_with_agent = {**source, "from_agent": result.agent_name}
                sources.append(source_with_agent)
        return sources

    @staticmethod
    def _calculate_confidence(results: list[AgentResult]) -> float:
        """Calculate weighted average confidence across agents."""
        if not results:
            return 0.0
        total_weight = sum(r.confidence for r in results)
        if total_weight == 0:
            return 0.0
        weighted = sum(r.confidence * r.confidence for r in results)
        return min(weighted / total_weight, 1.0)
