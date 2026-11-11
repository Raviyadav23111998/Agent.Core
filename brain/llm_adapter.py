"""
Agent Brain — LLM Adapter
========================================
Bridges AG's GenericLLMClient to the Brain's expected LLMClient interface.

The Brain modules (router, graph_agent, synthesizer, extractor) expect methods:
  - chat(messages, purpose, temperature, max_tokens, response_format)
  - extract_json(system_prompt, user_prompt)
  - generate_text(system_prompt, user_prompt, context, temperature, max_tokens)

This adapter wraps AG's existing requests-based client to provide those methods.
"""

import json
import re
import logging

logger = logging.getLogger(__name__)


class BrainLLMAdapter:
    """Wraps AG's GenericLLMClient to match the Brain's LLMClient interface.
    
    The Brain was built for LiteLLM. AG uses a raw `requests`-based client.
    This adapter translates between the two interfaces without adding dependencies.
    """

    def __init__(self, ag_client):
        """
        Args:
            ag_client: The GenericLLMClient instance from client/cli.py
        """
        self._client = ag_client

    async def chat(self, messages, purpose="routing", temperature=0.1,
                   max_tokens=4096, response_format=None):
        """Send a chat request through the shared LLM client.

        Delegates to `client.generate()` — the SAME code path as agent-loop
        calls, so brain-internal calls inherit retries, fallback,
        Retry-After discipline, provider selection (incl. the native
        Anthropic adapter), and telemetry, instead of a duplicated
        one-shot HTTP request with none of that.

        Returns the response text string.
        """
        system = ""
        rest = []
        for m in messages or []:
            if m.get("role") == "system" and not rest:
                part = m.get("content") or ""
                system = f"{system}\n{part}".strip() if system else part
            else:
                rest.append(m)
        if not rest:
            rest = [{"role": "user", "content": ""}]

        try:
            from core.model_routing import resolve_model
            model = resolve_model(purpose, getattr(self._client, "model", None))
        except Exception:
            model = getattr(self._client, "model", None)

        resp = await self._client.generate(
            system or "You are a precise assistant.",
            rest,
            model=model,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if (resp or {}).get("error"):
            raise RuntimeError((resp or {}).get("content") or "LLM call failed")
        return (resp or {}).get("content") or ""

    async def extract_json(self, system_prompt, user_prompt, use_extraction_model=True):
        """Call the LLM and parse the response as JSON.
        
        Used for entity extraction, query routing, etc.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response_text = await self.chat(
            messages=messages,
            purpose="extraction",
            temperature=0.0,
            max_tokens=2048,
        )

        cleaned = response_text.strip()
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if json_match:
            cleaned = json_match.group(1).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error("brain_llm.json_parse_failed: %s", response_text[:200])
            return {}

    async def generate_text(self, system_prompt, user_prompt, context="",
                            temperature=0.3, max_tokens=4096, purpose="synthesis"):
        """Generate a natural language response.
        
        Used for answer synthesis, explanations, etc.
        """
        messages = [{"role": "system", "content": system_prompt}]

        if context:
            messages.append({
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{user_prompt}"
            })
        else:
            messages.append({"role": "user", "content": user_prompt})

        return await self.chat(
            messages=messages,
            purpose=purpose,
            temperature=temperature,
            max_tokens=max_tokens,
        )
