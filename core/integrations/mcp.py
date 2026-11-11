"""
MCP server → normalized Integration (the second source, Phase 6).
═══════════════════════════════════════════════════════════════════
An MCP server exposes `tools/list` (each tool = name + description + JSON-Schema
`inputSchema`) and `tools/call`. We lower each tool into the SAME
`IntegrationOperation` the OpenAPI path produces, so the runtime/governance/
selection are identical — MCP is just another `spec_kind`, not a parallel path.

Transport: JSON-RPC 2.0 over HTTP POST to the server URL (the streamable-HTTP
transport). The executor wraps a call as `tools/call`; this module handles
discovery (`tools/list`) and normalization.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from core.integrations.model import (
    Integration, IntegrationOperation, tool_name_for,
)
from core.integrations.openapi import SpecError

LIST_RPC = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}


def call_envelope(operation_id: str, arguments: dict) -> dict:
    """The JSON-RPC body the executor POSTs for an MCP tool call."""
    return {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": operation_id, "arguments": arguments or {}}}


def fetch_tools(url: str, *, http_post=None, headers: Optional[dict] = None) -> List[dict]:
    """Discover an MCP server's tools via `tools/list`. `http_post(url, json, headers)
    -> dict` is injectable for tests; defaults to requests."""
    if http_post is not None:
        result = http_post(url, LIST_RPC, headers)
    else:
        import requests
        r = requests.post(url, json=LIST_RPC, timeout=15,
                          headers={**(headers or {}), "Accept": "application/json"})
        r.raise_for_status()
        result = r.json()
    if isinstance(result, dict) and "error" in result and result["error"]:
        raise SpecError(f"MCP server error on tools/list: {result['error']}")
    tools = ((result or {}).get("result") or {}).get("tools")
    if not isinstance(tools, list):
        raise SpecError("MCP tools/list returned no tools array.")
    return tools


def normalize_mcp(tools: List[dict], slug: str, base_url: str,
                  display_name: Optional[str] = None
                  ) -> Tuple[Integration, Dict[str, dict]]:
    """Lower MCP tools into an Integration. MCP tools are actions → default to
    write + approval (admin can relax per-op). Args go in the JSON-RPC envelope,
    so request_map is empty (the executor branches on spec_kind='mcp')."""
    integ = Integration(
        slug=slug, display_name=display_name or slug, base_url=base_url.rstrip("/"),
        spec_kind="mcp", auth_provider=None, security_scheme=None, auth_config={})
    req_maps: Dict[str, dict] = {}
    seen = set()
    for t in tools:
        name = t.get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        schema = t.get("inputSchema") or t.get("input_schema") or {"type": "object", "properties": {}}
        integ.operations.append(IntegrationOperation(
            operation_id=name,
            tool_name=tool_name_for(slug, name),
            http_method="POST",
            path_template="",
            summary=(t.get("title") or "")[:200],
            description=(t.get("description") or name).strip(),
            parameters_schema=schema,
            side_effect="write",
            requires_approval=True,
            enabled=True,
        ))
        req_maps[name] = {"path": [], "query": [], "header": [], "has_body": True, "mcp": True}
    if not integ.operations:
        raise SpecError("MCP server exposes no tools.")
    return integ, req_maps


def parse_mcp_result(payload: Any) -> str:
    """Unwrap an MCP `tools/call` result into text. Prefers result.content[].text;
    falls back to structuredContent / the raw payload."""
    if not isinstance(payload, dict):
        return str(payload)
    if payload.get("error"):
        return f"MCP error: {json.dumps(payload['error'])[:1000]}"
    result = payload.get("result", payload)
    content = result.get("content") if isinstance(result, dict) else None
    if isinstance(content, list):
        texts = [c.get("text", "") for c in content
                 if isinstance(c, dict) and c.get("type") == "text"]
        if texts:
            joined = "\n".join(t for t in texts if t)
            if result.get("isError"):
                return f"[tool error]\n{joined}"
            return joined
    if isinstance(result, dict) and "structuredContent" in result:
        return json.dumps(result["structuredContent"], indent=2)[:8000]
    return json.dumps(result, indent=2)[:8000]
