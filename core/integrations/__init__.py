"""
core.integrations — dynamic, spec-derived external API integrations as agent tools.
═══════════════════════════════════════════════════════════════════════════════════

Turns any external HTTP API (OpenAPI 3.x / Swagger 2.0 / MCP) — public SaaS OR a
tenant's own internal business API — into governed, retrieval-selected tools with
NO per-provider Python class. Backed by the ag_api schema (V031) and reusing the
existing Connections (credentials), ToolRegistry/ToolExecutor (governance),
EmbeddingClient + pgvector (tool-RAG), and TenantDBProxy (multi-tenancy).

Phase 0 ships only the normalized model + policy mapping (this package's model.py);
ingestion (ingest.py) and runtime (runtime.py) land in later phases. Everything is
gated by the `agent.dynamic_integrations` feature flag (default off).

See API_INTEGRATION_IMPLEMENTATION_PLAN.md.
"""

from core.integrations.model import (
    Integration,
    IntegrationOperation,
    READ,
    WRITE,
    IRREVERSIBLE,
    side_effect_for_method,
    requires_approval_for_method,
    policy_for_method,
    tool_name_for,
    slugify,
)

__all__ = [
    "Integration",
    "IntegrationOperation",
    "READ",
    "WRITE",
    "IRREVERSIBLE",
    "side_effect_for_method",
    "requires_approval_for_method",
    "policy_for_method",
    "tool_name_for",
    "slugify",
]


def __getattr__(name):
    if name in ("SpecIngestor", "IngestReport"):
        from core.integrations import ingest
        return getattr(ingest, name)
    if name in ("normalize", "SpecError", "coerce_to_openapi3", "detect_kind"):
        from core.integrations import openapi
        return getattr(openapi, name)
    if name == "IntegrationStore":
        from core.integrations.store import IntegrationStore
        return IntegrationStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
