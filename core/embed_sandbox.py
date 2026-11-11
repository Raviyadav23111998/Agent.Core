"""
Embed tool sandbox — hard server-side containment for embedded agents.
═══════════════════════════════════════════════════════════════════════════
An embedded integration (chatbot / copilot running on a third-party site) must
NEVER be able to reach admin, filesystem, shell, or server-management tools —
no matter what its persona's allowed_tools happen to contain. This module is
that boundary: it builds a scoped ToolRegistry as

    (persona.allowed_tools, or all)  ∩  EMBED_SAFE[mode]

EMBED_SAFE is a hard ceiling — a misconfigured persona can only ever LOSE tools
through the intersection, never gain a privileged one. This is the security
core of Phase 1 (see EMBED_AGENT_PLATFORM_ARCHITECTURE.md §4, Appendix A #3).
"""
from typing import Iterable, Optional, Set

_CHAT_TOOLS: Set[str] = {"brain_ask", "brain_search"}

_COPILOT_EXTRA: Set[str] = {
    "page_observe", "page_act", "page_navigate", "page_read", "page_wait",
}


def embed_safe_tools(mode: str) -> Set[str]:
    """The hard ceiling of tools an embedded session may use, by mode."""
    tools = set(_CHAT_TOOLS)
    if mode == "copilot":
        tools |= _COPILOT_EXTRA
    return tools


def build_embed_registry(full_registry, mode: str,
                         persona_allowed: Optional[Iterable[str]] = None,
                         dynamic_embed_safe: Optional[Iterable[str]] = None):
    """Return a NEW ToolRegistry containing ONLY embed-safe tools.

    scoped = (persona_allowed ∩ embed_safe[mode])  — or embed_safe[mode] if no
    persona allow-list is given. Admin/filesystem/shell/server tools can never
    appear, even if the persona lists them, because they are not in the ceiling.

    `dynamic_embed_safe` are dynamic API-integration ops the admin EXPLICITLY
    marked embed-safe AND granted to this app. They are allowed IN ADDITION to
    the static ceiling — the ONLY way an external/business API reaches an
    anonymous chatbot visitor, so an internal API is never exposed by default.

    Args:
        full_registry: the complete ToolRegistry (all server tools registered).
        mode: 'chat' | 'copilot'.
        persona_allowed: the integration persona's allowed_tools (optional).
        dynamic_embed_safe: admin-opted-in dynamic op tool_names (optional).
    """
    from tools.registry import ToolRegistry

    ceiling = embed_safe_tools(mode)
    allowed = (set(persona_allowed) & ceiling) if persona_allowed else set(ceiling)
    if dynamic_embed_safe:
        allowed |= set(dynamic_embed_safe)

    scoped = ToolRegistry()
    for name in allowed:
        if name in full_registry._tools:
            scoped._tools[name] = full_registry._tools[name]
            scoped._callbacks[name] = full_registry._callbacks[name]
            if name in getattr(full_registry, "_policies", {}):
                scoped._policies[name] = full_registry._policies[name]
    return scoped
