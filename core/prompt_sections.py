"""
Prompt-section renderers for the engine's built-in capabilities.

These turn feature DATA into the fully-rendered text that contributors register
with `PromptBuilder.set_section(key, text, order)`. The wording lives HERE (with
the capability), not inside the generic `PromptBuilder` — the engine's prompt
assembler stays feature-agnostic.

Section order convention (lower = earlier in the dynamic-context zone):
    10  copilot_routes      (api/services/copilot_prompt)
    20  copilot_playbook    (api/services/copilot_prompt)
    30  skills_index        (SKILLS_ORDER)
    40  memory              (MEMORY_ORDER)
    50  knowledge           (KNOWLEDGE_ORDER)
"""
from typing import Iterable

SKILLS_ORDER = 30
MEMORY_ORDER = 40
KNOWLEDGE_ORDER = 50

SKILLS_KEY = "skills_index"
MEMORY_KEY = "memory"
KNOWLEDGE_KEY = "knowledge"


def render_skills_index(rows) -> str:
    """Procedural-memory index — one line per skill, sorted by name so an
    unchanged library renders identical bytes (stays in the cached prefix).
    Bodies load on demand via skill_view. Empty library → '' (no block)."""
    lines = [f"- {r['name']} — {r.get('description', '')}"
             for r in sorted(rows or [], key=lambda r: r.get("name", ""))]
    if not lines:
        return ""
    return ("<skills_index>\n"
            "Learned skills (procedural memory). BEFORE starting a task "
            "a skill covers, load it with skill_view(name):\n"
            + "\n".join(lines) + "\n</skills_index>")


def render_knowledge(items: Iterable[str]) -> str:
    """Always-injected knowledge items (artifact summaries) as a <context>
    block. Empty → ''."""
    items = [i for i in (items or []) if i]
    if not items:
        return ""
    return "<context>\n" + "\n".join(f"- {ki}" for ki in items) + "\n</context>"


def render_memory(text: str) -> str:
    """Episodic/durable memory context block. Empty → ''."""
    text = (text or "").strip()
    if not text:
        return ""
    return f"<memory>\n{text}\n</memory>"
