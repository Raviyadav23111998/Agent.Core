"""
Agent — Skills MCP Tools (procedural memory)
============================================
Design: docs/SKILLS_LEARNING_IMPLEMENTATION_PLAN_2026-07-04.md — a port of the
hermes-agent learning loop (MIT, Nous Research) onto the tenant DB.

A skill is a SKILL.md-shaped markdown document (name, one-line description,
tags, body) plus optional support files (references/ templates/ scripts/).
The system prompt carries only an INDEX (name — description per skill); the
model opens a body on demand with skill_view. The background reviewer and the
curator maintain the library through skill_manage.

Layering: the engine is DB-agnostic. The host injects a SkillStore
implementation (api layer: tenant-DB-backed); absent a store, the tools
degrade to a clear "not available" answer — same fail-soft contract as
llm_scope. Skill content is untrusted-adjacent (agents write it): every body
is wrapped by injection_guard on load, and scanned on write.

Tools:
  skills_list   — index of available skills (name, description, tags)
  skill_view    — full body + support-file listing; marks the skill "in play"
  skill_manage  — create | edit | patch | write_file | remove_file | archive
"""

import logging
from contextvars import ContextVar
from typing import Optional, Set

from .native_mcp import NativeMCP

logger = logging.getLogger(__name__)

# The canonical skill-tool names, referenced by EVERY tool-gating point so
# there is one source of truth (not per-filter literals). skill_manage is a
# write; the read tools are always-safe engine infrastructure over the
# tenant's own procedural memory.
SKILL_READ_TOOLS = ("skills_list", "skill_view")
SKILL_TOOL_NAMES = ("skills_list", "skill_view", "skill_manage")

# Set per turn by AgentLoop: skill_view records names here so the loop can
# (a) attribute effectiveness (a correction in a conversation where a skill
# was in play counts against that skill) and (b) trigger the review pass.
current_skill_tracker: ContextVar[Optional[Set[str]]] = ContextVar(
    "current_skill_tracker", default=None)


class SkillStore:
    """Contract the host implements (api layer, tenant-DB-backed).

    All methods are synchronous and may raise — the tool wrappers translate
    failures into readable tool errors. Scope filtering (tenant|app|user) is
    the STORE's responsibility: the engine never sees other scopes' skills.
    """

    def index(self):  # -> List[dict] with name, description, tags, pinned, origin
        raise NotImplementedError

    def get(self, name: str):  # -> dict|None with body, files: List[{kind,path}]
        raise NotImplementedError

    def get_file(self, name: str, path: str):  # -> str|None
        raise NotImplementedError

    def write(self, action: str, name: str, *, description: str = None,
              body: str = None, tags=None, file_path: str = None,
              file_content: str = None, edited_by: str = "agent"):  # -> str
        raise NotImplementedError

    def record_use(self, name: str):  # best-effort
        raise NotImplementedError

    def index_for(self, app_id, user_id):  # explicit scope (session creation)
        raise NotImplementedError


class SkillsMCPTools:
    """MCP tool wrappers over the injected SkillStore.

    The store is resolved LAZILY per call: tools register once at CLI
    startup, but the host may attach the store later (api layer,
    attach_skills seam) — same late-binding contract as llm_scope.
    """

    _NO_STORE = ("Skills are not available in this environment "
                 "(no skill store attached).")

    def __init__(self, store_resolver=None):
        self._resolve = store_resolver or (lambda: None)

    @property
    def _store(self):
        try:
            return self._resolve()
        except Exception:
            return None

    def register_tools(self, mcp_server: NativeMCP):

        @mcp_server.tool()
        def skills_list(tag: str = None) -> str:
            """List available skills (procedural memory): name, one-line
            description, and tags. Use skill_view to read one in full.

            Args:
                tag: Optional tag filter (e.g. "email", "jira").
            """
            if self._store is None:
                return self._NO_STORE
            try:
                rows = self._store.index() or []
            except Exception as e:
                return f"Error listing skills: {e}"
            if tag:
                t = tag.strip().lower()
                rows = [r for r in rows if t in [x.lower() for x in (r.get("tags") or [])]]
            if not rows:
                return "No skills recorded yet."
            lines = [f"- {r['name']} — {r.get('description', '')}"
                     + (f"  [tags: {', '.join(r['tags'])}]" if r.get("tags") else "")
                     for r in rows]
            return "Available skills:\n" + "\n".join(lines)

        @mcp_server.tool()
        def skill_view(name: str, file_path: str = None) -> str:
            """Read a skill's full instructions (or one of its support files).
            Load a skill BEFORE doing a task it covers.

            Args:
                name: The skill name from skills_list.
                file_path: Optional support file (e.g. "references/x.md").
            """
            if self._store is None:
                return self._NO_STORE
            try:
                if file_path:
                    content = self._store.get_file(name.strip(), file_path.strip())
                    if content is None:
                        return f"Skill '{name}' has no file '{file_path}'."
                    body, label = content, f"{name}/{file_path}"
                else:
                    skill = self._store.get(name.strip())
                    if not skill:
                        return f"No skill named '{name}'. Use skills_list to see what exists."
                    files = skill.get("files") or []
                    listing = ("\n\nSupport files (open with skill_view(name, file_path)):\n"
                               + "\n".join(f"  - {f['path']}" for f in files)) if files else ""
                    body, label = skill["body"] + listing, name
                    try:
                        self._store.record_use(name.strip())
                    except Exception:
                        pass
                    tracker = current_skill_tracker.get()
                    if tracker is not None:
                        tracker.add(name.strip())
            except Exception as e:
                return f"Error reading skill: {e}"
            # Skill bodies are agent-written content: frame, never trust.
            try:
                from core.injection_guard import guard_retrieved_content
                return guard_retrieved_content(f"skill:{label}", body)
            except Exception:
                return body

        @mcp_server.tool()
        def skill_manage(action: str, name: str, description: str = None,
                         body: str = None, tags: str = None,
                         file_path: str = None, file_content: str = None) -> str:
            """Create or maintain a skill (procedural memory for a CLASS of
            task — never a one-session artifact).

            Args:
                action: create | edit | patch | write_file | remove_file | archive.
                        create needs name+description+body; edit replaces the
                        body; patch appends a section to the body; write_file
                        adds/updates a support file (file_path must start with
                        references/, templates/ or scripts/).
                name: lowercase-hyphenated class-level skill name.
                description: One line, <=100 chars — it IS the index entry.
                body: Markdown instructions (for create/edit/patch).
                tags: Comma-separated tags.
                file_path: For write_file/remove_file.
                file_content: For write_file.
            """
            if self._store is None:
                return self._NO_STORE
            act = (action or "").strip().lower()
            if act not in ("create", "edit", "patch", "write_file",
                           "remove_file", "archive"):
                return ("Unknown action. Use create|edit|patch|write_file|"
                        "remove_file|archive.")
            try:
                tag_list = ([t.strip() for t in tags.split(",") if t.strip()]
                            if tags else None)
                result = self._store.write(
                    act, (name or "").strip(),
                    description=(description or "").strip() or None,
                    body=body, tags=tag_list,
                    file_path=(file_path or "").strip() or None,
                    file_content=file_content,
                )
                return result or f"Skill '{name}': {act} done."
            except Exception as e:
                return f"Error ({act} '{name}'): {e}"

        logger.info("skills.tools: registered (store=%s)",
                    "attached" if self._store is not None else "none")
