import os
from enum import Enum
from typing import List


class ReasoningEffort(Enum):
    """Task complexity classifier — determines which prompt sections to load."""
    TRIVIAL = "trivial"
    STANDARD = "standard"
    COMPLEX = "complex"


_COMPLEX_SIGNALS = frozenset([
    "refactor", "debug", "architecture", "redesign", "migrate", "overhaul",
    "rewrite", "optimize", "performance", "security", "deploy", "implement",
    "create a plan", "implementation plan", "break down", "step by step",
    "multiple files", "across the codebase", "all files", "entire project",
    "summarize", "analyze", "go through", "review", "audit",
    "explain the", "walk through", "read the project", "what does",
])

_TRIVIAL_SIGNALS = frozenset([
    "hi", "hello", "hey", "thanks", "ok", "okay", "sure",
    "yes", "no", "bye", "goodbye", "help",
])
# Multi-word phrases can't match via word-set intersection — checked as
# substrings instead.
_TRIVIAL_PHRASES = ("thank you", "what is", "who are you")


def estimate_tokens(text: str) -> int:
    """Fast token estimation: ~4 chars per token."""
    return len(text) // 4


def classify_effort(user_input: str) -> ReasoningEffort:
    """Classify task complexity from user input."""
    lower = user_input.lower().strip()
    words = set(lower.split())
    
    if len(lower) < 30 and (words & _TRIVIAL_SIGNALS
                            or any(p in lower for p in _TRIVIAL_PHRASES)):
        return ReasoningEffort.TRIVIAL
    
    for signal in _COMPLEX_SIGNALS:
        if signal in lower:
            return ReasoningEffort.COMPLEX
    
    if len(lower) > 500 or lower.count("```") >= 2:
        return ReasoningEffort.COMPLEX
    
    return ReasoningEffort.STANDARD


class PromptBuilder:
    """Config-driven system prompt assembler.
    
    Architecture (identical to Agent):
    - Prompt sections live in `prompts/` directory as .md files
    - PromptBuilder reads and assembles them at runtime
    - No hardcoded prompt text in Python code
    - Runtime variables interpolated via {variable} syntax
    - Conditional sections loaded based on effort level
    - Dynamic context (memory, KIs) appended only when populated
    
    This makes prompts:
    - Editable without code changes
    - Versionable independently
    - Extensible (plugins can add section files)
    - Testable per section
    """
    
    _CORE_SECTIONS = ["identity", "constraints", "communication"]
    
    _EFFORT_SECTIONS = {
        ReasoningEffort.COMPLEX: ["planning_mode"],
    }

    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        # The ONE generic injection seam: key -> (order, fully-rendered text).
        # Skills, knowledge, copilot playbook/routes, etc. all register here via
        # set_section(); the engine renders them without knowing what any of
        # them is. Feature wording lives with the feature, never in the engine.
        self._sections = {}
        self._compressed_tools_fallback = "{}"
        self._use_fallback_tools = False
        self._available_tools: frozenset = frozenset()

        self._prompts_dir = self._find_prompts_dir()

        self._section_cache = {}

    def set_section(self, key: str, text: str, order: int = 100) -> None:
        """THE generic prompt-injection seam.

        Register a named, fully-rendered prompt section. The engine renders
        registered sections in the dynamic-context zone, ordered by (order, key)
        so the assembled prefix is deterministic and stays in the provider's
        cache. The contributor owns the wording; the engine never learns what a
        section IS. Empty/blank text removes the section.
        """
        if text and text.strip():
            self._sections[key] = (order, text)
        else:
            self._sections.pop(key, None)

    def set_available_tools(self, tool_names):
        """Tool names available this session — gates doctrine blocks so the
        model is never instructed to use a tool its persona doesn't expose.
        Fixed per session, so the assembled prompt stays byte-stable."""
        self._available_tools = frozenset(tool_names or ())

    _DOCTRINE_IF_RE = None

    def _build_doctrine(self) -> str:
        """Assemble the ALWAYS-ON engine doctrine (prompts/doctrine.md).

        Layering contract (the Claude Code model): personas own identity,
        voice, and access constraints — they may REPLACE those sections.
        The engine owns its operating doctrine — how tools are used, task
        management, verification, safety — and no persona can delete it.

        `[[if:tool_name]] … [[endif]]` blocks are kept only when the tool is
        available in this session's registry.
        """
        raw = self._load_section("doctrine")
        if not raw:
            return ""
        import re as _re
        if PromptBuilder._DOCTRINE_IF_RE is None:
            PromptBuilder._DOCTRINE_IF_RE = _re.compile(
                r"\[\[if:([\w]+)\]\](.*?)\[\[endif\]\]", _re.DOTALL)

        def _keep(m):
            return m.group(2) if m.group(1) in self._available_tools else ""

        text = PromptBuilder._DOCTRINE_IF_RE.sub(_keep, raw)
        text = _re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
        
    def _find_prompts_dir(self) -> str:
        """Find the prompts/ directory relative to the package root."""
        pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prompts_dir = os.path.join(pkg_root, "prompts")
        if os.path.isdir(prompts_dir):
            return prompts_dir
        
        prompts_dir = os.path.join(os.path.abspath(self.workspace_path), "prompts")
        if os.path.isdir(prompts_dir):
            return prompts_dir
        
        return ""

    def _load_section(self, name: str) -> str:
        """Load a prompt section from file. Cached after first read."""
        if name in self._section_cache:
            return self._section_cache[name]
        
        if not self._prompts_dir:
            return ""
        
        filepath = os.path.join(self._prompts_dir, f"{name}.md")
        if not os.path.exists(filepath):
            return ""
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        self._section_cache[name] = content
        return content
    
    def reload_sections(self):
        """Clear cache — forces re-read from disk on next build.
        
        Useful for hot-reloading prompt changes without restarting.
        """
        self._section_cache.clear()
    

    def set_compressed_tools(self, compressed_json: str):
        self._compressed_tools_fallback = compressed_json
    
    def set_fallback_mode(self, enabled: bool):
        self._use_fallback_tools = enabled


    def build_system_prompt(self, effort: ReasoningEffort = ReasoningEffort.STANDARD,
                            custom_identity: str = None,
                            custom_constraints: str = None,
                            custom_communication: str = None) -> str:
        """Assemble system prompt from section files + dynamic context.
        
        Assembly order:
        1. Core sections (identity, constraints, communication)
           — can be overridden by custom_* parameters (e.g., from agent_personas DB)
        2. Runtime environment block (computed, not from file)
        3. Effort-conditional sections (planning_mode for COMPLEX)
        4. Dynamic context (memory, knowledge items)
        5. Fallback tools (non-FC models only)
        
        When custom_* overrides are provided (from ag_system.agent_personas),
        they replace the corresponding .md file content entirely. This allows
        per-Application prompt customization without modifying the prompt files.
        """
        parts = []

        identity = custom_identity or self._load_section("identity")
        if identity:
            if custom_identity and not identity.startswith("<identity>"):
                identity = f"<identity>\n{identity}\n</identity>"
            parts.append(identity)

        # Engine doctrine is NOT overridable — personas replace identity,
        # constraints, and communication, never how the engine operates.
        doctrine = self._build_doctrine()
        if doctrine:
            parts.append(doctrine)

        constraints = custom_constraints or self._load_section("constraints")
        if constraints:
            if custom_constraints and not constraints.startswith("<constraints>"):
                constraints = f"<constraints>\n{constraints}\n</constraints>"
            parts.append(constraints)
        
        communication = custom_communication or self._load_section("communication")
        if communication:
            if custom_communication and not communication.startswith("<communication"):
                communication = f"<communication_style>\n{communication}\n</communication_style>"
            parts.append(communication)
        
        parts.insert(1, self._build_environment())
        
        effort_sections = self._EFFORT_SECTIONS.get(effort, [])
        for section in effort_sections:
            content = self._load_section(section)
            if content:
                parts.append(content)
        
        # Contributed sections — skills, knowledge, copilot playbook/routes, …
        # The engine renders whatever the host registered via set_section(),
        # in a deterministic (order, key) order for cache-stability. It does
        # NOT know what any of these are.
        for _key in sorted(self._sections,
                           key=lambda k: (self._sections[k][0], k)):
            parts.append(self._sections[_key][1])

        if self._use_fallback_tools:
            parts.append(
                f"<tools>\n{self._compressed_tools_fallback}\n"
                f"Use <call:tool_name>{{...}}</call:tool_name> format.\n</tools>"
            )
        
        return "\n\n".join(parts)
    
    def _build_environment(self) -> str:
        """Build the environment section — fully computed at runtime.
        
        Zero hardcoded values. Everything is detected from the OS.
        """
        import platform
        import getpass
        
        os_name = platform.system()
        if os_name == "Darwin":
            os_name = "macOS"
        shell = os.path.basename(os.environ.get("SHELL", "unknown"))
        user = getpass.getuser()
        
        return (
            f"<environment>\n"
            f"Your workspace: {self.workspace_path}\n"
            f"OS: {os_name} | Shell: {shell}\n"
            f"Filesystem access: RESTRICTED to your workspace only\n"
            f"You CANNOT access files outside: {self.workspace_path}\n"
            f"All file paths must be within your workspace directory.\n"
            f"</environment>"
        )
    
    def get_prompt_token_estimate(self) -> int:
        return estimate_tokens(self.build_system_prompt())
