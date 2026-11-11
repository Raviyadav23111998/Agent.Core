"""
Tool spec — the declarative way to define a tool.
═══════════════════════════════════════════════════════════════════════════════
A tool is a `Tool` value object; a module exposes a LIST of them and calls
`register_specs(registry, specs)`. Adding a tool is appending one `Tool(...)` —
never editing a register() body or a dispatch branch. The ToolRegistry stays the
lookup-dispatched executor underneath; `Tool` is the schema/handler in one place.

    from tools.tool_spec import Tool, register_specs

    def register(self, registry):
        register_specs(registry, [
            Tool("page_observe", "Look at the screen…", {}, self.page_observe),
            Tool("page_act", "Do several actions…", {"steps": "string"}, self.page_act,
                 param_descriptions={"steps": "JSON array of {do,target,value}."}),
        ])
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List


@dataclass(frozen=True)
class Tool:
    """A declarative tool definition.

    name:        the tool name the model calls.
    description: what it does (shown to the model).
    parameters:  {param_name: "type"} shorthand ("string"/"int"/"bool"/…); a
                 trailing "?" on the key marks it optional (same as register()).
    handler:     the callable that runs the tool.
    param_descriptions: optional {param: rich description}.
    side_effect: the tool's KIND ("read"/"write"/"irreversible") — the preferred
                 one-word declaration; the framework maps it to the canonical
                 policy. `policy` is the escape hatch for custom mechanics.
    policy:      optional full ToolPolicy; name-inferred when both are omitted.
    """
    name: str
    description: str
    parameters: Dict[str, str]
    handler: Callable
    param_descriptions: Dict[str, str] = field(default_factory=dict)
    policy: Any = None
    side_effect: str = None

    def register(self, registry) -> None:
        """Register this tool's schema + handler on a ToolRegistry."""
        registry.register(
            self.name, self.description, self.parameters, self.handler,
            param_descriptions=self.param_descriptions or None,
            policy=self.policy, side_effect=self.side_effect,
        )


def register_specs(registry, specs: Iterable[Tool]) -> List[str]:
    """Register every Tool in `specs`; return the registered names (for logs)."""
    names = []
    for spec in specs:
        spec.register(registry)
        names.append(spec.name)
    return names
