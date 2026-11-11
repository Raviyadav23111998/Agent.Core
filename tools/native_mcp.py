import inspect
import json
import re
from typing import Callable, Dict

_DOC_SECTIONS = ("returns:", "return:", "raises:", "yields:", "examples:",
                 "example:", "note:", "notes:")


def parse_docstring(doc: str):
    """Parse a Google-style docstring into (description, {param: description}).

    The leading paragraph(s) before `Args:` become the description; the
    indented `name: text` lines under `Args:` (with continuation lines)
    become per-parameter descriptions. Tolerant of missing sections.
    """
    if not doc:
        return "", {}
    lines = doc.strip("\n").splitlines()

    desc_parts, i = [], 0
    while i < len(lines):
        if lines[i].strip().lower() in ("args:", "arguments:", "params:", "parameters:"):
            break
        desc_parts.append(lines[i].strip())
        i += 1
    description = " ".join(p for p in desc_parts if p).strip()

    arg_desc: Dict[str, str] = {}
    if i < len(lines):
        i += 1
        current = None
        for line in lines[i:]:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.lower() in _DOC_SECTIONS:
                break
            m = re.match(r"^([A-Za-z_]\w*)\s*(?:\([^)]*\))?\s*:\s*(.*)$", stripped)
            if m:
                current = m.group(1)
                arg_desc[current] = m.group(2).strip()
            elif current:
                arg_desc[current] = (arg_desc[current] + " " + stripped).strip()
    return description, arg_desc


class NativeMCP:
    """
    A lightweight, token-optimized native implementation of the Model Context Protocol.
    Compresses standard tool schemas down to bare-minimum atomic representations.
    """

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._dense_schemas: Dict[str, Dict[str, str]] = {}

    def tool(self, *, side_effect: str = None, shell: bool = False):
        """Decorator to register a function as an ultra-dense MCP tool.

        `side_effect` ("read"/"write"/"irreversible") declares the tool's KIND.
        It is stamped ON the function, so whichever ToolRegistry later wires
        this callback derives the right policy from the declaration instead of
        guessing from the name (`shell=True` = irreversible whose identical
        re-runs are normal, e.g. run_command).
        """
        def decorator(func: Callable):
            name = func.__name__
            self._tools[name] = func
            if side_effect is not None:
                func._tool_side_effect = side_effect
            if shell:
                func._tool_shell = True

            doc = func.__doc__ or ""
            description, arg_desc = parse_docstring(doc)
            if not description:
                description = name

            self._dense_schemas[name] = {
                "desc": description,
                "args": self._compress_signature(func),
                "arg_descriptions": arg_desc,
            }
            return func
        return decorator
        
    def _compress_signature(self, func: Callable) -> Dict[str, str]:
        """
        Compresses a Python function signature into an ultra-dense format.
        Example: search(query: str, limit: int = 10) -> {"query": "str", "limit?": "int"}
        """
        sig = inspect.signature(func)
        compressed = {}
        
        for param_name, param in sig.parameters.items():
            type_str = param.annotation.__name__ if param.annotation != inspect.Parameter.empty else "any"
            
            key = f"{param_name}?" if param.default != inspect.Parameter.empty else param_name
            compressed[key] = type_str
            
        return compressed

    def get_compressed_registry(self) -> str:
        """
        Returns the entire registry as a single, minified JSON string.
        e.g., {"search":{"q":"str"},"read":{"path":"str"}}
        """
        return json.dumps(self._dense_schemas, separators=(',', ':'))
        
    def execute(self, tool_name: str, **kwargs) -> str:
        """Executes a registered tool."""
        if tool_name not in self._tools:
            raise ValueError(f"Unknown MCP tool: {tool_name}")
        return self._tools[tool_name](**kwargs)
