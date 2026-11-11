import json
from typing import Dict, Any, Callable, List


_TYPE_MAP = {
    "string": "string",
    "str": "string",
    "int": "integer",
    "integer": "integer",
    "bool": "boolean",
    "boolean": "boolean",
    "float": "number",
    "number": "number",
    "any": "string",
}


class ToolRegistry:
    """Central registry for all agent tools.
    
    Architecture (identical to Agent):
    - Tools registered with name, description, typed parameters, callback
    - Exports OpenAI-compatible function definitions for native function calling
    - Strict schema validation with type coercion on execution
    - Keeps compressed text format as fallback for non-FC models
    """
    
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._callbacks: Dict[str, Callable] = {}
        self._policies: Dict[str, Any] = {}
        
    def register(self, name: str, description: str, parameters: Dict[str, Any],
                 callback: Callable, param_descriptions: Dict[str, str] = None,
                 policy=None, side_effect: str = None,
                 json_schema: Dict[str, Any] = None):
        """Register a tool with its schema and execution callback.

        Args:
            param_descriptions: Optional dict mapping param_name → rich description.
            policy: A full ToolPolicy — only for the rare tool that needs custom
                    mechanics. Prefer declaring `side_effect` instead.
            side_effect: The tool's KIND ("read"/"write"/"irreversible"). The
                    scalable, preferred declaration: the framework maps it to
                    the canonical policy (tool_policy.policy_for_kind), so a tool
                    never hardcodes timeouts/cacheability. The kind may also be
                    declared on the callback itself (``@tool(side_effect=…)``)
                    and travels with it through any registration path.
            json_schema: Optional FULL JSON Schema for the tool's parameters
                    (nested objects/arrays/enums the flat `{param:type}` cannot
                    express — the dynamic API-integration path).
        """
        schema = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "param_descriptions": param_descriptions or {},
            "json_schema": json_schema,
        }
        self._tools[name] = schema
        self._callbacks[name] = callback
        self._policies[name] = self._resolve_policy(
            name, callback, policy, side_effect)

    @staticmethod
    def _resolve_policy(name, callback, policy, side_effect):
        """Priority: explicit policy > declared kind (arg, else carried on the
        tool fn) > legacy name-inference. A tool declares WHAT it is; the
        framework owns HOW it behaves."""
        if policy is not None:
            return policy
        from tools.tool_policy import policy_for_kind, default_policy_for
        kind = side_effect or getattr(callback, "_tool_side_effect", None)
        if kind:
            return policy_for_kind(
                kind, shell=bool(getattr(callback, "_tool_shell", False)))
        return default_policy_for(name)

    def get_policy(self, name: str):
        """Return the ToolPolicy for a tool (inferring a default if unseen)."""
        pol = self._policies.get(name)
        if pol is None:
            from tools.tool_policy import default_policy_for
            pol = default_policy_for(name)
            self._policies[name] = pol
        return pol

    
    def get_openai_tools(self) -> List[dict]:
        """Generate OpenAI-compatible tool definitions for the API `tools` parameter.
        
        This is the CORRECT way to expose tools — via API parameter, not system prompt.
        The model was trained to produce structured `tool_calls` in response to these.
        
        Output format:
        [
            {
                "type": "function",
                "function": {
                    "name": "view_file",
                    "description": "Read file contents...",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "absolute_path": {"type": "string", "description": "Path to file"}
                        },
                        "required": ["absolute_path"]
                    }
                }
            }
        ]
        """
        tools = []
        for name, schema in self._tools.items():
            full = schema.get("json_schema")
            if full:
                tools.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": schema.get("description", ""),
                        "parameters": full,
                    },
                })
                continue

            raw_params = schema.get("parameters", {})

            if "desc" in raw_params and "args" in raw_params:
                param_dict = raw_params["args"]
                desc = raw_params.get("desc", schema.get("description", ""))
            else:
                param_dict = raw_params
                desc = schema.get("description", "")
            
            properties = {}
            required = []
            pdesc = schema.get("param_descriptions", {})
            
            for param_key, type_str in param_dict.items():
                is_optional = param_key.endswith("?")
                clean_key = param_key.rstrip("?")
                base_type = type_str.replace("?", "")
                
                json_type = _TYPE_MAP.get(base_type, "string")
                description = pdesc.get(clean_key, clean_key.replace("_", " "))
                properties[clean_key] = {
                    "type": json_type,
                    "description": description
                }
                
                if not is_optional:
                    required.append(clean_key)
            
            tool_def = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                    }
                }
            }
            
            if required:
                tool_def["function"]["parameters"]["required"] = required
            
            tools.append(tool_def)
        
        return tools
    
        
    def get_compressed_schemas(self) -> str:
        """Returns schemas as minified JSON for fallback text-based tool calling.
        
        Only used when native function calling is not available.
        """
        compressed = {}
        for name, schema in self._tools.items():
            params = schema.get("parameters", {})
            if "desc" in params and "args" in params:
                compressed[name] = params
            else:
                compressed[name] = {
                    "desc": schema.get("description", ""),
                    "args": params
                }
        return json.dumps(compressed, separators=(',', ':'))
    
    def get_all_schemas(self) -> list:
        """Returns full internal schemas."""
        return list(self._tools.values())

        
    def is_async(self, tool_name: str) -> bool:
        """True when the tool's callback is a coroutine function.

        Async tools run directly ON the event loop (no worker thread, no
        nested loop) — the path sub-agent tools use so parallel tool_use
        batches become genuinely concurrent sub-agents.
        """
        import asyncio
        return asyncio.iscoroutinefunction(self._callbacks.get(tool_name))

    def _resolve_call(self, tool_name: str, args: Dict[str, Any]):
        """Validate/coerce args; returns (callback, cleaned_args).

        Raises ValueError for unknown tools / missing or mistyped args.
        """
        if tool_name not in self._callbacks:
            available = ", ".join(sorted(self._callbacks.keys()))
            raise ValueError(f"Tool '{tool_name}' not registered. Available: {available}")

        if self._tools[tool_name].get("json_schema"):
            return self._callbacks[tool_name], dict(args)

        cleaned = self._clean_args(tool_name, args)
        return self._callbacks[tool_name], cleaned

    async def execute_async(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Await an async tool callback (see is_async)."""
        callback, cleaned = self._resolve_call(tool_name, args)
        try:
            result = await callback(**cleaned)
            return str(result) if result is not None else "Done."
        except TypeError as e:
            raise ValueError(f"Argument mismatch for '{tool_name}': {e}")
        except Exception as e:
            return f"Error executing {tool_name}: {e}"

    def execute(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute a registered tool with schema validation and type coercion.

        Handles:
        - Missing required argument detection
        - Optional parameter '?' suffix stripping
        - Type coercion: str→int, str→bool, any→str
        - Native FC format: args come as correct types from JSON parse
        """
        callback, cleaned = self._resolve_call(tool_name, args)
        try:
            result = callback(**cleaned)
            return str(result) if result is not None else "Done."
        except TypeError as e:
            raise ValueError(f"Argument mismatch for '{tool_name}': {e}")
        except Exception as e:
            return f"Error executing {tool_name}: {e}"

    def _clean_args(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Dense-schema validation + coercion (the non-json_schema path)."""
        schema_params = self._tools[tool_name].get("parameters", {})
        if "desc" in schema_params and "args" in schema_params:
            schema_params = schema_params["args"]
        
        cleaned_args = {}
        
        for param_key, expected_type in schema_params.items():
            is_optional = param_key.endswith("?")
            clean_key = param_key.rstrip("?")
            base_type = expected_type.replace("?", "")
            
            val = None
            has_val = False
            
            if clean_key in args:
                val = args[clean_key]
                has_val = True
            elif param_key in args and is_optional:
                val = args[param_key]
                has_val = True
            
            if not has_val:
                if not is_optional:
                    raise ValueError(f"Missing required argument: '{clean_key}'")
                continue
            
            if base_type in ("int", "integer") and not isinstance(val, int):
                try:
                    val = int(val)
                except (ValueError, TypeError):
                    raise ValueError(f"Argument '{clean_key}' must be int, got {type(val).__name__}: {val!r}")
            elif base_type in ("bool", "boolean") and not isinstance(val, bool):
                s = str(val).lower()
                if s in ("true", "1", "yes"):
                    val = True
                elif s in ("false", "0", "no"):
                    val = False
                else:
                    raise ValueError(f"Argument '{clean_key}' must be bool, got: {val!r}")
            elif base_type in ("string", "str") and not isinstance(val, str):
                val = str(val)
            
            cleaned_args[clean_key] = val

        return cleaned_args
