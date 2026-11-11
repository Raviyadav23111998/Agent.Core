"""
Agent — Configuration Metadata Registry
=====================================================
Single source of truth for config key metadata — drives the
System Config Hub UI and API validation.

Adding a new config key to the codebase? Add its metadata here
and the UI picks it up automatically.
"""


GROUP_METADATA = {
    "llm": {
        "label": "LLM / Model Configuration",
        "icon": "🤖",
        "sort": 1,
        "description": "AI model provider, tokens, caching, embedding, and fallback settings",
    },
    "agent": {
        "label": "Agent Engine",
        "icon": "⚙️",
        "sort": 2,
        "description": "Core agent loop behavior, budgets, planning, memory, and safety controls",
    },
    "brain": {
        "label": "Brain / Knowledge Base",
        "icon": "🧠",
        "sort": 3,
        "description": "Knowledge base synchronization and retrieval settings",
    },
    "memory": {
        "label": "Memory System",
        "icon": "💾",
        "sort": 4,
        "description": "Session and long-term memory limits, decay, and consolidation",
    },
    "auth": {
        "label": "Authentication",
        "icon": "🔐",
        "sort": 5,
        "description": "Login rate limiting, lockout, and authentication policies",
    },
    "secret": {
        "label": "Security & Keys",
        "icon": "🔑",
        "sort": 6,
        "description": "JWT keys, encryption keys, token expiry. Changes require restart.",
    },
    "redis": {
        "label": "Redis",
        "icon": "🔴",
        "sort": 8,
        "description": "Redis connection for distributed locks, rate limiting, and caching",
    },
    "gateway": {
        "label": "Gateway / Platforms",
        "icon": "🌐",
        "sort": 9,
        "description": "Telegram bot, Teams, and platform gateway settings",
    },
    "workspace": {
        "label": "Workspaces",
        "icon": "📂",
        "sort": 10,
        "description": "User workspace sandboxing, paths, and quotas",
    },
    "shell": {
        "label": "Shell & Sandbox",
        "icon": "🖥️",
        "sort": 11,
        "description": "Shell command execution and OS-level sandboxing",
    },
    "db": {
        "label": "Database",
        "icon": "🗄️",
        "sort": 12,
        "description": "PostgreSQL connection settings. Changes require restart.",
    },
    "app": {
        "label": "Application",
        "icon": "📱",
        "sort": 13,
        "description": "Application-level environment and behavior settings",
    },
    "bot": {
        "label": "Bot Settings",
        "icon": "🤖",
        "sort": 14,
        "description": "Chatbot widget behavior and approval settings",
    },
}


CONFIG_METADATA = {
    "llm.provider": {
        "label": "Provider",
        "group": "llm",
        "type": "text",
        "description": "LLM API provider identifier (e.g., openrouter, anthropic, google).",
        "default": "openrouter",
        "managed_by": "/chat/settings/llm",
    },
    "llm.model": {
        "label": "Primary Model",
        "group": "llm",
        "type": "text",
        "description": "Default LLM model for the main agent loop.",
        "default": "deepseek/deepseek-v4-flash",
        "managed_by": "/chat/settings/llm",
    },
    "llm.base_url": {
        "label": "Base URL",
        "group": "llm",
        "type": "text",
        "description": "API endpoint URL for the LLM provider.",
        "default": "https://openrouter.ai/api/v1/chat/completions",
        "managed_by": "/chat/settings/llm",
    },
    "llm.api_key": {
        "label": "API Key",
        "group": "llm",
        "type": "password",
        "description": "API key for the LLM provider.",
        "default": "",
        "is_secret": True,
        "managed_by": "/chat/settings/llm",
    },
    "llm.max_tokens": {
        "label": "Max Tokens",
        "group": "llm",
        "type": "number",
        "description": "Maximum output tokens per LLM request.",
        "default": "16000",
        "validation": {"min": 100, "max": 200000},
        "managed_by": "/chat/settings/llm",
    },
    "llm.temperature": {
        "label": "Temperature",
        "group": "llm",
        "type": "text",
        "description": "Sampling temperature (0.0 = deterministic, 1.0 = creative).",
        "default": "0.7",
        "managed_by": "/chat/settings/llm",
    },
    "llm.provider_order": {
        "label": "Provider Order",
        "group": "llm",
        "type": "text",
        "description": "Comma-separated provider preference order for OpenRouter routing.",
        "default": "DeepInfra",
        "managed_by": "/chat/settings/llm",
    },
    "llm.fallback_model": {
        "label": "Fallback Model",
        "group": "llm",
        "type": "text",
        "description": "Model to try when the primary model fails after retries. Leave empty to disable.",
        "default": "",
    },
    "llm.small_model": {
        "label": "Small / Cheap Model",
        "group": "llm",
        "type": "text",
        "description": "Cost-efficient model for classification, routing, salience, and compaction calls. "
                       "Falls back to the primary model if empty.",
        "default": "",
    },
    "llm.context_window": {
        "label": "Context Window Size",
        "group": "llm",
        "type": "number",
        "description": "Total context window of the primary model in tokens. Used to auto-calculate "
                       "the history budget. Set 0 to use the fixed budget instead.",
        "default": "0",
        "validation": {"min": 0, "max": 2000000},
    },
    "llm.embedding_model": {
        "label": "Embedding Model",
        "group": "llm",
        "type": "text",
        "description": "Model name for generating embeddings (e.g., nomic-embed-text). "
                       "Required to activate vector RAG.",
        "default": "",
    },
    "llm.embedding_base_url": {
        "label": "Embedding Base URL",
        "group": "llm",
        "type": "text",
        "description": "API endpoint for the embedding model (e.g., http://localhost:11434/v1/embeddings).",
        "default": "",
    },
    "llm.embedding_api_key": {
        "label": "Embedding API Key",
        "group": "llm",
        "type": "password",
        "description": "API key for the embedding endpoint. Leave empty if not required.",
        "default": "",
        "is_secret": True,
    },
    "llm.embedding_dimensions": {
        "label": "Embedding Dimensions",
        "group": "llm",
        "type": "number",
        "description": "Dimensionality of the embedding vectors (e.g., 768 for nomic-embed-text, "
                       "1536 for text-embedding-ada-002). Must match the model output.",
        "default": "768",
        "validation": {"min": 64, "max": 4096},
    },
    "llm.prompt_caching": {
        "label": "Prompt Caching",
        "group": "llm",
        "type": "boolean",
        "description": "Enable Anthropic-style prompt caching to reduce token costs on repeated prefixes.",
        "default": "true",
    },
    "llm.debug_dump_payload": {
        "label": "Debug: Dump Payloads",
        "group": "llm",
        "type": "boolean",
        "description": "Write raw LLM request/response payloads to temp/ for debugging. "
                       "WARNING: May contain sensitive data.",
        "default": "false",
    },

    "agent.history_token_budget": {
        "label": "History Token Budget",
        "group": "agent",
        "type": "number",
        "description": "Maximum tokens for conversation history before compaction triggers. "
                       "Overridden by context_window calculation if llm.context_window is set.",
        "default": "120000",
        "validation": {"min": 4000, "max": 2000000},
    },
    "agent.max_tools": {
        "label": "Max Tools per Agent",
        "group": "agent",
        "type": "number",
        "description": "Maximum number of tools included in an agent definition.",
        "default": "20",
        "validation": {"min": 1, "max": 100},
    },
    "agent.timeout_seconds": {
        "label": "Agent Run Timeout",
        "group": "agent",
        "type": "number",
        "description": "Maximum seconds an agent run can execute before being terminated.",
        "default": "300",
        "validation": {"min": 10, "max": 3600},
    },
    "agent.max_concurrent_turns": {
        "label": "Max Concurrent Turns",
        "group": "agent",
        "type": "number",
        "description": "Maximum agent turns running in parallel per process. "
                       "Prevents resource exhaustion under load.",
        "default": "8",
        "validation": {"min": 1, "max": 50},
    },
    "agent.conversation_token_budget": {
        "label": "Conversation Token Budget",
        "group": "agent",
        "type": "number",
        "description": "Maximum cumulative tokens per conversation before the agent stops. "
                       "Set 0 for unlimited.",
        "default": "0",
        "validation": {"min": 0, "max": 10000000},
    },
    "agent.context_strategy": {
        "label": "Context Strategy",
        "group": "agent",
        "type": "select",
        "validation": {"options": ["cache_first", "legacy"]},
        "description": "cache_first (default): append-only history with one-shot compaction and a "
                       "byte-stable tools/system prefix, so provider prompt caches (Anthropic, "
                       "OpenAI, DeepSeek, Gemini) serve prior tokens at ~10% price. "
                       "legacy: per-turn compress/prune cascade (smaller single requests, but "
                       "defeats prefix caching).",
        "default": "cache_first",
    },
    "agent.tool_result_max_chars": {
        "label": "Tool Result Max Chars",
        "group": "agent",
        "type": "number",
        "description": "Cap on a single tool result at insertion into context (0 = uncapped). "
                       "Oversized output keeps head+tail with an explicit re-invoke affordance.",
        "default": "30000",
        "validation": {"min": 0, "max": 500000},
    },
    "agent.stable_tools_max": {
        "label": "Stable Tools Threshold",
        "group": "agent",
        "type": "number",
        "description": "cache_first only: send the FULL tool catalog every call when it has at most "
                       "this many tools (stable cached prefix). Larger catalogs use a grow-only "
                       "session subset.",
        "default": "80",
        "validation": {"min": 1, "max": 1000},
    },
    "agent.compaction_llm": {
        "label": "LLM Compaction",
        "group": "agent",
        "type": "boolean",
        "description": "Use the LLM (cheap model via llm.small_model) for context compaction instead of "
                       "heuristic summarization. Default ON — a high-fidelity continuation summary is "
                       "what makes one-shot compaction safe.",
        "default": "true",
    },
    "agent.behavior_nudges": {
        "label": "Behavior Nudges",
        "group": "agent",
        "type": "boolean",
        "description": "Re-prompt the model when it refuses without trying tools or narrates without "
                       "acting. Self-limiting: never fires on frontier models (~zero cost), keeps "
                       "mid-tier models (deepseek-class) from stalling mid-task. Default ON.",
        "default": "true",
    },
    "llm.reasoning_effort": {
        "label": "Reasoning Effort",
        "group": "llm",
        "type": "select",
        "validation": {"options": ["", "off", "low", "medium", "high"]},
        "description": "Thinking-effort control for reasoning models via OpenRouter's unified "
                       "reasoning parameter (llm.provider=openrouter only). Unset = provider "
                       "default. Measured on prod: default effort burned 4-10k completion tokens "
                       "per call on triage-grade work.",
        "default": "",
    },
    "agent.subagent_concurrency": {
        "label": "Sub-agent Concurrency",
        "group": "agent",
        "type": "number",
        "description": "Maximum sub-agents (delegate_task / spawn_agent) running concurrently per process.",
        "default": "4",
        "validation": {"min": 1, "max": 16},
    },
    "agent.session_resume": {
        "label": "Session Resume",
        "group": "agent",
        "type": "boolean",
        "description": "Persist conversation history at each turn end and rehydrate it when a session "
                       "is recreated (process restart / pool eviction).",
        "default": "true",
    },
    "llm.thinking_budget": {
        "label": "Thinking Budget",
        "group": "llm",
        "type": "number",
        "description": "Extended-thinking token budget for the native Anthropic provider "
                       "(llm.provider=anthropic). 0 disables thinking.",
        "default": "0",
        "validation": {"min": 0, "max": 128000},
    },
    "agent.memory_llm_salience": {
        "label": "LLM Memory Salience",
        "group": "agent",
        "type": "boolean",
        "description": "Use the LLM to distill auto-memories into clean facts instead of storing raw text.",
        "default": "false",
    },
    "agent.planning_enabled": {
        "label": "Planning Enabled (legacy)",
        "group": "agent",
        "type": "boolean",
        "description": "Legacy one-shot planner LLM call for complex tasks. Superseded by the "
                       "model-maintained todo_write list; default OFF.",
        "default": "false",
    },
    "agent.tool_result_guard": {
        "label": "Tool Result Injection Guard",
        "group": "agent",
        "type": "boolean",
        "description": "Scan tool output for prompt-injection patterns and frame suspicious "
                       "content as untrusted.",
        "default": "true",
    },
    "agent.approval_mode": {
        "label": "Approval Mode",
        "group": "agent",
        "type": "select",
        "description": "When to require human approval before executing tools. "
                       "'irreversible' = only for destructive actions, 'all' = every tool call, "
                       "'none' = fully autonomous.",
        "default": "irreversible",
        "validation": {"options": ["none", "irreversible", "all"]},
    },
    "agent.data_dir": {
        "label": "Data Directory",
        "group": "agent",
        "type": "text",
        "description": "Path to the agent's data directory for file storage.",
        "default": ".ag_data",
    },

    "brain.auto_sync": {
        "label": "Auto Sync",
        "group": "brain",
        "type": "boolean",
        "description": "Automatically sync data sources on a schedule.",
        "default": "false",
    },
    "brain.sync_interval": {
        "label": "Sync Interval",
        "group": "brain",
        "type": "text",
        "description": "How often to sync data sources (e.g., '30m', '1h', '6h').",
        "default": "30m",
    },

    "memory.session_limit": {
        "label": "Session Memory Limit",
        "group": "memory",
        "type": "number",
        "description": "Maximum memories retained per session layer before consolidation.",
        "default": "50",
        "validation": {"min": 5, "max": 500},
    },
    "memory.decay_days": {
        "label": "Memory Decay Days",
        "group": "memory",
        "type": "number",
        "description": "Days before unused session memories start decaying in relevance scoring.",
        "default": "7",
        "validation": {"min": 1, "max": 365},
    },

    "auth.login_max_failures": {
        "label": "Max Login Failures",
        "group": "auth",
        "type": "number",
        "description": "Number of failed login attempts before account lockout.",
        "default": "5",
        "validation": {"min": 3, "max": 20},
    },
    "auth.login_window_seconds": {
        "label": "Login Window (seconds)",
        "group": "auth",
        "type": "number",
        "description": "Sliding time window for counting failed login attempts.",
        "default": "300",
        "validation": {"min": 60, "max": 3600},
    },
    "auth.login_lockout_seconds": {
        "label": "Lockout Duration (seconds)",
        "group": "auth",
        "type": "number",
        "description": "Duration of account lockout after exceeding max failures.",
        "default": "900",
        "validation": {"min": 60, "max": 86400},
    },

    "secret.encryption_key": {
        "label": "Encryption Key",
        "group": "secret",
        "type": "password",
        "description": "Fernet encryption key for credential vault. REQUIRED in production. "
                       "Changing this will make existing encrypted secrets unreadable!",
        "default": "",
        "is_secret": True,
        "requires_restart": True,
    },
    "secret.private_key": {
        "label": "JWT Private Key (RSA)",
        "group": "secret",
        "type": "textarea",
        "description": "RSA private key in PEM format for signing JWT tokens.",
        "default": "",
        "is_secret": True,
        "requires_restart": True,
    },
    "secret.public_key": {
        "label": "JWT Public Key (RSA)",
        "group": "secret",
        "type": "textarea",
        "description": "RSA public key in PEM format for verifying JWT tokens.",
        "default": "",
        "is_secret": True,
        "requires_restart": True,
    },
    "secret.issuer": {
        "label": "JWT Issuer",
        "group": "secret",
        "type": "text",
        "description": "JWT issuer claim value.",
        "default": "Storeglance.Agent",
        "requires_restart": True,
    },
    "secret.audience": {
        "label": "JWT Audience",
        "group": "secret",
        "type": "text",
        "description": "JWT audience claim value.",
        "default": "Storeglance.Agent.Clients",
        "requires_restart": True,
    },
    "secret.field_encryption_key": {
        "label": "Field Encryption Key (MSystem-shared)",
        "group": "secret",
        "type": "password",
        "description": "32-byte base64 AES-256-GCM key, MUST equal MSystem's "
                       "Security:FieldEncryptionKey for this environment. Used to "
                       "decrypt the agent DB password from tenant_mgmt.app_databases. "
                       "Also settable via AG_FIELD_ENCRYPTION_KEY env var.",
        "default": "",
        "is_secret": True,
        "requires_restart": True,
    },
    "secret.service_token": {
        "label": "Service-to-Service Token",
        "group": "secret",
        "type": "password",
        "description": "Shared bearer for internal S2S calls (MSystem → agent "
                       "DELETE /admin/tenants/{id}/stores on offboard). Must match "
                       "MSystem's configured token. Also settable via AG_SERVICE_TOKEN.",
        "default": "",
        "is_secret": True,
        "requires_restart": False,
    },
    "secret.token_expiry_minutes": {
        "label": "Access Token Expiry (min)",
        "group": "secret",
        "type": "number",
        "description": "JWT access token lifetime in minutes.",
        "default": "1440",
        "validation": {"min": 5, "max": 43200},
        "requires_restart": True,
    },
    "secret.refresh_token_expiry_days": {
        "label": "Refresh Token Expiry (days)",
        "group": "secret",
        "type": "number",
        "description": "JWT refresh token lifetime in days.",
        "default": "7",
        "validation": {"min": 1, "max": 365},
        "requires_restart": True,
    },


    "redis.url": {
        "label": "Redis URL",
        "group": "redis",
        "type": "text",
        "description": "Redis connection URL (e.g., redis://localhost:6379/0). "
                       "Leave empty to use in-process fallback.",
        "default": "",
    },

    "gateway.platform": {
        "label": "Gateway Platform",
        "group": "gateway",
        "type": "select",
        "description": "Primary platform adapter for the gateway process.",
        "default": "telegram",
        "validation": {"options": ["telegram", "teams", "both"]},
    },
    "gateway.telegram.bot_token": {
        "label": "Telegram Bot Token",
        "group": "gateway",
        "type": "password",
        "description": "BotFather-issued token for the Telegram bot.",
        "default": "",
        "is_secret": True,
        "managed_by": "/chat/settings/integrations",
    },
    "gateway.telegram.allowed_users": {
        "label": "Telegram Allowed Users",
        "group": "gateway",
        "type": "text",
        "description": "Comma-separated Telegram user IDs allowed to interact with the bot.",
        "default": "",
        "managed_by": "/chat/settings/integrations",
    },
    "gateway.telegram.home_channel": {
        "label": "Telegram Home Channel",
        "group": "gateway",
        "type": "text",
        "description": "Telegram channel ID for broadcast notifications.",
        "default": "",
        "managed_by": "/chat/settings/integrations",
    },

    "workspace.root": {
        "label": "Workspace Root Path",
        "group": "workspace",
        "type": "text",
        "description": "Filesystem root for per-user sandboxed workspaces.",
        "default": "/var/agent-workspaces",
    },
    "workspace.default_quota_mb": {
        "label": "Default Quota (MB)",
        "group": "workspace",
        "type": "number",
        "description": "Default disk quota per user workspace in megabytes.",
        "default": "500",
        "validation": {"min": 10, "max": 100000},
    },
    "workspace.tenant_id": {
        "label": "Tenant ID",
        "group": "workspace",
        "type": "text",
        "description": "Tenant identifier for workspace path scoping.",
        "default": "default",
    },

    "shell.os_sandbox": {
        "label": "OS Sandbox Mode",
        "group": "shell",
        "type": "select",
        "description": "Controls whether shell commands run inside a bubblewrap sandbox. "
                       "'auto' = use bwrap if available, 'require' = fail without bwrap, "
                       "'off' = regex-only sandbox.",
        "default": "auto",
        "validation": {"options": ["auto", "off", "require"]},
    },

    "db.url": {
        "label": "Database URL",
        "group": "db",
        "type": "password",
        "description": "PostgreSQL connection URL. Set via AG_DATABASE_URL env var. "
                       "Changing this requires a full restart.",
        "default": "",
        "is_secret": True,
        "requires_restart": True,
    },
    "db.port": {
        "label": "Database Port",
        "group": "db",
        "type": "number",
        "description": "PostgreSQL port (informational — derived from db.url).",
        "default": "5432",
        "requires_restart": True,
    },
    "db.name": {
        "label": "Database Name",
        "group": "db",
        "type": "text",
        "description": "PostgreSQL database name.",
        "default": "agent_uat",
        "requires_restart": True,
    },

    "app.env": {
        "label": "Environment",
        "group": "app",
        "type": "select",
        "description": "Application environment. 'production' enforces stricter security checks.",
        "default": "development",
        "validation": {"options": ["development", "staging", "production"]},
    },

    "bot.approval_mode": {
        "label": "Bot Approval Mode",
        "group": "bot",
        "type": "select",
        "description": "Approval gate behavior for chatbot widget interactions.",
        "default": "irreversible",
        "validation": {"options": ["none", "irreversible", "all"]},
    },
}


def get_config_type(key: str) -> str:
    """Get the expected type for a config key."""
    meta = CONFIG_METADATA.get(key, {})
    return meta.get("type", "text")


def get_config_group(key: str) -> str:
    """Get the group for a config key (falls back to first prefix segment)."""
    meta = CONFIG_METADATA.get(key)
    if meta:
        return meta.get("group", key.split(".")[0])
    return key.split(".")[0]


def get_group_label(group: str) -> str:
    """Get the display label for a config group."""
    return GROUP_METADATA.get(group, {}).get("label", group.title())


def validate_config_value(key: str, value: str) -> tuple:
    """Validate a config value against its metadata.

    Returns:
        (is_valid: bool, error_message: str | None)
    """
    meta = CONFIG_METADATA.get(key)
    if not meta:
        return True, None

    cfg_type = meta.get("type", "text")
    validation = meta.get("validation", {})

    if cfg_type == "number":
        try:
            num = float(value) if "." in value else int(value)
        except (ValueError, TypeError):
            return False, f"'{key}' must be a number, got: '{value}'"
        if "min" in validation and num < validation["min"]:
            return False, f"'{key}' must be >= {validation['min']}, got: {num}"
        if "max" in validation and num > validation["max"]:
            return False, f"'{key}' must be <= {validation['max']}, got: {num}"

    elif cfg_type == "boolean":
        if value.lower() not in ("true", "false", "1", "0", "yes", "no", "on", "off"):
            return False, f"'{key}' must be a boolean (true/false/yes/no/1/0), got: '{value}'"

    elif cfg_type == "select":
        options = validation.get("options", [])
        if options and value not in options:
            return False, f"'{key}' must be one of {options}, got: '{value}'"

    return True, None


def get_all_known_keys() -> list:
    """Get all config keys defined in metadata, sorted by group then key."""
    return sorted(CONFIG_METADATA.keys(), key=lambda k: (
        GROUP_METADATA.get(CONFIG_METADATA[k].get("group", ""), {}).get("sort", 99),
        k
    ))
