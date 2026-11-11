"""
Shared pytest fixtures + a stub app_config so core modules import without a DB.

Run from agent.core/:   python -m pytest tests/ -q
"""

import sys
import os
import types

_CORE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)


class _StubConfig:
    _vals = {}

    def get(self, key, default=None, env=None):
        return self._vals.get(key, default)

    def get_int(self, key, default=0):
        v = self._vals.get(key)
        return int(v) if v is not None else default

    def get_bool(self, key, default=False):
        v = self._vals.get(key, default)
        return str(v).lower() in ("1", "true", "yes", "on") if isinstance(v, str) else bool(v)

    def resolve_env(self, env_var, default=""):
        return self._vals.get(env_var, default)

    def set(self, key, value, is_secret=False):
        self._vals[key] = value


if "core.config" not in sys.modules:
    sys.modules["core.config"] = types.SimpleNamespace(app_config=_StubConfig())
