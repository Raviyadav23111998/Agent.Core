"""Structured JSON logging for centralized log aggregation (Loki/Promtail).

Writes NLog-compatible JSON to /logs/<AG_ENVIRONMENT>/agent/agent.log so the
existing promtail `dotnet_logs` pipeline ingests it with environment/app/level
labels and correlationId->Tempo linking — the same convention the .NET apps use.

Console logging is kept human-readable. If the log directory is not writable
(e.g. local dev without /logs), the file handler is skipped gracefully.
"""
import datetime
import json
import logging
import logging.handlers
import os

_OPTIONAL_FIELDS = ("correlationId", "sessionId", "company", "email", "ip", "userAgent")


class JsonLogFormatter(logging.Formatter):
    """Emit one JSON object per record matching the NLog JsonLayout schema."""

    def __init__(self, app: str, environment: str, version: str):
        super().__init__()
        self._app = app
        self._environment = environment
        self._version = version

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": datetime.datetime.fromtimestamp(record.created).astimezone().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "app": self._app,
            "environment": self._environment,
            "version": self._version,
            "message": record.getMessage(),
        }
        for field in _OPTIONAL_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_structured_logging(app: str = "agent", level: int = logging.INFO) -> None:
    """Configure root logging: human-readable console + JSON file for promtail.

    Idempotent — safe to call once per process at startup. Reads AG_ENVIRONMENT
    (Production/UAT/SIT, default 'dev') and AG_LOG_DIR (default '/logs').
    """
    environment = os.environ.get("AG_ENVIRONMENT", "dev")
    version = os.environ.get("AG_VERSION", "1.0.0")
    root = logging.getLogger()
    root.setLevel(level)

    if not any(getattr(h, "_ag_console", False) for h in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
        console._ag_console = True
        root.addHandler(console)

    if any(getattr(h, "_ag_json_file", False) for h in root.handlers):
        return

    base_dir = os.environ.get("AG_LOG_DIR", "/logs")
    log_dir = os.path.join(base_dir, environment, app)
    try:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, f"{app}.log"),
            maxBytes=20 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(JsonLogFormatter(app, environment, version))
        file_handler._ag_json_file = True
        root.addHandler(file_handler)
        logging.getLogger(__name__).info(
            "structured JSON logging -> %s (env=%s)", log_dir, environment)
    except OSError as exc:
        logging.getLogger(__name__).warning(
            "structured JSON logging disabled (cannot write %s): %s", log_dir, exc)
