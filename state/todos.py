"""Model-maintained todo list — the Claude Code TodoWrite mechanism.
═══════════════════════════════════════════════════════════════════

Long tasks stay coherent because the MODEL maintains a live task list (one
`todo_write` tool call replaces the whole list), and the harness nudges it
with a suffix-only reminder when the list goes stale. This replaces the
old one-shot planner LLM call: no extra request, and the plan evolves with
the work instead of going stale after iteration 1.

The store is per-conversation state that lives on the AgentLoop. The tool
callback is registered once on a (shared) registry, so it reaches the
current conversation's store through a ContextVar the loop binds for the
duration of each turn — the same pattern used for user/tenant context.
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

VALID_STATUSES = ("pending", "in_progress", "completed")

MAX_ITEMS = 50
MAX_CONTENT_CHARS = 300

current_todo_store: ContextVar[Optional["TodoStore"]] = ContextVar(
    "current_todo_store", default=None)


class TodoStore:
    """The live task list for one conversation."""

    def __init__(self):
        self.items: List[Dict[str, str]] = []
        self.update_count = 0

    def update(self, todos: Any) -> str:
        """Replace the list. Returns a rendered confirmation for the model."""
        if isinstance(todos, str):
            try:
                todos = json.loads(todos)
            except json.JSONDecodeError as e:
                return f"Error: todos must be a JSON array of items: {e}"
        if not isinstance(todos, list):
            return "Error: todos must be an array of {content, status} items."

        parsed: List[Dict[str, str]] = []
        for item in todos[:MAX_ITEMS]:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            status = str(item.get("status", "pending")).strip().lower()
            if status not in VALID_STATUSES:
                status = "pending"
            parsed.append({"content": content[:MAX_CONTENT_CHARS],
                           "status": status})

        self.items = parsed
        self.update_count += 1

        counts = self.counts()
        note = ""
        if counts["in_progress"] > 1:
            note = " Keep exactly ONE item in_progress at a time."
        return (f"Todo list updated — {counts['pending']} pending, "
                f"{counts['in_progress']} in progress, "
                f"{counts['completed']} completed.{note}\n{self.render()}")

    def counts(self) -> Dict[str, int]:
        c = {s: 0 for s in VALID_STATUSES}
        for t in self.items:
            c[t["status"]] += 1
        return c

    @property
    def open_count(self) -> int:
        return sum(1 for t in self.items if t["status"] != "completed")

    def render(self) -> str:
        if not self.items:
            return "(todo list is empty)"
        marks = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
        return "\n".join(f"{marks[t['status']]} {t['content']}"
                         for t in self.items)


TODO_WRITE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "todos": {
            "type": "array",
            "description": ("The COMPLETE todo list (replaces the previous "
                            "list entirely — include unchanged items)."),
            "items": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The task, imperative form.",
                    },
                    "status": {
                        "type": "string",
                        "enum": list(VALID_STATUSES),
                    },
                },
                "required": ["content", "status"],
            },
        },
    },
    "required": ["todos"],
}


def todo_write(todos: Any = None) -> str:
    """Tool callback — routes to the current conversation's store."""
    store = current_todo_store.get()
    if store is None:
        return "Error: no active conversation for the todo list."
    return store.update(todos or [])
