"""
Page Tools — the in-app web copilot's hands on the browser.
═══════════════════════════════════════════════════════════════════════════
These tools let the agent perceive and drive the user's CURRENT screen in the
web app. Like the VS Code filesystem tools, they are CLIENT-EXECUTED: the agent
runs on the server, but each call is delegated over the WebSocket to the browser
(RemoteToolExecutor → tool_request → CopilotEngineService → tool_result). The
tool names are listed in core.tool_executor.LOCAL_TOOLS so they route to the
client; the server-side bodies here are only a graceful fallback for when no
browser copilot is attached.

The copilot drives the screen with ONE addressing model — by human LABEL:
  1. page_observe()              → see the screen's fields/buttons/NAV LINKS by LABEL
  2. To reach another screen, CLICK the matching sidebar/menu link found in
     page_observe (via page_act) — do NOT guess route URLs. Use page_navigate
     only when you already know the app's exact route string.
  3. page_act([{do,target,value}…]) → fill/select/click SEVERAL elements (by label)
                                    in one call, e.g. fill the form + click Save.
After every click/navigation the page changes — re-observe before acting again.
"""

_BROWSER_ONLY = (
    "page tools run in the user's browser (web copilot). No browser session is "
    "attached to this run, so this action cannot be performed here."
)


class PageTools:
    """Registers the page_* tools. Execution is delegated to the browser."""

    def page_observe(self) -> str:
        return _BROWSER_ONLY

    def page_act(self, steps: str) -> str:
        return _BROWSER_ONLY

    def page_navigate(self, route: str) -> str:
        return _BROWSER_ONLY

    def page_wait(self) -> str:
        return _BROWSER_ONLY

    def page_read(self) -> str:
        return _BROWSER_ONLY

    def tool_specs(self) -> list:
        """Declarative list of the page_* tools (see tools/tool_spec.Tool).

        Adding a page tool = append one Tool(...) here. The copilot targets
        elements by their human LABEL — page_observe to see them, page_act to
        fill/click them. Each tool declares its `kind` (read | write) — the
        framework maps that to the canonical policy (see tools/tool_policy.py).
        """
        from tools.tool_spec import Tool
        return [
            Tool(
                "page_observe",
                "Look at the user's CURRENT screen. Returns the interactive "
                "elements (fields, dropdowns, checkboxes, buttons, NAV/SIDEBAR "
                "LINKS) by their human LABEL (and a dropdown's options). You "
                "target elements by label — there are no ids to track. ALWAYS "
                "call this first to find how to reach another screen: locate the "
                "matching menu/sidebar link here and click it — do NOT guess a "
                "route URL. Call again after any navigation or click that changes "
                "the page.",
                {}, self.page_observe, side_effect="read",
            ),
            Tool(
                "page_act",
                "Perform SEVERAL actions in ONE call — the efficient way to fill "
                "a form and submit. `steps` is a JSON array of "
                "{\"do\",\"target\",\"value\"} executed in order on the live page. "
                "do = fill | select | check | click. target = the element's LABEL "
                "from page_observe (matched fuzzily, case-insensitive). value = "
                "the text (fill) or option (select). Returns each step's result + "
                "the updated screen. Example: "
                "[{\"do\":\"fill\",\"target\":\"Application Name\",\"value\":\"Web Copilot\"},"
                "{\"do\":\"select\",\"target\":\"Type\",\"value\":\"Internal\"},"
                "{\"do\":\"click\",\"target\":\"Create Application\"}]",
                {"steps": "string"}, self.page_act,
                param_descriptions={"steps": "A JSON array of action steps {do, target, value}."},
                side_effect="write",
            ),
            Tool(
                "page_navigate",
                "Jump the web app DIRECTLY to a route string — use this ONLY when "
                "you already KNOW the app's exact route (e.g. the user gave it, or "
                "you saw it). Do NOT guess routes: if you're unsure of the path, "
                "call page_observe and CLICK the matching sidebar/menu link "
                "instead — guessing routes is the main cause of getting stuck. "
                "Waits for the destination to load and RETURNS the new screen — "
                "re-observe after; any element references from the previous screen "
                "are invalid.",
                {"route": "string"}, self.page_navigate,
                param_descriptions={
                    "route": "The app route path to go to (starts with '/'). Only "
                             "pass a route you actually know — never a guess.",
                },
                side_effect="write",
            ),
            Tool(
                "page_wait",
                "Wait for the current screen to finish loading (spinners gone, "
                "form rendered). Use this if page_observe looks empty or "
                "incomplete right after navigating — many screens load their "
                "fields after an async data fetch.",
                {}, self.page_wait, side_effect="write",
            ),
            Tool(
                "page_read",
                "Read the VISIBLE TEXT of the current screen (labels, values, "
                "status, info) — use this to inspect/report what a screen shows "
                "when the data isn't an editable field (e.g. 'check the current "
                "LLM configuration').",
                {}, self.page_read, side_effect="read",
            ),
        ]

    def register(self, registry) -> None:
        """Register page_* schemas on the given ToolRegistry."""
        from tools.tool_spec import register_specs
        register_specs(registry, self.tool_specs())
