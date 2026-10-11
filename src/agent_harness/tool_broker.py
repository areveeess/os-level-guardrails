"""
Tool Broker — the only path from the agent to external app tools
=================================================================
App tools (email, banking, smart home, ...) are external services. The
sandbox has no network, so `call_tool` actions cannot run inside it; they go
to this broker, which sits outside the sandbox and is therefore a complete-
mediation point for every app-level action.

In the protected configuration the broker enforces **task-scoped
capabilities** — an access-matrix row for "the agent working on this task":
only the tools the user's task needs may be invoked. Everything else is
denied and logged as layer `tool_broker`.

Tool backends are simulated (fixed responses), as in InjecAgent.
"""

from dataclasses import dataclass, field

from src.agent_harness.executors import Executor, Observation

DEFAULT_RESPONSE = '{"success": true}'


@dataclass
class ToolCall:
    tool: str
    args: dict
    allowed: bool
    response: str = ""


class ToolBroker(Executor):

    def __init__(self, inner: Executor, responses: dict = None, known_tools: set = None,
                 capabilities: set = None, policy=None):
        """
        Args:
            inner: Executor for every non-tool action (shell, files, http).
            responses: tool name -> simulated response text.
            known_tools: tools that exist; calls to others return an error.
            capabilities: tools this task may invoke. None = no restriction (unprotected).
            policy: Optional PolicyEngine, used only to log decisions to the dashboard.
        """
        self.inner = inner
        self.responses = responses or {}
        self.known_tools = known_tools
        self.capabilities = capabilities
        self.policy = policy
        self.calls: list = []

    def execute(self, action: dict) -> Observation:
        if action.get("action") != "call_tool":
            return self.inner.execute(action)

        tool = str(action.get("tool", ""))
        args = action.get("args") or {}
        if self.known_tools is not None and tool not in self.known_tools:
            self.calls.append(ToolCall(tool, args, allowed=False))
            return Observation("error", reason=f"Unknown tool: {tool}")

        if self.capabilities is not None and tool not in self.capabilities:
            self.calls.append(ToolCall(tool, args, allowed=False))
            reason = f"Tool '{tool}' is not in this task's capability list"
            if self.policy:
                self.policy._log_decision("call_tool", tool, "BLOCK", reason, layer="tool_broker")
            return Observation("blocked", layer="tool_broker", reason=reason)

        response = self.responses.get(tool, DEFAULT_RESPONSE)
        self.calls.append(ToolCall(tool, args, allowed=True, response=response))
        if self.policy and self.capabilities is not None:
            self.policy._log_decision("call_tool", tool, "ALLOW", "Tool is in this task's capability list",
                                      layer="tool_broker")
        return Observation("success", output=response)

    def close(self):
        self.inner.close()
