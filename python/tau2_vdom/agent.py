"""Official τ² HalfDuplexAgent that delegates each turn to the vdom graph sidecar."""

from __future__ import annotations

import os
from typing import Any, Optional

from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
)
from tau2.environment.tool import Tool
from tau2.registry import registry

from tau2_vdom.sidecar import VdomSidecar, default_sidecar

DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"
DEFAULT_TECHNIQUE = "one-shot"

# Harvested by the runner after a simulation (per-process).
TURN_TRACES: list[dict[str, Any]] = []
TURN_TRACES_BY_TASK: dict[str, list[dict[str, Any]]] = {}


def reset_turn_traces() -> None:
    TURN_TRACES.clear()
    TURN_TRACES_BY_TASK.clear()


def _openai_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in tools:
        schema = t.openai_schema
        fn = schema.get("function", schema)
        out.append(
            {
                "name": fn.get("name") or t.name,
                "description": fn.get("description") or t.name,
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return out


def _msg_to_json(m: Message) -> dict[str, Any]:
    if isinstance(m, ToolMessage):
        return {
            "role": "tool",
            "content": m.content or "",
            "name": getattr(m, "id", None) or "tool",
            "tool_call_id": m.id,
        }
    payload: dict[str, Any] = {
        "role": getattr(m, "role", "user"),
        "content": getattr(m, "content", None) or "",
    }
    tool_calls = getattr(m, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = [
            {
                "id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments or {},
            }
            for tc in tool_calls
        ]
    return payload


class VdomAgentState:
    def __init__(self, messages: list[Message]):
        self.messages = messages
        self.traces: list[dict[str, Any]] = []


class VdomAgent(HalfDuplexAgent["VdomAgentState"]):
    """vdom AgentGraph + provider.completeTurn, driven by τ²'s orchestrator."""

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        sidecar: Optional[VdomSidecar] = None,
        technique: Optional[str] = None,
        task_id: Optional[str] = None,
        **_kwargs,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.llm = llm or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL
        self.llm_args = llm_args or {}
        self.technique = technique or os.environ.get("VDOM_TAU2_TECHNIQUE") or DEFAULT_TECHNIQUE
        self.task_id = task_id
        self._sidecar = sidecar or default_sidecar()
        self._tool_schemas = _openai_tools(tools)
        self.last_traces: list[dict[str, Any]] = []

    def set_seed(self, seed: int) -> None:
        return

    def get_init_state(self, message_history: Optional[list[Message]] = None) -> VdomAgentState:
        return VdomAgentState(messages=list(message_history or []))

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: VdomAgentState
    ) -> tuple[AssistantMessage, VdomAgentState]:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        elif message is not None:
            state.messages.append(message)

        payload = {
            "op": "turn",
            "policy": self.domain_policy,
            "tools": self._tool_schemas,
            "messages": [_msg_to_json(m) for m in state.messages],
            "technique": self.technique,
            "model": self.llm,
        }
        data = self._sidecar.request(payload)
        traces = data.get("traces") or []
        state.traces.extend(traces)
        self.last_traces = traces
        TURN_TRACES.extend(traces)
        key = self.task_id or "_"
        TURN_TRACES_BY_TASK.setdefault(key, []).extend(traces)

        tool_calls_raw = data.get("tool_calls") or []
        if tool_calls_raw:
            tool_calls = [
                ToolCall(
                    id=tc.get("id") or f"call_{i}",
                    name=tc["name"],
                    arguments=tc.get("arguments") or {},
                    requestor="assistant",
                )
                for i, tc in enumerate(tool_calls_raw)
            ]
            assistant = AssistantMessage(role="assistant", content=None, tool_calls=tool_calls)
        else:
            content = data.get("content") or ""
            assistant = AssistantMessage(role="assistant", content=content)

        state.messages.append(assistant)
        return assistant, state


def create_vdom_agent(tools, domain_policy, **kwargs):
    """Factory for `registry.register_agent_factory` / `tau2 run --agent vdom`."""
    task = kwargs.get("task")
    return VdomAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        technique=(kwargs.get("llm_args") or {}).get("technique")
        if isinstance(kwargs.get("llm_args"), dict)
        else None,
        task_id=getattr(task, "id", None),
    )


def register_vdom_agent() -> None:
    if "vdom" not in registry.get_agents():
        registry.register_agent_factory(create_vdom_agent, "vdom")
