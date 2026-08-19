"""Kernel C tools. Intercepted by the sidecar — never forwarded to the τ² gym."""

from __future__ import annotations

from typing import Any, Iterable

KERNEL_SELF_TOOLS = frozenset({"get_agent_graph", "set_agent_graph"})


def is_kernel_self_tool(name: str | None) -> bool:
    return bool(name) and name in KERNEL_SELF_TOOLS


def strip_kernel_self_tools(tool_calls: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Drop leaked get/set_agent_graph calls so the gym env never sees them."""
    out: list[dict[str, Any]] = []
    for tc in tool_calls or []:
        name = tc.get("name") if isinstance(tc, dict) else None
        if is_kernel_self_tool(name):
            continue
        out.append(tc)
    return out
