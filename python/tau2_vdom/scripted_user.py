"""Credential-free user for `--domain mock` smoke. Not a retail user simulator."""

from __future__ import annotations

from typing import Optional

from tau2.data_model.message import AssistantMessage, Message, MultiToolMessage, UserMessage
from tau2.registry import registry
from tau2.user.user_simulator_base import (
    STOP,
    HalfDuplexUser,
    UserState,
    ValidUserInputMessage,
)


class ScriptedUser(HalfDuplexUser[UserState]):
    """Sends the task instructions once, then ###STOP### after the agent speaks."""

    def __init__(
        self,
        tools=None,
        instructions: Optional[str] = None,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        **_kwargs,
    ):
        super().__init__(instructions=instructions, tools=tools)
        self._uttered = False

    def set_seed(self, seed: int) -> None:
        return

    def get_init_state(self, message_history: Optional[list[Message]] = None) -> UserState:
        return UserState(system_messages=[], messages=list(message_history or []))

    def _first_line(self) -> str:
        raw = str(self.instructions or "").strip()
        if "Important Meeting" in raw and "user_1" in raw:
            return "Please create a new task called Important Meeting for user_1."
        if "task_1" in raw and "completed" in raw.lower():
            return "Please mark task_1 as completed."
        if "task_2" in raw and "completed" in raw.lower():
            return "Please mark task_2 as completed."
        if "delete" in raw.lower():
            return "Please delete all of my current tasks."
        if raw:
            # Last non-empty line is usually the concrete instruction.
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            return lines[-1] if lines else "I need help with a task."
        return "I need help with a task."

    def generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> tuple[UserMessage, UserState]:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        elif isinstance(message, AssistantMessage):
            state.messages.append(message)

        if not self._uttered:
            self._uttered = True
            user = UserMessage(role="user", content=self._first_line())
            state.messages.append(user)
            return user, state

        user = UserMessage(role="user", content=STOP)
        state.messages.append(user)
        return user, state


def register_scripted_user() -> None:
    if "scripted_user" not in registry.get_users():
        registry.register_user(ScriptedUser, "scripted_user")
