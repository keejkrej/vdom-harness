"""τ²-bench agent: official HalfDuplexAgent wrapping the vdom TypeScript graph."""

__all__ = ["register", "DEFAULT_MODEL"]

DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"


def register() -> None:
    from tau2_vdom.agent import register_vdom_agent
    from tau2_vdom.scripted_user import register_scripted_user

    register_vdom_agent()
    register_scripted_user()
