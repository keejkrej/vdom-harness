"""Run official τ² evaluations with the vdom agent. Writes eval/tau2/*.json."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "eval" / "tau2"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
METRIC_NOTE = (
    "pass^k is reliability across k independent trials (Yao et al. 2024; "
    "Barres et al. 2025). The accompanying paper treats pass^k as first-passage p_hit."
)
PAPER_REPO = "https://github.com/keejkrej/agent-stochastic-dynamics"
TAU2_REPO = "https://github.com/sierra-research/tau2-bench"
# Airline write tools whose miss should select the policy-checklist I_loop graph.
# Mock update_task_status is intentionally excluded so 0 → 0.5 → 1.0 still holds.
POLICY_WRITE_TOOLS = frozenset(
    {
        "cancel_reservation",
        "update_reservation_flights",
        "update_reservation_baggages",
        "update_reservation_passengers",
        "book_reservation",
    }
)
_REFUSE_CANCEL = re.compile(
    r"unable to cancel|cannot cancel|can't cancel|can not cancel|"
    r"no way for me to|no mechanism|not possible to cancel|"
    r"i(?:'m| am) unable to|unfortunately.{0,60}cancel|"
    r"i(?:'m| am) afraid.{0,60}cancel|i(?:'m| am) sorry.{0,60}unable",
    re.I | re.S,
)
_INVENTED_POLICY = re.compile(
    r"no-?show|no mechanism|i have no (?:way|mechanism)|"
    r"there is no way for me to|not possible to (?:make|process) (?:a )?no-?show|"
    r"i have no (?:tool|api) to cancel|personal reason|"
    r"change of plan is not (?:covered|eligible)|"
    r"not a (?:valid|covered) (?:personal )?reason",
    re.I,
)


def _has_live_key() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"))


def _apply_openrouter_defaults() -> None:
    """Map OPENROUTER_API_KEY onto the OpenAI-compatible env the TS provider reads."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = key
    if key:
        os.environ.setdefault("OPENAI_BASE_URL", OPENROUTER_BASE)
        os.environ.setdefault("OPENAI_MODEL", DEFAULT_MODEL)
        # LiteLLM user simulator (live retail) uses openrouter/<slug>
        os.environ.setdefault("OR_SITE_URL", "https://github.com/keejkrej/vdom-harness")
        os.environ.setdefault("OR_APP_NAME", "vdom-harness")
        _pin_tau2_judges(os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL)


def _pin_tau2_judges(model: str) -> str:
    """Force tau2 NL-assertion / user-sim defaults off gpt-4.1 onto the live model."""
    judge = _litellm_user_model(model)
    try:
        import tau2.config as tau2_config

        tau2_config.DEFAULT_LLM_NL_ASSERTIONS = judge
        tau2_config.DEFAULT_LLM_USER = judge
        tau2_config.DEFAULT_LLM_ENV_INTERFACE = judge
    except Exception:
        pass
    try:
        import tau2.evaluator.evaluator_nl_assertions as nla

        nla.DEFAULT_LLM_NL_ASSERTIONS = judge
    except Exception:
        pass
    os.environ["VDOM_PINNED_JUDGE"] = judge
    _install_model_audit()
    return judge


def _install_model_audit() -> None:
    """Log every LiteLLM model name (never keys) so we can detect gpt-4.1 leakage."""
    if os.environ.get("VDOM_MODEL_AUDIT") == "1":
        return
    os.environ["VDOM_MODEL_AUDIT"] = "1"
    path = EVAL_DIR / "model-audit.log"
    try:
        import litellm

        orig = litellm.completion

        def wrapped(*args, **kwargs):
            model = kwargs.get("model") or (args[0] if args else "?")
            EVAL_DIR.mkdir(parents=True, exist_ok=True)
            with path.open("a") as fh:
                fh.write(f"{model}\n")
            if "gpt-4.1" in str(model):
                raise RuntimeError(f"blocked gpt-4.1 leakage: {model}")
            return orig(*args, **kwargs)

        litellm.completion = wrapped
    except Exception:
        pass


def _litellm_user_model(model: str) -> str:
    if "/" in model and not model.startswith("openrouter/"):
        # deepseek/deepseek-v4-flash-0731 → openrouter/deepseek/deepseek-v4-flash-0731
        if model.startswith("deepseek/") or os.environ.get("OPENROUTER_API_KEY"):
            return f"openrouter/{model}"
    return model


def _serialize_message(m: Any) -> dict[str, Any]:
    if hasattr(m, "model_dump"):
        return m.model_dump(mode="json")
    return {"repr": str(m)}


def _actions_from_messages(messages: list[Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for m in messages:
        role = getattr(m, "role", None)
        if role == "assistant":
            tcs = getattr(m, "tool_calls", None) or []
            if tcs:
                for tc in tcs:
                    key = f"tool:{tc.name}:{json.dumps(tc.arguments or {}, sort_keys=True)}"
                    seen[key] = seen.get(key, 0) + 1
                    actions.append(
                        {
                            "kind": "tool",
                            "text": tc.name,
                            "toolName": tc.name,
                            "toolArgs": tc.arguments or {},
                            "ok": True,
                            "repeat": seen[key] > 1,
                        }
                    )
            elif getattr(m, "content", None):
                actions.append({"kind": "text", "text": m.content, "ok": True, "repeat": False})
        elif role == "tool":
            err = bool(getattr(m, "error", False))
            if err and actions:
                actions[-1]["ok"] = False
    return actions


def _assistant_text(
    actions: list[dict[str, Any]],
    messages: list[Any] | None = None,
) -> str:
    parts = [str(a.get("text") or "") for a in actions if a.get("kind") == "text"]
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") == "assistant":
            parts.append(str(m.get("content") or ""))
        elif getattr(m, "role", None) == "assistant":
            parts.append(str(getattr(m, "content", "") or ""))
    return "\n".join(parts)


def _missed_actions_from_reward_info(reward_info: Any) -> list[dict[str, Any]]:
    if reward_info is None:
        return []
    if isinstance(reward_info, dict):
        if reward_info.get("missedActions"):
            return [
                {"name": a.get("name"), "arguments": a.get("arguments") or {}}
                for a in reward_info["missedActions"]
                if a.get("name")
            ]
        checks = reward_info.get("action_checks") or []
    else:
        checks = getattr(reward_info, "action_checks", None) or []
    missed: list[dict[str, Any]] = []
    for check in checks:
        if isinstance(check, dict):
            if check.get("action_match") is not False:
                continue
            action = check.get("action") or {}
            name = action.get("name") or check.get("name")
            args = action.get("arguments") or check.get("arguments") or {}
        else:
            if getattr(check, "action_match", True) is not False:
                continue
            action = getattr(check, "action", None)
            name = getattr(action, "name", None) if action is not None else None
            args = getattr(action, "arguments", None) if action is not None else {}
        if name:
            missed.append({"name": name, "arguments": args or {}})
    return missed


def serialize_reward_info(reward_info: Any) -> dict[str, Any] | None:
    """Persist tau2 RewardInfo. On airline, ACTION / nl_assertions are diagnostics only
    (reward_basis is DB × COMMUNICATE); still keep them so I_loop can see missed writes."""
    if reward_info is None:
        return None
    if hasattr(reward_info, "model_dump"):
        raw = reward_info.model_dump(mode="json")
    elif isinstance(reward_info, dict):
        raw = reward_info
    else:
        raw = {}

    def _compact_action_check(check: Any) -> dict[str, Any]:
        if isinstance(check, dict):
            action = check.get("action") or {}
            return {
                "name": action.get("name") or check.get("name"),
                "arguments": action.get("arguments") or check.get("arguments") or {},
                "action_match": check.get("action_match"),
                "action_reward": check.get("action_reward"),
                "tool_type": check.get("tool_type"),
            }
        action = getattr(check, "action", None)
        return {
            "name": getattr(action, "name", None) if action is not None else None,
            "arguments": (getattr(action, "arguments", None) or {}) if action is not None else {},
            "action_match": getattr(check, "action_match", None),
            "action_reward": getattr(check, "action_reward", None),
            "tool_type": getattr(check, "tool_type", None),
        }

    def _compact_comm(check: Any) -> dict[str, Any]:
        if isinstance(check, dict):
            return {
                "info": check.get("info"),
                "met": check.get("met"),
                "justification": check.get("justification"),
            }
        return {
            "info": getattr(check, "info", None),
            "met": getattr(check, "met", None),
            "justification": getattr(check, "justification", None),
        }

    def _compact_nl(check: Any) -> dict[str, Any]:
        if isinstance(check, dict):
            return {
                "nl_assertion": check.get("nl_assertion"),
                "met": check.get("met"),
                "justification": check.get("justification"),
            }
        return {
            "nl_assertion": getattr(check, "nl_assertion", None),
            "met": getattr(check, "met", None),
            "justification": getattr(check, "justification", None),
        }

    action_checks = raw.get("action_checks") or getattr(reward_info, "action_checks", None) or []
    communicate = raw.get("communicate_checks") or getattr(reward_info, "communicate_checks", None) or []
    nl = raw.get("nl_assertions") or getattr(reward_info, "nl_assertions", None) or []
    db = raw.get("db_check") if "db_check" in raw else getattr(reward_info, "db_check", None)
    if db is not None and hasattr(db, "model_dump"):
        db = db.model_dump(mode="json")
    missed = _missed_actions_from_reward_info(reward_info)
    return {
        "reward": raw.get("reward", getattr(reward_info, "reward", None)),
        "action_checks": [_compact_action_check(c) for c in action_checks] or None,
        "communicate_checks": [_compact_comm(c) for c in communicate] or None,
        "nl_assertions": [_compact_nl(c) for c in nl] or None,
        "db_check": db,
        "missedActions": missed,
        "reward_basis": raw.get("reward_basis") or getattr(reward_info, "reward_basis", None),
        "reward_breakdown": raw.get("reward_breakdown")
        or getattr(reward_info, "reward_breakdown", None),
    }


_HARD_INCOMPLETE_TERM = re.compile(r"timeout|hung|crash|error", re.I)


def called_write_tools(obs: dict[str, Any]) -> bool:
    return any(a and not str(a).startswith("text:") for a in (obs.get("lastActions") or []))


def has_loop_attractor(obs: dict[str, Any]) -> bool:
    if obs.get("inventedPolicy") or obs.get("refusedCancel"):
        return True
    if obs.get("techniqueRecommendation") == "policy-checklist":
        return True
    missed = obs.get("missedActions") or []
    if any(isinstance(a, dict) and a.get("name") for a in missed):
        return True
    return any(a in POLICY_WRITE_TOOLS for a in (obs.get("lastActions") or []))


def is_incomplete_obs(obs: dict[str, Any]) -> bool:
    """Hung / timeout / crash / transfer-without-writes / no-write without an attractor."""
    if obs.get("nSuccessProxy") == 1 and not obs.get("hung"):
        return False
    if obs.get("hung"):
        return True
    term = str(obs.get("termination") or "").lower()
    if term and _HARD_INCOMPLETE_TERM.search(term):
        return True
    if has_loop_attractor(obs):
        return False
    if "transfer" in term:
        return True
    if "user_stop" in term:
        return False
    if not called_write_tools(obs):
        return True
    return False


def is_sku_arm(arm: str | None) -> bool:
    """Official incomplete arm is I_sku. I_catalog / I_weight are prior/stub aliases."""
    return arm in {"I_sku", "I_catalog", "I_weight"}


def intervention_license(obs: dict[str, Any], *, loop_exhausted: bool = False) -> str:
    """License is hung/incomplete, not pick a pricier model."""
    if obs.get("hung"):
        return "hung"
    if obs.get("nSuccessProxy") == 1:
        return "hit"
    if is_incomplete_obs(obs):
        return "incomplete"
    if loop_exhausted:
        return "exhausted"
    return "attractor"


def recommend_intervention(obs: dict[str, Any], *, loop_exhausted: bool = False) -> str:
    """Hung is first-class here. Hit→wait; hung|crash|no-write→I_sku; attractor→I_loop."""
    if obs.get("hung"):
        return "I_sku"
    if obs.get("nSuccessProxy") == 1:
        return "wait"
    if is_incomplete_obs(obs):
        return "I_sku"
    if loop_exhausted:
        return "I_sku"
    return "I_loop"


def recommend_slice_intervention(
    obs_list: list[dict[str, Any]],
    *,
    loop_exhausted: bool = False,
) -> str:
    """I_sku if ANY episode is hung or incomplete; wait only if every episode hit."""
    if any(o.get("hung") for o in obs_list):
        return "I_sku"
    if obs_list and all(o.get("nSuccessProxy") == 1 for o in obs_list):
        return "wait"
    if any(is_incomplete_obs(o) for o in obs_list):
        return "I_sku"
    if loop_exhausted:
        return "I_sku"
    return "I_loop"


def control_batch(
    obs_list: list[dict[str, Any]],
    *,
    loop_exhausted: bool = False,
) -> dict[str, Any]:
    """Landed controller. Applies BOTH buckets; slice alone drops I_loop on mixed 39/44."""
    from tau2_vdom.improve import apply_scope_from_obs

    episodes = [
        {
            "taskId": o.get("taskId"),
            "hung": bool(o.get("hung")),
            "arm": recommend_intervention(o, loop_exhausted=loop_exhausted),
            "license": intervention_license(o, loop_exhausted=loop_exhausted),
            "serving": {"sku": DEFAULT_MODEL, "servingPaused": False},
        }
        for o in obs_list
    ]
    scope = apply_scope_from_obs(obs_list)
    buckets = {str(e["taskId"]): e["arm"] for e in episodes if e.get("taskId")}
    applied: list[str] = []
    if scope.get("looped"):
        applied.append("I_loop")
    if scope.get("weighted"):
        applied.append("I_sku")
    serving = {"sku": DEFAULT_MODEL, "servingPaused": False}
    return {
        "episodes": episodes,
        "slice": recommend_slice_intervention(obs_list, loop_exhausted=loop_exhausted),
        "buckets": buckets,
        "applied": applied,
        "applyScope": scope,
        "serving": serving,
        "servingSku": serving,
        "servingPaused": False,
        "trained": False,
        "notFineTuning": True,
    }


def _obs(
    actions: list[dict[str, Any]],
    reward: float | None,
    traces: list[Any],
    reward_info: Any = None,
    hung: bool = False,
    messages: list[Any] | None = None,
    task_id: str | None = None,
    termination: str | None = None,
) -> dict[str, Any]:
    last = [
        a.get("toolName") or f"text:{(a.get('text') or '')[:80]}"
        for a in actions
    ]
    repeats = sum(1 for a in actions if a.get("repeat"))
    failures = sum(1 for a in actions if a.get("ok") is False)
    p_hit = 0 if hung else (1 if reward is not None and reward >= 1 - 1e-6 else 0)
    blob = _assistant_text(actions, messages)
    missed = _missed_actions_from_reward_info(reward_info)
    refused_cancel = bool(_REFUSE_CANCEL.search(blob))
    invented_policy = bool(_INVENTED_POLICY.search(blob))
    missed_policy = [a for a in missed if a.get("name") in POLICY_WRITE_TOOLS]
    recommend_policy = (not p_hit) and (
        refused_cancel or invented_policy or bool(missed_policy)
    )
    draft = {
        "nSuccessProxy": p_hit,
        "lastActions": last,
        "hung": hung,
        "termination": termination,
        "missedActions": missed,
        "refusedCancel": refused_cancel,
        "inventedPolicy": invented_policy,
        "techniqueRecommendation": "policy-checklist" if recommend_policy else None,
    }
    arm = recommend_intervention(draft)
    if p_hit:
        critique = "path measure hits S; wait"
    elif hung:
        critique = "trial hung or skipped; keep task in the set (null reward), retry once"
    elif is_sku_arm(arm):
        critique = (
            "episode incomplete (hung / crash / no-write); I_sku "
            "(catalog rebind, not fine-tuning)"
        )
    elif recommend_policy:
        names = ", ".join(a["name"] for a in missed_policy) or "cancel/update"
        critique = (
            f"user asked cancel/update and agent refused or never called the tool "
            f"({names}); I_loop policy-checklist"
        )
    elif failures:
        critique = "tool failures in trajectory; inspect env channel"
    elif repeats:
        critique = "repeat actions; loop mutation or wait"
    else:
        critique = "episode unfinished or miss; inspect cascade / tools"
    return {
        "nSteps": max(len(traces), len(actions)),
        "nSuccessProxy": p_hit,
        "lastActions": last,
        "channels": ["env"] if any(a.get("kind") == "tool" for a in actions) else ["samp"],
        "critique": critique,
        "toolFailures": failures,
        "repeatActions": repeats,
        "arm": arm,
        "missedActions": missed,
        "refusedCancel": refused_cancel,
        "inventedPolicy": invented_policy,
        "hung": hung,
        "termination": termination,
        "techniqueRecommendation": "policy-checklist" if recommend_policy else None,
        "taskId": str(task_id) if task_id else None,
    }


class HungSimulation:
    """Placeholder so a skipped trial stays in the requested task set."""

    def __init__(self, task_id: str, trial: int, reason: str = "timeout"):
        self.task_id = task_id
        self.trial = trial
        self.messages: list[Any] = []
        self.reward_info = None
        self.termination_reason = reason
        self.hung = True


def write_eval_file(
    *,
    domain: str,
    model: str,
    provider: str,
    technique: str,
    live: bool,
    smoke: bool,
    simulations: list[Any],
    pass_hat_k: dict[str, float] | None,
    avg_reward: float | None,
    extra_traces: dict[str, list] | None = None,
    tag: str | None = None,
) -> Path:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    kind = "smoke" if smoke else ("live" if live else "offline")
    if tag:
        kind = f"{kind}-{tag}"
    path = EVAL_DIR / f"{domain}-{kind}-{stamp}.json"

    logs = []
    for i, sim in enumerate(simulations):
        reward = None
        reward_info = getattr(sim, "reward_info", None)
        if reward_info is not None:
            reward = float(reward_info.reward)
        hung = bool(getattr(sim, "hung", False))
        actions = _actions_from_messages(getattr(sim, "messages", []) or [])
        traces = (extra_traces or {}).get(getattr(sim, "task_id", str(i)), [])
        messages = [_serialize_message(m) for m in (getattr(sim, "messages", []) or [])]
        compact_ri = serialize_reward_info(reward_info)
        logs.append(
            {
                "taskId": getattr(sim, "task_id", None),
                "trial": getattr(sim, "trial", i),
                "reward": reward,
                "pHit": (1 if reward is not None and reward >= 1 - 1e-6 else 0)
                if reward is not None
                else None,
                "termination": getattr(getattr(sim, "termination_reason", None), "value", None)
                or str(getattr(sim, "termination_reason", "") or None),
                "actions": actions,
                "traces": traces,
                "obs": _obs(
                    actions,
                    reward,
                    traces,
                    reward_info=reward_info,
                    hung=hung,
                    messages=messages,
                    task_id=str(getattr(sim, "task_id", "") or "") or None,
                    termination=(
                        getattr(getattr(sim, "termination_reason", None), "value", None)
                        or str(getattr(sim, "termination_reason", "") or "")
                        or None
                    ),
                ),
                "rewardInfo": compact_ri,
                "hung": hung,
                "messages": messages,
            }
        )

    payload = {
        "benchmark": "tau2-bench",
        "domain": domain,
        "agent": "vdom",
        "model": model,
        "provider": provider,
        "technique": technique,
        "paperRepo": PAPER_REPO,
        "tau2Repo": TAU2_REPO,
        "metricNote": METRIC_NOTE,
        "live": live,
        "smoke": smoke,
        "passHatK": pass_hat_k,
        "avgReward": avg_reward,
        "simulations": logs,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    latest = EVAL_DIR / ("latest-smoke.json" if smoke else "latest.json")
    latest.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def run_smoke(technique: str = "one-shot", task_id: str = "create_task_1") -> Path:
    """Official τ² mock domain, scripted user, deterministic vdom agent. No API key."""
    from tau2.data_model.simulation import TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.runner import get_tasks, run_single_task

    from tau2_vdom import register
    from tau2_vdom.agent import TURN_TRACES, TURN_TRACES_BY_TASK, reset_turn_traces

    register()
    reset_turn_traces()
    tasks = get_tasks("mock", task_ids=[task_id])
    if not tasks:
        raise SystemExit(f"mock task {task_id} not found")
    config = TextRunConfig(
        domain="mock",
        agent="vdom",
        user="scripted_user",
        llm_agent="deterministic",
        llm_user="scripted",
        num_trials=1,
        max_steps=20,
        max_concurrency=1,
        max_retries=0,
        log_level="ERROR",
        task_ids=[task_id],
        num_tasks=1,
    )
    result = run_single_task(config, tasks[0], seed=42, evaluation_type=EvaluationType.ALL)
    reward = None
    if result.reward_info is not None:
        reward = float(result.reward_info.reward)
    path = write_eval_file(
        domain="mock",
        model="deterministic",
        provider="deterministic",
        technique=technique,
        live=False,
        smoke=True,
        simulations=[result],
        pass_hat_k={"1": 1.0} if reward is not None and reward >= 1 - 1e-6 else {"1": 0.0}
        if reward is not None
        else None,
        avg_reward=reward,
        extra_traces=dict(TURN_TRACES_BY_TASK) or {result.task_id: list(TURN_TRACES)},
    )
    print(f"smoke task={result.task_id} reward={reward} termination={result.termination_reason}")
    print(f"wrote {path}")
    return path


def run_live(
    *,
    domain: str,
    num_tasks: int | None,
    num_trials: int,
    task_ids: list[str] | None,
    model: str,
    technique: str,
    user: str,
    max_steps: int,
) -> Path:
    from tau2.data_model.simulation import TextRunConfig
    from tau2.metrics.agent_metrics import compute_metrics
    from tau2.runner import run_domain

    from tau2_vdom import register
    from tau2_vdom.agent import TURN_TRACES, TURN_TRACES_BY_TASK, reset_turn_traces

    if not _has_live_key():
        raise SystemExit(
            "Live τ² needs OPENROUTER_API_KEY (or OPENAI_API_KEY). "
            "For a key-free check: python -m tau2_vdom --domain mock --smoke"
        )

    _apply_openrouter_defaults()
    register()
    reset_turn_traces()
    user_llm = _litellm_user_model(model)
    config = TextRunConfig(
        domain=domain,
        agent="vdom",
        user=user,
        llm_agent=model,
        llm_user=user_llm,
        num_trials=num_trials,
        num_tasks=num_tasks,
        task_ids=task_ids,
        max_steps=max_steps,
        max_concurrency=1,
        log_level="ERROR",
    )
    results = run_domain(config)
    metrics = compute_metrics(results)
    pass_hat = {str(k): float(v) for k, v in (metrics.pass_hat_ks or {}).items()} or None
    path = write_eval_file(
        domain=domain,
        model=model,
        provider="openrouter" if os.environ.get("OPENROUTER_API_KEY") else "openai",
        technique=technique,
        live=True,
        smoke=False,
        simulations=list(results.simulations),
        pass_hat_k=pass_hat,
        avg_reward=float(metrics.avg_reward) if metrics.avg_reward is not None else None,
        extra_traces=dict(TURN_TRACES_BY_TASK),
    )
    print(f"avg_reward={metrics.avg_reward} pass^k={pass_hat}")
    print(f"wrote {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate the vdom agent on official τ²-bench (not a reimplemented domain)."
    )
    p.add_argument("--domain", default="mock", help="tau2 domain: mock, retail, airline, telecom")
    p.add_argument("--smoke", action="store_true", help="mock domain + scripted user, no API key")
    p.add_argument("--num-tasks", type=int, default=None)
    p.add_argument("--num-trials", type=int, default=1)
    p.add_argument("--task-ids", nargs="*", default=None)
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL))
    p.add_argument("--technique", default=os.environ.get("VDOM_TAU2_TECHNIQUE", "one-shot"))
    p.add_argument("--user", default="user_simulator")
    p.add_argument("--max-steps", type=int, default=200)
    return p


def _ensure_tau2_data_dir() -> None:
    if os.environ.get("TAU2_DATA_DIR"):
        return
    for cand in (REPO_ROOT / ".tau2-bench" / "data", Path("/tmp/tau2-bench/data")):
        if cand.is_dir():
            os.environ["TAU2_DATA_DIR"] = str(cand)
            return


def main(argv: list[str] | None = None) -> int:
    _ensure_tau2_data_dir()
    args = build_parser().parse_args(argv)
    os.environ["VDOM_TAU2_TECHNIQUE"] = args.technique
    try:
        if args.smoke or (args.domain == "mock" and not _has_live_key()):
            run_smoke(technique=args.technique)
            return 0
        run_live(
            domain=args.domain,
            num_tasks=args.num_tasks,
            num_trials=args.num_trials,
            task_ids=args.task_ids,
            model=args.model,
            technique=args.technique,
            user=args.user,
            max_steps=args.max_steps,
        )
        return 0
    except ModuleNotFoundError as exc:
        if exc.name in {"tau2", "tau2.registry", "tau2.runner"} or (
            exc.name and exc.name.startswith("tau2")
        ):
            print(
                "tau2 is not installed. From the repo root:\n"
                "  bash scripts/setup-tau2.sh\n"
                "  PYTHONPATH=python python -m tau2_vdom --domain mock --smoke",
                file=sys.stderr,
            )
            return 2
        raise


if __name__ == "__main__":
    raise SystemExit(main())
