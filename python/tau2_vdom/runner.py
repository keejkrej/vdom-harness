"""Run official τ² evaluations with the vdom agent. Writes eval/tau2/*.json."""

from __future__ import annotations

import argparse
import json
import os
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


def _obs(actions: list[dict[str, Any]], reward: float | None, traces: list[Any]) -> dict[str, Any]:
    last = [
        a.get("toolName") or f"text:{(a.get('text') or '')[:80]}"
        for a in actions
    ]
    repeats = sum(1 for a in actions if a.get("repeat"))
    failures = sum(1 for a in actions if a.get("ok") is False)
    p_hit = 1 if reward is not None and reward >= 1 - 1e-6 else 0
    if p_hit:
        critique = "path measure hits S; wait"
        arm = "wait"
    elif failures:
        critique = "tool failures in trajectory; inspect env channel"
        arm = "I_loop"
    elif repeats:
        critique = "repeat actions; loop mutation or wait"
        arm = "I_loop"
    else:
        critique = "episode unfinished or miss; inspect cascade / tools"
        arm = "I_loop"
    return {
        "nSteps": max(len(traces), len(actions)),
        "nSuccessProxy": p_hit,
        "lastActions": last,
        "channels": ["env"] if any(a.get("kind") == "tool" for a in actions) else ["samp"],
        "critique": critique,
        "toolFailures": failures,
        "repeatActions": repeats,
        "arm": arm,
    }


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
        if getattr(sim, "reward_info", None) is not None:
            reward = float(sim.reward_info.reward)
        actions = _actions_from_messages(getattr(sim, "messages", []) or [])
        traces = (extra_traces or {}).get(getattr(sim, "task_id", str(i)), [])
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
                "obs": _obs(actions, reward, traces),
                "messages": [_serialize_message(m) for m in (getattr(sim, "messages", []) or [])],
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
