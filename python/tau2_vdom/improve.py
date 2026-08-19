"""Runtime self-improvement on τ²: naive graph → Obs → I_loop | I_weight → re-run.

The claim is before/after on the *same* tasks, not the saturated retail 0–4
one-shot slice (pass^k=1.0). Default no-key path uses official mock
``update_task_1``, which the naive one-shot fails and Self-Refine recovers.

    PYTHONPATH=python python3 -m tau2_vdom.improve
    npm run eval:tau2:improve

Live (needs a key). Do not default to retail tasks 0–4:

    export OPENROUTER_API_KEY=...
    export OPENAI_BASE_URL=https://openrouter.ai/api/v1
    export OPENAI_MODEL=deepseek/deepseek-v4-flash-0731
    PYTHONPATH=python python3 -m tau2_vdom.improve --domain airline --num-tasks 4 --num-trials 1
    PYTHONPATH=python python3 -m tau2_vdom.improve --domain retail --task-ids 5 6 7 8 9 --num-trials 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tau2_vdom.runner import (
    DEFAULT_MODEL,
    EVAL_DIR,
    METRIC_NOTE,
    PAPER_REPO,
    REPO_ROOT,
    TAU2_REPO,
    _apply_openrouter_defaults,
    _ensure_tau2_data_dir,
    _has_live_key,
    write_eval_file,
)

SATURATED_NOTE = (
    "Slice already pass^k=1.0 under the naive graph — it cannot show improvement. "
    "The 5×4 retail one-shot (tasks 0–4) is this kind of slice. "
    "Use mock update_task_1 (no key), airline, or retail tasks beyond 0–4."
)
CLAIM = (
    "Runtime self-improvement: Obs from traces, then I_loop (AgentGraph mutation) "
    "or gated I_weight (async trainer; serving does not pause). "
    "Not the saturated 5×4 retail one-shot pass^k=1.0."
)
DEFAULT_MOCK_TASKS = ["update_task_1"]
# Official retail tasks 0–4 already scored pass^k=1.0 one-shot. Held-out only.
RETAIL_HELD_OUT = ["5", "6", "7", "8", "9"]


def _success(reward: float | None) -> bool:
    return reward is not None and reward >= 1 - 1e-6


def pass_hat_k_from_rewards(by_task: dict[str, list[float | None]]) -> dict[str, float]:
    """Official pass^k estimator (Yao et al. 2024): C(c,k)/C(n,k), averaged over tasks.

    Built only from measured rewards. Empty if there are no trials.
    """
    if not by_task:
        return {}
    lengths = {len(v) for v in by_task.values()}
    n = min(lengths) if lengths else 0
    if n <= 0:
        return {}
    out: dict[str, float] = {}
    for k in range(1, n + 1):
        scores: list[float] = []
        for rewards in by_task.values():
            trials = rewards[:n]
            c = sum(1 for r in trials if _success(r))
            if n < k:
                continue
            scores.append(math.comb(c, k) / math.comb(n, k) if c >= k else 0.0)
        if scores:
            out[str(k)] = sum(scores) / len(scores)
    return out


def _rewards_by_task(simulations: list[Any]) -> dict[str, list[float | None]]:
    by_task: dict[str, list[float | None]] = {}
    for sim in simulations:
        reward = None
        if getattr(sim, "reward_info", None) is not None:
            reward = float(sim.reward_info.reward)
        tid = str(getattr(sim, "task_id", "?"))
        by_task.setdefault(tid, []).append(reward)
    return by_task


def _avg_reward(simulations: list[Any]) -> float | None:
    vals = [r for rewards in _rewards_by_task(simulations).values() for r in rewards]
    nums = [v for v in vals if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _saturated(pass_hat: dict[str, float]) -> bool:
    if not pass_hat:
        return False
    return all(v >= 1 - 1e-6 for v in pass_hat.values())


def run_slice(
    *,
    domain: str,
    task_ids: list[str],
    technique: str,
    model: str,
    live: bool,
    num_trials: int,
    user: str,
    max_steps: int,
) -> tuple[list[Any], dict[str, float], float | None, Path]:
    from tau2.data_model.simulation import TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.runner import get_tasks, run_single_task

    from tau2_vdom import register
    from tau2_vdom.agent import TURN_TRACES_BY_TASK, reset_turn_traces

    os.environ["VDOM_TAU2_TECHNIQUE"] = technique
    register()
    reset_turn_traces()
    tasks = get_tasks(domain, task_ids=task_ids)
    if not tasks:
        raise SystemExit(f"no tasks for domain={domain} ids={task_ids}")

    config = TextRunConfig(
        domain=domain,
        agent="vdom",
        user=user,
        llm_agent=model,
        llm_user=model if live else "scripted",
        num_trials=num_trials,
        max_steps=max_steps,
        max_concurrency=1,
        max_retries=0,
        log_level="ERROR",
        task_ids=task_ids,
        num_tasks=len(task_ids),
    )
    simulations: list[Any] = []
    for trial in range(num_trials):
        for task in tasks:
            simulations.append(
                run_single_task(
                    config,
                    task,
                    seed=42 + trial,
                    evaluation_type=EvaluationType.ALL,
                )
            )

    pass_hat = pass_hat_k_from_rewards(_rewards_by_task(simulations))
    avg = _avg_reward(simulations)
    path = write_eval_file(
        domain=domain,
        model=model,
        provider="openrouter" if live and os.environ.get("OPENROUTER_API_KEY") else (
            "openai" if live else "deterministic"
        ),
        technique=technique,
        live=live,
        smoke=False,
        simulations=simulations,
        pass_hat_k=pass_hat,
        avg_reward=avg,
        extra_traces=dict(TURN_TRACES_BY_TASK),
        tag=technique,
    )
    return simulations, pass_hat, avg, path


def _sidecar_improve(sidecar: Any) -> dict[str, Any]:
    """I_loop via the live sidecar. Serving stays up (ping after mutate)."""
    mutated = sidecar.request({"op": "i_loop"})
    ping = sidecar.request({"op": "ping"})
    return {
        "technique": mutated.get("technique"),
        "graphDiff": mutated.get("graphDiff") or [],
        "graph": mutated.get("graph"),
        "servingPaused": bool(mutated.get("servingPaused")) or bool(ping.get("servingPaused")),
        "ping": ping.get("content"),
    }


def _sidecar_weight(sidecar: Any, *, before: float, after: float) -> dict[str, Any]:
    """Spawn FakeTrainer asynchronously; serving continues; mount only if gate says so."""
    spawned = sidecar.request({"op": "i_weight_spawn"})
    ping = sidecar.request({"op": "ping"})
    deadline = time.time() + 5.0
    status: dict[str, Any] = {}
    while time.time() < deadline:
        status = sidecar.request({"op": "i_weight_status"})
        if status.get("done"):
            break
        time.sleep(0.02)
    gate = sidecar.request({"op": "i_weight_gate", "before": before, "after": after})
    return {
        "spawned": bool(spawned.get("spawned")),
        "done": bool(status.get("done")),
        "servingPaused": bool(spawned.get("servingPaused"))
        or bool(ping.get("servingPaused")),
        "ping": ping.get("content"),
        "gate": gate.get("gate") or {},
        "mounted": (gate.get("gate") or {}).get("action") == "mount",
    }


class ThreadFakeTrainer:
    """Python-side stub if the sidecar is unavailable. Serving thread stays free."""

    def __init__(self) -> None:
        self.done = threading.Event()
        self.artifact: dict[str, Any] | None = None

    def spawn(self) -> None:
        def work() -> None:
            time.sleep(0.05)
            self.artifact = {"id": "fake-lora", "stub": True}
            self.done.set()

        threading.Thread(target=work, daemon=True).start()


def write_improve_report(payload: dict[str, Any]) -> Path:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = EVAL_DIR / f"improve-{payload.get('domain', 'mock')}-{stamp}.json"
    text = json.dumps(payload, indent=2) + "\n"
    path.write_text(text)
    (EVAL_DIR / "latest-improve.json").write_text(text)
    return path


def run_improve(
    *,
    domain: str,
    task_ids: list[str],
    model: str,
    live: bool,
    num_trials: int,
    user: str,
    max_steps: int,
    weight: bool,
) -> Path:
    from tau2_vdom.agent import reset_turn_traces
    from tau2_vdom.sidecar import default_sidecar

    sidecar = default_sidecar()
    sidecar.request({"op": "set_technique", "technique": "one-shot"})
    reset_turn_traces()

    before_sims, pass_before, avg_before, before_path = run_slice(
        domain=domain,
        task_ids=task_ids,
        technique="one-shot",
        model=model,
        live=live,
        num_trials=num_trials,
        user=user,
        max_steps=max_steps,
    )

    from tau2_vdom.runner import _obs, _actions_from_messages

    obs_list = []
    for sim in before_sims:
        reward = None
        if getattr(sim, "reward_info", None) is not None:
            reward = float(sim.reward_info.reward)
        actions = _actions_from_messages(getattr(sim, "messages", []) or [])
        obs_list.append(_obs(actions, reward, []))

    saturated = _saturated(pass_before)
    intervention = "none"
    intervention_detail = "wait"
    graph_diff: list[Any] = []
    serving_paused = False
    weight_info: dict[str, Any] | None = None
    after_sims = before_sims
    pass_after = pass_before
    avg_after = avg_before
    after_path = before_path
    note = CLAIM

    if saturated:
        intervention = "none"
        intervention_detail = "wait (naive already hits; I_loop would not be identified)"
        note = SATURATED_NOTE
        # Still ping so the report records that serving did not pause.
        ping = sidecar.request({"op": "ping"})
        serving_paused = bool(ping.get("servingPaused"))
    else:
        loop = _sidecar_improve(sidecar)
        graph_diff = loop.get("graphDiff") or []
        serving_paused = bool(loop.get("servingPaused"))
        os.environ["VDOM_TAU2_TECHNIQUE"] = "self-refine"
        intervention = "I_loop"
        intervention_detail = "self-refine critic+refine (scientist / applySelfRefineMutation)"

        after_sims, pass_after, avg_after, after_path = run_slice(
            domain=domain,
            task_ids=task_ids,
            technique="self-refine",
            model=model,
            live=live,
            num_trials=num_trials,
            user=user,
            max_steps=max_steps,
        )

        if weight:
            before_score = avg_after if avg_after is not None else 0.0
            # Candidate eval is the current after-I_loop score: FakeTrainer does
            # not change the τ² script. Gate must refuse unless after > before.
            weight_info = _sidecar_weight(
                sidecar, before=before_score, after=before_score
            )
            if weight_info.get("mounted"):
                intervention = "I_weight"
                intervention_detail = "adapter mounted after gate"
            graph_diff = list(graph_diff) + [
                {
                    "op": "mount" if weight_info.get("mounted") else "reject",
                    "key": "adapter",
                }
            ]

    payload = {
        "benchmark": "tau2-bench",
        "kind": "runtime-self-improvement",
        "claim": CLAIM,
        "note": note,
        "paperRepo": PAPER_REPO,
        "tau2Repo": TAU2_REPO,
        "metricNote": METRIC_NOTE,
        "domain": domain,
        "taskIds": task_ids,
        "numTrials": num_trials,
        "agent": "vdom",
        "model": model,
        "provider": (
            "openrouter"
            if live and os.environ.get("OPENROUTER_API_KEY")
            else ("openai" if live else "deterministic")
        ),
        "live": live,
        "passHatKBefore": pass_before,
        "passHatKAfter": pass_after,
        "avgRewardBefore": avg_before,
        "avgRewardAfter": avg_after,
        "intervention": intervention,
        "interventionDetail": intervention_detail,
        "graphDiff": graph_diff,
        "obs": obs_list,
        "weight": weight_info,
        "servingPaused": serving_paused,
        "beforeFile": str(before_path.relative_to(REPO_ROOT)),
        "afterFile": str(after_path.relative_to(REPO_ROOT)),
        "command": (
            "PYTHONPATH=python python3 -m tau2_vdom.improve"
            + ("" if domain == "mock" else f" --domain {domain}")
        ),
    }
    path = write_improve_report(payload)
    print(json.dumps({
        "wrote": str(path),
        "passHatKBefore": pass_before,
        "passHatKAfter": pass_after,
        "intervention": intervention,
        "graphDiff": graph_diff,
        "servingPaused": serving_paused,
        "saturated": saturated,
    }, indent=2))
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Runtime self-improvement on τ²: naive one-shot → Obs → I_loop "
            "(or gated I_weight) → same tasks. Not the 5×4 retail one-shot 1.0."
        )
    )
    p.add_argument("--domain", default=None, help="mock (default, no key), airline, retail, telecom")
    p.add_argument("--task-ids", nargs="*", default=None)
    p.add_argument("--num-tasks", type=int, default=None)
    p.add_argument("--num-trials", type=int, default=1)
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL))
    p.add_argument("--user", default=None)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument(
        "--weight",
        action="store_true",
        help="Also spawn FakeTrainer (async). Mount only if after-eval beats before.",
    )
    return p


def _resolve_slice(args: argparse.Namespace) -> tuple[str, list[str], bool, str]:
    live = _has_live_key()
    if args.domain is None:
        domain = "airline" if live else "mock"
    else:
        domain = args.domain

    if domain == "mock" and not live:
        task_ids = list(args.task_ids) if args.task_ids else list(DEFAULT_MOCK_TASKS)
        user = args.user or "scripted_user"
        return domain, task_ids, False, user

    if not live:
        raise SystemExit(
            "Live τ² improve needs OPENROUTER_API_KEY (or OPENAI_API_KEY). "
            "Key-free figure: PYTHONPATH=python python3 -m tau2_vdom.improve"
        )

    _apply_openrouter_defaults()
    if args.task_ids:
        task_ids = list(args.task_ids)
    elif domain == "retail":
        n = args.num_tasks or len(RETAIL_HELD_OUT)
        task_ids = RETAIL_HELD_OUT[:n]
    elif args.num_tasks:
        # Probe first N of the domain (airline / telecom). Caller chose the domain.
        from tau2.runner import get_tasks

        all_tasks = get_tasks(domain)
        task_ids = [t.id for t in all_tasks[: args.num_tasks]]
    else:
        from tau2.runner import get_tasks

        all_tasks = get_tasks(domain)
        task_ids = [t.id for t in all_tasks[:4]]

    if domain == "retail" and set(task_ids) <= set("01234"):
        print(
            "warning: retail tasks 0–4 already scored pass^k=1.0 one-shot; "
            "they cannot show improvement. Prefer --task-ids 5 6 7 8 9.",
            file=sys.stderr,
        )

    user = args.user or "user_simulator"
    return domain, task_ids, True, user


def main(argv: list[str] | None = None) -> int:
    _ensure_tau2_data_dir()
    args = build_parser().parse_args(argv)
    try:
        domain, task_ids, live, user = _resolve_slice(args)
        model = args.model if live else "deterministic"
        run_improve(
            domain=domain,
            task_ids=task_ids,
            model=model,
            live=live,
            num_trials=args.num_trials,
            user=user,
            max_steps=args.max_steps,
            weight=args.weight,
        )
        return 0
    except ModuleNotFoundError as exc:
        if exc.name in {"tau2", "tau2.registry", "tau2.runner"} or (
            exc.name and exc.name.startswith("tau2")
        ):
            print(
                "tau2 is not installed. From the repo root:\n"
                "  bash scripts/setup-tau2.sh\n"
                "  PYTHONPATH=python python3 -m tau2_vdom.improve",
                file=sys.stderr,
            )
            return 2
        raise


if __name__ == "__main__":
    raise SystemExit(main())
