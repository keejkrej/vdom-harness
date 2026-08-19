"""Closed-loop runtime self-improvement on τ².

self-observe → I_loop | I_weight → run again → self-observe, until pass^k
saturates or a round budget. Not a single before/after.

Default no-key slice: official mock ``update_task_1`` + ``impossible_task_1``.
Round 0 naive one-shot misses both. Round 1 Self-Refine recovers update only.
Round 2 validator recovers transfer. Then Obs sees saturation and waits.

    PYTHONPATH=python python3 -m tau2_vdom.improve
    npm run eval:tau2:improve

Live (needs a key). Do not default to retail tasks 0–4. Do not invent scores.

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
    HungSimulation,
    _actions_from_messages,
    _apply_openrouter_defaults,
    _ensure_tau2_data_dir,
    _has_live_key,
    _litellm_user_model,
    _obs,
    _pin_tau2_judges,
    serialize_reward_info,
    write_eval_file,
)

SATURATED_NOTE = (
    "Slice already pass^k=1.0 under the naive graph — it cannot show improvement. "
    "The 5×4 retail one-shot (tasks 0–4) is this kind of slice. "
    "Use mock update_task_1 + impossible_task_1 (no key), airline, or retail beyond 0–4."
)
CLAIM = (
    "Closed loop: self-observe → I_loop or I_weight → run again → self-observe, "
    "until pass^k saturates or the round budget. Serving does not pause. "
    "I_loop is failure-aware (Obs / reward_info → typed graph), not a fixed "
    "self-refine → validator ladder. Not the saturated 5×4 retail one-shot pass^k=1.0."
)
SKIP_POLICY = (
    "A hung trial is retried once. If it still hangs, the simulation is recorded "
    "with reward=null and hung=true; the task stays in taskPHit as null. pass^k "
    "averages only tasks that completed at least one trial, so a skip cannot "
    "disappear from the reported set and is not treated as a measured 0."
)
# Two official mock tasks so the loop needs two I_loop rounds (not one before/after).
DEFAULT_MOCK_TASKS = ["update_task_1", "impossible_task_1"]
RETAIL_HELD_OUT = ["5", "6", "7", "8", "9"]


def _success(reward: float | None) -> bool:
    return reward is not None and reward >= 1 - 1e-6


def pass_hat_k_from_rewards(by_task: dict[str, list[float | None]]) -> dict[str, float]:
    """Official pass^k estimator (Yao et al. 2024): C(c,k)/C(n,k), averaged over tasks.

    Null rewards (hung / skipped after retry) are not completed trials. Tasks with
    no completed trial stay in the set for taskPHit but are omitted from this
    average so a skip is not a measured 0.
    """
    completed: dict[str, list[float]] = {}
    for tid, rewards in by_task.items():
        done = [r for r in rewards if r is not None]
        if done:
            completed[tid] = done
    if not completed:
        return {}
    lengths = {len(v) for v in completed.values()}
    n = min(lengths) if lengths else 0
    if n <= 0:
        return {}
    out: dict[str, float] = {}
    for k in range(1, n + 1):
        scores: list[float] = []
        for rewards in completed.values():
            trials = rewards[:n]
            c = sum(1 for r in trials if _success(r))
            scores.append(math.comb(c, k) / math.comb(n, k) if c >= k else 0.0)
        if scores:
            out[str(k)] = sum(scores) / len(scores)
    return out


def _rewards_by_task(
    simulations: list[Any],
    task_ids: list[str] | None = None,
) -> dict[str, list[float | None]]:
    by_task: dict[str, list[float | None]] = {str(t): [] for t in (task_ids or [])}
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


def _p_hit(pass_hat: dict[str, float]) -> float | None:
    """First-passage proxy: pass^1 when present."""
    if "1" in pass_hat:
        return float(pass_hat["1"])
    return None


def _task_p_hit(by_task: dict[str, list[float | None]]) -> dict[str, float | None]:
    """Always include every requested task. All-null (skipped) → null, not dropped."""
    out: dict[str, float | None] = {}
    for tid, rewards in by_task.items():
        completed = [r for r in rewards if r is not None]
        if not completed:
            out[tid] = None
            continue
        hits = sum(1 for r in completed if _success(r))
        out[tid] = hits / len(completed)
    return out


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
    trial_timeout_s: int = 480,
) -> tuple[list[Any], dict[str, float], float | None, Path, list[str]]:
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
        llm_user=_litellm_user_model(model) if live else "scripted",
        num_trials=num_trials,
        max_steps=max_steps,
        max_concurrency=1,
        max_retries=0,
        log_level="ERROR",
        task_ids=task_ids,
        num_tasks=len(task_ids),
    )
    import concurrent.futures

    simulations: list[Any] = []
    skipped: list[str] = []
    for trial in range(num_trials):
        for task in tasks:
            tid = str(getattr(task, "id", "?"))
            t0 = time.time()
            print(f"[improve] start task={tid} trial={trial} technique={technique}", flush=True)
            sim = None
            last_err: str | None = None
            for attempt in (0, 1):
                pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                try:
                    fut = pool.submit(
                        run_single_task,
                        config,
                        task,
                        seed=42 + trial + attempt,
                        evaluation_type=EvaluationType.ALL,
                    )
                    sim = fut.result(timeout=trial_timeout_s)
                    last_err = None
                    break
                except concurrent.futures.TimeoutError:
                    last_err = "timeout"
                    print(
                        f"[improve] HUNG task={tid} trial={trial} attempt={attempt} "
                        f"> {trial_timeout_s}s"
                        + ("; retrying once" if attempt == 0 else "; keeping null reward"),
                        flush=True,
                    )
                except Exception as exc:
                    last_err = f"err:{type(exc).__name__}"
                    print(
                        f"[improve] FAIL task={tid} trial={trial} attempt={attempt}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    break
                finally:
                    pool.shutdown(wait=False, cancel_futures=True)
            if sim is None:
                skipped.append(f"{tid}:t{trial}:{last_err or 'timeout'}")
                simulations.append(HungSimulation(tid, trial, last_err or "timeout"))
                continue
            elapsed = time.time() - t0
            reward = None
            if getattr(sim, "reward_info", None) is not None:
                reward = float(sim.reward_info.reward)
            print(
                f"[improve] done task={tid} trial={trial} reward={reward} "
                f"term={getattr(sim, 'termination_reason', None)} elapsed={elapsed:.1f}s",
                flush=True,
            )
            simulations.append(sim)
    if skipped:
        print(f"[improve] skipped={skipped} ({SKIP_POLICY})", flush=True)

    by_task = _rewards_by_task(simulations, task_ids)
    pass_hat = pass_hat_k_from_rewards(by_task)
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
        tag=f"{technique}-r{int(time.time())}",
    )
    return simulations, pass_hat, avg, path, skipped


def _collect_obs(simulations: list[Any]) -> list[dict[str, Any]]:
    obs_list = []
    for sim in simulations:
        reward = None
        if getattr(sim, "reward_info", None) is not None:
            reward = float(sim.reward_info.reward)
        actions = _actions_from_messages(getattr(sim, "messages", []) or [])
        hung = bool(getattr(sim, "hung", False))
        obs_list.append(
            _obs(
                actions,
                reward,
                [],
                reward_info=getattr(sim, "reward_info", None),
                hung=hung,
                messages=getattr(sim, "messages", None) or [],
            )
        )
    return obs_list


def _sidecar_i_loop(sidecar: Any, obs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"op": "i_loop"}
    if obs is not None:
        payload["obs"] = obs
    mutated = sidecar.request(payload)
    ping = sidecar.request({"op": "ping"})
    return {
        "applied": mutated.get("content") == "applied",
        "technique": mutated.get("technique"),
        "graphDiff": mutated.get("graphDiff") or [],
        "graph": mutated.get("graph"),
        "servingPaused": bool(mutated.get("servingPaused")) or bool(ping.get("servingPaused")),
        "ping": ping.get("content"),
    }


def _sidecar_weight(sidecar: Any, *, before: float, after: float) -> dict[str, Any]:
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


def write_improve_report(payload: dict[str, Any]) -> Path:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = EVAL_DIR / f"improve-{payload.get('domain', 'mock')}-{stamp}.json"
    text = json.dumps(payload, indent=2) + "\n"
    path.write_text(text)
    (EVAL_DIR / "latest-improve.json").write_text(text)
    return path


def _round_record(
    *,
    round_i: int,
    technique: str,
    pass_hat: dict[str, float],
    avg: float | None,
    obs: list[dict[str, Any]],
    by_task: dict[str, list[float | None]],
    intervention: str | None,
    graph_diff: list[Any],
    eval_file: Path,
    skipped: list[str] | None = None,
    reward_infos: list[dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    return {
        "round": round_i,
        "technique": technique,
        "pHit": _p_hit(pass_hat),
        "passHatK": pass_hat,
        "avgReward": avg,
        "taskPHit": _task_p_hit(by_task),
        "obs": obs,
        "rewardInfo": reward_infos or [],
        "intervention": intervention,
        "graphDiff": graph_diff,
        "evalFile": str(eval_file.relative_to(REPO_ROOT)),
        "skipped": skipped or [],
        "skipPolicy": SKIP_POLICY,
    }


def run_improve(
    *,
    domain: str,
    task_ids: list[str],
    model: str,
    live: bool,
    num_trials: int,
    user: str,
    max_steps: int,
    max_rounds: int,
    weight: bool,
    trial_timeout_s: int = 480,
) -> Path:
    from tau2_vdom.agent import reset_turn_traces
    from tau2_vdom.sidecar import default_sidecar

    sidecar = default_sidecar()
    sidecar.request({"op": "set_technique", "technique": "one-shot"})
    reset_turn_traces()

    technique = "one-shot"
    serving_paused = False
    rounds: list[dict[str, Any]] = []

    sims, pass_hat, avg, path, skipped = run_slice(
        domain=domain,
        task_ids=task_ids,
        technique=technique,
        model=model,
        live=live,
        num_trials=num_trials,
        user=user,
        max_steps=max_steps,
        trial_timeout_s=trial_timeout_s,
    )
    obs = _collect_obs(sims)
    rounds.append(
        _round_record(
            round_i=0,
            technique=technique,
            pass_hat=pass_hat,
            avg=avg,
            obs=obs,
            by_task=_rewards_by_task(sims, task_ids),
            intervention=None,
            graph_diff=[],
            eval_file=path,
            skipped=skipped,
            reward_infos=[serialize_reward_info(getattr(s, "reward_info", None)) for s in sims],
        )
    )

    stop_reason = "saturated" if _saturated(pass_hat) else "budget"
    note = SATURATED_NOTE if _saturated(pass_hat) else CLAIM
    if _saturated(pass_hat):
        ping = sidecar.request({"op": "ping"})
        serving_paused = bool(ping.get("servingPaused"))

    for r in range(1, max_rounds + 1):
        if _saturated(pass_hat):
            stop_reason = "saturated"
            break

        miss = any(o.get("nSuccessProxy", 0) != 1 for o in obs)
        loop = _sidecar_i_loop(sidecar, obs=obs)
        serving_paused = serving_paused or bool(loop.get("servingPaused"))

        if loop.get("applied"):
            intervention = "I_loop"
            graph_diff = loop.get("graphDiff") or []
            technique = str(loop.get("technique") or technique)
            os.environ["VDOM_TAU2_TECHNIQUE"] = technique
        elif weight:
            current = avg if avg is not None else 0.0
            w = _sidecar_weight(sidecar, before=current, after=current)
            serving_paused = serving_paused or bool(w.get("servingPaused"))
            if w.get("mounted"):
                intervention = "I_weight"
                graph_diff = [{"op": "mount", "key": "adapter"}]
            else:
                intervention = "I_weight"
                graph_diff = [{"op": "reject", "key": "adapter"}]
                rounds.append(
                    _round_record(
                        round_i=r,
                        technique=technique,
                        pass_hat=pass_hat,
                        avg=avg,
                        obs=obs,
                        by_task=_rewards_by_task(sims, task_ids),
                        intervention=intervention,
                        graph_diff=graph_diff,
                        eval_file=path,
                        skipped=skipped,
                        reward_infos=[
                            serialize_reward_info(getattr(s, "reward_info", None)) for s in sims
                        ],
                    )
                )
                stop_reason = "weight-rejected"
                break
        else:
            stop_reason = "loop-exhausted"
            break

        if not miss and intervention != "I_loop":
            stop_reason = "saturated"
            break

        sims, pass_hat, avg, path, skipped = run_slice(
            domain=domain,
            task_ids=task_ids,
            technique=technique,
            model=model,
            live=live,
            num_trials=num_trials,
            user=user,
            max_steps=max_steps,
            trial_timeout_s=trial_timeout_s,
        )
        obs = _collect_obs(sims)
        rounds.append(
            _round_record(
                round_i=r,
                technique=technique,
                pass_hat=pass_hat,
                avg=avg,
                obs=obs,
                by_task=_rewards_by_task(sims, task_ids),
                intervention=intervention,
                graph_diff=graph_diff,
                eval_file=path,
                skipped=skipped,
                reward_infos=[serialize_reward_info(getattr(s, "reward_info", None)) for s in sims],
            )
        )
        if _saturated(pass_hat):
            stop_reason = "saturated"
            break
    else:
        if not _saturated(pass_hat):
            stop_reason = "budget"

    first = rounds[0]
    last = rounds[-1]
    improve_rounds = [x for x in rounds if x.get("intervention")]
    payload = {
        "benchmark": "tau2-bench",
        "kind": "runtime-self-improvement",
        "closedLoop": True,
        "claim": CLAIM,
        "note": note,
        "paperRepo": PAPER_REPO,
        "tau2Repo": TAU2_REPO,
        "metricNote": METRIC_NOTE,
        "domain": domain,
        "taskIds": task_ids,
        "numTrials": num_trials,
        "maxRounds": max_rounds,
        "stopReason": stop_reason,
        "agent": "vdom",
        "model": model,
        "provider": (
            "openrouter"
            if live and os.environ.get("OPENROUTER_API_KEY")
            else ("openai" if live else "deterministic")
        ),
        "live": live,
        "passHatKBefore": first["passHatK"],
        "passHatKAfter": last["passHatK"],
        "pHitSequence": [x["pHit"] for x in rounds],
        "avgRewardBefore": first["avgReward"],
        "avgRewardAfter": last["avgReward"],
        "interventions": [x["intervention"] for x in improve_rounds],
        "graphDiffs": [x["graphDiff"] for x in improve_rounds],
        "rounds": rounds,
        "servingPaused": serving_paused,
        "skipPolicy": SKIP_POLICY,
        "command": (
            "PYTHONPATH=python python3 -m tau2_vdom.improve"
            + ("" if domain == "mock" else f" --domain {domain}")
        ),
    }
    out = write_improve_report(payload)
    print(json.dumps({
        "wrote": str(out),
        "closedLoop": True,
        "stopReason": stop_reason,
        "pHitSequence": payload["pHitSequence"],
        "interventions": payload["interventions"],
        "graphDiffs": payload["graphDiffs"],
        "servingPaused": serving_paused,
        "passHatKBefore": first["passHatK"],
        "passHatKAfter": last["passHatK"],
    }, indent=2))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Closed-loop τ² self-improvement: Obs → I_loop|I_weight → Obs, "
            "until pass^k saturates or --max-rounds. Not the 5×4 retail 1.0."
        )
    )
    p.add_argument("--domain", default=None, help="mock (default, no key), airline, retail, telecom")
    p.add_argument("--task-ids", nargs="*", default=None)
    p.add_argument("--num-tasks", type=int, default=None)
    p.add_argument("--num-trials", type=int, default=1)
    p.add_argument("--max-rounds", type=int, default=4, help="Improve-round budget after the first Obs")
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL))
    p.add_argument("--user", default=None)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument(
        "--trial-timeout",
        type=int,
        default=480,
        help=(
            "Retry once if a trial exceeds this many seconds (default 480). "
            "A second hang keeps the task in taskPHit with null reward."
        ),
    )
    p.add_argument(
        "--weight",
        action="store_true",
        help="If I_loop is exhausted and the slice still misses, spawn FakeTrainer (async).",
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
        if live:
            _pin_tau2_judges(model)
            print(f"[improve] pinned judge/user={_litellm_user_model(model)} model={model}", flush=True)
        run_improve(
            domain=domain,
            task_ids=task_ids,
            model=model,
            live=live,
            num_trials=args.num_trials,
            user=user,
            max_steps=args.max_steps,
            max_rounds=args.max_rounds,
            weight=args.weight,
            trial_timeout_s=args.trial_timeout,
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
