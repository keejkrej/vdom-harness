"""Closed-loop runtime self-improvement on τ².

self-observe → I_loop | I_sku → run again → self-observe, until pass^k
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
    is_incomplete_obs,
    control_batch,
    recommend_intervention,
    recommend_slice_intervention,
    serialize_reward_info,
    write_eval_file,
)

SATURATED_NOTE = (
    "Slice already pass^k=1.0 under the naive graph — it cannot show improvement. "
    "The 5×4 retail one-shot (tasks 0–4) is this kind of slice. "
    "Use mock update_task_1 + impossible_task_1 (no key), airline, or retail beyond 0–4."
)
CLAIM = (
    "Closed loop: self-observe → I_loop or I_sku → run again → self-observe, "
    "until pass^k saturates or the round budget. Serving does not pause. "
    "The agent may get_agent_graph / set_agent_graph mid-turn (local intercept) "
    "and may rewrite C on the slow-clock Obs; host I_loop is fallback if it never "
    "called set. A mixed batch applies C1 only to miss / I_loop tasks; wait+hit "
    "keeps C0 (applyScope). Canned airline checklist is fallback when self-Obs JSON "
    "is invalid. Not the saturated 5×4 retail one-shot pass^k=1.0."
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
# Deterministic incomplete episode for the I_weight TrainJob stub (no live key).
INCOMPLETE_FIXTURE_ID = "incomplete_fixture_1"
SERVING_MODEL = "deepseek/deepseek-v4-flash-0731"
CATALOG_JUMP_MODEL = "deepseek/deepseek-v4-pro-0813"
I_SKU_NOTE = (
    "I_sku is a gated catalog rebind licensed by hung/incomplete, not pick a pricier model. "
    "Base SKU deepseek/deepseek-v4-flash-0731. Slow arm proposes "
    "deepseek/deepseek-v4-pro-0813 (OpenRouter, GA 2026-08-12). "
    "Gate is a measured after-eval; 0813 existing is not a gate. "
    "Jump iff later serving model id is 0813. servingPaused stays false. "
    "I_sku writes S (catalog pointer beside C), not n.model. "
    "Catalog rebind, not fine-tuning. FakeTrainer and LoRA are not this arm."
)
I_CATALOG_NOTE = I_SKU_NOTE
I_WEIGHT_NOTE = (
    "I_weight TrainJob is a protocol stub, not the paper slow arm and not fine-tuning. "
    "The official incomplete arm is I_sku (catalog rebind to pro-0813). "
    + I_SKU_NOTE
)


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
            os.environ["VDOM_TAU2_TASK_ID"] = tid
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


def apply_scope_from_obs(obs_list: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Wait+hit keep C0; completed I_loop miss get C1; incomplete / I_sku keep C0."""
    hits: set[str] = set()
    incompletes: set[str] = set()
    order: list[str] = []
    for o in obs_list:
        tid = str(o.get("taskId") or "")
        if not tid:
            continue
        if tid not in order:
            order.append(tid)
        arm = o.get("arm") or recommend_intervention(o)
        hit = arm == "wait" and o.get("nSuccessProxy") == 1 and not o.get("hung")
        if hit:
            hits.add(tid)
        elif arm == "I_loop":
            continue
        elif arm in {"I_sku", "I_catalog", "I_weight"} or is_incomplete_obs(o):
            incompletes.add(tid)
    wait_kept = [tid for tid in order if tid in hits]
    weighted = [tid for tid in order if tid in incompletes]
    looped = [tid for tid in order if tid not in hits and tid not in incompletes]
    return {"waitKept": wait_kept, "looped": looped, "weighted": weighted}


def _collect_obs(simulations: list[Any]) -> list[dict[str, Any]]:
    obs_list = []
    for sim in simulations:
        reward = None
        if getattr(sim, "reward_info", None) is not None:
            reward = float(sim.reward_info.reward)
        actions = _actions_from_messages(getattr(sim, "messages", []) or [])
        hung = bool(getattr(sim, "hung", False))
        term = _termination(sim)
        if hung and not term:
            term = "timeout"
        obs_list.append(
            _obs(
                actions,
                reward,
                [],
                reward_info=getattr(sim, "reward_info", None),
                hung=hung,
                messages=getattr(sim, "messages", None) or [],
                task_id=str(getattr(sim, "task_id", "") or "") or None,
                termination=term or None,
            )
        )
    return obs_list


def missed_tool_names_only(obs_list: list[dict[str, Any]]) -> list[str]:
    """Missed *tool names* for self-Obs. Never gold reservation IDs or write args."""
    names: list[str] = []
    for o in obs_list:
        for a in o.get("missedActions") or []:
            name = a.get("name") if isinstance(a, dict) else None
            if name:
                names.append(str(name))
    return names


def _self_obs_ctx(
    simulations: list[Any],
    obs: list[dict[str, Any]],
    extra_traces: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    tool_names: list[str] = []
    rewards: list[int] = []
    terminations: list[str] = []
    traces: list[dict[str, Any]] = []
    for sim in simulations:
        reward = _sim_reward(sim)
        rewards.append(1 if _success(reward) else 0)
        term = _termination(sim)
        if bool(getattr(sim, "hung", False)) and not term:
            term = "timeout"
        terminations.append(term or "user_stop")
        for a in _actions_from_messages(getattr(sim, "messages", []) or []):
            name = a.get("toolName") if isinstance(a, dict) else None
            if name:
                tool_names.append(str(name))
    extra = extra_traces or {}
    for rows in extra.values():
        for t in rows:
            if isinstance(t, dict):
                traces.append(
                    {
                        "nodeKey": t.get("nodeKey") or t.get("role") or "solve",
                        "role": t.get("role") or "solve",
                        "output": t.get("output") or "",
                    }
                )
    task_ids = [str(getattr(sim, "task_id", "?")) for sim in simulations]
    return {
        "toolNames": tool_names,
        "rewards": rewards,
        "terminations": terminations,
        "missedToolNames": missed_tool_names_only(obs),
        "taskIds": task_ids,
        "traces": traces,
    }


def _sidecar_i_loop(
    sidecar: Any,
    obs: list[dict[str, Any]] | None = None,
    *,
    ctx: dict[str, Any] | None = None,
    model: str = "deterministic",
) -> dict[str, Any]:
    payload: dict[str, Any] = {"op": "self_obs", "model": model}
    if obs is not None:
        payload["obs"] = obs
    if ctx:
        payload.update(ctx)
    mutated = sidecar.request(payload)
    ping = sidecar.request({"op": "ping"})
    applied = mutated.get("content") == "applied" or bool(mutated.get("applied"))
    return {
        "applied": applied,
        "technique": mutated.get("technique"),
        "graphDiff": mutated.get("graphDiff") or [],
        "graph": mutated.get("graph"),
        "servingPaused": bool(mutated.get("servingPaused")) or bool(ping.get("servingPaused")),
        "ping": ping.get("content"),
        "selfObsPath": mutated.get("path") or "fallback",
        "action": mutated.get("action"),
        "rationale": mutated.get("rationale"),
        "applyScope": mutated.get("applyScope") or apply_scope_from_obs(obs or []),
    }


def _termination(sim: Any) -> str:
    raw = getattr(sim, "termination_reason", None)
    if raw is None:
        return ""
    val = getattr(raw, "value", None)
    return str(val or raw or "").lower()


def _sim_reward(sim: Any) -> float | None:
    info = getattr(sim, "reward_info", None)
    if info is None:
        return None
    try:
        return float(info.reward)
    except (TypeError, ValueError, AttributeError):
        return None


def is_incomplete_episode(sim: Any) -> bool:
    """Incomplete (hung / crash / no-write) licenses I_sku, not a train."""
    if bool(getattr(sim, "hung", False)):
        return True
    term = _termination(sim)
    if any(tok in term for tok in ("transfer", "crash", "error", "timeout")):
        return True
    reward = _sim_reward(sim)
    if reward is not None and reward <= 0 and "transfer" in term:
        return True
    return False


def incomplete_reason(sim: Any) -> str:
    if bool(getattr(sim, "hung", False)):
        return "hung"
    term = _termination(sim)
    if "crash" in term or "error" in term:
        return "crash"
    if "transfer" in term:
        reward = _sim_reward(sim)
        if reward is not None and reward <= 0:
            return "reward0-early-transfer"
        return "transfer"
    return term or "incomplete"


def incomplete_fixture_traces() -> list[dict[str, Any]]:
    """Deterministic incomplete episode. Not a live 0731 trial."""
    return [
        {
            "taskId": INCOMPLETE_FIXTURE_ID,
            "trial": 0,
            "reward": 0.0,
            "hung": False,
            "termination": "transfer_to_human",
            "reason": "reward0-early-transfer",
            "nodeKey": "solve",
            "role": "solve",
            "input": "complete the remaining airline write; do not transfer",
            "output": "transfer_to_human_agents",
            "ts": 0,
        }
    ]


def incomplete_train_traces(
    simulations: list[Any],
    extra_traces: dict[str, list[Any]] | None = None,
) -> list[dict[str, Any]]:
    extra = extra_traces or {}
    out: list[dict[str, Any]] = []
    for sim in simulations:
        if not is_incomplete_episode(sim):
            continue
        tid = str(getattr(sim, "task_id", "?"))
        reason = incomplete_reason(sim)
        harvested = extra.get(tid) or []
        if harvested:
            for t in harvested:
                row = dict(t) if isinstance(t, dict) else {"output": str(t)}
                row.setdefault("nodeKey", row.get("nodeKey") or "solve")
                row.setdefault("role", row.get("role") or "solve")
                row.setdefault("input", row.get("input") or tid)
                row.setdefault("output", row.get("output") or "")
                row.setdefault("ts", row.get("ts") or 0)
                row["taskId"] = tid
                row["trial"] = getattr(sim, "trial", 0)
                row["reward"] = _sim_reward(sim)
                row["hung"] = bool(getattr(sim, "hung", False))
                row["termination"] = _termination(sim)
                row["reason"] = reason
                out.append(row)
            continue
        out.append(
            {
                "taskId": tid,
                "trial": getattr(sim, "trial", 0),
                "reward": _sim_reward(sim),
                "hung": bool(getattr(sim, "hung", False)),
                "termination": _termination(sim),
                "reason": reason,
                "nodeKey": "solve",
                "role": "solve",
                "input": tid,
                "output": _termination(sim) or reason,
                "ts": 0,
            }
        )
    return out


def _sidecar_catalog_jump(
    sidecar: Any,
    *,
    before: float,
    after: float | None = None,
) -> dict[str, Any]:
    """I_sku actuator: propose pro-0813, gate, maybe write S. Catalog rebind, not fine-tuning."""
    req: dict[str, Any] = {
        "op": "i_sku",
        "model": CATALOG_JUMP_MODEL,
        "before": before,
    }
    if after is not None:
        req["after"] = after
    res = sidecar.request(req)
    ping = sidecar.request({"op": "ping"})
    catalog = res.get("catalog") or {}
    jumped = bool(res.get("jumped") or catalog.get("jumped"))
    mounted = catalog.get("action") == "mount"
    return {
        "arm": "I_sku",
        "kind": "catalog-rebind",
        "notPostTraining": True,
        "notFineTuning": True,
        "trained": False,
        "proposed": CATALOG_JUMP_MODEL,
        "from": SERVING_MODEL,
        "to": CATALOG_JUMP_MODEL,
        "jumped": jumped,
        "mounted": mounted,
        "rejected": not mounted,
        "servingPaused": bool(res.get("servingPaused")) or bool(ping.get("servingPaused")),
        "servingModel": res.get("servingModel") or catalog.get("servingModelId") or SERVING_MODEL,
        "serving": res.get("serving")
        or catalog.get("serving")
        or {"sku": res.get("servingModel") or SERVING_MODEL, "servingPaused": False},
        "gate": res.get("gate") or {},
        "catalog": catalog,
        "graph": res.get("graph") or catalog.get("graph"),
        "honestNote": I_SKU_NOTE,
        "not0731Weights": True,
        "spawned": True,
        "done": True,
    }


def _sidecar_weight(
    sidecar: Any,
    *,
    traces: list[dict[str, Any]] | None = None,
    before: float,
    after: float | None = None,
    trainer: str = "surrogate",
    base_model: str = "surrogate-theta",
) -> dict[str, Any]:
    """I_weight TrainJob stub: spawn, poll, gate. Not I_sku and not a θ jump."""
    used = list(traces) if traces else incomplete_fixture_traces()
    spawned = sidecar.request(
        {
            "op": "i_weight_spawn",
            "traces": used,
            "trainer": trainer,
            "baseModel": base_model,
        }
    )
    ping = sidecar.request({"op": "ping"})
    job_id = (spawned.get("job") or {}).get("id")
    deadline = time.time() + 8.0
    status: dict[str, Any] = {}
    while time.time() < deadline:
        status = sidecar.request({"op": "i_weight_status", "jobId": job_id})
        if status.get("done"):
            break
        ping = sidecar.request({"op": "ping"})
        if ping.get("servingPaused"):
            break
        time.sleep(0.02)
    gate_req: dict[str, Any] = {
        "op": "i_weight_gate",
        "before": before,
        "jobId": job_id,
    }
    if after is not None:
        gate_req["after"] = after
    gate = sidecar.request(gate_req)
    job = gate.get("job") or status.get("job") or spawned.get("job") or {}
    gate_body = gate.get("gate") or {}
    serving_paused = bool(spawned.get("servingPaused")) or bool(ping.get("servingPaused"))
    mounted = gate_body.get("action") == "mount"
    return {
        "spawned": bool(spawned.get("spawned")),
        "done": bool(status.get("done")),
        "servingPaused": serving_paused,
        "ping": ping.get("content"),
        "gate": gate_body,
        "job": job,
        "mounted": mounted,
        "rejected": not mounted,
        "tracesUsed": len(used),
        "incompleteFixture": all(
            t.get("taskId") == INCOMPLETE_FIXTURE_ID for t in used
        ),
        "surrogate": bool(job.get("surrogate", trainer == "surrogate")),
        "not0731Weights": True,
        "honestNote": I_WEIGHT_NOTE,
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
    i_weight: dict[str, Any] | None = None,
    i_sku: dict[str, Any] | None = None,
    self_obs_path: str | None = None,
    self_obs: dict[str, Any] | None = None,
    apply_scope: dict[str, list[str]] | None = None,
    controller: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
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
    if i_weight is not None:
        rec["iWeight"] = i_weight
    if i_sku is not None:
        rec["iSku"] = i_sku
    if self_obs_path:
        rec["selfObsPath"] = self_obs_path
    if self_obs:
        rec["selfObs"] = self_obs
    if apply_scope is not None:
        rec["applyScope"] = apply_scope
    if controller is not None:
        rec["controller"] = controller
    return rec


def _weight_graph_diff(w: dict[str, Any]) -> list[dict[str, str]]:
    action = "mount" if w.get("mounted") else "reject"
    return [{"op": action, "key": "adapter", "note": "surrogate-or-reject"}]


def _run_weight_round(
    *,
    sidecar: Any,
    sims: list[Any],
    extra_traces: dict[str, list[Any]] | None,
    before: float,
    trainer: str = "surrogate",
    base_model: str = "surrogate-theta",
) -> dict[str, Any]:
    traces = incomplete_train_traces(sims, extra_traces)
    fixture = not traces
    if fixture:
        traces = incomplete_fixture_traces()
    # Held-out: surrogate cannot complete incomplete episodes. Honest after=0
    # unless FakeTrainer is explicitly requested (protocol unit test).
    after: float | None = 1.0 if trainer == "fake" else 0.0
    w = _sidecar_weight(
        sidecar,
        traces=traces,
        before=before,
        after=after,
        trainer=trainer,
        base_model=base_model,
    )
    w["incompleteFixture"] = fixture or bool(w.get("incompleteFixture"))
    return w


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
    i_weight_report: dict[str, Any] | None = None
    i_sku_report: dict[str, Any] | None = None

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
        from tau2_vdom.agent import TURN_TRACES_BY_TASK

        ctrl = control_batch(obs)
        slice_arm = ctrl["slice"]
        apply_scope = ctrl["applyScope"]
        applied = list(ctrl["applied"])
        # Consume buckets, not slice. Mixed 39/44 is slice=I_sku but 39 still I_loop.
        sku_w: dict[str, Any] | None = None
        if "I_sku" in applied:
            current = _p_hit(pass_hat)
            before = 0.0 if current is None else current
            # Fixture after only at the controller; omit here so live airline
            # cannot invent p_hit(0813)>p_hit(0731). 0813 existing is not a gate.
            sku_w = _sidecar_catalog_jump(sidecar, before=before)
            i_sku_report = sku_w
            serving_paused = serving_paused or bool(sku_w.get("servingPaused"))

        if "I_loop" not in applied and "I_sku" in applied:
            rounds.append(
                _round_record(
                    round_i=r,
                    technique=technique,
                    pass_hat=pass_hat,
                    avg=avg,
                    obs=obs,
                    by_task=_rewards_by_task(sims, task_ids),
                    intervention="I_sku",
                    graph_diff=[{"op": "catalog-rebind" if sku_w.get("jumped") else "reject", "key": "solve", "note": "pro-0813"}],
                    eval_file=path,
                    skipped=skipped,
                    reward_infos=[
                        serialize_reward_info(getattr(s, "reward_info", None)) for s in sims
                    ],
                    i_sku=sku_w,
                    apply_scope=apply_scope,
                    controller=ctrl,
                )
            )
            stop_reason = "catalog-mounted" if sku_w.get("jumped") else "catalog-rejected"
            break

        loop = _sidecar_i_loop(
            sidecar,
            obs=obs,
            ctx=_self_obs_ctx(sims, obs, dict(TURN_TRACES_BY_TASK)),
            model=model,
        )
        serving_paused = serving_paused or bool(loop.get("servingPaused"))
        self_obs_path = str(loop.get("selfObsPath") or "fallback")
        apply_scope = loop.get("applyScope") or apply_scope_from_obs(obs)
        self_obs_rec = {
            "path": self_obs_path,
            "action": loop.get("action"),
            "rationale": loop.get("rationale"),
            "applyScope": apply_scope,
        }

        if loop.get("action") == "wait" and not loop.get("applied"):
            if weight and miss:
                current = _p_hit(pass_hat)
                before = 0.0 if current is None else current
                w = _run_weight_round(
                    sidecar=sidecar,
                    sims=sims,
                    extra_traces=dict(TURN_TRACES_BY_TASK),
                    before=before,
                )
                i_weight_report = w
                serving_paused = serving_paused or bool(w.get("servingPaused"))
                rounds.append(
                    _round_record(
                        round_i=r,
                        technique=technique,
                        pass_hat=pass_hat,
                        avg=avg,
                        obs=obs,
                        by_task=_rewards_by_task(sims, task_ids),
                        intervention="I_weight",
                        graph_diff=_weight_graph_diff(w),
                        eval_file=path,
                        skipped=skipped,
                        reward_infos=[
                            serialize_reward_info(getattr(s, "reward_info", None)) for s in sims
                        ],
                        i_weight=w,
                        self_obs_path=self_obs_path,
                        self_obs=self_obs_rec,
                        apply_scope=apply_scope,
                    )
                )
                stop_reason = "weight-mounted" if w.get("mounted") else "weight-rejected"
                break
            stop_reason = "wait" if miss else "saturated"
            break

        if loop.get("applied"):
            intervention = "I_loop+I_sku" if sku_w is not None else "I_loop"
            graph_diff = list(loop.get("graphDiff") or [])
            if sku_w is not None:
                graph_diff.append(
                    {"op": "catalog-rebind" if sku_w.get("jumped") else "reject", "key": "solve", "note": "pro-0813"}
                )
            technique = str(loop.get("technique") or technique)
            os.environ["VDOM_TAU2_TECHNIQUE"] = technique
        elif weight:
            current = _p_hit(pass_hat)
            before = 0.0 if current is None else current
            w = _run_weight_round(
                sidecar=sidecar,
                sims=sims,
                extra_traces=dict(TURN_TRACES_BY_TASK),
                before=before,
            )
            i_weight_report = w
            serving_paused = serving_paused or bool(w.get("servingPaused"))
            intervention = "I_weight"
            graph_diff = _weight_graph_diff(w)
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
                    i_weight=w,
                    self_obs_path=self_obs_path,
                    self_obs=self_obs_rec,
                    apply_scope=apply_scope,
                )
            )
            stop_reason = "weight-mounted" if w.get("mounted") else "weight-rejected"
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
                self_obs_path=self_obs_path,
                self_obs=self_obs_rec,
                apply_scope=apply_scope,
                i_sku=sku_w,
                controller=ctrl,
            )
        )
        if _saturated(pass_hat):
            stop_reason = "saturated"
            break
    else:
        if not _saturated(pass_hat):
            stop_reason = "budget"

    if weight and i_weight_report is None and i_sku_report is None:
        from tau2_vdom.agent import TURN_TRACES_BY_TASK

        current = _p_hit(pass_hat)
        before = 0.0 if current is None else 0.0
        # Official slice may have saturated; I_weight TrainJob stub still runs on incompletes
        # or the deterministic fixture. Do not invent a new official p_hit.
        w = _run_weight_round(
            sidecar=sidecar,
            sims=sims,
            extra_traces=dict(TURN_TRACES_BY_TASK),
            before=before,
        )
        i_weight_report = w
        serving_paused = serving_paused or bool(w.get("servingPaused"))
        rounds.append(
            _round_record(
                round_i=len(rounds),
                technique=technique,
                pass_hat=pass_hat,
                avg=avg,
                obs=obs,
                by_task=_rewards_by_task(sims, task_ids),
                intervention="I_weight",
                graph_diff=_weight_graph_diff(w),
                eval_file=path,
                skipped=skipped,
                reward_infos=[
                    serialize_reward_info(getattr(s, "reward_info", None)) for s in sims
                ],
                i_weight=w,
            )
        )
        stop_reason = "weight-mounted" if w.get("mounted") else "weight-rejected"

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
        "selfObsPaths": [x.get("selfObsPath") for x in improve_rounds],
        "applyScopes": [x.get("applyScope") for x in improve_rounds],
        "rounds": rounds,
        "servingPaused": serving_paused,
        "iWeight": i_weight_report,
        "iSku": i_sku_report,
        "skipPolicy": SKIP_POLICY,
        "command": (
            "PYTHONPATH=python python3 -m tau2_vdom.improve"
            + ("" if domain == "mock" else f" --domain {domain}")
            + (" --weight" if weight else "")
        ),
    }
    if i_sku_report is not None:
        payload["honestNote"] = I_SKU_NOTE
        payload["note"] = (note + " " + I_SKU_NOTE).strip()
    elif weight:
        payload["honestNote"] = I_WEIGHT_NOTE
        payload["note"] = (note + " " + I_WEIGHT_NOTE).strip()
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
        "iWeight": {
            "spawned": bool((i_weight_report or {}).get("spawned")),
            "done": bool((i_weight_report or {}).get("done")),
            "mounted": bool((i_weight_report or {}).get("mounted")),
            "rejected": bool((i_weight_report or {}).get("rejected")),
            "servingPaused": bool((i_weight_report or {}).get("servingPaused")),
        }
        if i_weight_report
        else None,
        "iSku": {
            "arm": "I_sku",
            "kind": "catalog-rebind",
            "trained": False,
            "notFineTuning": True,
            "jumped": bool((i_sku_report or {}).get("jumped")),
            "mounted": bool((i_sku_report or {}).get("mounted")),
            "rejected": bool((i_sku_report or {}).get("rejected")),
            "servingPaused": bool((i_sku_report or {}).get("servingPaused")),
            "servingModel": (i_sku_report or {}).get("servingModel"),
        }
        if i_sku_report
        else None,
    }, indent=2))
    return out


def run_weight_fixture_improve(
    *,
    trainer: str = "surrogate",
    base_model: str = "surrogate-theta",
) -> Path:
    """I_weight TrainJob stub on the deterministic incomplete fixture. Not I_sku."""
    from tau2_vdom.sidecar import default_sidecar

    sidecar = default_sidecar()
    sidecar.request({"op": "ping"})
    traces = incomplete_fixture_traces()
    w = _sidecar_weight(
        sidecar,
        traces=traces,
        before=0.0,
        after=1.0 if trainer == "fake" else 0.0,
        trainer=trainer,
        base_model=base_model,
    )
    w["incompleteFixture"] = True
    serving_paused = bool(w.get("servingPaused"))
    empty_pass: dict[str, float] = {}
    rec = _round_record(
        round_i=0,
        technique="one-shot",
        pass_hat=empty_pass,
        avg=0.0,
        obs=[],
        by_task={INCOMPLETE_FIXTURE_ID: [0.0]},
        intervention="I_weight",
        graph_diff=_weight_graph_diff(w),
        eval_file=EVAL_DIR / "latest-improve.json",
        i_weight=w,
    )
    rec["evalFile"] = "eval/tau2/latest-improve.json"
    payload = {
        "benchmark": "tau2-bench",
        "kind": "runtime-self-improvement",
        "closedLoop": True,
        "claim": CLAIM,
        "note": I_WEIGHT_NOTE,
        "honestNote": I_WEIGHT_NOTE,
        "paperRepo": PAPER_REPO,
        "tau2Repo": TAU2_REPO,
        "metricNote": METRIC_NOTE,
        "domain": "mock",
        "taskIds": [INCOMPLETE_FIXTURE_ID],
        "numTrials": 1,
        "maxRounds": 0,
        "stopReason": "weight-mounted" if w.get("mounted") else "weight-rejected",
        "agent": "vdom",
        "model": "deterministic",
        "provider": "deterministic",
        "live": False,
        "incompleteFixture": True,
        "passHatKBefore": {},
        "passHatKAfter": {},
        "pHitSequence": [0.0],
        "avgRewardBefore": 0.0,
        "avgRewardAfter": 0.0,
        "interventions": ["I_weight"],
        "graphDiffs": [rec["graphDiff"]],
        "rounds": [rec],
        "servingPaused": serving_paused,
        "iWeight": w,
        "skipPolicy": SKIP_POLICY,
        "command": "PYTHONPATH=python python3 -m tau2_vdom.improve --weight --weight-fixture",
    }
    out = write_improve_report(payload)
    print(json.dumps({
        "wrote": str(out),
        "closedLoop": True,
        "stopReason": payload["stopReason"],
        "interventions": payload["interventions"],
        "servingPaused": serving_paused,
        "iWeight": {
            "spawned": w.get("spawned"),
            "done": w.get("done"),
            "mounted": w.get("mounted"),
            "rejected": w.get("rejected"),
            "servingPaused": serving_paused,
        },
    }, indent=2))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Closed-loop τ² self-improvement: Obs → I_loop|I_sku → Obs, "
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
        help=(
            "On I_sku (hung / crash / no-write), propose catalog rebind to "
            "deepseek/deepseek-v4-pro-0813 and gate it. Not I_weight-as-trainer "
            "and not fine-tuning. Serving is never paused. --weight-fixture is "
            "the I_weight TrainJob stub, not this arm."
        ),
    )
    p.add_argument(
        "--weight-fixture",
        action="store_true",
        help=(
            "Run only the I_weight TrainJob stub on the deterministic incomplete "
            "fixture (no tau2 slice, no API key). Not I_sku and not a θ jump."
        ),
    )
    p.add_argument(
        "--isku-mount-cell",
        action="store_true",
        help=(
            "Honest I_sku mount protocol cell: replay hung-44 license through "
            "controlBatch / Obs (same sources as #12 reject), call i_sku WITH "
            "fixture after, then one live OpenRouter completion on 0813. "
            "Not a τ² score. Not invented p_hit(0813). Writes "
            "eval/tau2/improve-live-0731-isku-44-mount.json."
        ),
    )
    p.add_argument(
        "--hybrid-state-s-dump",
        action="store_true",
        help=(
            "X_n.S dump after licensed I_sku write. Reuses the #15 mount-cell "
            "controller (hung-44 + fixture after). Does not re-run a live 0813 "
            "ping. jumped is the S write. Not a score. Writes "
            "eval/tau2/hybrid-state-s-dump.json."
        ),
    )
    p.add_argument(
        "--hybrid-state-serving-step-dump",
        action="store_true",
        help=(
            "Serving-step X_n dump after licensed I_sku write (hole (1) after "
            "#17). One runTau2Turn on the existing HybridState; H/M from that "
            "turn. licenseE is the hung/timeout LICENSE; servingE is the "
            "greeting turn. Not a score. Writes "
            "eval/tau2/hybrid-state-serving-step-dump.json."
        ),
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


def run_isku_mount_cell(*, live_serve: bool = True, write: bool = True) -> Path:
    """Dedicated I_sku mount protocol cell. Does not run the live airline improveLoop.

    Live airline improveLoop omits after so it cannot invent p_hit(0813).
    This path calls I_sku WITH fixture after, then one live 0813 serve.
    """
    import shutil
    import subprocess

    _apply_openrouter_defaults()
    npx = shutil.which("npx")
    if npx is None:
        raise SystemExit("npx not found; install Node.js >= 18")
    script = REPO_ROOT / "src" / "eval" / "tau2-isku-mount-cell.ts"
    cmd = [npx, "--yes", "tsx", str(script)]
    if not live_serve:
        cmd.append("--no-live")
    if not write:
        cmd.append("--no-write")
    print("[isku-mount-cell] " + " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    out = EVAL_DIR / "improve-live-0731-isku-44-mount.json"
    if completed.returncode not in {0, 2}:
        raise SystemExit(completed.returncode)
    if write and not out.is_file():
        raise SystemExit(f"mount cell did not write {out}")
    return out


def run_hybrid_state_s_dump(*, write: bool = True) -> Path:
    """X_n.S dump after licensed write. No live 0813 ping. No airline improveLoop."""
    import shutil
    import subprocess

    npx = shutil.which("npx")
    if npx is None:
        raise SystemExit("npx not found; install Node.js >= 18")
    script = REPO_ROOT / "src" / "eval" / "tau2-hybrid-state-s-dump.ts"
    cmd = [npx, "--yes", "tsx", str(script)]
    if not write:
        cmd.append("--no-write")
    print("[hybrid-state-s-dump] " + " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    out = EVAL_DIR / "hybrid-state-s-dump.json"
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    if write and not out.is_file():
        raise SystemExit(f"hybrid-state dump did not write {out}")
    return out


def run_hybrid_state_serving_step_dump(*, write: bool = True) -> Path:
    """Serving-step X_n dump. One runTau2Turn on the existing X. No airline improveLoop."""
    import shutil
    import subprocess

    npx = shutil.which("npx")
    if npx is None:
        raise SystemExit("npx not found; install Node.js >= 18")
    script = REPO_ROOT / "src" / "eval" / "tau2-hybrid-state-serving-step-dump.ts"
    cmd = [npx, "--yes", "tsx", str(script)]
    if not write:
        cmd.append("--no-write")
    print("[hybrid-state-serving-step-dump] " + " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    out = EVAL_DIR / "hybrid-state-serving-step-dump.json"
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    if write and not out.is_file():
        raise SystemExit(f"serving-step dump did not write {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.hybrid_state_serving_step_dump:
        run_hybrid_state_serving_step_dump()
        return 0
    if args.hybrid_state_s_dump:
        run_hybrid_state_s_dump()
        return 0
    if args.isku_mount_cell:
        run_isku_mount_cell()
        return 0
    if args.weight_fixture or (args.weight and os.environ.get("VDOM_WEIGHT_FIXTURE") == "1"):
        run_weight_fixture_improve()
        return 0
    _ensure_tau2_data_dir()
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
            if args.weight:
                print(
                    "tau2 is not installed; running I_weight TrainJob stub on the deterministic "
                    "incomplete fixture (protocol smoke, no live p_hit).",
                    file=sys.stderr,
                )
                run_weight_fixture_improve()
                return 0
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
