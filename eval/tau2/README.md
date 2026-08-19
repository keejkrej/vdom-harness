# tau2-bench scores (vdom agent)

Official scores come from measured rewards and the Yao et al. `pass^k` estimator `C(c,k)/C(n,k)`. This harness does not invent `pass^k`.

## The claim is runtime self-improvement

A vdom agent observes its traces (Obs), then either mutates the AgentGraph (`I_loop`) or dispatches an async trainer and mounts weights when a gate passes (`I_weight`). Serving does not pause.

The 5×4 retail one-shot on tasks 0–4 scored `pass^k=1.0`. That slice is **saturated** — it cannot show improvement. Do not lead with it. The figure is `python -m tau2_vdom.improve`.

## Mock-domain before / after (no API key)

Naive one-shot fails official `update_task_1` (create_task attractor). `I_loop` mounts critic+refine (Self-Refine); the same task is re-run.

```
bash scripts/setup-tau2.sh
PYTHONPATH=python python3 -m tau2_vdom.improve
# or: npm run eval:tau2:improve
```

Writes `eval/tau2/latest-improve.json`: `passHatKBefore`, `passHatKAfter`, `intervention`, `graphDiff`, `obs`. Optional `--weight` spawns `FakeTrainer` without pausing serving and mounts only if after-eval beats before.

Measured (deterministic, no key, official `update_task_1`, tau2 evaluator):

- `passHatKBefore` / `avgRewardBefore` = 0.0 (naive one-shot, create_task attractor)
- `passHatKAfter` / `avgRewardAfter` = 1.0 (`I_loop` Self-Refine)
- `graphDiff`: retain solve, mount critic, mount refine
- `I_weight` FakeTrainer spawned; servingPaused=false; gate **rejected** (after did not beat current 1.0)

These are measured. No live airline/retail after-scores are recorded here.

## Live (needs a key)

Default live model: `deepseek/deepseek-v4-flash-0731` via OpenRouter. Probe airline, or retail tasks **beyond 0–4**. If the naive graph already scores 1.0, the report records `intervention: none` and does not invent an after-score.

```
export OPENROUTER_API_KEY=...
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export OPENAI_MODEL=deepseek/deepseek-v4-flash-0731
PYTHONPATH=python python3 -m tau2_vdom.improve --domain airline --num-tasks 4 --num-trials 1
PYTHONPATH=python python3 -m tau2_vdom.improve --domain retail --task-ids 5 6 7 8 9 --num-trials 1
```

## Saturated one-shot retail (not the claim)

Command:

    PYTHONPATH=python python3 -m tau2_vdom --domain retail --num-tasks 5 --num-trials 4

Model: deepseek/deepseek-v4-flash-0731
Provider: openrouter
N = 5 tasks x 4 trials = 20 simulations.
Finished: 2026-08-19 10:36 CEST.

compute_metrics (copied):
- avgReward = 1.0
- pass^1 = 1.0
- pass^2 = 1.0
- pass^3 = 1.0
- pass^4 = 1.0

All 20 simulations: user_stop, reward 1.0. Compact: retail-live-metrics.json.
This slice cannot demonstrate runtime improvement.
