# tau2-bench scores (vdom agent)

Official scores come from measured rewards and the Yao et al. `pass^k` estimator `C(c,k)/C(n,k)`. This harness does not invent `pass^k`.

## The claim is a closed loop

self-observe → `I_loop` or `I_weight` → run the same tasks → self-observe again, until `pass^k` saturates or a round budget. Serving does not pause. Not a single before/after.

The 5×4 retail one-shot on tasks 0–4 scored `pass^k=1.0`. That slice is **saturated** — it cannot show improvement. Do not lead with it. The figure is `python -m tau2_vdom.improve`.

## Mock-domain closed loop (no API key)

Two official mock tasks so two `I_loop` rounds actually change `p_hit`:

1. `update_task_1` — naive create_task attractor; Self-Refine recovers.
2. `impossible_task_1` — still misses after Self-Refine; validator node transfers.

```
bash scripts/setup-tau2.sh
PYTHONPATH=python python3 -m tau2_vdom.improve
# or: npm run eval:tau2:improve
```

Writes `eval/tau2/latest-improve.json`: `rounds[]` with `pHit`, `passHatK`, `taskPHit`, `obs`, `intervention`, `graphDiff` per round. `--max-rounds` is the improve budget (default 4). `--weight` only if topology is exhausted and the slice still misses.

Measured (deterministic, no key, official tau2 evaluator):

| round | technique | p_hit (pass^1) | update_task_1 | impossible_task_1 | intervention | graphDiff |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | one-shot | 0.0 | 0 | 0 | — | — |
| 1 | self-refine | 0.5 | 1 | 0 | I_loop | mount critic, refine |
| 2 | validator | 1.0 | 1 | 1 | I_loop | mount validator |

Then Obs saturates and the loop stops. These are measured (deterministic mock).

## Live closed loop on 0731 (airline 39, 44, 41)

Command:

    PYTHONPATH=python python3 -m tau2_vdom.improve --domain airline --task-ids 39 44 41 --num-trials 1 --max-rounds 3 --max-steps 80 --trial-timeout 480 --model deepseek/deepseek-v4-flash-0731

Model: `deepseek/deepseek-v4-flash-0731` via OpenRouter. User simulator and (if used) NL-assertion judge pinned to `openrouter/deepseek/deepseek-v4-flash-0731`. Airline `reward_basis` is DB+COMMUNICATE, so EvaluationType.ALL does not call the NL judge. No gpt-4.1 calls observed.

Finished: 2026-08-19 14:11 CEST. Compact: `improve-live-0731.json`.

| round | technique | p_hit (pass^1) | 39 | 41 | 44 | intervention | graphDiff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | one-shot | 0.333 | 0 | 1 | 0 | — | — |
| 1 | self-refine | 0.0 | 0 | skip (hung > 8 min) | 0 | I_loop | mount critic, refine |
| 2 | validator | 0.333 | 0 | 1 | 0 | I_loop | mount validator |

`stopReason: loop-exhausted`. `servingPaused: false`. I_loop did **not** raise p_hit vs naive one-shot. Round-1 0.0 is a smaller denominator (task 41 skipped). Tasks 39 and 44 stayed 0.0. Not invented.

Probe (one-shot, same model): airline 0/1/2 all hit 1.0 (too easy); 23 and 18 hit; 39, 44, 41 missed (41 later hit on the loop's one-shot trial).

## Live (needs a key)

Default live model: `deepseek/deepseek-v4-flash-0731` via OpenRouter. Probe airline, or retail tasks **beyond 0–4**. If the naive graph already scores 1.0, the report records `stopReason: saturated` after the first Obs and does not invent an after-score.

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
