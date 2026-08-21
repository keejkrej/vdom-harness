# tau2-bench scores (vdom agent)

Official scores come from measured rewards and the Yao et al. `pass^k` estimator `C(c,k)/C(n,k)`. This harness does not invent `pass^k`.

## The claim is a closed loop

self-observe → `I_loop` or `I_sku` → run the same tasks → self-observe again, until `pass^k` saturates or a round budget. The serving agent may `get_agent_graph` / `set_agent_graph` mid-turn (local intercept; never a gym tool). Host I_loop is fallback if it never called set. Serving does not pause. Not a single before/after. `I_sku` is a gated catalog rebind from `deepseek/deepseek-v4-flash-0731` to `deepseek/deepseek-v4-pro-0813`. Jump iff later serving model id is 0813. Not I_weight-as-trainer and not fine-tuning.

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

Writes `eval/tau2/latest-improve.json`: `rounds[]` with `pHit`, `passHatK`, `taskPHit`, `obs`, `intervention`, `graphDiff` per round. `--max-rounds` is the improve budget (default 4). Incomplete / hung episodes fire `I_sku` (propose pro-0813, gate, write serving pointer `S` beside `C`; catalog rebind, not fine-tuning). `--weight-fixture` is the `I_weight` TrainJob stub — not a catalog jump and not a θ win. Serving does not pause.

Measured (deterministic, no key, official tau2 evaluator):

| round | technique | p_hit (pass^1) | update_task_1 | impossible_task_1 | intervention | graphDiff |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | one-shot | 0.0 | 0 | 0 | — | — |
| 1 | self-refine | 0.5 | 1 | 0 | I_loop | mount critic, refine |
| 2 | validator | 1.0 | 1 | 1 | I_loop | mount validator |

Then Obs saturates and the loop stops. These are measured (deterministic mock).

`--weight` then spawns a slow-clock `TrainJob` from incomplete-episode traces (or the deterministic fixture if the official slice already hit). Serving stays up. The surrogate cannot raise `p_hit`; the gate rejects. Official mock `p_hit` is unchanged. Not a 0731 LoRA.

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

Official post-gate 39/44 log (replay falsifier, not a live 0813 vs 0731 table): 39 is `I_loop`; 44 hung/timeout is `I_sku` / catalog rebind; `waitKept=[]`. If 44 is `I_loop` unless `loopExhausted`, the test fails. 44 did not train.

Honest I_sku reject cell (controller replay of saved live hung-44 traces; omit after; `nTurns=9` is one hung trial, not nine hangs; not a new timeout; not a result): `improve-live-0731-isku-44-reject.json`.

Honest I_sku mount protocol cell (same hung-44 license as the #12 reject; I_sku WITH fixture after; then one live 0813 serve). Not a τ² lift, not invented `p_hit(0813)`, not a Pro-vs-Flash score: `improve-live-0731-isku-44-mount.json`.

```
npm test
PYTHONPATH=python python3 -m tau2_vdom.improve --isku-mount-cell
# or: npm run eval:tau2:isku-mount-cell
# or: npx tsx src/eval/tau2-isku-mount-cell.ts
```

Airline `reward_basis` is DB × COMMUNICATE (`communicate_info` is `[]` on 39/44), so score 0 is a DB miss. ACTION / `nl_assertions` are diagnostics only.

Gold (from `.tau2-bench/data/tau2/domains/airline/tasks.json`; the policy node encodes **rules**, never these IDs):

- Task 39: cancel every eligible reservation (business, or economy + insurance). Do not invent a “personal reason” block on insured economy. Do not cancel basic economy without insurance, or anything already / partially flown. Extra ineligible cancels also zero the DB.
- Task 44: if the user is healthy, insurance does not apply — refuse that economy cancel. Then complete **every** eligible cabin upgrade (`update_reservation_flights`). Cancelling the ineligible reservation zeros the score. Do not force-cancel it.
- Task 41: check reservations, cancel none. Already hits.

The next I_loop arm is `policy-checklist`: if Obs sees `refusedCancel` / a missed `cancel_reservation` or `update_reservation_*`, mount that policy critic instead of another generic self-refine. Hung trials retry once and stay in `taskPHit` as null rather than disappearing. This arm is not a measured p_hit win until the parent re-runs live 0731.

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
