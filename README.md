# vdom — virtual DOM for agents

Pi customizes an agent. DSH composes a runtime. vdom reconciles a society that can rewrite itself.

A vdom agent **observes itself** — traces, scores, failures — and **reengineers itself**. It can mutate its AgentGraph (the loop), mount capabilities (harness / tools), or dispatch an **async** trainer and switch `f_θ` / adapters when an eval gate passes. It does this however it wants.

The paper lives in [agent-stochastic-dynamics](https://github.com/keejkrej/agent-stochastic-dynamics) — theory, ICLR draft, typed kernel, OpenRouter traces. This repo is the accompanying runtime submitted with that paper: the self-observing agent that can reengineer its loop and dispatch async weight updates. vdom is not the paper.

You do not `spawnAgent()`. You declare an AgentGraph. A reconciler diffs topology. The agent reads its own traces and emits a new graph, a capability, or an async weight job. Improvement does not patch the host runtime.

Sources of "what to become" are untrusted text: papers, blogs, GitHub repos, X posts, other research agents, conversation, traces. A compiler turns some of that into a graph. That compiler is a convenience, not the product.

## Demo

```
npm run export && npm run viz
```

Open http://127.0.0.1:4173. The page replays `public/run.json`, an event log from a real TypeScript run — not hardcoded scores.

## Coding agent (ACP)

`vdom` is also a headless coding agent that speaks the [Agent Client Protocol](https://agentclientprotocol.com) on stdio. No TUI, no GUI: an ACP client (T3 Code, Zed, …) drives it. Its system prompt is compiled from an AgentGraph, and the agent can rewrite that graph mid-session with `set_agent_graph` (reconciled, diff reported). Executable kinds (`capability` / `adapter`) cannot be mounted that way.

```
npm i && npm run build && npm link     # puts `vdom` on PATH
vdom doctor                            # config + one test completion
vdom run --cwd ~/code/repo "fix the failing test"   # headless, auto-approves
vdom                                   # ACP server on stdio
```

Model endpoint is any OpenAI-compatible `/chat/completions`. Precedence: CLI flags > `VDOM_BASE_URL`/`VDOM_API_KEY` > `~/.vdom/config.json` > generic provider keys > local Ollama.

| Source | Endpoint | Default model |
| --- | --- | --- |
| `VDOM_BASE_URL` (+ `VDOM_API_KEY`) | that URL | `VDOM_MODEL` |
| `~/.vdom/config.json` (`baseUrl`, `apiKey` or `apiKeyEnv`, `model`, `diagnosisModel`, `sentimentModel`, `models`, `contextTokens`, `guards`, …) | `baseUrl` | `model` |
| `OLLAMA_API_KEY` | `https://ollama.com/v1` | `gpt-oss:120b` |
| `OPENROUTER_API_KEY` | OpenRouter | `deepseek/deepseek-v4-flash-0731` |
| `OPENAI_API_KEY` | `OPENAI_BASE_URL` or OpenAI | — |
| none | local Ollama `http://127.0.0.1:11434/v1` | `gpt-oss:20b` |

Tools (Pi/DSH-grade): `read` (text + images, 2000 lines/50KB pages), `edit` (multiple exact replacements per call, fuzzy fallback, BOM/CRLF preserved), `write`, `bash` (tail-truncated, full output spilled to a temp file, `run_in_background` jobs + `job_output`/`job_kill`), `grep`/`find`/`ls` (ripgrep), `web_fetch`, `todo_write` (ACP plan), `subagent` (fresh context, parallel), `history`, `get_agent_graph`/`set_agent_graph`, plus MCP servers from the ACP client, `~/.vdom/mcp.json`, `.vdom/mcp.json` or `.mcp.json`. Habitual names (`search`, `read_file`, `str_replace`, `shell`, …) are aliased to the real tools.

Context: `AGENTS.md`/`CLAUDE.md` from `~/.vdom` and every directory root → cwd, `SYSTEM.md`/`APPEND_SYSTEM.md`, skills (`SKILL.md` in `.vdom/skills`, `.agents/skills`, `.claude/skills`), prompt templates (`.vdom/prompts`, `.claude/commands`; `/name args`), `/compact [focus]`, `/reload`. Thinking level (`off|low|medium|high`), auto-compaction with overflow recovery, steering (a prompt sent mid-turn joins the running turn), fork/load/resume/list.

Modes (ACP session modes and the `mode` config option): `agent` (edits inside the workspace run; shell commands and edits outside it ask), `ask` (everything asks), `plan` (read-only). `--force` never asks.

### Adaptive model routing

Cheap-first, escalate when needed, decay when it doesn't. Set a ladder with `--ladder cheap,mid,big`, `VDOM_ROUTE_LADDER`, or `routeLadder` in `~/.vdom/config.json` (off by default; `VDOM_ROUTING=off` disables). New sessions start on the cheapest rung. Within a turn, one of these steps the ladder up one rung (at most once per turn):

- a model error, an empty response, or three failed tool calls,
- a `verify_claims` guard nudge (the model claimed success it never checked),
- bad-turn feedback on the previous turn (keyword detector or sentiment interpreter): the *next* turn starts one rung up.

An escalated turn keeps the rung for itself and the next one; after that, each clean turn steps back down one rung until the session is back on the cheapest model. Bad turns never decay — only clean ones do. The position and decay counters persist in the session record, so a resumed session keeps its rung. A manually picked model stands routing down; if the top rung fails, the turn ends with a clear message — no silent fallback spending.

The router also learns, simply and deterministically: turns are bucketed by request kind (question/fix/feature/refactor/other, plus whether the cwd is a vdom harness checkout), and each bucket tracks per-starting-rung outcomes. When rung 0 goes bad in over 40% of ≥5 recorded turns, new turns in that bucket start one rung up (re-evaluated every turn — learning never permanently elevates). The reason is logged in the routing event (`learned: fix/harness`). Stats persist in `~/.vdom/routing/stats.json`; `vdom doctor` shows calls + escalations per model and a per-bucket table while routing is on. Every step (up or down) is logged as a `routing` event in the session log.

### Driving and debugging

```
vdom client --cwd <repo> -V --trace /tmp/t "<prompt>" [-p "<next>"] [-s <session> | -c]   # spawn an ACP agent (--agent "<cmd>" for any) and print the transcript
vdom run   --cwd <repo> "<prompt>"          # same, in-process
vdom sessions [list | show <id> [A-B] [--faults] | analyze <id>]
vdom issues [list | show <id>]
```

Every session is an append-only `events.jsonl` (`~/.vdom/sessions/--<cwd>--/<id>/`): turns, model requests/responses with usage and timings, tool calls with arguments, results, status, errors and durations, permission decisions, compactions with trigger and tokens before/after, feedback, lessons, issues. Transcripts, findings and the resume snapshot are derived from it.

### Self-improvement

- **Real time (in-session).** A `verify_claims` guard stops a turn from ending on an unverified success claim. A keyword detector and a small sentiment model read each user message; a bad turn triggers a read-only background diagnostician (`diagnosisModel`) that inspects the recorded turn and the harness source. Its rule is reconciled into the live session's AgentGraph as a `lesson-N` node and applies from the next model call.
- **Long term.** The same diagnosis files an issue (`~/.vdom/issues`). `vdom fix <issue> [--promote]` cuts a staging worktree from prod, lets vdom fix itself there, and gates the result: the new regression test must fail on the prod base and build + `npm test` must pass on staging. Only then does prod fast-forward.
- **Environments.** `vdom env init --repo <this repo>` creates prod (`~/.vdom/env/prod`, launched through `~/.vdom/bin/vdom`); `vdom env status | stage | gate | promote | rollback | drop`.

### T3 Code

Use the `vdom` provider in the T3 Code fork (branch `vdom-driver`), or in stock T3 set **Settings → Providers → Cursor → Binary path** to `~/.vdom/bin/vdom.cmd` (vdom also speaks the Cursor CLI's spawn contract).

## Not Pi, not DSH

Pi: you customize an agent. Abstraction stops at AgentSession + extensions.
DeepSeek Harness: you compose what a runtime is. Cordis plugins, profiles, bundles.
vdom: you do not spawnAgent(). You declare an AgentGraph. A reconciler diffs topology.
The agent reads its own traces and emits a replacement graph, a capability, or a weight job. The topology is a value.

Agents should not spawnAgent() any more than React should document.createElement() by hand.

They emit a virtual agent graph. A reconciler diffs desired vs current topology and mounts, updates, or unmounts physical nodes. A scientist can emit a replacement graph after reading benchmark traces. A compiler can turn a paper, a blog, a repo, or an X post into a starting graph — that is one input, not the ontology. Improvement does not patch the host runtime.

    untrusted source -> compiler (optional) -> Virtual Agent Graph -> reconciler -> execution
      -> traces + scores -> observe / fail -> new graph | capability | async trainer

The IR is plain objects (JSX in comments is fine). A node is the primitive; an agent is one executor kind.

## Run


Install dependencies, then run the test script and the demo script.

demo is deterministic. No network, no API key. The word-reverse puzzle is the fixture: a naive one-shot reverses the whole string and scores 0; Self-Refine and Reflexion recover to 1.00.

## Sources as programs

Untrusted text can propose a desired topology. This repo's compiler currently encodes two inference-time mechanisms as fixtures — not complete reproductions, and not the only legal input:

- Self-Refine (Madaan et al.): generator -> feedback -> refinement
- Reflexion (Shinn et al.): actor -> reflection -> episodic memory -> retry

`compilePaper(text)` / `compileSource(text)` (same function) routes on those names. Anything else — a blog, a repo README, an X post, another agent's transcript, generic conversation — becomes a one-shot solve node. The scientist mutates a failing one-shot into the Self-Refine topology; the reconciler prints the diff.

A paper is a convenient resource. The product is the society that can rewrite itself.

## Real models

If OPENAI_API_KEY is set, createProvider() uses an OpenAI-compatible chat adapter. Optional OPENAI_BASE_URL (default https://api.openai.com/v1) and OPENAI_MODEL (default gpt-4o-mini). Then researchLoop can compile arbitrary source text.

Without a key, the deterministic provider stays active.

## Evaluation

The result this repo claims is a **closed loop**: self-observe → `I_loop` or `I_sku` → run again → self-observe, until `pass^k` saturates or a round budget. The serving agent sees kernel C and may `get_agent_graph` / `set_agent_graph` mid-turn (intercepted locally; never forwarded to the τ² gym); host I_loop is fallback if it never called set. The canned airline checklist is fallback when self-Obs JSON is invalid. Not a static one-shot τ² score and not a single before/after. Serving does not pause. The 5×4 retail one-shot slice on tasks 0–4 scored `pass^k=1.0` — that slice is saturated and cannot show improvement; do not lead with it.

Toys in `src/benchmarks.ts` (word-reverse, and friends) are **unit fixtures**. They prove the reconciler and DeterministicProvider, not the agent. The paper that accompanies this runtime is [agent-stochastic-dynamics](https://github.com/keejkrej/agent-stochastic-dynamics). This repo is the submitted runtime.

Established tool–agent–user eval is **[τ²-bench](https://github.com/sierra-research/tau2-bench)** (Yao et al. 2024; Barres et al. 2025). We implement their `HalfDuplexAgent` (`python/tau2_vdom/`) and keep the TypeScript AgentGraph. Each turn calls `runTau2Turn` → `complete()` / `completeTurn()`; official tau2 owns domains, tools, user simulator, orchestrator, and `pass^k`. We do not reimplement retail.

```
# Fixtures (no key) — already in npm test
npm test

# Official mock-domain smoke (no key). Installs nothing if tau2 is present.
bash scripts/setup-tau2.sh
npm run eval:tau2:smoke

# Closed loop (no key): observe → I_loop → observe → I_loop → observe
# until pass^k saturates. Mock uses update_task_1 + impossible_task_1
# so two rounds actually change p_hit (0 → 0.5 → 1.0).
PYTHONPATH=python python3 -m tau2_vdom.improve
npm run eval:tau2:improve

# I_sku slow arm (no key): hung / crash / no-write proposes catalog rebind
# from deepseek/deepseek-v4-flash-0731 to deepseek/deepseek-v4-pro-0813.
# Not I_weight-as-trainer and not fine-tuning. --weight-fixture is the stub.
PYTHONPATH=python python3 -m tau2_vdom.improve --weight
PYTHONPATH=python python3 -m tau2_vdom.improve --weight --weight-fixture

# I_sku mount protocol cell (hung-44 license + fixture after + one live 0813 serve).
# Not a τ² score. Needs OPENROUTER_API_KEY for the live serve; without it the
# controller still mounts S for 44 and writes a reject-with-reason JSON.
PYTHONPATH=python python3 -m tau2_vdom.improve --isku-mount-cell

# X_n.S dump after licensed I_sku write (controller; no new 0813 ping).
# Not a score. Writes eval/tau2/hybrid-state-s-dump.json.
npm run eval:tau2:hybrid-state-s-dump
# or: PYTHONPATH=python python3 -m tau2_vdom.improve --hybrid-state-s-dump

# Serving-step X_n dump (holes (1)+(2) after #18): licensed write, then ONE
# runTau2Turn on that same X. licenseE is an own field (hung fixture). Dump
# refuses if that own field is missing (does not invent licenseE from X.E).
# X.E / servingE is attached from that turn (not a dump overlay). Not a score.
# No after= on live airline improveLoop. No key → mock provider still runs a
# real turn; live serving id is not faked.
npm run eval:tau2:hybrid-state-serving-step-dump
# or: PYTHONPATH=python python3 -m tau2_vdom.improve --hybrid-state-serving-step-dump
```

`eval/tau2/latest-improve.json` records the **sequence** of rounds: `pHit` / `passHatK` / `taskPHit`, Obs, intervention, and graph diff per round. Scores are not invented. If a live slice is already 1.0 under the naive graph, the report stops after the first Obs (`stopReason: saturated`).

Live self-improvement (needs a key). Default live domain is **airline**, or retail **held-out** tasks 5–9 — not retail 0–4:

```
export OPENROUTER_API_KEY=...
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export OPENAI_MODEL=deepseek/deepseek-v4-flash-0731
PYTHONPATH=python python3 -m tau2_vdom.improve --domain airline --num-tasks 4 --num-trials 1
PYTHONPATH=python python3 -m tau2_vdom.improve --domain retail --task-ids 5 6 7 8 9 --num-trials 1
```

One-shot retail slice (saturated; not the claim):

```
PYTHONPATH=python python3 -m tau2_vdom --domain retail --num-tasks 5 --num-trials 4
```

The runner registers `--agent vdom` on the official tau2 registry, then calls `run_domain` / `run_single_task`. Use `python -m tau2_vdom` / `python -m tau2_vdom.improve` (not a stock `tau2 run`) so the factory is imported.

Trajectories (actions, tool failures, repeats, reward, `obs` for the paper's `p_hit`) write to `eval/tau2/*.json`. `pass^k` is computed from measured rewards with the official `C(c,k)/C(n,k)` estimator — this harness does not invent scores.

## Limitation

Any source can propose topology, prompts, memory, and routing. The next primitive is already Capability: executable tools still go through that gate — scientist JSON is never trusted as code.

## Runtime improvement

Agents improve the society by emitting graph changes. The reconciler mounts, updates, or unmounts — agents do not patch `RuntimeDOM` internals.

Two gated paths sit beside topology mutation (`researchLoop` / Self-Refine):

1. **Harness (capability)** — propose a `kind: "capability"` node with a `source` ref → sandbox validation → eval (`runBenchmark`) → mount on success. Raw scientist JSON is never executed; only `module:<id>` refs (or exact fingerprints) against a pre-approved capability registry pass the sandbox. Failed eval leaves the live graph unchanged.

2. **Weights (adapter)** — an injectable `Trainer` runs out-of-process (tests use `FakeTrainer`) and returns an `AdapterArtifact`. A `kind: "adapter"` node carries `adapterRef` / `modelRef`; on a passing gate the target agent's `model` pointer updates (same binding as AgentNode.model → PhysicalNode.provider). Failed eval rejects the candidate; a later regression can `rollbackAdapter` (unmount + restore previous model). FakeTrainer is a protocol stub, not the paper slow arm. Official incomplete arm is `I_sku`: gated catalog rebind from `deepseek/deepseek-v4-flash-0731` to `deepseek/deepseek-v4-pro-0813`. Not I_weight-as-trainer and not fine-tuning. Never a fake LoRA.

`improveLoop` chooses topology, capability, or adapter (or `auto`). Real LoRA / Hugging Face Jobs stay behind the `Trainer` port — see `describeHfJobsExtension` in `src/trainer.ts`. No in-process GPU training.

## Layout

- src/ir.ts -- AgentGraph / Node / flatten / clone
- src/reconciler.ts -- mount / update / retain / unmount
- src/providers.ts -- deterministic + OpenAI-compatible
- src/runtime.ts -- walk the mounted graph, collect traces
- src/papers.ts -- source text to graph (`compilePaper` / `compileSource`)
- src/benchmarks.ts -- word-reverse fixtures (not the paper eval)
- src/eval/ -- τ² turn loop, sidecar, Obs, I_loop / I_sku catalog rebind
- python/tau2_vdom/ -- official HalfDuplexAgent + runner + `python -m tau2_vdom.improve`
- src/scientist.ts -- evolve the graph from traces
- src/capability.ts -- approved capability registry + sandbox gate
- src/trainer.ts -- Trainer port, FakeTrainer, adapter artifacts
- src/lifecycle.ts -- propose → sandbox → eval → mount | reject | rollback
- src/improve.ts -- improveLoop (topology | capability | adapter)
- src/acp/ -- ACP coding agent: agent loop, tools, event log, feedback/diagnosis, envs + fix gate, client
- src/demo.ts -- the loop, printed
- src/export-run.ts -- real run to public/run.json
- src/serve.ts -- static viz on :4173

## Scripts

package.json defines demo, test, build, export, and viz. After installing packages: test then demo.

    npm i && npm test && npm run demo
    npm run eval:tau2:smoke     # official τ² mock create_task_1, no API key
    npm run eval:tau2:improve   # naive → Obs → I_loop → same tasks (update_task_1)
    npm run eval:tau2:improve:weight  # I_sku catalog rebind; --weight-fixture is TrainJob stub
