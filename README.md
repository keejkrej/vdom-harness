# vdom — virtual DOM for agents

Pi customizes an agent. DSH composes a runtime. vdom reconciles a society.

## Demo

```
npm run export && npm run viz
```

Open http://127.0.0.1:4173. The page replays `public/run.json`, an event log from a real TypeScript run — not hardcoded scores.

## Not Pi, not DSH

Pi: you customize an agent. Abstraction stops at AgentSession + extensions.
DeepSeek Harness: you compose what a runtime is. Cordis plugins, profiles, bundles.
vdom: you do not spawnAgent(). You declare an AgentGraph. A reconciler diffs topology.
Papers compile to graphs. A scientist emits a replacement graph. The topology is a value.

Agents should not spawnAgent() any more than React should document.createElement() by hand.

They emit a virtual agent graph. A reconciler diffs desired vs current topology and mounts, updates, or unmounts physical nodes. Papers compile into graphs. A scientist agent can emit a replacement graph after reading benchmark traces. Improvement does not patch the host runtime — it changes the graph.

    paper -> paper compiler -> Virtual Agent Graph -> reconciler -> execution
      -> benchmark + traces -> scientist agent -> modified graph

The IR is plain objects (JSX in comments is fine). A node is the primitive; an agent is one executor kind.

## Run


Install dependencies, then run the test script and the demo script.

demo is deterministic. No network, no API key. The word-reverse puzzle is the fixture: a naive one-shot reverses the whole string and scores 0; Self-Refine and Reflexion recover to 1.00.

## Papers as programs

A paper is a program that targets a cognitive IR. This repo encodes two inference-time mechanisms -- not complete reproductions:

- Self-Refine (Madaan et al.): generator -> feedback -> refinement
- Reflexion (Shinn et al.): actor -> reflection -> episodic memory -> retry

compilePaper(text) routes on those names. Anything else becomes a one-shot solve node. The scientist mutates a failing one-shot into the Self-Refine topology; the reconciler prints the diff.

## Real models

If OPENAI_API_KEY is set, createProvider() uses an OpenAI-compatible chat adapter. Optional OPENAI_BASE_URL (default https://api.openai.com/v1) and OPENAI_MODEL (default gpt-4o-mini). Then researchLoop can compile arbitrary paper text.

Without a key, the deterministic provider stays active.

## Limitation

Papers can synthesize topology, prompts, memory, and routing. Arbitrary executable tools must go through the Capability gate below — scientist JSON is never trusted as code.

## Runtime improvement

Agents improve the society by emitting graph changes. The reconciler mounts, updates, or unmounts — agents do not patch `RuntimeDOM` internals.

Two gated paths sit beside topology mutation (`researchLoop` / Self-Refine):

1. **Harness (capability)** — propose a `kind: "capability"` node with a `source` ref → sandbox validation → eval (`runBenchmark`) → mount on success. Raw scientist JSON is never executed; only `module:<id>` refs (or exact fingerprints) against a pre-approved capability registry pass the sandbox. Failed eval leaves the live graph unchanged.

2. **Weights (adapter)** — an injectable `Trainer` runs out-of-process (tests use `FakeTrainer`) and returns an `AdapterArtifact`. A `kind: "adapter"` node carries `adapterRef` / `modelRef`; on a passing gate the target agent's `model` pointer updates (same binding as AgentNode.model → PhysicalNode.provider). Failed eval rejects the candidate; a later regression can `rollbackAdapter` (unmount + restore previous model).

`improveLoop` chooses topology, capability, or adapter (or `auto`). Real LoRA / Hugging Face Jobs stay behind the `Trainer` port — see `describeHfJobsExtension` in `src/trainer.ts`. No in-process GPU training.

## Layout

- src/ir.ts -- AgentGraph / Node / flatten / clone
- src/reconciler.ts -- mount / update / retain / unmount
- src/providers.ts -- deterministic + OpenAI-compatible
- src/runtime.ts -- walk the mounted graph, collect traces
- src/papers.ts -- paper to graph
- src/benchmarks.ts -- word-reverse and friends
- src/scientist.ts -- evolve the graph from traces
- src/capability.ts -- approved capability registry + sandbox gate
- src/trainer.ts -- Trainer port, FakeTrainer, adapter artifacts
- src/lifecycle.ts -- propose → sandbox → eval → mount | reject | rollback
- src/improve.ts -- improveLoop (topology | capability | adapter)
- src/demo.ts -- the loop, printed
- src/export-run.ts -- real run to public/run.json
- src/serve.ts -- static viz on :4173

## Scripts

package.json defines demo, test, build, export, and viz. After installing packages: test then demo.

    npm i && npm test && npm run demo
