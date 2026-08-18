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

Papers can synthesize topology, prompts, memory, and routing. They cannot yet emit arbitrary executable tools. The next primitive is a Capability node: generate source -> sandbox -> tests -> mount.

## Layout

- src/ir.ts -- AgentGraph / Node / flatten / clone
- src/reconciler.ts -- mount / update / retain / unmount
- src/providers.ts -- deterministic + OpenAI-compatible
- src/runtime.ts -- walk the mounted graph, collect traces
- src/papers.ts -- paper to graph
- src/benchmarks.ts -- word-reverse and friends
- src/scientist.ts -- evolve the graph from traces
- src/demo.ts -- the loop, printed
- src/export-run.ts -- real run to public/run.json
- src/serve.ts -- static viz on :4173

## Scripts

package.json defines demo, test, build, export, and viz. After installing packages: test then demo.

    npm i && npm test && npm run demo
