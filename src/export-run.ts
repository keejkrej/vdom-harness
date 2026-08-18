import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { type AgentGraph, flatten } from "./ir.js";
import { DeterministicProvider } from "./providers.js";
import {
  compilePaper,
  SELF_REFINE_ABSTRACT,
  REFLEXION_ABSTRACT,
  oneShotGraph,
} from "./papers.js";
import { WORD_REVERSE, runBenchmark } from "./benchmarks.js";
import { applySelfRefineMutation } from "./scientist.js";
import { reconcile, type ReconcileOp } from "./reconciler.js";

const MARK: Record<ReconcileOp["op"], string> = {
  mount: "+",
  update: "~",
  retain: "=",
  unmount: "-",
};

export type GraphNodeSnap = {
  key: string;
  role: string;
  parentKey: string | null;
  kind: string;
};

export type OpSnap = {
  op: ReconcileOp["op"];
  mark: string;
  key: string;
  role: string;
  parentKey: string | null;
  kind: string;
};

export type RunEvent = {
  t: number;
  type:
    | "title"
    | "paper"
    | "compile"
    | "mount"
    | "trace"
    | "score"
    | "scientist"
    | "reconcile"
    | "compare"
    | "done";
  [key: string]: unknown;
};

function snap(g: AgentGraph): GraphNodeSnap[] {
  return flatten(g).map(({ node, parentKey }) => ({
    key: node.key,
    role: node.role,
    parentKey: parentKey ?? null,
    kind: node.kind ?? "agent",
  }));
}

function snapOps(ops: ReconcileOp[]): OpSnap[] {
  return ops.map((o) => ({
    op: o.op,
    mark: MARK[o.op],
    key: o.node.key,
    role: o.node.role,
    parentKey: o.parentKey ?? null,
    kind: o.node.kind ?? "agent",
  }));
}

class Timeline {
  t = 0;
  events: RunEvent[] = [];

  emit(dt: number, event: Omit<RunEvent, "t">): void {
    this.t += dt;
    this.events.push({ t: this.t, ...event } as RunEvent);
  }

  hold(dt: number): void {
    this.t += dt;
  }
}

async function main(): Promise<void> {
  const provider = new DeterministicProvider();

  const selfRefine = compilePaper(SELF_REFINE_ABSTRACT);
  const sr = await runBenchmark(selfRefine, WORD_REVERSE, provider);

  const reflexion = compilePaper(REFLEXION_ABSTRACT);
  const rx = await runBenchmark(reflexion, WORD_REVERSE, provider);

  const v1 = oneShotGraph();
  const v1b = await runBenchmark(v1, WORD_REVERSE, provider);

  const v2 = applySelfRefineMutation(v1);
  const v2b = await runBenchmark(v2, WORD_REVERSE, provider);

  const recEmptySr = reconcile(undefined, selfRefine);
  const recSrRx = reconcile(selfRefine, reflexion);
  const recRxV1 = reconcile(reflexion, v1);
  const recV1V2 = reconcile(v1, v2);

  const tl = new Timeline();

  tl.emit(0, {
    type: "title",
    text: "Pi customizes an agent. DSH composes a runtime. vdom reconciles a society.",
  });

  tl.emit(1400, {
    type: "paper",
    name: "Self-Refine",
    technique: "self-refine",
    authors: "Madaan et al.",
    abstract: SELF_REFINE_ABSTRACT,
    phase: "self-refine",
  });
  tl.emit(1600, {
    type: "compile",
    name: "Self-Refine",
    graphId: selfRefine.id,
    nodes: snap(selfRefine),
    phase: "self-refine",
  });
  tl.emit(400, {
    type: "mount",
    graphId: selfRefine.id,
    version: selfRefine.version,
    label: "Self-Refine",
    phase: "self-refine",
    nodes: snap(selfRefine),
    ops: snapOps(recEmptySr.ops),
  });
  for (const tr of sr.traces) {
    tl.emit(800, {
      type: "trace",
      role: tr.role,
      input: tr.input,
      output: tr.output,
      nodeKey: tr.nodeKey,
      phase: "self-refine",
    });
  }
  tl.emit(600, {
    type: "score",
    value: sr.score,
    label: "Self-Refine",
    nodes: sr.nodes,
    phase: "self-refine",
  });

  tl.emit(1200, {
    type: "paper",
    name: "Reflexion",
    technique: "reflexion",
    authors: "Shinn et al.",
    abstract: REFLEXION_ABSTRACT,
    phase: "reflexion",
  });
  tl.emit(900, {
    type: "compile",
    name: "Reflexion",
    graphId: reflexion.id,
    nodes: snap(reflexion),
    phase: "reflexion",
  });
  tl.emit(300, {
    type: "reconcile",
    from: "self-refine",
    to: "reflexion",
    graphId: reflexion.id,
    version: reflexion.version,
    label: "Reflexion",
    phase: "reflexion",
    nodes: snap(reflexion),
    ops: snapOps(recSrRx.ops),
  });
  for (const tr of rx.traces) {
    tl.emit(350, {
      type: "trace",
      role: tr.role,
      input: tr.input,
      output: tr.output,
      nodeKey: tr.nodeKey,
      phase: "reflexion",
    });
  }
  tl.emit(500, {
    type: "score",
    value: rx.score,
    label: "Reflexion",
    nodes: rx.nodes,
    phase: "reflexion",
  });

  tl.emit(1100, {
    type: "paper",
    name: "Runtime evolution",
    technique: "one-shot",
    authors: "",
    abstract: "A one-shot solve node. No critic. No refine. The society is a single agent.",
    phase: "v1",
  });
  tl.emit(500, {
    type: "compile",
    name: "one-shot",
    graphId: v1.id,
    nodes: snap(v1),
    phase: "v1",
  });
  tl.emit(300, {
    type: "reconcile",
    from: "reflexion",
    to: "oneshot",
    graphId: v1.id,
    version: v1.version,
    label: "v1 one-shot",
    phase: "v1",
    nodes: snap(v1),
    ops: snapOps(recRxV1.ops),
  });
  for (const tr of v1b.traces) {
    tl.emit(700, {
      type: "trace",
      role: tr.role,
      input: tr.input,
      output: tr.output,
      nodeKey: tr.nodeKey,
      phase: "v1",
    });
  }
  tl.emit(700, {
    type: "score",
    value: v1b.score,
    label: "v1 one-shot",
    nodes: v1b.nodes,
    phase: "v1",
  });

  tl.emit(1100, {
    type: "scientist",
    text: "emit replacement graph",
    from: v1.id,
    to: v2.id,
    phase: "scientist",
  });

  tl.emit(1400, {
    type: "reconcile",
    from: "v1",
    to: "v2",
    graphId: v2.id,
    version: v2.version,
    label: "v2 one-shot+critic",
    phase: "v2",
    nodes: snap(v2),
    ops: snapOps(recV1V2.ops),
  });
  for (const tr of v2b.traces) {
    tl.emit(800, {
      type: "trace",
      role: tr.role,
      input: tr.input,
      output: tr.output,
      nodeKey: tr.nodeKey,
      phase: "v2",
    });
  }
  tl.emit(700, {
    type: "score",
    value: v2b.score,
    label: "v2 one-shot+critic",
    nodes: v2b.nodes,
    phase: "v2",
  });

  tl.emit(1400, { type: "compare", phase: "compare" });
  tl.hold(2800);
  tl.emit(0, { type: "done", phase: "done" });

  const doc = {
    generatedAt: new Date().toISOString(),
    provider: provider.name,
    task: {
      id: WORD_REVERSE.id,
      input: WORD_REVERSE.input,
      expected: WORD_REVERSE.expected,
    },
    duration: tl.t,
    scores: {
      selfRefine: sr.score,
      reflexion: rx.score,
      v1: v1b.score,
      v2: v2b.score,
    },
    events: tl.events,
  };

  const here = dirname(fileURLToPath(import.meta.url));
  const out = join(here, "..", "public", "run.json");
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, JSON.stringify(doc, null, 2) + "\n");

  const counts: Record<string, number> = {};
  for (const e of tl.events) {
    counts[e.type] = (counts[e.type] ?? 0) + 1;
  }
  console.log(`wrote ${out}`);
  console.log(`duration ${tl.t}ms`);
  console.log(`events ${tl.events.length} ${JSON.stringify(counts)}`);
  console.log(
    `scores self-refine=${sr.score.toFixed(2)} reflexion=${rx.score.toFixed(2)} v1=${v1b.score.toFixed(2)} v2=${v2b.score.toFixed(2)}`,
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
