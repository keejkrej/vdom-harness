import { type AgentGraph, type Trace, cloneGraph, node } from "./ir.js";
import { type Provider, type Message } from "./providers.js";
import { reconcile, formatOps, type ReconcileResult } from "./reconciler.js";
import { compilePaper, oneShotGraph } from "./papers.js";
import { runBenchmark, type Task, type BenchmarkResult } from "./benchmarks.js";

export function applySelfRefineMutation(g: AgentGraph): AgentGraph {
  const next = cloneGraph(g);
  next.version = g.version + 1;
  next.id = `${g.id}-self-refine`;
  next.meta = { ...(g.meta ?? {}), technique: "self-refine", mutated: true };
  const root = next.root;
  if (!root.children || root.children.length === 0) {
    root.children = [
      node({
        key: "critic",
        role: "critic",
        objective: "Critique the generator output",
        children: [
          node({
            key: "refine",
            role: "refine",
            objective: "Produce a corrected solution from the critique",
          }),
        ],
      }),
    ];
  }
  return next;
}

/** Second I_loop step: mount a validator that forbids the last failed action. */
export function applyValidatorMutation(g: AgentGraph): AgentGraph {
  const next = cloneGraph(g);
  next.version = g.version + 1;
  next.id = `${g.id}-validator`;
  next.meta = { ...(g.meta ?? {}), technique: "validator", mutated: true };
  const kids = next.root.children ?? [];
  if (!kids.some((c) => c.key === "validator")) {
    next.root = {
      ...next.root,
      children: [
        ...kids,
        node({
          key: "validator",
          role: "validator",
          objective: "Forbid the last failed action; transfer when policy requires a human",
        }),
      ],
    };
  }
  return next;
}

function tryParseGraph(raw: string): AgentGraph | undefined {
  const fence = raw.match(/```(?:json)?\s*([\s\S]*?)```/);
  const text = fence?.[1] ?? raw;
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start < 0 || end < 0) return undefined;
  try {
    const obj = JSON.parse(text.slice(start, end + 1)) as Partial<AgentGraph> & {
      root?: AgentGraph["root"];
    };
    if (obj.root && typeof obj.root.key === "string") {
      return {
        id: String(obj.id ?? "evolved"),
        version: Number(obj.version ?? 2),
        root: obj.root,
        meta: obj.meta,
      };
    }
  } catch {
    return undefined;
  }
  return undefined;
}

export async function evolveOnce(
  g: AgentGraph,
  traces: Trace[],
  score: number,
  provider: Provider,
): Promise<AgentGraph> {
  if (score >= 1) return g;

  const msgs: Message[] = [
    {
      role: "system",
      content: "Role: scientist\nObjective: Propose an improved AgentGraph as JSON",
    },
    {
      role: "user",
      content: [
        `The current graph scored ${score}. Improve a failing one-shot graph.`,
        `Traces:\n${traces.map((t) => `[${t.role}] ${t.output}`).join("\n")}`,
        `Current graph:\n${JSON.stringify(g, null, 2)}`,
        "Return a JSON AgentGraph that adds critic + refiner children (Self-Refine topology).",
      ].join("\n\n"),
    },
  ];

  const raw = await provider.complete(msgs, { role: "scientist" });
  return tryParseGraph(raw) ?? applySelfRefineMutation(g);
}

export type ResearchIter = {
  graph: AgentGraph;
  benchmark: BenchmarkResult;
  reconcile?: ReconcileResult;
};

export async function researchLoop(opts: {
  papers?: string[];
  task: Task;
  provider: Provider;
  maxIters: number;
}): Promise<ResearchIter[]> {
  const { papers, task, provider, maxIters } = opts;
  let g = papers?.[0] ? compilePaper(papers[0]) : oneShotGraph();
  const history: ResearchIter[] = [];
  let prev: AgentGraph | undefined;

  for (let i = 0; i < maxIters; i++) {
    const rec = prev ? reconcile(prev, g) : undefined;
    if (rec) {
      console.log(`reconcile v${prev!.version} → v${g.version}`);
      console.log(formatOps(rec.ops));
    }
    const benchmark = await runBenchmark(g, task, provider);
    history.push({ graph: g, benchmark, reconcile: rec });
    if (benchmark.score >= 1) break;
    prev = g;
    g = await evolveOnce(g, benchmark.traces, benchmark.score, provider);
  }

  return history;
}
