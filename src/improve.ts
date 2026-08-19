import { type AgentGraph, type Trace } from "./ir.js";
import { type Provider } from "./providers.js";
import { type Task, type BenchmarkResult, runBenchmark } from "./benchmarks.js";
import { RuntimeDOM, reconcile, formatOps, type ReconcileResult } from "./reconciler.js";
import { applySelfRefineMutation, evolveOnce } from "./scientist.js";
import { oneShotGraph } from "./papers.js";
import { proposeCapability } from "./capability.js";
import {
  type Trainer,
  FakeTrainer,
  spawnTrainJob,
  waitTrainJob,
  recordTrainJobGate,
  trainerKindOf,
} from "./trainer.js";
import {
  type GateDecision,
  gateCapability,
  gateAdapter,
  unmountAdapterOnFailure,
} from "./lifecycle.js";

export type ImproveMode = "topology" | "capability" | "adapter" | "auto";

export type ImproveIter = {
  mode: ImproveMode | "eval";
  graph: AgentGraph;
  benchmark: BenchmarkResult;
  reconcile?: ReconcileResult;
  gate?: GateDecision;
};

/**
 * Sibling to researchLoop: choose topology mutation, capability mount, or
 * adapter mount based on mode / traces. All non-topology paths go through
 * the eval gate before the live graph changes.
 *
 * Adapter / I_weight uses two clocks: serving keeps old θ (fast) while
 * `spawnTrainJob` trains from incomplete traces (slow). The live graph
 * changes only after `gateAdapter` / a held-out eval. servingPaused is never
 * set. FakeTrainer is the protocol unit test; SurrogateTrainer is an explicit
 * non-0731 θ (prompt-prefix stand-in), not a claimed API LoRA.
 */
export async function improveLoop(opts: {
  task: Task;
  provider: Provider;
  maxIters: number;
  mode?: ImproveMode;
  trainer?: Trainer;
  /** Pre-approved capability source (`module:<id>` or registered fingerprint). */
  capabilitySource?: string;
  capabilityKey?: string;
  /** Agent key whose model pointer updates after a successful adapter gate. */
  targetKey?: string;
  baseModel?: string;
  start?: AgentGraph;
  dom?: RuntimeDOM;
  threshold?: number;
}): Promise<ImproveIter[]> {
  const mode: ImproveMode = opts.mode ?? "auto";
  const threshold = opts.threshold ?? 1;
  const dom = opts.dom ?? new RuntimeDOM();
  const trainer = opts.trainer ?? new FakeTrainer();
  const targetKey = opts.targetKey ?? "solve";
  const capabilityKey = opts.capabilityKey ?? "harness-cap";

  let g = opts.start ?? oneShotGraph();
  const history: ImproveIter[] = [];
  let prev: AgentGraph | undefined;

  for (let i = 0; i < opts.maxIters; i++) {
    const rec = prev ? reconcile(prev, g) : undefined;
    if (rec) {
      console.log(`improve reconcile v${prev!.version} → v${g.version}`);
      console.log(formatOps(rec.ops));
    }

    const benchmark = await runBenchmark(g, opts.task, opts.provider, dom);
    history.push({ mode: "eval", graph: g, benchmark, reconcile: rec });
    if (benchmark.score >= threshold) break;

    prev = g;
    const chosen = pickMode(mode, i, benchmark.traces);

    if (chosen === "topology") {
      g = await evolveOnce(g, benchmark.traces, benchmark.score, opts.provider);
      // Fall back if scientist somehow returns same topology without children.
      if (g === prev || g.version === prev.version) {
        g = applySelfRefineMutation(prev);
      }
      history.push({
        mode: "topology",
        graph: g,
        benchmark,
      });
      continue;
    }

    if (chosen === "capability") {
      const source = opts.capabilitySource ?? "module:reverse-each-word";
      const proposal = proposeCapability({
        key: capabilityKey,
        source,
        objective: "Harness capability proposed from improveLoop",
      });
      const gate = await gateCapability({
        base: g,
        proposal,
        task: opts.task,
        provider: opts.provider,
        threshold,
        dom,
      });
      g = gate.graph;
      history.push({ mode: "capability", graph: g, benchmark, gate });
      if (gate.action === "reject") {
        // Try topology next iteration if auto; otherwise stop improving this path.
        if (mode !== "auto") break;
      }
      continue;
    }

    // adapter — slow clock. Fast clock is untouched (no serving pause).
    const job = spawnTrainJob({
      trainer,
      traces: benchmark.traces,
      trainOpts: {
        baseModel: opts.baseModel ?? "base",
        technique: trainerKindOf(trainer) === "surrogate" ? "surrogate-prefix" : "fake-lora",
      },
      persist: false,
    });
    const finished = await waitTrainJob(job.id);
    if (finished.status !== "done" || !finished.artifact) {
      history.push({ mode: "adapter", graph: g, benchmark });
      if (mode !== "auto") break;
      continue;
    }
    const artifact = finished.artifact;
    const gate = await gateAdapter({
      base: g,
      artifact,
      targetKey,
      task: opts.task,
      provider: opts.provider,
      threshold,
      dom,
    });
    recordTrainJobGate(
      finished.id,
      {
        arm: "I_weight",
        action: gate.action === "mount" ? "mount" : "reject",
        before: benchmark.score,
        after: gate.score,
        reason: gate.action === "mount" ? "held-out eval did not regress" : gate.reason,
      },
      false,
    );
    g = gate.graph;
    history.push({ mode: "adapter", graph: g, benchmark, gate });

    if (gate.action === "reject") {
      if (mode !== "auto") break;
      continue;
    }

    // Optional post-mount re-eval: if live score regresses, rollback.
    const live = await runBenchmark(g, opts.task, opts.provider, dom);
    if (live.score < threshold) {
      const rollback = await unmountAdapterOnFailure({
        current: g,
        adapterKey: "adapter",
        targetKey,
        previousModel: g.meta?.previousModel as AgentGraph["root"]["model"],
        task: opts.task,
        provider: opts.provider,
        dom,
      });
      g = rollback.graph;
      history.push({ mode: "adapter", graph: g, benchmark: live, gate: rollback });
    }
  }

  return history;
}

function pickMode(mode: ImproveMode, iter: number, _traces: Trace[]): ImproveMode {
  if (mode !== "auto") return mode;
  // auto: capability → adapter → topology
  if (iter === 0) return "capability";
  if (iter === 1) return "adapter";
  return "topology";
}
