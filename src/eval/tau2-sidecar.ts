/**
 * Stdio JSONL sidecar. Python HalfDuplexAgent sends one turn per line.
 * Logs go to stderr so stdout stays machine-readable.
 *
 * Two-clock I_weight: set_technique / i_loop / turn keep answering (fast clock).
 * i_weight_spawn returns immediately with a TrainJob; servingPaused is always
 * false. 0731 cannot take an adapter — default trainer is SurrogateTrainer.
 */
import { createProvider, DeterministicProvider, type Message, type ToolSpec } from "../providers.js";
import { runTau2Turn } from "./tau2-turn.js";
import { type AgentGraph } from "../ir.js";
import {
  FakeTrainer,
  SurrogateTrainer,
  type TrainJob,
  type TrainerKind,
  activeTrainJob,
  getTrainJob,
  localHeldOutScore,
  persistTrainJob,
  recordTrainJobGate,
  spawnTrainJob,
} from "../trainer.js";
import { type Tau2Obs, type Tau2Technique, type Tau2TurnResponse } from "./tau2-types.js";
import { gateWeightMount, type GraphDiffOp } from "./tau2-improve.js";
import { runSelfObs } from "./tau2-self-obs.js";
import { tau2Graph } from "./tau2-graph.js";
import { createInterface } from "node:readline";

type Incoming = {
  op?: string;
  id?: string;
  policy?: string;
  tools?: ToolSpec[];
  messages?: Message[];
  technique?: Tau2Technique;
  model?: string;
  graph?: AgentGraph;
  before?: number;
  after?: number;
  obs?: Tau2Obs | Tau2Obs[];
  traces?: unknown[];
  toolNames?: string[];
  rewards?: Array<number | null>;
  terminations?: string[];
  missedToolNames?: string[];
  trainer?: TrainerKind;
  jobId?: string;
  baseModel?: string;
};

type SidecarReply = Tau2TurnResponse & {
  technique?: Tau2Technique;
  graph?: AgentGraph;
  graphDiff?: GraphDiffOp[];
  servingPaused?: boolean;
  spawned?: boolean;
  done?: boolean;
  gate?: ReturnType<typeof gateWeightMount>;
  job?: TrainJob;
  path?: "self" | "fallback";
  action?: "wait" | "I_loop";
  rationale?: string;
  applied?: boolean;
};

function providerFor(model?: string): ReturnType<typeof createProvider> {
  if (!model || model === "deterministic" || model === "scripted") {
    return new DeterministicProvider();
  }
  return createProvider();
}

let currentTechnique: Tau2Technique = "one-shot";
let currentGraph: AgentGraph = tau2Graph("one-shot");

function write(obj: SidecarReply): void {
  process.stdout.write(`${JSON.stringify(obj)}\n`);
}

function setTechnique(next: Tau2Technique, graph?: AgentGraph): void {
  currentTechnique = next;
  currentGraph = graph ?? tau2Graph(next);
}

async function handle(line: string): Promise<void> {
  let req: Incoming;
  try {
    req = JSON.parse(line) as Incoming;
  } catch (err) {
    write({
      op: "error",
      id: "parse",
      error: err instanceof Error ? err.message : "invalid json",
    });
    return;
  }

  const id = req.id ?? req.op ?? "?";

  if (req.op === "ping") {
    write({
      op: "ok",
      id,
      content: "pong",
      technique: currentTechnique,
      servingPaused: false,
    });
    return;
  }

  if (req.op === "get_state") {
    write({
      op: "ok",
      id,
      technique: currentTechnique,
      graph: currentGraph,
      servingPaused: false,
    });
    return;
  }

  if (req.op === "set_technique") {
    const next = (req.technique ?? "one-shot") as Tau2Technique;
    setTechnique(next, req.graph);
    write({
      op: "ok",
      id,
      technique: currentTechnique,
      graph: currentGraph,
      servingPaused: false,
    });
    return;
  }

  if (req.op === "i_loop" || req.op === "self_obs") {
    const result = await runSelfObs({
      graph: req.graph ?? currentGraph,
      traces: req.traces as Array<{ nodeKey?: string; role?: string; output?: string }>,
      toolNames: req.toolNames,
      rewards: req.rewards,
      terminations: req.terminations,
      missedToolNames: req.missedToolNames,
      obs: req.obs,
      provider: providerFor(req.model),
      model: req.model,
    });
    if (result.applied) {
      currentTechnique = result.techniqueAfter;
      currentGraph = result.graphAfter;
    }
    write({
      op: "ok",
      id,
      technique: currentTechnique,
      graph: currentGraph,
      graphDiff: result.graphDiff,
      servingPaused: false,
      content: result.applied ? "applied" : result.action === "wait" ? "wait" : "exhausted",
      path: result.path,
      action: result.action,
      rationale: result.rationale,
      applied: result.applied,
    });
    return;
  }

  if (req.op === "i_weight_spawn") {
    const trainerKind: TrainerKind = req.trainer === "fake" ? "fake" : "surrogate";
    const trainer = trainerKind === "fake" ? new FakeTrainer() : new SurrogateTrainer();
    const job = spawnTrainJob({
      trainer,
      traces: req.traces ?? [],
      trainOpts: {
        baseModel: req.baseModel ?? req.model ?? "surrogate-theta",
        technique: trainerKind === "fake" ? "fake-lora" : "surrogate-prefix",
      },
      trainerKind,
      persist: true,
    });
    write({
      op: "ok",
      id,
      spawned: true,
      done: job.status === "done",
      servingPaused: false,
      job,
      content: job.id,
    });
    return;
  }

  if (req.op === "i_weight_status") {
    const job = (req.jobId ? getTrainJob(req.jobId) : undefined) ?? activeTrainJob();
    write({
      op: "ok",
      id,
      spawned: Boolean(job),
      done: job?.status === "done" || job?.status === "failed",
      servingPaused: false,
      job,
      content: job?.artifactPointer ?? job?.id,
    });
    return;
  }

  if (req.op === "i_weight_gate") {
    const job = (req.jobId ? getTrainJob(req.jobId) : undefined) ?? activeTrainJob();
    const before = Number(req.before ?? 0);
    const after =
      req.after !== undefined && req.after !== null
        ? Number(req.after)
        : job
          ? localHeldOutScore(job)
          : before;
    const gate = gateWeightMount(before, after);
    if (job) {
      const updated = recordTrainJobGate(job.id, gate, true);
      persistTrainJob(updated ?? job);
      write({
        op: "ok",
        id,
        gate,
        job: updated ?? getTrainJob(job.id) ?? job,
        servingPaused: false,
      });
      return;
    }
    write({
      op: "ok",
      id,
      gate,
      servingPaused: false,
    });
    return;
  }

  if (req.op !== "turn") {
    write({ op: "error", id, error: `unknown op ${req.op ?? ""}` });
    return;
  }

  try {
    const technique = req.technique ?? currentTechnique;
    const result = await runTau2Turn({
      policy: req.policy ?? "",
      tools: req.tools ?? [],
      messages: req.messages ?? [],
      technique,
      graph: req.graph ?? currentGraph,
      model: req.model,
      provider: req.model === "deterministic" ? new DeterministicProvider() : createProvider(),
    });
    write({
      op: "ok",
      id,
      content: result.content,
      tool_calls: result.toolCalls,
      traces: result.traces,
      technique,
      servingPaused: false,
    });
  } catch (err) {
    write({
      op: "error",
      id,
      error: err instanceof Error ? err.message : String(err),
    });
  }
}

async function main(): Promise<void> {
  process.stderr.write("vdom tau2 sidecar ready\n");
  const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of rl) {
    const trimmed = line.trim();
    if (trimmed.length === 0) continue;
    await handle(trimmed);
  }
}

main().catch((err) => {
  process.stderr.write(String(err) + "\n");
  process.exit(1);
});
