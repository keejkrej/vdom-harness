/**
 * Stdio JSONL sidecar. Python HalfDuplexAgent sends one turn per line.
 * Logs go to stderr so stdout stays machine-readable.
 *
 * Fast clock: set_technique / i_loop / turn keep answering.
 * I_sku (i_sku) is the slow arm: propose pro-0813, gate, rebind n.model
 * (catalog rebind, not I_weight-as-trainer, not fine-tuning).
 * servingPaused is always false. FakeTrainer / i_weight_* are stubs.
 */
import { createProvider, DeterministicProvider, resolveProvider, type Message, type ToolSpec } from "../providers.js";
import { filterGymToolCalls, runTau2Turn } from "./tau2-turn.js";
import { type AgentGraph } from "../ir.js";
import {
  selectServingGraph,
  servingTechnique,
  techniqueOfGraph,
  type ApplyScope,
} from "./tau2-improve.js";
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
import { applyISku, servingModelOfGraph } from "./tau2-weight.js";
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
  taskId?: string;
  taskIds?: string[];
};

type SidecarReply = Tau2TurnResponse & {
  technique?: Tau2Technique;
  graph?: AgentGraph;
  graphDiff?: GraphDiffOp[];
  servingPaused?: boolean;
  spawned?: boolean;
  done?: boolean;
  gate?: ReturnType<typeof gateWeightMount> | { arm: "I_sku"; action: "mount" | "reject"; before: number; after: number; reason: string };
  job?: TrainJob;
  path?: "self" | "fallback";
  action?: "wait" | "I_loop";
  rationale?: string;
  applied?: boolean;
  graphEdits?: unknown;
  applyScope?: ApplyScope;
  jumped?: boolean;
  servingModel?: string;
  trained?: false;
  catalog?: ReturnType<typeof applyISku>;
};

function providerFor(model?: string): ReturnType<typeof createProvider> {
  if (!model || model === "deterministic" || model === "scripted") {
    return new DeterministicProvider();
  }
  return createProvider();
}

let currentTechnique: Tau2Technique = "one-shot";
let currentGraph: AgentGraph = tau2Graph("one-shot");
let graphC0: AgentGraph = currentGraph;
let graphC1: AgentGraph | undefined;
let applyScope: ApplyScope | undefined;

function resetApplyScope(): void {
  applyScope = undefined;
  graphC1 = undefined;
  graphC0 = currentGraph;
}

function liveGraph(taskId?: string, reqGraph?: AgentGraph): AgentGraph {
  return selectServingGraph({
    taskId,
    reqGraph,
    currentGraph,
    graphBefore: graphC0,
    graphAfter: graphC1 ?? currentGraph,
    applyScope,
  });
}

function write(obj: SidecarReply): void {
  process.stdout.write(`${JSON.stringify(obj)}\n`);
}

function setTechnique(next: Tau2Technique, graph?: AgentGraph): void {
  currentTechnique = next;
  currentGraph = graph ?? tau2Graph(next);
  resetApplyScope();
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
      taskIds: req.taskIds,
      obs: req.obs,
      provider: providerFor(req.model),
      model: req.model,
    });
    applyScope = result.applyScope;
    graphC0 = result.graphBefore;
    graphC1 = result.applied ? result.graphAfter : undefined;
    if (result.applied) {
      if (
        result.applyScope &&
        (result.applyScope.waitKept.length > 0 || (result.applyScope.weighted ?? []).length > 0) &&
        result.applyScope.looped.length > 0
      ) {
        // Mixed batch: default C stays C0. Never a silent global mount.
        currentGraph = result.graphBefore;
        currentTechnique = techniqueOfGraph(result.graphBefore);
      } else {
        currentTechnique = result.techniqueAfter;
        currentGraph = result.graphAfter;
      }
    }
    write({
      op: "ok",
      id,
      technique: result.applied ? result.techniqueAfter : currentTechnique,
      graph: result.applied ? result.graphAfter : currentGraph,
      graphDiff: result.graphDiff,
      servingPaused: false,
      content: result.applied ? "applied" : result.action === "wait" ? "wait" : "exhausted",
      path: result.path,
      action: result.action,
      rationale: result.rationale,
      applied: result.applied,
      applyScope: result.applyScope,
    });
    return;
  }

  if (req.op === "i_sku" || req.op === "i_catalog" || req.op === "i_weight_catalog") {
    const before = Number(req.before ?? 0);
    const after = req.after !== undefined && req.after !== null ? Number(req.after) : before;
    const catalog = applyISku({
      graph: req.graph ?? currentGraph,
      before,
      after,
    });
    if (catalog.action === "mount") {
      currentGraph = catalog.graph;
      graphC0 = catalog.graph;
    }
    write({
      op: "ok",
      id,
      servingPaused: false,
      jumped: catalog.jumped,
      servingModel: catalog.servingModelId,
      catalog,
      trained: false,
      gate: {
        arm: "I_sku",
        action: catalog.action,
        before: catalog.before,
        after: catalog.after,
        reason: catalog.reason,
      },
      graph: catalog.graph,
      content: catalog.action,
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
    const live = liveGraph(req.taskId, req.graph);
    const technique = servingTechnique(live, {
      taskId: req.taskId,
      applyScope,
      reqTechnique: req.technique,
      currentTechnique,
    });
    const servingModel = servingModelOfGraph(live, req.model);
    const provider =
      servingModel === "deterministic" || servingModel === "scripted" || req.model === "deterministic"
        ? new DeterministicProvider(servingModel)
        : resolveProvider(servingModel);
    const result = await runTau2Turn({
      policy: req.policy ?? "",
      tools: req.tools ?? [],
      messages: req.messages ?? [],
      technique,
      graph: live,
      model: servingModel,
      provider,
    });
    if (
      req.taskId &&
      (applyScope?.waitKept.includes(req.taskId) || (applyScope?.weighted ?? []).includes(req.taskId))
    ) {
      graphC0 = result.graph;
    } else if (req.taskId && applyScope?.looped.includes(req.taskId)) {
      graphC1 = result.graph;
      currentGraph = result.graph;
      currentTechnique = techniqueOfGraph(result.graph);
    } else if (!(applyScope && (applyScope.waitKept.length > 0 || (applyScope.weighted ?? []).length > 0))) {
      currentGraph = result.graph;
      currentTechnique = techniqueOfGraph(result.graph);
    } else {
      graphC0 = result.graph;
    }
    for (const edit of result.graphEdits) {
      process.stderr.write(
        `graph-tool ${edit.tool} ${edit.rejected ? "rejected" : "applied"} ${edit.reason}\n`,
      );
    }
    const gymCalls = filterGymToolCalls(result.toolCalls);
    write({
      op: "ok",
      id,
      content: result.content,
      tool_calls: gymCalls,
      traces: result.traces,
      technique: currentTechnique,
      graph: currentGraph,
      graphEdits: result.graphEdits,
      servingPaused: false,
      servingModel: servingModelOfGraph(result.graph, servingModel),
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
