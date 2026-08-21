/**
 * Stdio JSONL sidecar. Python HalfDuplexAgent sends one turn per line.
 * Logs go to stderr so stdout stays machine-readable.
 *
 * Fast clock: set_technique / i_loop / turn keep answering.
 * I_sku (i_sku) is the slow arm: propose pro-0813, gate, write S
 * onto existing HybridState objects (X.S). servingByTask is a derived
 * cache from X.S, not the lookup. servingPaused is always false.
 * FakeTrainer / i_weight_* are stubs.
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
import { type HybridState, type Tau2Obs, type Tau2Technique, type Tau2TurnResponse } from "./tau2-types.js";
import { gateWeightMount, type GraphDiffOp } from "./tau2-improve.js";
import {
  applyISku,
  catalogPointer,
  SERVING_MODEL,
  type CatalogPointer,
} from "./tau2-weight.js";
import {
  createHybridStore,
  derivedServingByTask,
  hybridRecord,
  hybridState,
  installHybridState,
  servingFromHybrid,
  writeSOnStore,
  type HybridStore,
} from "./tau2-hybrid-state.js";
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
  gate?: ReturnType<typeof gateWeightMount> | { arm: "I_sku"; action: "mount" | "reject"; before: number; after: number | null; reason: string };
  job?: TrainJob;
  path?: "self" | "fallback";
  action?: "wait" | "I_loop";
  rationale?: string;
  applied?: boolean;
  graphEdits?: unknown;
  applyScope?: ApplyScope;
  jumped?: boolean;
  servingModel?: string;
  serving?: CatalogPointer;
  servingSku?: CatalogPointer;
  servingByTask?: Record<string, CatalogPointer>;
  servingByTaskIs?: "derived cache from X.S, not the lookup";
  X?: Record<string, HybridState>;
  sourceOfTruth?: "X_n.S";
  dumpIsNot?: "ping / get_state S0";
  notAssembledFromServingByTask?: true;
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
let graphSku: AgentGraph | undefined;
let applyScope: ApplyScope | undefined;
/** Paper S0. New batches / resets start here. Not a leftover process sku. */
let servingS0: CatalogPointer = catalogPointer(SERVING_MODEL);
/**
 * Live HybridState objects. I_sku writes X.S on these.
 * servingByTask is derived from X.S — this Map is not the lookup.
 */
const hybridByTask: HybridStore = createHybridStore();

function resetServingState(): void {
  servingS0 = catalogPointer(SERVING_MODEL);
  hybridByTask.clear();
}

function resetApplyScope(): void {
  applyScope = undefined;
  graphC1 = undefined;
  graphSku = undefined;
  graphC0 = currentGraph;
  resetServingState();
}

function servingForRequest(taskId?: string): CatalogPointer {
  return servingFromHybrid(hybridByTask, taskId, servingS0);
}

/** Derived cache from X.S. Not the source of truth. */
function servingByTaskRecord(): Record<string, CatalogPointer> {
  return derivedServingByTask(hybridByTask);
}

function hybridReply(): {
  X: Record<string, HybridState>;
  servingByTask: Record<string, CatalogPointer>;
  servingByTaskIs: "derived cache from X.S, not the lookup";
} {
  return {
    X: hybridRecord(hybridByTask),
    servingByTask: servingByTaskRecord(),
    servingByTaskIs: "derived cache from X.S, not the lookup",
  };
}

/** Per-task S pick from X.S. Omit task / omit SKU → S0, not last I_sku mount. */
function servingModelForRequest(taskId?: string, reqModel?: string): string {
  if (taskId) return servingForRequest(taskId).sku;
  return servingS0.sku || reqModel || SERVING_MODEL;
}

function installHybridsFromObs(obsList: Tau2Obs[]): void {
  hybridByTask.clear();
  for (const o of obsList) {
    if (!o.taskId) continue;
    installHybridState(
      hybridByTask,
      o.taskId,
      hybridState({
        E: o,
        C: liveGraph(o.taskId),
        S: catalogPointer(servingS0.sku),
      }),
    );
  }
}

function liveGraph(taskId?: string, reqGraph?: AgentGraph): AgentGraph {
  return selectServingGraph({
    taskId,
    reqGraph,
    currentGraph,
    graphBefore: graphC0,
    graphAfter: graphC1 ?? currentGraph,
    graphSku,
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
      serving: servingS0,
      servingSku: servingS0,
      servingModel: servingS0.sku,
      ...hybridReply(),
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
      serving: servingS0,
      servingSku: servingS0,
      servingModel: servingS0.sku,
      ...hybridReply(),
    });
    return;
  }

  if (req.op === "dump_hybrid") {
    write({
      op: "ok",
      id,
      content: "X_n.S dump from HybridState objects; not ping / get_state S0",
      servingPaused: false,
      sourceOfTruth: "X_n.S",
      dumpIsNot: "ping / get_state S0",
      notAssembledFromServingByTask: true,
      ...hybridReply(),
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
      serving: servingS0,
      servingSku: servingS0,
      servingModel: servingS0.sku,
      ...hybridReply(),
    });
    return;
  }

  if (req.op === "i_loop" || req.op === "self_obs") {
    // New I_loop batch starts at S0. I_loop never writes S; do not inherit last mount.
    resetServingState();
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
    const obsList = Array.isArray(req.obs) ? req.obs : req.obs ? [req.obs] : [];
    installHybridsFromObs(obsList);
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
      serving: servingS0,
      servingSku: servingS0,
      servingModel: servingS0.sku,
      ...hybridReply(),
    });
    return;
  }

  if (req.op === "i_sku" || req.op === "i_catalog" || req.op === "i_weight_catalog") {
    const before = Number(req.before ?? 0);
    const after =
      req.after !== undefined && req.after !== null ? Number(req.after) : undefined;
    const catalog = applyISku({
      graph: req.graph ?? currentGraph,
      before,
      after,
      serving: servingS0,
    });
    if (catalog.action === "mount") {
      // C topology stays C0. Write S onto existing HybridState objects only.
      graphSku = catalog.graph;
      const weighted = applyScope?.weighted ?? [];
      for (const taskId of weighted) {
        writeSOnStore(hybridByTask, taskId, catalog.serving);
      }
    }
    write({
      op: "ok",
      id,
      servingPaused: false,
      jumped: catalog.jumped,
      servingModel: catalog.servingModelId,
      serving: catalog.serving,
      servingSku: catalog.serving,
      ...hybridReply(),
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
    const Xn = req.taskId ? hybridByTask.get(req.taskId) : undefined;
    const servingModel = Xn ? Xn.S.sku : servingModelForRequest(req.taskId, req.model);
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
      servingModel,
      serving: Xn?.S ?? servingForRequest(req.taskId),
      servingSku: Xn?.S ?? servingForRequest(req.taskId),
      ...hybridReply(),
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
