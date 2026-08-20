import { flatten, type AgentGraph } from "../ir.js";
import {
  applyPolicyChecklistMutation,
  applySelfRefineMutation,
  applyValidatorMutation,
} from "../scientist.js";
import { reconcile, type ReconcileOp } from "../reconciler.js";
import { tau2Graph } from "./tau2-graph.js";
import { type ApplyScope, type Tau2Obs, type Tau2Technique } from "./tau2-types.js";
import { AIRLINE_POLICY_CHECKLIST, isPolicyWriteTool, shouldRecommendPolicy } from "./tau2-policy.js";

export type { ApplyScope } from "./tau2-types.js";

/** Official incomplete / hung arm is I_catalog (catalog-rebind). I_weight is a trainer stub only. */
export type InterventionArm = "I_loop" | "I_catalog" | "wait";

export function isCatalogArm(arm: string | undefined): boolean {
  return arm === "I_catalog" || arm === "I_weight";
}

export const REFUSED_GLOBAL_ILOOP = "refused global I_loop: wait-hit in batch";

export type GraphDiffOp = {
  op: ReconcileOp["op"];
  key: string;
  parentKey?: string;
};

export type ILoopResult = {
  arm: "I_loop";
  applied: boolean;
  techniqueBefore: Tau2Technique;
  techniqueAfter: Tau2Technique;
  graphBefore: AgentGraph;
  graphAfter: AgentGraph;
  graphDiff: GraphDiffOp[];
  /** self = LLM Obs patch; fallback = host technique ladder / canned checklist. */
  path?: "self" | "fallback";
  action?: "wait" | "I_loop";
  rationale?: string;
  /** Per-task C: wait+hit keep graphBefore (C0); miss / I_loop get graphAfter (C1). */
  applyScope?: ApplyScope;
};

/** FakeTrainer / TrainJob eval gate. Not the paper slow arm and not a θ jump. */
export type WeightGateDecision = {
  arm: "I_weight";
  action: "mount" | "reject";
  before: number;
  after: number;
  reason: string;
};

export function graphHas(g: AgentGraph, key: string): boolean {
  return flatten(g).some((f) => f.node.key === key);
}

function asObsList(obs?: Tau2Obs | Tau2Obs[] | null): Tau2Obs[] {
  if (!obs) return [];
  return Array.isArray(obs) ? obs : [obs];
}

const HARD_INCOMPLETE_TERM = /timeout|hung|crash|error/;

/** Env / write tool names in lastActions (text:… is not a write). */
export function calledWriteTools(obs: Tau2Obs): boolean {
  return (obs.lastActions ?? []).some((a) => Boolean(a) && !a.startsWith("text:"));
}

/** Completed-miss topology / policy attractor (not incompleteness). */
export function hasLoopAttractor(obs: Tau2Obs): boolean {
  if (obs.inventedPolicy || obs.refusedCancel) return true;
  if (shouldRecommendPolicy(obs)) return true;
  if ((obs.missedActions ?? []).length > 0) return true;
  return (obs.lastActions ?? []).some((a) => isPolicyWriteTool(a));
}

/**
 * Episode did not complete: hang, timeout, crash, transfer-without-writes,
 * or no-write without a diagnosed missed-tool / policy attractor.
 */
export function isIncompleteEpisode(obs: Tau2Obs): boolean {
  if (obs.nSuccessProxy === 1 && !obs.hung) return false;
  if (obs.hung) return true;
  const term = (obs.termination ?? "").toLowerCase();
  if (term && HARD_INCOMPLETE_TERM.test(term)) return true;
  if (hasLoopAttractor(obs)) return false;
  if (term.includes("transfer")) return true;
  if (term.includes("user_stop")) return false;
  if (!calledWriteTools(obs)) return true;
  return false;
}

/** Wait + official hit. Hung is never a hit. */
export function isWaitHit(obs: Tau2Obs): boolean {
  if (obs.hung) return false;
  const hit = obs.nSuccessProxy === 1;
  const arm = obs.arm ?? (hit ? "wait" : "I_loop");
  return hit && arm === "wait";
}

export function episodeTaskId(obs: Tau2Obs, index: number, taskIds?: string[]): string {
  if (obs.taskId && obs.taskId.length > 0) return obs.taskId;
  const fromList = taskIds?.[index];
  if (fromList && fromList.length > 0) return fromList;
  return String(index);
}

/**
 * Split a batch so a global C mutation cannot land on wait+hit or I_catalog
 * (incomplete) episodes. Optional patchTaskIds scopes C1 further (wait-hit
 * and incomplete are never included).
 */
export function computeApplyScope(
  obs?: Tau2Obs | Tau2Obs[] | null,
  opts?: { patchTaskIds?: string[]; taskIds?: string[] },
): ApplyScope {
  const list = asObsList(obs);
  const hits = new Set<string>();
  const order: string[] = [];
  for (let i = 0; i < list.length; i++) {
    const id = episodeTaskId(list[i]!, i, opts?.taskIds);
    if (!order.includes(id)) order.push(id);
    if (isWaitHit(list[i]!)) hits.add(id);
  }
  const scoped = (opts?.patchTaskIds ?? []).filter((id) => id.length > 0);
  const waitKept: string[] = [];
  const looped: string[] = [];
  const weighted: string[] = [];
  const incomplete = new Set<string>();
  for (let i = 0; i < list.length; i++) {
    const o = list[i]!;
    const id = episodeTaskId(o, i, opts?.taskIds);
    if (isIncompleteEpisode(o) && !hits.has(id)) incomplete.add(id);
  }
  for (const id of order) {
    if (hits.has(id)) {
      waitKept.push(id);
      continue;
    }
    if (incomplete.has(id)) {
      weighted.push(id);
      continue;
    }
    if (scoped.length > 0 && !scoped.includes(id)) {
      waitKept.push(id);
      continue;
    }
    looped.push(id);
  }
  return { waitKept, looped, weighted };
}

export function graphsByApplyScope(
  graphBefore: AgentGraph,
  graphAfter: AgentGraph,
  scope: ApplyScope,
): Record<string, AgentGraph> {
  const out: Record<string, AgentGraph> = {};
  for (const id of scope.waitKept) out[id] = graphBefore;
  for (const id of scope.weighted ?? []) out[id] = graphBefore;
  for (const id of scope.looped) out[id] = graphAfter;
  return out;
}

export function graphForScopedTask(
  graphBefore: AgentGraph,
  graphAfter: AgentGraph,
  scope: ApplyScope,
  taskId: string,
): AgentGraph {
  if (scope.waitKept.includes(taskId) || (scope.weighted ?? []).includes(taskId)) return graphBefore;
  if (scope.looped.includes(taskId)) return graphAfter;
  if (scope.waitKept.length > 0 || (scope.weighted ?? []).length > 0) return graphBefore;
  return graphAfter;
}

/** Serving pick: never silently hand C1 to a wait+hit (or an unknown id in a mixed batch). */
export function selectServingGraph(opts: {
  taskId?: string;
  reqGraph?: AgentGraph;
  currentGraph: AgentGraph;
  graphBefore: AgentGraph;
  graphAfter?: AgentGraph;
  applyScope?: ApplyScope;
}): AgentGraph {
  if (opts.reqGraph) return opts.reqGraph;
  const scope = opts.applyScope;
  if (opts.taskId && scope) {
    if (scope.waitKept.includes(opts.taskId) || (scope.weighted ?? []).includes(opts.taskId)) {
      return opts.graphBefore;
    }
    if (scope.looped.includes(opts.taskId) && opts.graphAfter) return opts.graphAfter;
  }
  if (scope && (scope.waitKept.length > 0 || (scope.weighted ?? []).length > 0)) return opts.graphBefore;
  return opts.currentGraph;
}

export function servingTechnique(
  live: AgentGraph,
  opts: {
    taskId?: string;
    applyScope?: ApplyScope;
    reqTechnique?: Tau2Technique;
    currentTechnique: Tau2Technique;
  },
): Tau2Technique {
  const scope = opts.applyScope;
  const scoped =
    Boolean(opts.taskId) &&
    Boolean(scope) &&
    (scope!.waitKept.includes(opts.taskId!) ||
      scope!.looped.includes(opts.taskId!) ||
      (scope!.weighted ?? []).includes(opts.taskId!));
  if (
    scoped ||
    live.meta?.selfEdit === true ||
    (scope && (scope.waitKept.length > 0 || (scope.weighted ?? []).length > 0) && !opts.taskId)
  ) {
    return techniqueOfGraph(live);
  }
  return opts.reqTechnique ?? opts.currentTechnique;
}

export function loopExhausted(g: AgentGraph, obs?: Tau2Obs | Tau2Obs[]): boolean {
  if (graphHas(g, "policy-checklist")) return true;
  if (obsNeedsPolicy(obs)) return false;
  return graphHas(g, "critic") && graphHas(g, "validator");
}

export function obsNeedsPolicy(obs?: Tau2Obs | Tau2Obs[] | null): boolean {
  if (!obs) return false;
  const list = asObsList(obs);
  return list.some((o) => !isWaitHit(o) && !isIncompleteEpisode(o) && shouldRecommendPolicy(o));
}

/**
 * Per-episode arm from traces.
 * Hit → wait. Hung / timeout / transfer / crash / no-write → I_catalog
 * (gated catalog rebind; catalog swap, not post-training).
 * Completed miss with invented-policy / refused-cancel / extra-write (or
 * generic topology) attractor → I_loop. loopExhausted is a fallback to I_catalog.
 */
export function recommendIntervention(
  obs: Tau2Obs,
  opts?: { loopExhausted?: boolean },
): InterventionArm {
  if (obs.nSuccessProxy === 1 && !obs.hung) return "wait";
  if (isIncompleteEpisode(obs)) return "I_catalog";
  if (opts?.loopExhausted) return "I_catalog";
  return "I_loop";
}

/** Slice-level arm: I_catalog if ANY episode is incomplete; wait only if every episode hit. */
export function recommendSliceIntervention(
  obsList: Tau2Obs[],
  opts?: { loopExhausted?: boolean },
): InterventionArm {
  if (obsList.length > 0 && obsList.every((o) => o.nSuccessProxy === 1 && !o.hung)) {
    return "wait";
  }
  if (obsList.some((o) => isIncompleteEpisode(o))) return "I_catalog";
  if (opts?.loopExhausted) return "I_catalog";
  return "I_loop";
}

export function summarizeObs(obsList: Tau2Obs[]): Tau2Obs {
  if (obsList.length === 0) {
    return {
      nSteps: 0,
      nSuccessProxy: 0,
      lastActions: [],
      channels: [],
      critique: "no traces",
      toolFailures: 0,
      repeatActions: 0,
    };
  }
  const last = obsList[obsList.length - 1]!;
  const toolFailures = obsList.reduce((n, o) => n + o.toolFailures, 0);
  const repeatActions = obsList.reduce((n, o) => n + o.repeatActions, 0);
  const hits = obsList.reduce((n, o) => n + o.nSuccessProxy, 0);
  return {
    nSteps: obsList.reduce((n, o) => n + o.nSteps, 0),
    nSuccessProxy: hits,
    lastActions: last.lastActions,
    channels: [...new Set(obsList.flatMap((o) => o.channels))],
    critique:
      hits === obsList.length && !obsList.some((o) => o.hung)
        ? "path measure hits S; wait"
        : obsList.some((o) => isIncompleteEpisode(o))
          ? "episode incomplete (hung / timeout / no-write); I_catalog (catalog swap, not post-training)"
          : obsNeedsPolicy(obsList)
            ? "user asked cancel/update and agent refused or never called the tool; I_loop policy-checklist"
            : toolFailures > 0
              ? "tool failures in trajectory; inspect env channel"
              : repeatActions > 0
                ? "repeat actions; loop mutation or wait"
                : "episode unfinished or miss; inspect cascade / tools",
    toolFailures,
    repeatActions,
    missedActions: obsList.flatMap((o) => o.missedActions ?? []),
    refusedCancel: obsList.some((o) => o.refusedCancel),
    inventedPolicy: obsList.some((o) => o.inventedPolicy),
    hung: obsList.some((o) => o.hung),
    techniqueRecommendation: obsNeedsPolicy(obsList) ? "policy-checklist" : last.techniqueRecommendation,
  };
}

export function diffOps(ops: ReconcileOp[]): GraphDiffOp[] {
  return ops.map((o) => ({
    op: o.op,
    key: o.node.key,
    ...("parentKey" in o && o.parentKey ? { parentKey: o.parentKey } : {}),
  }));
}

export function techniqueOfGraph(g: AgentGraph): Tau2Technique {
  const t = g.meta?.technique;
  if (
    t === "self-refine" ||
    t === "validator" ||
    t === "reflexion" ||
    t === "one-shot" ||
    t === "policy-checklist"
  ) {
    return t;
  }
  return "one-shot";
}

/**
 * Next I_loop step on the live graph. Reconcile is deterministic; serving does not restart.
 *
 * Default ladder (mock / no typed miss): one-shot → Self-Refine → validator → exhausted.
 * If Obs says refusedCancel / inventedPolicy / missed cancel_reservation or
 * update_reservation_*, emit the policy-checklist graph instead of another
 * generic self-refine / validator step.
 */
export function applyILoop(start?: AgentGraph, obs?: Tau2Obs | Tau2Obs[]): ILoopResult {
  const graphBefore = start ?? tau2Graph("one-shot");
  const techniqueBefore = techniqueOfGraph(graphBefore);
  const applyScope = computeApplyScope(obs);
  let graphAfter: AgentGraph;
  let techniqueAfter: Tau2Technique;
  let applied = true;
  let rationale: string;

  if (applyScope.looped.length === 0 && asObsList(obs).length > 0) {
    graphAfter = graphBefore;
    techniqueAfter = techniqueBefore;
    applied = false;
    rationale =
      applyScope.waitKept.length > 0 ? REFUSED_GLOBAL_ILOOP : "incomplete licenses I_catalog; no I_loop";
  } else if (obsNeedsPolicy(obs) && !graphHas(graphBefore, "policy-checklist")) {
    graphAfter = applyPolicyChecklistMutation(graphBefore, AIRLINE_POLICY_CHECKLIST);
    techniqueAfter = "policy-checklist";
    rationale = "host I_loop ladder";
  } else if (obsNeedsPolicy(obs)) {
    graphAfter = graphBefore;
    techniqueAfter = techniqueBefore;
    applied = false;
    rationale = "host I_loop exhausted";
  } else if (!graphHas(graphBefore, "critic")) {
    graphAfter = applySelfRefineMutation(graphBefore);
    techniqueAfter = "self-refine";
    rationale = "host I_loop ladder";
  } else if (!graphHas(graphBefore, "validator")) {
    graphAfter = applyValidatorMutation(graphBefore);
    techniqueAfter = "validator";
    rationale = "host I_loop ladder";
  } else {
    graphAfter = graphBefore;
    techniqueAfter = techniqueBefore;
    applied = false;
    rationale = "host I_loop exhausted";
  }

  if (applied) {
    graphAfter.meta = {
      ...(graphAfter.meta ?? {}),
      technique: techniqueAfter,
      intervention: "I_loop",
    };
  }
  const rec = reconcile(graphBefore, graphAfter);
  return {
    arm: "I_loop",
    applied,
    techniqueBefore,
    techniqueAfter,
    graphBefore,
    graphAfter,
    graphDiff: diffOps(rec.ops),
    path: "fallback",
    action: applied
      ? "I_loop"
      : applyScope.looped.length === 0 && applyScope.waitKept.length > 0
        ? "wait"
        : "I_loop",
    rationale,
    applyScope,
  };
}

/**
 * FakeTrainer / TrainJob eval gate (protocol stub). Not the paper slow arm.
 * The official incomplete actuator is applyICatalog: propose pro-0813, gate,
 * rebind n.model (catalog swap, not post-training). FakeTrainer / surrogate-prefix
 * are not jumps and must not be reported as a θ win.
 */
export function gateWeightMount(before: number, after: number): WeightGateDecision {
  if (after > before) {
    return {
      arm: "I_weight",
      action: "mount",
      before,
      after,
      reason: "after-eval beats before; catalog-swap may mount pro-0813",
    };
  }
  return {
    arm: "I_weight",
    action: "reject",
    before,
    after,
    reason: "after-eval did not beat before; serving keeps flash-0731 (not a jump)",
  };
}
