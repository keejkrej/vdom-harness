import { flatten, type AgentGraph } from "../ir.js";
import { applySelfRefineMutation, applyValidatorMutation } from "../scientist.js";
import { reconcile, type ReconcileOp } from "../reconciler.js";
import { tau2Graph } from "./tau2-graph.js";
import { type Tau2Obs, type Tau2Technique } from "./tau2-types.js";

export type InterventionArm = "I_loop" | "I_weight" | "wait";

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
};

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

export function loopExhausted(g: AgentGraph): boolean {
  return graphHas(g, "critic") && graphHas(g, "validator");
}

export function recommendIntervention(
  obs: Tau2Obs,
  opts?: { loopExhausted?: boolean },
): InterventionArm {
  if (obs.nSuccessProxy === 1) return "wait";
  if (opts?.loopExhausted) return "I_weight";
  return "I_loop";
}

/** Slice-level arm: wait only if every episode hit; else I_loop until topology is exhausted. */
export function recommendSliceIntervention(
  obsList: Tau2Obs[],
  opts?: { loopExhausted?: boolean },
): InterventionArm {
  if (obsList.length > 0 && obsList.every((o) => o.nSuccessProxy === 1)) return "wait";
  if (opts?.loopExhausted) return "I_weight";
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
      hits === obsList.length
        ? "path measure hits S; wait"
        : toolFailures > 0
          ? "tool failures in trajectory; inspect env channel"
          : repeatActions > 0
            ? "repeat actions; loop mutation or wait"
            : "episode unfinished or miss; inspect cascade / tools",
    toolFailures,
    repeatActions,
  };
}

export function diffOps(ops: ReconcileOp[]): GraphDiffOp[] {
  return ops.map((o) => ({
    op: o.op,
    key: o.node.key,
    ...("parentKey" in o && o.parentKey ? { parentKey: o.parentKey } : {}),
  }));
}

function techniqueOf(g: AgentGraph): Tau2Technique {
  const t = g.meta?.technique;
  if (t === "self-refine" || t === "validator" || t === "reflexion" || t === "one-shot") {
    return t;
  }
  return "one-shot";
}

/**
 * Next I_loop step on the live graph. Reconcile is deterministic; serving does not restart.
 * one-shot → Self-Refine (critic+refine); self-refine → validator node; then exhausted.
 */
export function applyILoop(start?: AgentGraph): ILoopResult {
  const graphBefore = start ?? tau2Graph("one-shot");
  const techniqueBefore = techniqueOf(graphBefore);
  let graphAfter: AgentGraph;
  let techniqueAfter: Tau2Technique;
  let applied = true;

  if (!graphHas(graphBefore, "critic")) {
    graphAfter = applySelfRefineMutation(graphBefore);
    techniqueAfter = "self-refine";
  } else if (!graphHas(graphBefore, "validator")) {
    graphAfter = applyValidatorMutation(graphBefore);
    techniqueAfter = "validator";
  } else {
    graphAfter = graphBefore;
    techniqueAfter = techniqueBefore;
    applied = false;
  }

  graphAfter.meta = {
    ...(graphAfter.meta ?? {}),
    technique: techniqueAfter,
    intervention: applied ? "I_loop" : "exhausted",
  };
  const rec = reconcile(graphBefore, graphAfter);
  return {
    arm: "I_loop",
    applied,
    techniqueBefore,
    techniqueAfter,
    graphBefore,
    graphAfter,
    graphDiff: diffOps(rec.ops),
  };
}

/**
 * I_weight eval gate. Mount only if the after-eval strictly beats before.
 * Serving keeps the old f_θ on reject. FakeTrainer is enough to exercise this.
 */
export function gateWeightMount(before: number, after: number): WeightGateDecision {
  if (after > before) {
    return {
      arm: "I_weight",
      action: "mount",
      before,
      after,
      reason: "after-eval beats before; mount adapter",
    };
  }
  return {
    arm: "I_weight",
    action: "reject",
    before,
    after,
    reason: "after-eval did not beat before; serving keeps old f_theta",
  };
}
