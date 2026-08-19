import { flatten, type AgentGraph } from "../ir.js";
import {
  applyPolicyChecklistMutation,
  applySelfRefineMutation,
  applyValidatorMutation,
} from "../scientist.js";
import { reconcile, type ReconcileOp } from "../reconciler.js";
import { tau2Graph } from "./tau2-graph.js";
import { type Tau2Obs, type Tau2Technique } from "./tau2-types.js";
import { AIRLINE_POLICY_CHECKLIST, shouldRecommendPolicy } from "./tau2-policy.js";

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

export function loopExhausted(g: AgentGraph, obs?: Tau2Obs | Tau2Obs[]): boolean {
  if (graphHas(g, "policy-checklist")) return true;
  if (obsNeedsPolicy(obs)) return false;
  return graphHas(g, "critic") && graphHas(g, "validator");
}

export function obsNeedsPolicy(obs?: Tau2Obs | Tau2Obs[] | null): boolean {
  if (!obs) return false;
  const list = Array.isArray(obs) ? obs : [obs];
  return list.some((o) => shouldRecommendPolicy(o));
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

function techniqueOf(g: AgentGraph): Tau2Technique {
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
  const techniqueBefore = techniqueOf(graphBefore);
  let graphAfter: AgentGraph;
  let techniqueAfter: Tau2Technique;
  let applied = true;

  if (obsNeedsPolicy(obs) && !graphHas(graphBefore, "policy-checklist")) {
    graphAfter = applyPolicyChecklistMutation(graphBefore, AIRLINE_POLICY_CHECKLIST);
    techniqueAfter = "policy-checklist";
  } else if (obsNeedsPolicy(obs)) {
    graphAfter = graphBefore;
    techniqueAfter = techniqueBefore;
    applied = false;
  } else if (!graphHas(graphBefore, "critic")) {
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
 * I_weight eval gate (slow clock). Mount only if the after-eval strictly beats
 * before. Serving keeps the old f_θ on reject (`servingPaused` stays false).
 * FakeTrainer exercises the protocol; a surrogate that cannot raise p_hit is
 * an honest reject — never a claimed 0731 LoRA.
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
