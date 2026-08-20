import { type AgentGraph } from "../ir.js";
import { RuntimeDOM } from "../reconciler.js";
import { type Provider } from "../providers.js";
import {
  applyILoop,
  computeApplyScope,
  graphForScopedTask,
  interventionLicense,
  recommendIntervention,
  recommendSliceIntervention,
  type ILoopResult,
  type InterventionArm,
  type InterventionLicense,
} from "./tau2-improve.js";
import { type ApplyScope, type CatalogPointer, type Tau2Obs } from "./tau2-types.js";
import {
  applyISku,
  CATALOG_JUMP_MODEL,
  catalogPointer,
  proposeCatalogJump,
  SERVING_MODEL,
  type CatalogJumpDecision,
  type CatalogJumpProposal,
} from "./tau2-weight.js";

/**
 * Landed controller.
 * Hung/incomplete licenses I_sku — not “pick a pricier model.”
 * Mixed batches apply BOTH buckets: slice=I_sku does not drop I_loop.
 * S is a CatalogPointer beside C, not n.model and not a rebound sibling graph.
 * Gate is a measured after-eval, not “0813 exists.”
 */
export const CONTROLLER_NOTE =
  "License is hung/incomplete, not pick a pricier model. " +
  "Mixed 39/44 applies BOTH buckets (39 I_loop C1, 44 I_sku); consuming only slice drops I_loop. " +
  "I_sku writes S (catalog pointer beside C) to deepseek/deepseek-v4-pro-0813 after a measured after-eval; " +
  "0813 existing is not a gate. Jump iff later serving model id is 0813. " +
  "I_loop never writes S. servingPaused stays false. Catalog rebind, not fine-tuning. No LoRA.";

export type ControlledEpisode = {
  taskId?: string;
  hung: boolean;
  arm: InterventionArm;
  license: InterventionLicense;
};

export type ControlledBatch = {
  episodes: ControlledEpisode[];
  /** Slice prefers I_sku if any hung. Not the only consumer. */
  slice: InterventionArm;
  /** Per-task arms. A runner that only reads slice drops I_loop on mixed 39/44. */
  buckets: Record<string, InterventionArm>;
  /** Arms actually applied from buckets, not from slice alone. */
  applied: InterventionArm[];
  applyScope: ApplyScope;
  loop?: ILoopResult;
  proposal?: CatalogJumpProposal;
  gate?: CatalogJumpDecision;
  graphC0?: AgentGraph;
  graphC1?: AgentGraph;
  /** C topology for the weighted bucket (C0). Not a rebound-n.model clone. */
  graphSku?: AgentGraph;
  /** Paper S0. I_loop never writes this. */
  serving: CatalogPointer;
  /** S after I_sku (0813 if mounted). Weighted serving reads this, not n.model. */
  servingSku: CatalogPointer;
  servingPaused: false;
  trained: false;
  notFineTuning: true;
};

export function controlEpisode(
  obs: Tau2Obs,
  opts?: { loopExhausted?: boolean },
): ControlledEpisode {
  return {
    taskId: obs.taskId,
    hung: Boolean(obs.hung),
    arm: recommendIntervention(obs, opts),
    license: interventionLicense(obs, opts),
  };
}

function bucketsFromEpisodes(episodes: ControlledEpisode[]): Record<string, InterventionArm> {
  const buckets: Record<string, InterventionArm> = {};
  for (const e of episodes) {
    if (e.taskId) buckets[e.taskId] = e.arm;
  }
  return buckets;
}

/** Applied arms from per-task buckets. Fails the mixed-batch contract if only slice is used. */
export function appliedFromScope(scope: ApplyScope): InterventionArm[] {
  const applied: InterventionArm[] = [];
  if (scope.looped.length > 0) applied.push("I_loop");
  if ((scope.weighted ?? []).length > 0) applied.push("I_sku");
  return applied;
}

/**
 * Batch controller. Applies BOTH buckets when a graph is supplied:
 * looped → I_loop (C1); weighted → I_sku writes S (C topology stays C0).
 * Omit after → sku reject. Fixture after=before+ε may mount.
 */
export function controlBatch(
  obsList: Tau2Obs[],
  opts?: {
    loopExhausted?: boolean;
    graph?: AgentGraph;
    before?: number;
    after?: number | null;
    provider?: Provider;
    dom?: RuntimeDOM;
    serving?: CatalogPointer;
  },
): ControlledBatch {
  const episodes = obsList.map((o) => controlEpisode(o, opts));
  const slice = recommendSliceIntervention(obsList, opts);
  const applyScope = computeApplyScope(obsList);
  const buckets = bucketsFromEpisodes(episodes);
  const applied = appliedFromScope(applyScope);
  const serving = opts?.serving ?? catalogPointer(SERVING_MODEL);
  const out: ControlledBatch = {
    episodes,
    slice,
    buckets,
    applied,
    applyScope,
    serving,
    servingSku: serving,
    servingPaused: false,
    trained: false,
    notFineTuning: true,
    graphC0: opts?.graph,
  };

  if (opts?.graph && applyScope.looped.length > 0) {
    out.loop = applyILoop(opts.graph, obsList);
    out.graphC1 = out.loop.applied ? out.loop.graphAfter : opts.graph;
    // I_loop mutates C only. S stays serving / servingSku.
  }

  if (applyScope.weighted.length > 0 || slice === "I_sku") {
    out.proposal = proposeCatalogJump(serving.sku);
    if (opts?.graph) {
      out.gate = applyISku({
        graph: opts.graph,
        before: opts.before ?? 0,
        after: opts.after,
        provider: opts.provider,
        dom: opts.dom,
        serving,
      });
      out.servingSku = out.gate.serving;
      out.graphSku = out.gate.graph;
    }
  }
  return out;
}

/** Serving graph for one task after both buckets. */
export function servingGraphForTask(ctrl: ControlledBatch, taskId: string): AgentGraph | undefined {
  if (!ctrl.graphC0) return undefined;
  return graphForScopedTask(
    ctrl.graphC0,
    ctrl.graphC1 ?? ctrl.graphC0,
    ctrl.applyScope,
    taskId,
    ctrl.graphSku,
  );
}

export function servingModelForTask(ctrl: ControlledBatch, taskId: string): string | undefined {
  if ((ctrl.applyScope.weighted ?? []).includes(taskId)) {
    return ctrl.servingSku.sku;
  }
  return ctrl.serving.sku;
}

export { CATALOG_JUMP_MODEL, recommendIntervention, recommendSliceIntervention };
