import { type AgentGraph } from "../ir.js";
import { RuntimeDOM } from "../reconciler.js";
import { type Provider } from "../providers.js";
import {
  computeApplyScope,
  interventionLicense,
  recommendIntervention,
  recommendSliceIntervention,
  type InterventionArm,
  type InterventionLicense,
} from "./tau2-improve.js";
import { type ApplyScope, type Tau2Obs } from "./tau2-types.js";
import {
  applyISku,
  CATALOG_JUMP_MODEL,
  proposeCatalogJump,
  type CatalogJumpDecision,
  type CatalogJumpProposal,
} from "./tau2-weight.js";

/**
 * Landed controller.
 * Hung/incomplete licenses I_sku — not “pick a pricier model.”
 * Slow arm proposes 0813; the gate is a measured after-eval, not “0813 exists.”
 * servingPaused stays false. Jump iff later serving model id is 0813.
 */
export const CONTROLLER_NOTE =
  "License is hung/incomplete, not pick a pricier model. " +
  "I_sku proposes deepseek/deepseek-v4-pro-0813 and gates on a measured after-eval; " +
  "0813 existing is not a gate. Jump iff later serving model id is 0813. " +
  "servingPaused stays false. Catalog rebind, not fine-tuning. No LoRA.";

export type ControlledEpisode = {
  taskId?: string;
  hung: boolean;
  arm: InterventionArm;
  license: InterventionLicense;
};

export type ControlledBatch = {
  episodes: ControlledEpisode[];
  slice: InterventionArm;
  applyScope: ApplyScope;
  proposal?: CatalogJumpProposal;
  gate?: CatalogJumpDecision;
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

/**
 * Batch controller. 39/44: completed attractor → I_loop; hung → I_sku.
 * If the slice arm is I_sku and a graph is supplied, propose 0813 and gate.
 * Omit after → reject (no measured eval). Passing after requires after > before.
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
  },
): ControlledBatch {
  const episodes = obsList.map((o) => controlEpisode(o, opts));
  const slice = recommendSliceIntervention(obsList, opts);
  const applyScope = computeApplyScope(obsList);
  const out: ControlledBatch = {
    episodes,
    slice,
    applyScope,
    servingPaused: false,
    trained: false,
    notFineTuning: true,
  };
  if (slice !== "I_sku") return out;
  out.proposal = proposeCatalogJump();
  if (!opts?.graph) return out;
  out.gate = applyISku({
    graph: opts.graph,
    before: opts.before ?? 0,
    after: opts.after,
    provider: opts.provider,
    dom: opts.dom,
  });
  return out;
}

export { CATALOG_JUMP_MODEL, recommendIntervention, recommendSliceIntervention };
