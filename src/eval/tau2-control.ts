import { findNode, type AgentGraph } from "../ir.js";
import { RuntimeDOM } from "../reconciler.js";
import { resolveProvider, type Provider } from "../providers.js";
import { providerForNode } from "../runtime.js";
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
import { type ApplyScope, type CatalogPointer, type HybridState, type Tau2Obs } from "./tau2-types.js";
import {
  applyISku,
  CATALOG_JUMP_MODEL,
  catalogPointer,
  proposeCatalogJump,
  SERVING_MODEL,
  type CatalogJumpDecision,
  type CatalogJumpProposal,
} from "./tau2-weight.js";
import { defaultHybridC, hybridState, writeHybridS } from "./tau2-hybrid-state.js";

/**
 * Landed controller.
 * Hung/incomplete licenses I_sku — not “pick a pricier model.”
 * Mixed batches apply BOTH buckets: slice=I_sku does not drop I_loop.
 * S is HybridState.S on the X_n object (also projected as episode.serving).
 * Not n.model, not a rebound sibling graph, and not a process-global servingSku.
 * Gate is a measured after-eval, not “0813 exists.”
 */
export const CONTROLLER_NOTE =
  "License is hung/incomplete, not pick a pricier model. " +
  "Mixed 39/44 applies BOTH buckets (39 I_loop C1, 44 I_sku); consuming only slice drops I_loop. " +
  "I_sku writes X.S (HybridState.S on the episode) to deepseek/deepseek-v4-pro-0813 after a measured after-eval; " +
  "0813 existing is not a gate. Jump iff later serving model id is 0813. " +
  "I_loop never writes any episode's S. A new batch starts at S0 and does not inherit process servingSku. " +
  "servingPaused stays false. Catalog rebind, not fine-tuning. No LoRA. " +
  "servingByTask is a derived cache from X.S, not the lookup.";

export type ControlledEpisode = {
  taskId?: string;
  hung: boolean;
  arm: InterventionArm;
  license: InterventionLicense;
  /** Paper X_n. I_sku mount writes X.S on this object. */
  X: HybridState;
  /** Derived view of X.S. Source of truth is X.S. */
  readonly serving: CatalogPointer;
};

export type ControlledBatch = {
  episodes: ControlledEpisode[];
  /**
   * Live HybridState objects keyed by task. Same references as episodes[i].X.
   * I_sku writes X.S on these objects. Not assembled from servingByTask.
   */
  X: Record<string, HybridState>;
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
  /** Paper S0 for this batch. New batches start here; not a leftover process sku. */
  serving: CatalogPointer;
  /**
   * I_sku cell result for this batch (0813 if that gate mounted).
   * Not the per-task source of truth — read X_n.S / servingModelForTask.
   */
  servingSku: CatalogPointer;
  servingPaused: false;
  trained: false;
  notFineTuning: true;
};

export function controlEpisode(
  obs: Tau2Obs,
  opts?: { loopExhausted?: boolean; graph?: AgentGraph; serving?: CatalogPointer },
): ControlledEpisode {
  const s0 = opts?.serving ?? catalogPointer(SERVING_MODEL);
  const X = hybridState({
    E: obs,
    C: opts?.graph ?? defaultHybridC(s0.sku),
    S: catalogPointer(s0.sku),
  });
  return {
    taskId: obs.taskId,
    hung: Boolean(obs.hung),
    arm: recommendIntervention(obs, opts),
    license: interventionLicense(obs, opts),
    X,
    get serving() {
      return this.X.S;
    },
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
 * looped → I_loop (C1); weighted → I_sku writes only those episodes' S
 * (C topology stays C0). A fresh batch starts every episode at S0.
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
    /** S0 for this batch only. Not a leftover process servingSku. */
    serving?: CatalogPointer;
  },
): ControlledBatch {
  const s0 = opts?.serving ?? catalogPointer(SERVING_MODEL);
  const episodes = obsList.map((o) =>
    controlEpisode(o, { ...opts, serving: catalogPointer(s0.sku), graph: opts?.graph }),
  );
  const X: Record<string, HybridState> = {};
  for (const e of episodes) {
    if (e.taskId) X[e.taskId] = e.X;
  }
  const slice = recommendSliceIntervention(obsList, opts);
  const applyScope = computeApplyScope(obsList);
  const buckets = bucketsFromEpisodes(episodes);
  const applied = appliedFromScope(applyScope);
  const out: ControlledBatch = {
    episodes,
    X,
    slice,
    buckets,
    applied,
    applyScope,
    serving: catalogPointer(s0.sku),
    servingSku: catalogPointer(s0.sku),
    servingPaused: false,
    trained: false,
    notFineTuning: true,
    graphC0: opts?.graph,
  };

  if (opts?.graph && applyScope.looped.length > 0) {
    out.loop = applyILoop(opts.graph, obsList);
    out.graphC1 = out.loop.applied ? out.loop.graphAfter : opts.graph;
    // I_loop mutates C only. No episode's S is written.
    for (const e of episodes) {
      if (e.taskId && applyScope.looped.includes(e.taskId) && out.graphC1) {
        e.X.C = out.graphC1;
      }
    }
  }

  if (applyScope.weighted.length > 0 || slice === "I_sku") {
    out.proposal = proposeCatalogJump(s0.sku);
    if (opts?.graph) {
      out.gate = applyISku({
        graph: opts.graph,
        before: opts.before ?? 0,
        after: opts.after,
        provider: opts.provider,
        dom: opts.dom,
        serving: catalogPointer(s0.sku),
      });
      out.servingSku = out.gate.serving;
      out.graphSku = out.gate.graph;
      if (out.gate.action === "mount") {
        writeISkuServing(out.episodes, applyScope.weighted, out.gate.serving);
      }
    }
  }
  return out;
}

/** I_sku mount writes only the weighted episodes' X.S. I_loop never calls this. */
export function writeISkuServing(
  episodes: ControlledEpisode[],
  weighted: string[],
  serving: CatalogPointer,
): void {
  for (const e of episodes) {
    if (e.taskId && weighted.includes(e.taskId)) {
      writeHybridS(e.X, serving);
    }
  }
}

/** JSON/controller log: 39.sku=0731 44.sku=0813 after a fixture mount. */
export function controllerServingLog(ctrl: ControlledBatch): {
  text: string;
  tasks: Record<string, CatalogPointer>;
} {
  const tasks: Record<string, CatalogPointer> = {};
  const parts: string[] = [];
  for (const e of ctrl.episodes) {
    if (!e.taskId) continue;
    tasks[e.taskId] = e.X.S;
    const tag = e.X.S.sku.includes("0813")
      ? "0813"
      : e.X.S.sku.includes("0731")
        ? "0731"
        : e.X.S.sku;
    parts.push(`${e.taskId}.sku=${tag}`);
  }
  return { text: parts.join(" "), tasks };
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
  const X = ctrl.X[taskId] ?? ctrl.episodes.find((e) => e.taskId === taskId)?.X;
  if (X) return X.S.sku;
  return ctrl.serving.sku;
}

/** Later serving client for one task. Reads per-task S; does not use a sprayed bind. */
export function servingProviderForTask(
  ctrl: ControlledBatch,
  taskId: string,
  fallback: Provider,
  dom?: RuntimeDOM,
  key = "solve",
): Provider {
  const sku = servingModelForTask(ctrl, taskId);
  const g = servingGraphForTask(ctrl, taskId);
  const n = g ? (findNode(g, key) ?? g.root) : undefined;
  if (!n) {
    return sku ? resolveProvider(sku) : fallback;
  }
  return providerForNode(n, fallback, dom, sku);
}

export { CATALOG_JUMP_MODEL, recommendIntervention, recommendSliceIntervention };
