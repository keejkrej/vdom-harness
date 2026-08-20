import { flatten, findNode, modelId, type AgentGraph } from "../ir.js";
import { DeterministicProvider, registerProvider, resolveChatConfig, type Provider } from "../providers.js";
import { RuntimeDOM } from "../reconciler.js";
import { providerForNode } from "../runtime.js";
import { type CatalogPointer } from "./tau2-types.js";

export type { CatalogPointer, ServingSku } from "./tau2-types.js";

/**
 * I_sku actuator: gated catalog rebind of the serving pointer S, not I_weight-as-trainer.
 * Base SKU is flash-0731 until gate=mount writes S to pro-0813.
 * C topology / n.model are not rewritten. servingPaused stays false.
 * Jump iff later serving model id is 0813. Not fine-tuning. No LoRA.
 */
export const SERVING_MODEL = "deepseek/deepseek-v4-flash-0731";
export const CATALOG_JUMP_MODEL = "deepseek/deepseek-v4-pro-0813";
export const BASE_SKU = SERVING_MODEL;
export const ESCALATE_SKU = CATALOG_JUMP_MODEL;
export const CATALOG_JUMP_GA = "2026-08-12";

export const CATALOG_JUMP_NOTE =
  "I_sku is a gated catalog rebind licensed by hung/incomplete, not pick a pricier model. " +
  "Base SKU deepseek/deepseek-v4-flash-0731. Slow arm proposes " +
  "deepseek/deepseek-v4-pro-0813 (OpenRouter, GA 2026-08-12). " +
  "Gate is a measured after-eval; 0813 existing is not a gate. " +
  "Jump iff later serving model id is 0813. servingPaused stays false. " +
  "I_sku writes S (catalog pointer beside C), not n.model. " +
  "Catalog rebind, not fine-tuning. FakeTrainer / LoRA are not this arm.";

export type CatalogJumpProposal = {
  arm: "I_sku";
  kind: "catalog-rebind";
  from: string;
  to: string;
  model: string;
  servingPaused: false;
  notPostTraining: true;
  notFineTuning: true;
  trained: false;
};

export type CatalogJumpDecision = {
  arm: "I_sku";
  kind: "catalog-rebind";
  action: "mount" | "reject";
  /** Author lock: true only when gate=mount and serving model id is 0813. */
  jumped: boolean;
  from: string;
  to: string;
  before: number;
  after: number | null;
  reason: string;
  servingPaused: false;
  notPostTraining: true;
  notFineTuning: true;
  trained: false;
  /** C after I_sku. Topology / n.model equal graphBefore; S is `serving`. */
  graph: AgentGraph;
  graphBefore: AgentGraph;
  /** Paper S. Source of truth for the serving SKU. */
  serving: CatalogPointer;
  servingModelId: string;
};

export function catalogPointer(sku: string = SERVING_MODEL): CatalogPointer {
  return { sku, servingPaused: false };
}

export function servingSkuOf(
  serving?: CatalogPointer | null,
  fallback: string = SERVING_MODEL,
): string {
  return serving?.sku ?? fallback;
}

/** C topology the ICLR critic compares: keys / objectives / policy prompts. Not n.model. */
export function cTopology(
  g: AgentGraph,
): Array<{ key: string; objective: string; prompt: string }> {
  return flatten(g).map((f) => ({
    key: f.node.key,
    objective: f.node.objective,
    prompt: f.node.prompt ?? "",
  }));
}

export function sameCTopology(a: AgentGraph, b: AgentGraph): boolean {
  return JSON.stringify(cTopology(a)) === JSON.stringify(cTopology(b));
}

export function proposeCatalogJump(from: string = SERVING_MODEL): CatalogJumpProposal {
  return {
    arm: "I_sku",
    kind: "catalog-rebind",
    from,
    to: CATALOG_JUMP_MODEL,
    model: CATALOG_JUMP_MODEL,
    servingPaused: false,
    notPostTraining: true,
    notFineTuning: true,
    trained: false,
  };
}

/**
 * Derived projection of C (n.model). Not paper S.
 * Serving reads CatalogPointer via servingSkuOf / servingModelId.
 */
export function servingModelOfGraph(g: AgentGraph, fallback?: string): string | undefined {
  return modelId(findNode(g, "solve")?.model) ?? modelId(g.root.model) ?? fallback;
}

export function thetaJumped(gate: CatalogJumpDecision): boolean {
  return gate.action === "mount" && gate.jumped && gate.serving.sku === CATALOG_JUMP_MODEL;
}

function bindCatalogProvider(to: string, provider?: Provider): void {
  if (provider) {
    registerProvider(to, provider);
    return;
  }
  // Unit / no-key: mock the bind. Live key: leave unregistered so resolveProvider
  // builds an OpenRouter client for 0813. Never a LoRA / FakeTrainer win.
  if (!resolveChatConfig()) {
    registerProvider(to, new DeterministicProvider(to));
  }
}

function measuredEval(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function rejectDecision(opts: {
  graph: AgentGraph;
  from: string;
  to: string;
  before: number;
  after: number | null;
  reason: string;
  serving: CatalogPointer;
}): CatalogJumpDecision {
  return {
    arm: "I_sku",
    kind: "catalog-rebind",
    action: "reject",
    jumped: false,
    from: opts.from,
    to: opts.to,
    before: opts.before,
    after: opts.after,
    reason: opts.reason,
    servingPaused: false,
    notPostTraining: true,
    notFineTuning: true,
    trained: false,
    graph: opts.graph,
    graphBefore: opts.graph,
    serving: opts.serving,
    servingModelId: opts.serving.sku,
  };
}

/**
 * Slow clock of I_sku: propose 0813, then gate on a measured after-eval.
 * 0813 existing is not a gate. Mount writes S to 0813. C is not rewritten.
 * Reject keeps S at 0731. Jump iff later serving model id is 0813.
 */
export function applyISku(opts: {
  graph: AgentGraph;
  before: number;
  after?: number | null;
  from?: string;
  to?: string;
  provider?: Provider;
  dom?: RuntimeDOM;
  /** Existing S. I_sku reads this, not n.model. */
  serving?: CatalogPointer;
}): CatalogJumpDecision {
  const from = opts.from ?? servingSkuOf(opts.serving, SERVING_MODEL);
  const to = opts.to ?? CATALOG_JUMP_MODEL;
  const proposal = proposeCatalogJump(from);
  const before = opts.before;
  const servingBefore = opts.serving ?? catalogPointer(from);
  if (!measuredEval(opts.after)) {
    return rejectDecision({
      graph: opts.graph,
      from,
      to,
      before,
      after: null,
      reason: "no measured after-eval; 0813 existing is not a gate",
      serving: catalogPointer(servingBefore.sku),
    });
  }
  const after = opts.after;
  if (after <= before) {
    return rejectDecision({
      graph: opts.graph,
      from,
      to,
      before,
      after,
      reason: "measured after-eval did not beat before; serving keeps flash-0731 (not fine-tuning)",
      serving: catalogPointer(servingBefore.sku),
    });
  }

  bindCatalogProvider(to, opts.provider);
  if (opts.dom) opts.dom.rebindServing(to, opts.provider);
  const serving = catalogPointer(to);
  const jumped = serving.sku === CATALOG_JUMP_MODEL;
  return {
    arm: "I_sku",
    kind: "catalog-rebind",
    action: "mount",
    jumped,
    from: proposal.from,
    to,
    before: opts.before,
    after: opts.after,
    reason: "measured after-eval beat before; wrote S to 0813 (C topology unchanged, not fine-tuning)",
    servingPaused: false,
    notPostTraining: true,
    notFineTuning: true,
    trained: false,
    graph: opts.graph,
    graphBefore: opts.graph,
    serving,
    servingModelId: serving.sku,
  };
}

/** Prior names; I_sku is the paper slow arm. */
export const applyICatalog = applyISku;
export const applyIWeightCatalog = applyISku;

/** Later serving step: bound provider from S (DOM / CatalogPointer), not n.model. */
export function servingProviderAfterJump(
  graph: AgentGraph,
  fallback: Provider,
  dom?: RuntimeDOM,
  key = "solve",
  serving?: CatalogPointer,
): Provider {
  const n = findNode(graph, key) ?? graph.root;
  return providerForNode(n, fallback, dom, serving?.sku);
}

/** True when S (not n.model) is the catalog jump SKU. */
export function catalogSwapOnServing(s: CatalogPointer): boolean {
  return s.sku === CATALOG_JUMP_MODEL;
}

/**
 * Derived: whether C's n.model projects 0813. Not the jump predicate.
 * Jump iff later serving model id (S) is 0813.
 */
export function catalogSwapOnGraph(g: AgentGraph): boolean {
  return flatten(g).some((f) => {
    const m = f.node.model;
    return typeof m === "string" && m === CATALOG_JUMP_MODEL;
  });
}
