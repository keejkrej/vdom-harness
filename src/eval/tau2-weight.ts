import { cloneGraph, findNode, flatten, modelId, type AgentGraph, type AgentNode } from "../ir.js";
import { DeterministicProvider, registerProvider, resolveChatConfig, type Provider } from "../providers.js";
import { RuntimeDOM } from "../reconciler.js";
import { providerForNode } from "../runtime.js";

/**
 * I_sku actuator: gated catalog rebind of the serving SKU, not I_weight-as-trainer.
 * Base SKU is flash-0731 until gate=mount rebinds PhysicalNode.provider
 * and n.model to pro-0813. servingPaused stays false.
 * Jump iff later serving model id is 0813. Not fine-tuning. No LoRA.
 */
export const SERVING_MODEL = "deepseek/deepseek-v4-flash-0731";
export const CATALOG_JUMP_MODEL = "deepseek/deepseek-v4-pro-0813";
export const BASE_SKU = SERVING_MODEL;
export const ESCALATE_SKU = CATALOG_JUMP_MODEL;
export const CATALOG_JUMP_GA = "2026-08-12";

export const CATALOG_JUMP_NOTE =
  "I_sku is a gated catalog rebind of the serving SKU, not I_weight-as-trainer. " +
  "Base SKU deepseek/deepseek-v4-flash-0731 until gate=mount rebinds " +
  "PhysicalNode.provider / n.model to deepseek/deepseek-v4-pro-0813 " +
  "(OpenRouter, GA 2026-08-12). Jump iff later serving model id is 0813. " +
  "servingPaused stays false. Catalog rebind, not fine-tuning. FakeTrainer / LoRA are not this arm.";

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
  after: number;
  reason: string;
  servingPaused: false;
  notPostTraining: true;
  notFineTuning: true;
  trained: false;
  graph: AgentGraph;
  graphBefore: AgentGraph;
  servingModelId: string;
};

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

export function servingModelOfGraph(g: AgentGraph, fallback?: string): string | undefined {
  return modelId(findNode(g, "solve")?.model) ?? modelId(g.root.model) ?? fallback;
}

export function thetaJumped(gate: CatalogJumpDecision): boolean {
  return gate.action === "mount" && gate.jumped && gate.servingModelId === CATALOG_JUMP_MODEL;
}

function rebindGraphModel(g: AgentGraph, model: string): AgentGraph {
  const copy = cloneGraph(g);
  const walk = (n: AgentNode): AgentNode => ({
    ...n,
    model,
    children: n.children?.map(walk),
  });
  copy.root = walk(copy.root);
  copy.version = g.version + 1;
  copy.id = `${g.id}-sku-0813`;
  copy.meta = {
    ...(copy.meta ?? {}),
    intervention: "I_sku",
    catalogRebind: true,
    servingModel: model,
    notPostTraining: true,
    notFineTuning: true,
    trained: false,
  };
  return copy;
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

/**
 * Slow clock of I_sku: propose escalate SKU 0813, then gate.
 * Mount rebinds n.model + PhysicalNode.provider to 0813. Reject keeps 0731.
 * Catalog rebind, not fine-tuning. Jump iff later serving model id is 0813.
 */
export function applyISku(opts: {
  graph: AgentGraph;
  before: number;
  after: number;
  from?: string;
  to?: string;
  provider?: Provider;
  dom?: RuntimeDOM;
}): CatalogJumpDecision {
  const from = opts.from ?? servingModelOfGraph(opts.graph) ?? SERVING_MODEL;
  const to = opts.to ?? CATALOG_JUMP_MODEL;
  const proposal = proposeCatalogJump(from);
  if (opts.after <= opts.before) {
    return {
      arm: "I_sku",
      kind: "catalog-rebind",
      action: "reject",
      jumped: false,
      from,
      to,
      before: opts.before,
      after: opts.after,
      reason: "catalog rebind gate rejected; serving keeps flash-0731 (not fine-tuning)",
      servingPaused: false,
      notPostTraining: true,
      notFineTuning: true,
      trained: false,
      graph: opts.graph,
      graphBefore: opts.graph,
      servingModelId: servingModelOfGraph(opts.graph, from) ?? from,
    };
  }

  bindCatalogProvider(to, opts.provider);
  const graphAfter = rebindGraphModel(opts.graph, to);
  if (opts.dom) opts.dom.reconcile(graphAfter);
  const servingModelId = servingModelOfGraph(graphAfter) ?? to;
  const jumped = servingModelId === CATALOG_JUMP_MODEL;
  return {
    arm: "I_sku",
    kind: "catalog-rebind",
    action: "mount",
    jumped,
    from: proposal.from,
    to,
    before: opts.before,
    after: opts.after,
    reason: "catalog rebind gate=mount; rebound serving SKU to pro-0813 (not fine-tuning)",
    servingPaused: false,
    notPostTraining: true,
    notFineTuning: true,
    trained: false,
    graph: graphAfter,
    graphBefore: opts.graph,
    servingModelId,
  };
}

/** Prior names; I_sku is the paper slow arm. */
export const applyICatalog = applyISku;
export const applyIWeightCatalog = applyISku;

/** Later serving step: bound provider for a node after a catalog mount. */
export function servingProviderAfterJump(
  graph: AgentGraph,
  fallback: Provider,
  dom?: RuntimeDOM,
  key = "solve",
): Provider {
  const n = findNode(graph, key) ?? graph.root;
  return providerForNode(n, fallback, dom);
}

export function catalogSwapOnGraph(g: AgentGraph): boolean {
  return flatten(g).some((f) => modelId(f.node.model) === CATALOG_JUMP_MODEL);
}
