import { cloneGraph, findNode, flatten, modelId, type AgentGraph, type AgentNode } from "../ir.js";
import { DeterministicProvider, registerProvider, resolveChatConfig, type Provider } from "../providers.js";
import { RuntimeDOM } from "../reconciler.js";
import { providerForNode } from "../runtime.js";

/**
 * I_catalog actuator: gated catalog rebind of f_θ (catalog swap, not post-training).
 * Serving stays on flash-0731 until gate=mount rebinds PhysicalNode.provider
 * and n.model to pro-0813. servingPaused stays false.
 * FakeTrainer / surrogate-prefix / LoRA are not jumps and are not this arm.
 */
export const SERVING_MODEL = "deepseek/deepseek-v4-flash-0731";
export const CATALOG_JUMP_MODEL = "deepseek/deepseek-v4-pro-0813";
export const CATALOG_JUMP_GA = "2026-08-12";

export const CATALOG_JUMP_NOTE =
  "I_catalog is a gated catalog rebind of f_θ (catalog swap, not post-training). " +
  "Serving stays on deepseek/deepseek-v4-flash-0731 until gate=mount rebinds " +
  "PhysicalNode.provider / n.model to deepseek/deepseek-v4-pro-0813 " +
  "(OpenRouter, GA 2026-08-12). servingPaused stays false. " +
  "FakeTrainer, surrogate-prefix, and LoRA are not jumps and are not this arm.";

export type CatalogJumpProposal = {
  arm: "I_catalog";
  kind: "catalog-rebind";
  from: string;
  to: string;
  model: string;
  servingPaused: false;
  notPostTraining: true;
  trained: false;
};

export type CatalogJumpDecision = {
  arm: "I_catalog";
  kind: "catalog-rebind";
  action: "mount" | "reject";
  /** Critic lock: true only when gate=mount and serving model id is 0813. */
  jumped: boolean;
  from: string;
  to: string;
  before: number;
  after: number;
  reason: string;
  servingPaused: false;
  notPostTraining: true;
  trained: false;
  graph: AgentGraph;
  graphBefore: AgentGraph;
  servingModelId: string;
};

export function proposeCatalogJump(from: string = SERVING_MODEL): CatalogJumpProposal {
  return {
    arm: "I_catalog",
    kind: "catalog-rebind",
    from,
    to: CATALOG_JUMP_MODEL,
    model: CATALOG_JUMP_MODEL,
    servingPaused: false,
    notPostTraining: true,
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
  copy.id = `${g.id}-catalog-0813`;
  copy.meta = {
    ...(copy.meta ?? {}),
    intervention: "I_catalog",
    catalogRebind: true,
    catalogSwap: true,
    servingModel: model,
    notPostTraining: true,
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
 * Slow clock of I_catalog: request a released checkpoint, then gate.
 * Mount rebinds n.model + PhysicalNode.provider to 0813. Reject keeps 0731.
 * Catalog swap, not post-training. θ jumped only if later serving model id is 0813.
 */
export function applyICatalog(opts: {
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
      arm: "I_catalog",
      kind: "catalog-rebind",
      action: "reject",
      jumped: false,
      from,
      to,
      before: opts.before,
      after: opts.after,
      reason: "catalog-rebind gate rejected; serving keeps flash-0731 (catalog swap, not post-training)",
      servingPaused: false,
      notPostTraining: true,
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
    arm: "I_catalog",
    kind: "catalog-rebind",
    action: "mount",
    jumped,
    from: proposal.from,
    to,
    before: opts.before,
    after: opts.after,
    reason: "catalog-rebind gate=mount; rebound f_θ to pro-0813 (catalog swap, not post-training)",
    servingPaused: false,
    notPostTraining: true,
    trained: false,
    graph: graphAfter,
    graphBefore: opts.graph,
    servingModelId,
  };
}

/** @deprecated name; I_catalog is the paper slow arm. */
export const applyIWeightCatalog = applyICatalog;

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
