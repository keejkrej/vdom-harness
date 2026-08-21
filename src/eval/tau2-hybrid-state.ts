/**
 * Runtime HybridState X=(H,M,E,C,S).
 *
 * S is a CatalogPointer on the state object itself. I_sku writes X.S.
 * servingByTask, if present, is a derived cache from X.S — not the lookup.
 * Do not assemble HybridState by reading servingByTask and stuffing S.
 */
import { createHash } from "node:crypto";
import { findNode, flatten, type AgentGraph } from "../ir.js";
import { type Message } from "../providers.js";
import { type Trace } from "../ir.js";
import {
  type CatalogPointer,
  type HybridState,
  type Tau2Obs,
} from "./tau2-types.js";
import { catalogPointer, SERVING_MODEL, cTopology } from "./tau2-weight.js";
import { tau2Graph } from "./tau2-graph.js";

export type { HybridState } from "./tau2-types.js";

export const HYBRID_STATE_KEYS = ["H", "M", "E", "C", "S"] as const;

export type HybridStore = Map<string, HybridState>;

export function hybridState(init: {
  H?: Message[];
  M?: Trace[];
  E: Tau2Obs;
  C: AgentGraph;
  S?: CatalogPointer;
}): HybridState {
  const X: HybridState = {
    H: init.H ?? [],
    M: init.M ?? [],
    E: init.E,
    C: init.C,
    S: catalogPointer(init.S?.sku ?? SERVING_MODEL),
  };
  return X;
}

/** Write S onto an existing X. Mutates the state object. Not a new assembly. */
export function writeHybridS(X: HybridState, serving: CatalogPointer): HybridState {
  X.S = catalogPointer(serving.sku);
  return X;
}

export function sOnState(X: HybridState | undefined | null): boolean {
  return Boolean(X && Object.prototype.hasOwnProperty.call(X, "S") && X.S && typeof X.S.sku === "string");
}

export function createHybridStore(): HybridStore {
  return new Map();
}

export function installHybridState(store: HybridStore, taskId: string, X: HybridState): HybridState {
  store.set(taskId, X);
  return X;
}

export function hybridOf(store: HybridStore, taskId: string): HybridState | undefined {
  return store.get(taskId);
}

/**
 * Write S onto the existing X for taskId. Returns undefined if no X exists —
 * caller must not invent a HybridState from a servingByTask Map.
 */
export function writeSOnStore(
  store: HybridStore,
  taskId: string,
  serving: CatalogPointer,
): HybridState | undefined {
  const X = store.get(taskId);
  if (!X) return undefined;
  return writeHybridS(X, serving);
}

/** Derived cache. Not the source of truth. */
export function derivedServingByTask(store: HybridStore): Record<string, CatalogPointer> {
  const out: Record<string, CatalogPointer> = {};
  for (const [id, X] of store) out[id] = X.S;
  return out;
}

export function servingFromHybrid(
  store: HybridStore,
  taskId?: string,
  s0: CatalogPointer = catalogPointer(SERVING_MODEL),
): CatalogPointer {
  if (taskId) {
    const X = store.get(taskId);
    if (X) return X.S;
  }
  return s0;
}

export function hybridRecord(store: HybridStore): Record<string, HybridState> {
  return Object.fromEntries(store);
}

export function nodeListOf(g: AgentGraph): string[] {
  return flatten(g).map((f) => f.node.key);
}

export function nModelOf(g: AgentGraph, fallback: string = SERVING_MODEL): string {
  const m = findNode(g, "solve")?.model ?? g.root.model;
  return typeof m === "string" && m.length > 0 ? m : fallback;
}

/** Topology + n.model. Critic compares this before vs after the S write. */
export function cGraphHash(g: AgentGraph): string {
  const payload = JSON.stringify({
    topology: cTopology(g),
    models: flatten(g).map((f) => ({ key: f.node.key, model: f.node.model ?? null })),
  });
  return createHash("sha256").update(payload).digest("hex");
}

export function compactC(g: AgentGraph): {
  nModel: string;
  nodeKeys: string[];
  graphHash: string;
} {
  return {
    nModel: nModelOf(g),
    nodeKeys: nodeListOf(g),
    graphHash: cGraphHash(g),
  };
}

export function defaultHybridC(sku: string = SERVING_MODEL): AgentGraph {
  return tau2Graph("one-shot", sku);
}

export function emptyObs(taskId: string): Tau2Obs {
  return {
    taskId,
    nSteps: 0,
    nSuccessProxy: 0,
    lastActions: [],
    channels: [],
    critique: "",
    toolFailures: 0,
    repeatActions: 0,
  };
}
