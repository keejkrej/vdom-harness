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
  SERVING_E_NOTE,
  type CatalogPointer,
  type HybridState,
  type LicenseE,
  type ServingE,
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
  licenseE?: LicenseE;
}): HybridState {
  const X: HybridState = {
    H: init.H ?? [],
    M: init.M ?? [],
    E: init.E,
    C: init.C,
    S: catalogPointer(init.S?.sku ?? SERVING_MODEL),
  };
  if (init.licenseE) {
    writeHybridLicenseE(X, init.licenseE);
  } else if (isHung44IskuLicense(init.E)) {
    writeHybridLicenseE(X, init.E);
  }
  return X;
}

/** Hung/timeout I_sku LICENSE for task 44. Not serving-step E. */
export function isHung44IskuLicense(E: HybridState["E"] | null | undefined): E is Tau2Obs & {
  taskId: "44";
  hung: true;
  arm: "I_sku";
  termination: "timeout";
} {
  if (!E) return false;
  return (
    "taskId" in E &&
    E.taskId === "44" &&
    E.hung === true &&
    E.termination === "timeout" &&
    (!("arm" in E) || E.arm === "I_sku" || E.arm === undefined)
  );
}

export function licenseEOnState(X: HybridState | undefined | null): boolean {
  return Boolean(
    X &&
      Object.prototype.hasOwnProperty.call(X, "licenseE") &&
      X.licenseE &&
      X.licenseE.kind === "license" &&
      X.licenseE.copiedIntoH === false,
  );
}

export function servingEOnState(X: HybridState | undefined | null): boolean {
  if (!X) return false;
  const ownServing = Object.prototype.hasOwnProperty.call(X, "servingE") && X.servingE;
  const ownE = Object.prototype.hasOwnProperty.call(X, "E") && isServingStepE(X.E);
  return Boolean(ownServing || ownE);
}

export function isServingStepE(E: HybridState["E"] | ServingE | null | undefined): E is ServingE {
  if (!E) return false;
  return (
    "kind" in E &&
    E.kind === "greeting-turn" &&
    E.hung === false &&
    E.termination !== "timeout" &&
    servingEHasTurnFact(E)
  );
}

export function servingEHasTurnFact(E: {
  servedModel?: string;
  ts?: number;
  content?: string;
} | null | undefined): boolean {
  if (!E) return false;
  return Boolean(
    (E.servedModel && E.servedModel.length > 0) ||
      E.ts != null ||
      (E.content != null && E.content.length > 0),
  );
}

export function licenseEFromHungObs(E: HybridState["E"] | LicenseE): LicenseE {
  if ("kind" in E && E.kind === "license") {
    if (E.copiedIntoH !== false) {
      throw new Error("dump refused: licenseE must stay out of H");
    }
    return E;
  }
  if (!isHung44IskuLicense(E)) {
    throw new Error("dump refused: X.E is not the reconstructed hung/timeout I_sku license");
  }
  return {
    kind: "license",
    taskId: "44",
    hung: true,
    arm: "I_sku",
    termination: "timeout",
    copiedIntoH: false,
  };
}

/** Write licenseE onto existing X. Not copied into H. */
export function writeHybridLicenseE(
  X: HybridState,
  source?: HybridState["E"] | LicenseE,
): HybridState {
  X.licenseE = licenseEFromHungObs(source ?? X.E);
  return X;
}

/** Turn record enough to attach serving-step E. Not a constant overlay. */
export type ServingTurnRecord = {
  content?: string;
  servedModel?: string;
  traces?: Array<{ ts?: number }>;
};

export function servingEFromTurn(
  turn: ServingTurnRecord,
  incoming: unknown[] = [],
): ServingE {
  if (incoming.length !== 0) {
    throw new Error("servingE refused: greeting turn incoming messages must be []");
  }
  const serving: ServingE = {
    kind: "greeting-turn",
    hung: false,
    termination: null,
    notTau2UserGymStep: true,
    incomingMessages: [],
    note: SERVING_E_NOTE,
  };
  if (turn.servedModel) serving.servedModel = turn.servedModel;
  if (turn.traces?.[0]?.ts != null) serving.ts = turn.traces[0].ts;
  if (turn.content != null && turn.content.length > 0) serving.content = turn.content;
  if (!servingEHasTurnFact(serving)) {
    throw new Error("servingE refused: no turn-derived fact (servedModel, ts, or content)");
  }
  return serving;
}

/**
 * Write serving-step E onto the same X the turn mutated.
 * X.E is the turn record; servingE is an alias of that object.
 */
export function writeHybridServingE(
  X: HybridState,
  turn: ServingTurnRecord,
  incoming: unknown[] = [],
): HybridState {
  const serving = servingEFromTurn(turn, incoming);
  X.E = serving;
  X.servingE = serving;
  return X;
}

/** Write S onto an existing X. Mutates the state object. Not a new assembly. */
export function writeHybridS(X: HybridState, serving: CatalogPointer): HybridState {
  X.S = catalogPointer(serving.sku);
  return X;
}

/** Append this turn's messages onto existing X.H. Mutates the same object. */
export function writeHybridH(X: HybridState, messages: Message[]): HybridState {
  if (messages.length === 0) return X;
  X.H.push(...messages);
  return X;
}

/** Append this turn's traces onto existing X.M. Mutates the same object. */
export function writeHybridM(X: HybridState, traces: Trace[]): HybridState {
  if (traces.length === 0) return X;
  X.M.push(...traces);
  return X;
}

/**
 * X["39"] must already exist. Do not assemble a HybridState or invent S=0731.
 */
export function requireHybridX(
  store: Record<string, HybridState> | HybridStore,
  taskId: string,
): HybridState {
  const X = store instanceof Map ? store.get(taskId) : store[taskId];
  if (!X) {
    throw new Error(
      `X["${taskId}"] missing; will not assemble HybridState or invent S=0731`,
    );
  }
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
