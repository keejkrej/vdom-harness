export type NodeKind =
  | "agent"
  | "goal"
  | "tool"
  | "memory"
  | "channel"
  | "policy"
  | "supervisor"
  | "barrier"
  | "capability"
  | "adapter"
  | "artifact";

export type Persistence = "ephemeral" | "session" | "durable";

/**
 * Gated lifecycle for capability / adapter / artifact nodes.
 * proposed → validated (sandbox) → mounted | rejected; unmounted = rollback.
 */
export type MountStatus =
  | "proposed"
  | "validated"
  | "mounted"
  | "rejected"
  | "unmounted";

export type AgentNode = {
  key: string;
  kind?: NodeKind;
  role: string;
  objective: string;
  capabilities?: string[];
  model?: { intelligence?: string; context?: number; cost?: string } | string;
  budget?: { tokens?: number; dollars?: number; children?: number };
  inputs?: string[];
  outputs?: string[];
  persistence?: Persistence;
  prompt?: string;
  children?: AgentNode[];
  paper?: string;
  technique?: string;
  /** Lifecycle status for capability / adapter / artifact nodes. */
  status?: MountStatus;
  /**
   * Proposed capability source. Never executed until sandboxed + eval-gated.
   * Prefer `module:<id>` refs to pre-approved modules over raw code.
   */
  source?: string;
  /** Approved capability module id after sandbox validation. */
  moduleId?: string;
  /** Base model an adapter trains against / remaps. */
  modelRef?: string;
  /** Mounted adapter artifact id (rollback target via unmount). */
  adapterRef?: string;
  /** Generic artifact ref (weights URI, bundle path, HF job id, …). */
  artifactRef?: string;
};

export type AgentGraph = {
  id: string;
  version: number;
  root: AgentNode;
  meta?: Record<string, unknown>;
};

export type Trace = {
  nodeKey: string;
  role: string;
  input: string;
  output: string;
  ts: number;
};

export type FlatNode = {
  node: AgentNode;
  parentKey?: string;
};

export function node(spec: AgentNode): AgentNode {
  return { ...spec, kind: spec.kind ?? "agent" };
}

export function graph(
  idOrSpec: string | AgentGraph,
  version?: number,
  root?: AgentNode,
  meta?: Record<string, unknown>,
): AgentGraph {
  if (typeof idOrSpec === "object") return idOrSpec;
  if (version === undefined || root === undefined) {
    throw new Error("graph(id, version, root) requires version and root");
  }
  return { id: idOrSpec, version, root, meta };
}

export function flatten(g: AgentGraph): FlatNode[] {
  const out: FlatNode[] = [];
  const walk = (n: AgentNode, parentKey?: string) => {
    out.push(parentKey === undefined ? { node: n } : { node: n, parentKey });
    for (const child of n.children ?? []) walk(child, n.key);
  };
  walk(g.root);
  return out;
}

export function cloneNode(n: AgentNode): AgentNode {
  return structuredClone(n);
}

export function cloneGraph(g: AgentGraph): AgentGraph {
  return structuredClone(g);
}

export function findNode(g: AgentGraph, key: string): AgentNode | undefined {
  for (const { node: n } of flatten(g)) {
    if (n.key === key) return n;
  }
  return undefined;
}

export function kindOf(n: AgentNode): NodeKind {
  return n.kind ?? "agent";
}

/** Stable id for AgentNode.model — string as-is, or object.intelligence. */
export function modelId(model?: AgentNode["model"]): string | undefined {
  if (model == null) return undefined;
  if (typeof model === "string") {
    const id = model.trim();
    return id.length > 0 ? id : undefined;
  }
  const id = model.intelligence?.trim();
  return id && id.length > 0 ? id : undefined;
}
