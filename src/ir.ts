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
  | "artifact";

export type Persistence = "ephemeral" | "session" | "durable";

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
