import {
  type AgentGraph,
  type AgentNode,
  type Trace,
  flatten,
  modelId,
} from "./ir.js";
import { type Provider, resolveProvider } from "./providers.js";
import { getCapability, type CapabilityFn } from "./capability.js";
import { getArtifact, type AdapterArtifact } from "./trainer.js";

export type ReconcileOp =
  | { op: "mount"; node: AgentNode; parentKey?: string }
  | { op: "update"; node: AgentNode; prev: AgentNode; parentKey?: string }
  | { op: "retain"; node: AgentNode; parentKey?: string }
  | { op: "unmount"; node: AgentNode; parentKey?: string };

export type ReconcileResult = {
  ops: ReconcileOp[];
  mounted: Map<string, AgentNode>;
};

export type PhysicalStatus = "mounted" | "updated" | "retained";

export type PhysicalNode = {
  descriptor: AgentNode;
  status: PhysicalStatus;
  traces: Trace[];
  /** Bound chat client for this node's model. Undefined when model is missing (runGraph uses its fallback). */
  provider?: Provider;
  /** Loaded capability runner when kind=capability and status=mounted. */
  capability?: CapabilityFn;
  /** Mounted adapter artifact snapshot (kind=adapter, status=mounted). */
  adapter?: AdapterArtifact;
};

const OP_MARK: Record<ReconcileOp["op"], string> = {
  mount: "+",
  update: "~",
  retain: "=",
  unmount: "-",
};

/** Props that, if changed, mark a node as update. Children are their own ops. */
export function propsChanged(a: AgentNode, b: AgentNode): boolean {
  return (
    a.role !== b.role ||
    a.objective !== b.objective ||
    a.prompt !== b.prompt ||
    (a.kind ?? "agent") !== (b.kind ?? "agent") ||
    JSON.stringify(a.capabilities ?? null) !== JSON.stringify(b.capabilities ?? null) ||
    JSON.stringify(a.model ?? null) !== JSON.stringify(b.model ?? null) ||
    JSON.stringify(a.budget ?? null) !== JSON.stringify(b.budget ?? null) ||
    (a.persistence ?? "ephemeral") !== (b.persistence ?? "ephemeral") ||
    (a.status ?? null) !== (b.status ?? null) ||
    (a.source ?? null) !== (b.source ?? null) ||
    (a.moduleId ?? null) !== (b.moduleId ?? null) ||
    (a.modelRef ?? null) !== (b.modelRef ?? null) ||
    (a.adapterRef ?? null) !== (b.adapterRef ?? null) ||
    (a.artifactRef ?? null) !== (b.artifactRef ?? null)
  );
}

export function reconcile(prev: AgentGraph | undefined, next: AgentGraph): ReconcileResult {
  const prevFlat = prev ? flatten(prev) : [];
  const prevByKey = new Map(prevFlat.map((f) => [f.node.key, f]));
  const nextFlat = flatten(next);
  const nextKeys = new Set(nextFlat.map((f) => f.node.key));

  const ops: ReconcileOp[] = [];
  const mounted = new Map<string, AgentNode>();

  for (const { node, parentKey } of nextFlat) {
    const prior = prevByKey.get(node.key);
    if (!prior) {
      ops.push(parentKey === undefined ? { op: "mount", node } : { op: "mount", node, parentKey });
    } else if (propsChanged(prior.node, node)) {
      ops.push(
        parentKey === undefined
          ? { op: "update", node, prev: prior.node }
          : { op: "update", node, prev: prior.node, parentKey },
      );
    } else {
      ops.push(parentKey === undefined ? { op: "retain", node } : { op: "retain", node, parentKey });
    }
    mounted.set(node.key, node);
  }

  for (const { node, parentKey } of [...prevFlat].reverse()) {
    if (!nextKeys.has(node.key)) {
      ops.push(parentKey === undefined ? { op: "unmount", node } : { op: "unmount", node, parentKey });
    }
  }

  return { ops, mounted };
}

export function formatOp(op: ReconcileOp): string {
  return `${OP_MARK[op.op]} ${op.node.key}`;
}

export function formatOps(ops: ReconcileOp[]): string {
  return ops.map((op) => `  ${formatOp(op)}`).join("\n");
}

function bindProvider(
  op: ReconcileOp,
  existing: PhysicalNode | undefined,
): Provider | undefined {
  if (op.op === "unmount") return undefined;

  const nextId = modelId(op.node.model);
  if (nextId == null) return undefined;

  if (op.op === "retain" && existing?.provider) {
    return existing.provider;
  }

  if (op.op === "update" && existing?.provider) {
    const prevId = modelId(op.prev.model);
    if (prevId === nextId) return existing.provider;
  }

  return resolveProvider(op.node.model);
}

function bindCapability(
  op: ReconcileOp,
  existing: PhysicalNode | undefined,
): CapabilityFn | undefined {
  if (op.op === "unmount") return undefined;
  const n = op.node;
  if ((n.kind ?? "agent") !== "capability") return existing?.capability;
  if (n.status !== "mounted" || !n.moduleId) return undefined;

  if (op.op === "retain" && existing?.capability && existing.descriptor.moduleId === n.moduleId) {
    return existing.capability;
  }

  return getCapability(n.moduleId)?.run;
}

function bindAdapter(
  op: ReconcileOp,
  existing: PhysicalNode | undefined,
): AdapterArtifact | undefined {
  if (op.op === "unmount") return undefined;
  const n = op.node;
  if ((n.kind ?? "agent") !== "adapter") return existing?.adapter;
  if (n.status !== "mounted" || !n.adapterRef) return undefined;

  if (
    op.op === "retain" &&
    existing?.adapter &&
    existing.descriptor.adapterRef === n.adapterRef
  ) {
    return existing.adapter;
  }

  return getArtifact(n.adapterRef);
}

export class RuntimeDOM {
  current = new Map<string, PhysicalNode>();
  private prev?: AgentGraph;

  reconcile(next: AgentGraph): ReconcileResult {
    const result = reconcile(this.prev, next);
    for (const op of result.ops) {
      if (op.op === "unmount") {
        this.current.delete(op.node.key);
        continue;
      }
      const status: PhysicalStatus =
        op.op === "mount" ? "mounted" : op.op === "update" ? "updated" : "retained";
      const existing = this.current.get(op.node.key);
      const provider = bindProvider(op, existing);
      const capability = bindCapability(op, existing);
      const adapter = bindAdapter(op, existing);
      this.current.set(op.node.key, {
        descriptor: op.node,
        status,
        traces: existing?.traces ?? [],
        provider,
        capability,
        adapter,
      });
    }
    this.prev = next;
    return result;
  }

  printOps(ops: ReconcileOp[]): void {
    for (const op of ops) console.log(`  ${formatOp(op)}`);
  }
}
