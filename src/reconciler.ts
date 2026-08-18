import {
  type AgentGraph,
  type AgentNode,
  type Trace,
  flatten,
} from "./ir.js";

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
    (a.persistence ?? "ephemeral") !== (b.persistence ?? "ephemeral")
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
      this.current.set(op.node.key, {
        descriptor: op.node,
        status,
        traces: existing?.traces ?? [],
      });
    }
    this.prev = next;
    return result;
  }

  printOps(ops: ReconcileOp[]): void {
    for (const op of ops) console.log(`  ${formatOp(op)}`);
  }
}
