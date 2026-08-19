import {
  type AgentGraph,
  type AgentNode,
  type MountStatus,
  cloneGraph,
  findNode,
  flatten,
  graph,
  node,
} from "./ir.js";
import {
  validateCapability,
  mountCapability,
  rejectCapability,
  getCapability,
} from "./capability.js";
import {
  type AdapterArtifact,
  getArtifact,
  registerArtifact,
  proposeAdapter,
} from "./trainer.js";
import { type Provider } from "./providers.js";
import { type Task, runBenchmark } from "./benchmarks.js";
import { RuntimeDOM } from "./reconciler.js";

export type GateDecision =
  | {
      action: "mount";
      score: number;
      graph: AgentGraph;
      previous?: AgentGraph;
    }
  | {
      action: "reject";
      score: number;
      reason: string;
      graph: AgentGraph;
      previous: AgentGraph;
    }
  | {
      action: "rollback";
      score: number;
      reason: string;
      graph: AgentGraph;
      previous: AgentGraph;
    };

function bump(g: AgentGraph, idSuffix: string, meta?: Record<string, unknown>): AgentGraph {
  return graph({
    id: `${g.id}-${idSuffix}`,
    version: g.version + 1,
    root: g.root,
    meta: { ...(g.meta ?? {}), ...(meta ?? {}) },
  });
}

function replaceNode(g: AgentGraph, key: string, next: AgentNode): AgentGraph {
  const copy = cloneGraph(g);
  const walk = (n: AgentNode): AgentNode => {
    if (n.key === key) return { ...next, children: next.children ?? n.children };
    return { ...n, children: n.children?.map(walk) };
  };
  copy.root = walk(copy.root);
  return copy;
}

function setNodeModel(g: AgentGraph, key: string, model: AgentNode["model"]): AgentGraph {
  const copy = cloneGraph(g);
  const walk = (n: AgentNode): AgentNode => {
    if (n.key === key) return { ...n, model };
    return { ...n, children: n.children?.map(walk) };
  };
  copy.root = walk(copy.root);
  return copy;
}

/**
 * Attach a capability as the graph executor (replace root) or as a child of root.
 */
export function attachCapability(
  base: AgentGraph,
  cap: AgentNode,
  mode: "replace-root" | "child-of-root" = "replace-root",
): AgentGraph {
  const copy = cloneGraph(base);
  if (mode === "replace-root") {
    copy.root = { ...cap, children: cap.children ?? [] };
  } else {
    copy.root = {
      ...copy.root,
      children: [...(copy.root.children ?? []), cap],
    };
  }
  return bump(copy, "cap-candidate", { capabilityGate: true, capabilityKey: cap.key });
}

/**
 * Eval-gate a proposed capability: sandbox → candidate reconcile → benchmark → mount | reject.
 * Scientist-emitted source never runs unless sandboxValidate accepts it.
 */
export async function gateCapability(opts: {
  base: AgentGraph;
  proposal: AgentNode;
  task: Task;
  provider: Provider;
  threshold?: number;
  attachMode?: "replace-root" | "child-of-root";
  dom?: RuntimeDOM;
}): Promise<GateDecision> {
  const threshold = opts.threshold ?? 1;
  const validated = validateCapability(opts.proposal);

  if (validated.status === "rejected" || !validated.moduleId) {
    return {
      action: "reject",
      score: 0,
      reason: "sandbox rejected capability source",
      graph: opts.base,
      previous: opts.base,
    };
  }

  if (!getCapability(validated.moduleId)) {
    return {
      action: "reject",
      score: 0,
      reason: `module not registered: ${validated.moduleId}`,
      graph: opts.base,
      previous: opts.base,
    };
  }

  // Eval on a scratch DOM with status=mounted so the module actually runs.
  // Live graph is untouched until the score clears the threshold.
  const evalNode = mountCapability({ ...validated, status: "validated" });
  const candidate = attachCapability(opts.base, evalNode, opts.attachMode);

  const scratch = new RuntimeDOM();
  const bench = await runBenchmark(candidate, opts.task, opts.provider, scratch);

  if (bench.score < threshold) {
    return {
      action: "reject",
      score: bench.score,
      reason: `eval score ${bench.score} < ${threshold}`,
      graph: opts.base,
      previous: replaceNode(candidate, validated.key, rejectCapability(validated)),
    };
  }

  const live = bump(candidate, "cap-mounted", {
    capabilityGate: "mounted",
    capabilityKey: evalNode.key,
    moduleId: evalNode.moduleId,
  });

  if (opts.dom) opts.dom.reconcile(live);
  return { action: "mount", score: bench.score, graph: live, previous: opts.base };
}

/**
 * Build a candidate graph that mounts an adapter and retargets an agent's model pointer.
 */
export function attachAdapter(
  base: AgentGraph,
  artifact: AdapterArtifact,
  targetKey: string,
  adapterKey = "adapter",
): { candidate: AgentGraph; previousModel: AgentNode["model"] } {
  const target = findNode(base, targetKey);
  if (!target) throw new Error(`attachAdapter: missing target ${targetKey}`);
  const previousModel = target.model;

  let copy = setNodeModel(cloneGraph(base), targetKey, artifact.resultModelId);
  const adapterNode = {
    ...proposeAdapter({ key: adapterKey, artifact }),
    status: "validated" as MountStatus,
  };
  copy.root = {
    ...copy.root,
    children: [...(copy.root.children ?? []), adapterNode],
  };
  copy = bump(copy, "adapter-candidate", {
    adapterGate: true,
    adapterRef: artifact.id,
    targetKey,
    previousModel,
  });
  return { candidate: copy, previousModel };
}

/**
 * Eval-gate an adapter: retarget model → bench → mount | reject.
 * On reject, the previous model pointer is preserved (no live mount).
 *
 * Slow clock of I_weight: the candidate is scored on a scratch DOM. The fast
 * clock keeps serving old θ (`servingPaused` stays false). Mount only if the
 * held-out score does not regress; otherwise reject and keep the live graph.
 */
export async function gateAdapter(opts: {
  base: AgentGraph;
  artifact: AdapterArtifact;
  targetKey: string;
  task: Task;
  provider: Provider;
  threshold?: number;
  adapterKey?: string;
  dom?: RuntimeDOM;
}): Promise<GateDecision> {
  const threshold = opts.threshold ?? 1;
  if (!getArtifact(opts.artifact.id)) {
    registerArtifact(opts.artifact);
  }

  const adapterKey = opts.adapterKey ?? "adapter";
  const { candidate, previousModel } = attachAdapter(
    opts.base,
    opts.artifact,
    opts.targetKey,
    adapterKey,
  );

  const scratch = new RuntimeDOM();
  const bench = await runBenchmark(candidate, opts.task, opts.provider, scratch);

  if (bench.score < threshold) {
    return {
      action: "reject",
      score: bench.score,
      reason: `adapter eval score ${bench.score} < ${threshold}`,
      graph: opts.base,
      previous: candidate,
    };
  }

  const mountedAdapter = node({
    key: adapterKey,
    kind: "adapter",
    role: "adapter",
    objective: `Mounted ${opts.artifact.technique}`,
    status: "mounted",
    adapterRef: opts.artifact.id,
    modelRef: opts.artifact.baseModel,
    artifactRef: opts.artifact.uri ?? opts.artifact.id,
    technique: opts.artifact.technique,
  });

  let live = setNodeModel(cloneGraph(opts.base), opts.targetKey, opts.artifact.resultModelId);
  live.root = {
    ...live.root,
    children: [
      ...(live.root.children ?? []).filter((c) => c.key !== adapterKey),
      mountedAdapter,
    ],
  };
  live = bump(live, "adapter-mounted", {
    adapterGate: "mounted",
    adapterRef: opts.artifact.id,
    targetKey: opts.targetKey,
    previousModel,
    resultModelId: opts.artifact.resultModelId,
  });

  if (opts.dom) opts.dom.reconcile(live);
  return { action: "mount", score: bench.score, graph: live, previous: opts.base };
}

/**
 * Unmount an adapter and restore the previous model pointer (rollback).
 */
export function rollbackAdapter(
  current: AgentGraph,
  opts: {
    adapterKey: string;
    targetKey: string;
    previousModel?: AgentNode["model"];
  },
): AgentGraph {
  const copy = cloneGraph(current);
  const strip = (n: AgentNode): AgentNode => ({
    ...n,
    model: n.key === opts.targetKey ? opts.previousModel : n.model,
    children: (n.children ?? [])
      .filter((c) => c.key !== opts.adapterKey)
      .map(strip),
  });
  copy.root = strip(copy.root);
  return bump(copy, "adapter-rollback", {
    adapterGate: "rollback",
    adapterKey: opts.adapterKey,
    targetKey: opts.targetKey,
  });
}

/**
 * After a mounted adapter fails a later eval, unmount and restore previous model.
 */
export async function unmountAdapterOnFailure(opts: {
  current: AgentGraph;
  adapterKey: string;
  targetKey: string;
  previousModel?: AgentNode["model"];
  task: Task;
  provider: Provider;
  dom?: RuntimeDOM;
}): Promise<GateDecision> {
  const previousModel =
    opts.previousModel ??
    (opts.current.meta?.previousModel as AgentNode["model"] | undefined);

  const rolled = rollbackAdapter(opts.current, {
    adapterKey: opts.adapterKey,
    targetKey: opts.targetKey,
    previousModel,
  });

  if (opts.dom) opts.dom.reconcile(rolled);
  const bench = await runBenchmark(opts.current, opts.task, opts.provider);
  return {
    action: "rollback",
    score: bench.score,
    reason: "adapter unmounted; model pointer restored",
    graph: rolled,
    previous: opts.current,
  };
}

/** Inspect whether any capability/adapter in the graph is live-mounted. */
export function mountedImprovementKeys(g: AgentGraph): {
  capabilities: string[];
  adapters: string[];
} {
  const capabilities: string[] = [];
  const adapters: string[] = [];
  for (const { node: n } of flatten(g)) {
    if (n.status !== "mounted") continue;
    if ((n.kind ?? "") === "capability") capabilities.push(n.key);
    if ((n.kind ?? "") === "adapter") adapters.push(n.key);
  }
  return { capabilities, adapters };
}
