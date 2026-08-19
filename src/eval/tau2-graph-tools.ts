import { cloneGraph, flatten, type AgentGraph, type AgentNode } from "../ir.js";
import { type ToolCallOut, type ToolSpec } from "../providers.js";
import { reconcile } from "../reconciler.js";
import { diffOps, type GraphDiffOp } from "./tau2-improve.js";
import {
  compactGraphJson,
  GRAPH_SELF_TOOL_NAMES,
  hasGoldReservationId,
  isGraphSelfTool,
  sanitizeGraphText,
} from "./tau2-kernel.js";
import { applyGraphPatch, type GraphPatch } from "./tau2-self-obs.js";

export { GRAPH_SELF_TOOL_NAMES, isGraphSelfTool };

const BLOCKED_KINDS = new Set(["capability", "adapter", "artifact"]);

export const GET_AGENT_GRAPH_TOOL: ToolSpec = {
  name: "get_agent_graph",
  description:
    "Read your live AgentGraph (kernel C): keys, roles, objectives, children, technique. Compact JSON. No reservation IDs.",
  parameters: { type: "object", properties: {} },
};

export const SET_AGENT_GRAPH_TOOL: ToolSpec = {
  name: "set_agent_graph",
  description:
    "Rewrite your live AgentGraph via reconcile. Pass graph (full IR) or graphPatch {technique, nodes:[{key,role,objective,prompt,parentKey}], remove:[keys]}. You write the node prompts. Get before set. Do not include reservation IDs.",
  parameters: {
    type: "object",
    properties: {
      graph: { type: "object", description: "Full AgentGraph {id, version, root}" },
      graphPatch: {
        type: "object",
        description: "Patch: nodes to add/replace plus optional remove keys",
      },
    },
  },
};

export const GRAPH_SELF_TOOLS: ToolSpec[] = [GET_AGENT_GRAPH_TOOL, SET_AGENT_GRAPH_TOOL];

export function withGraphSelfTools(tools: ToolSpec[]): ToolSpec[] {
  const names = new Set(tools.map((t) => t.name));
  return [...GRAPH_SELF_TOOLS.filter((t) => !names.has(t.name)), ...tools];
}

export function gymToolCalls(calls?: ToolCallOut[]): ToolCallOut[] | undefined {
  const gym = (calls ?? []).filter((c) => !isGraphSelfTool(c.name));
  return gym.length > 0 ? gym : undefined;
}

export type GraphToolEdit = {
  tool: "get_agent_graph" | "set_agent_graph";
  applied: boolean;
  rejected: boolean;
  reason: string;
  diff?: GraphDiffOp[];
};

export type SetGraphResult =
  | { ok: true; graph: AgentGraph; diff: GraphDiffOp[]; reason: string }
  | { ok: false; reason: string };

function asValidGraph(raw: unknown): AgentGraph | undefined {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return undefined;
  const obj = raw as Partial<AgentGraph> & { root?: Partial<AgentNode> };
  if (!obj.root || typeof obj.root.key !== "string" || typeof obj.root.role !== "string") {
    return undefined;
  }
  const root = obj.root as AgentNode;
  if (typeof root.objective !== "string") root.objective = root.key;
  return {
    id: String(obj.id ?? "set-agent-graph"),
    version: Number.isFinite(Number(obj.version)) ? Number(obj.version) : 1,
    root,
    meta: obj.meta,
  };
}

function graphHasBlockedKind(g: AgentGraph): boolean {
  return flatten(g).some((f) => BLOCKED_KINDS.has(f.node.kind ?? ""));
}

function graphHasGold(g: AgentGraph): boolean {
  return flatten(g).some((f) => {
    const blob = `${f.node.key}\n${f.node.objective ?? ""}\n${f.node.prompt ?? ""}`;
    return hasGoldReservationId(blob);
  });
}

function removeNodeByKey(g: AgentGraph, key: string): boolean {
  if (g.root.key === key) return false;
  let removed = false;
  const walk = (n: AgentNode): void => {
    const kids = n.children ?? [];
    const next = kids.filter((c) => {
      if (c.key === key) {
        removed = true;
        return false;
      }
      return true;
    });
    n.children = next;
    for (const c of next) walk(c);
  };
  walk(g.root);
  return removed;
}

function asPatch(args: Record<string, unknown>): (GraphPatch & { remove?: string[] }) | undefined {
  const raw = args.graphPatch ?? args.patch;
  if (raw && typeof raw === "object" && !Array.isArray(raw)) {
    return raw as GraphPatch & { remove?: string[] };
  }
  if (Array.isArray(args.nodes) || Array.isArray(args.remove) || args.technique) {
    return args as GraphPatch & { remove?: string[] };
  }
  return undefined;
}

/** Apply set_agent_graph. Reject gold IDs and invalid IR. Model writes nodes — no canned checklist default. */
export function applySetAgentGraph(current: AgentGraph, args: Record<string, unknown>): SetGraphResult {
  const raw = JSON.stringify(args ?? {});
  if (hasGoldReservationId(raw)) {
    return { ok: false, reason: "rejected: payload contains gold reservation IDs" };
  }

  if (args.graph != null) {
    const next = asValidGraph(args.graph);
    if (!next) return { ok: false, reason: "rejected: not valid AgentGraph IR" };
    if (graphHasBlockedKind(next)) return { ok: false, reason: "rejected: blocked node kind" };
    if (graphHasGold(next)) return { ok: false, reason: "rejected: gold reservation IDs in graph" };
    sanitizeGraphText(next);
    next.version = current.version + 1;
    next.meta = { ...(next.meta ?? {}), selfEdit: true, intervention: "set_agent_graph" };
    const rec = reconcile(current, next);
    return { ok: true, graph: next, diff: diffOps(rec.ops), reason: "applied" };
  }

  const patch = asPatch(args);
  if (!patch) return { ok: false, reason: "rejected: not valid IR (need graph or graphPatch)" };

  let base = cloneGraph(current);
  let removed = false;
  for (const key of patch.remove ?? []) {
    if (typeof key === "string" && removeNodeByKey(base, key)) removed = true;
  }
  const patched =
    patch.nodes && patch.nodes.length > 0 ? applyGraphPatch(base, patch) : removed ? base : undefined;
  if (!patched) return { ok: false, reason: "rejected: unusable graphPatch" };
  if (graphHasGold(patched)) return { ok: false, reason: "rejected: gold reservation IDs in graph" };
  if (removed && !patch.nodes?.length) {
    patched.version = current.version + 1;
    patched.meta = { ...(patched.meta ?? {}), selfEdit: true, intervention: "set_agent_graph" };
  } else {
    patched.meta = { ...(patched.meta ?? {}), selfEdit: true, intervention: "set_agent_graph" };
  }
  const rec = reconcile(current, patched);
  return { ok: true, graph: patched, diff: diffOps(rec.ops), reason: "applied" };
}

export function executeGraphSelfTool(
  call: ToolCallOut,
  graph: AgentGraph,
): { content: string; graph: AgentGraph; edit: GraphToolEdit } {
  if (call.name === "get_agent_graph") {
    return {
      content: JSON.stringify(compactGraphJson(graph)),
      graph,
      edit: { tool: "get_agent_graph", applied: true, rejected: false, reason: "read" },
    };
  }
  const result = applySetAgentGraph(graph, call.arguments ?? {});
  if (!result.ok) {
    return {
      content: JSON.stringify({ ok: false, rejected: true, reason: result.reason }),
      graph,
      edit: { tool: "set_agent_graph", applied: false, rejected: true, reason: result.reason },
    };
  }
  return {
    content: JSON.stringify({
      ok: true,
      applied: true,
      reason: result.reason,
      diff: result.diff,
      graph: compactGraphJson(result.graph),
    }),
    graph: result.graph,
    edit: {
      tool: "set_agent_graph",
      applied: true,
      rejected: false,
      reason: result.reason,
      diff: result.diff,
    },
  };
}
