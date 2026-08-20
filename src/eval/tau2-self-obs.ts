import { cloneGraph, findNode, flatten, node, type AgentGraph, type AgentNode, type Trace } from "../ir.js";
import { createProvider, type Provider } from "../providers.js";
import { reconcile } from "../reconciler.js";
import {
  applyILoop,
  computeApplyScope,
  diffOps,
  recommendIntervention,
  REFUSED_GLOBAL_ILOOP,
  techniqueOfGraph,
  type ILoopResult,
} from "./tau2-improve.js";
import { serializeKernelC, sanitizeGraphText, stripGoldIds, missedToolNamesOnly } from "./tau2-kernel.js";
import { type ApplyScope, type Tau2Obs, type Tau2Technique } from "./tau2-types.js";

export type GraphPatchNode = {
  key: string;
  role?: string;
  kind?: string;
  objective?: string;
  prompt?: string;
  parentKey?: string;
};

export type GraphPatch = {
  technique?: Tau2Technique;
  nodes?: GraphPatchNode[];
  /** If set, C1 applies only to these task ids. Wait-hit is never included. */
  taskIds?: string[];
};

export type SelfObsDecision = {
  action: "wait" | "I_loop";
  graphPatch?: GraphPatch;
  rationale?: string;
};

export type SelfObsInput = {
  graph: AgentGraph;
  traces?: Array<Trace | Record<string, unknown>>;
  toolNames?: string[];
  rewards?: Array<number | null>;
  terminations?: string[];
  missedToolNames?: string[];
  taskIds?: string[];
  obs?: Tau2Obs | Tau2Obs[];
  provider?: Provider;
  model?: string;
};

export type SelfObsResult = ILoopResult & {
  path: "self" | "fallback";
  action: "wait" | "I_loop";
  rationale?: string;
  servingPaused: false;
  raw?: string;
  applyScope?: ApplyScope;
};

const PATCH_ROLES = new Set(["critic", "refine", "validator", "policy", "policy-checklist"]);

/**
 * Slow-clock self-Obs. The agent may change its own AgentGraph when the path
 * measure misses; wait when it hits. Do not invent reservation IDs. Do not
 * transfer the rewrite to a hidden host script.
 */
export const SELF_OBS_WAIT_HIT_RULES = `Each episode lists taskId, reward, arm, hung, and write tool names (never reservation IDs).
If arm is wait and reward is 1, do not infer a missed cancel/update for that task.
If hung or incomplete (timeout / crash / no-write), arm is I_sku (catalog rebind, not I_weight-as-trainer); do not mount I_loop on those tasks.
Do not mount a cancel-always / upgrade-always node that applies to wait-hit tasks.
If you cannot write a gated patch (graphPatch.taskIds that leave wait-hit tasks on C0), return {"action":"wait"}.
Never include reservation IDs in the patch.
A mixed batch (wait-hit plus miss) must not apply a global C mutation to the wait-hit tasks.`;

export const SELF_OBS_SYSTEM = `You are observing your own AgentGraph — kernel C — on the slow clock.
You may change your own AgentGraph when the path measure misses; wait when it hits.
Do not invent reservation IDs.
Do not transfer the rewrite to a hidden host script.
${SELF_OBS_WAIT_HIT_RULES}
Return only JSON of the form {"action":"wait"|"I_loop","graphPatch":{"technique":"self-refine"|"validator"|"policy-checklist","taskIds":["miss-task-id"],"nodes":[{"key":"...","role":"critic"|"refine"|"validator"|"policy","kind":"policy","objective":"...","prompt":"...","parentKey":"..."}]},"rationale":"..."}.
If action is I_loop, you write the new critic/refine/validator/policy node prompt text yourself from this graph and these traces. The host must not fill a canned checklist for you.`;

function obsList(obs?: Tau2Obs | Tau2Obs[] | null): Tau2Obs[] {
  if (!obs) return [];
  return Array.isArray(obs) ? obs : [obs];
}

function rewardBits(input: SelfObsInput): Array<0 | 1> {
  if (input.rewards && input.rewards.length > 0) {
    return input.rewards.map((r) => (r != null && r >= 1 - 1e-6 ? 1 : 0));
  }
  return obsList(input.obs).map((o) => (o.nSuccessProxy === 1 ? 1 : 0));
}

function normalizeTermination(raw: string): string {
  const s = raw.toLowerCase();
  if (s.includes("user_stop") || s.includes("user-stop")) return "user_stop";
  if (s.includes("transfer")) return "transfer";
  if (s.includes("timeout") || s.includes("hung")) return "timeout";
  return s.slice(0, 40);
}

function compactTraces(traces?: Array<Trace | Record<string, unknown>>): string {
  if (!traces || traces.length === 0) return "";
  return traces
    .slice(0, 24)
    .map((t) => {
      const rec = t as Record<string, unknown>;
      const key = String(rec.nodeKey ?? rec.role ?? "?");
      const out = stripGoldIds(String(rec.output ?? "")).slice(0, 160);
      return `- ${key}: ${out}`;
    })
    .join("\n");
}

function writeToolNames(obs: Tau2Obs): string {
  const names = (obs.lastActions ?? [])
    .filter((a) => a && !a.startsWith("text:"))
    .map((a) => stripGoldIds(a));
  return [...new Set(names)].join(", ") || "none";
}

export function episodesFromInput(input: SelfObsInput): Tau2Obs[] {
  const list = obsList(input.obs).map((o, i) => ({
    ...o,
    taskId: o.taskId || input.taskIds?.[i],
  }));
  if (list.length > 0) return list;
  const rewards = input.rewards ?? [];
  return rewards.map((r, i) => {
    const hit = r != null && r >= 1 - 1e-6;
    const row: Tau2Obs = {
      nSteps: 0,
      nSuccessProxy: hit ? 1 : 0,
      lastActions: [],
      channels: [],
      critique: "",
      toolFailures: 0,
      repeatActions: 0,
      hung: false,
      taskId: input.taskIds?.[i],
    };
    if (input.terminations?.[i]) row.termination = input.terminations[i];
    if (!hit && (input.missedToolNames ?? []).length > 0) {
      row.missedActions = input.missedToolNames!.map((name) => ({ name }));
    }
    row.arm = recommendIntervention(row);
    return row;
  });
}

function formatEpisodes(input: SelfObsInput): string {
  const episodes = episodesFromInput(input);
  if (episodes.length === 0) return "";
  const lines = episodes.map((o, i) => {
    const id = o.taskId || String(i);
    const reward = o.nSuccessProxy;
    const arm = o.arm ?? recommendIntervention(o);
    return `- taskId=${id} reward=${reward} arm=${arm} hung=${Boolean(o.hung)} writeTools=${writeToolNames(o)}`;
  });
  return `Episodes:\n${lines.join("\n")}\n${SELF_OBS_WAIT_HIT_RULES}`;
}

export function formatSelfObsUser(input: SelfObsInput): string {
  const rewards = rewardBits(input);
  const terms = (input.terminations ?? []).map(normalizeTermination);
  const episodes = episodesFromInput(input);
  const fromObs = episodes.flatMap((o) => missedToolNamesOnly(o.missedActions));
  const missed = [...new Set([...(input.missedToolNames ?? []), ...fromObs])];
  const tools = [...new Set(input.toolNames ?? [])];
  const traces = compactTraces(input.traces);
  const lastActions = episodes.flatMap((o) => o.lastActions ?? []);
  return [
    "Current kernel C:",
    serializeKernelC(input.graph),
    "",
    formatEpisodes(input),
    `Official reward (0/1): ${rewards.length > 0 ? rewards.join(",") : "unknown"}`,
    `Termination (user_stop|transfer|timeout): ${terms.join(",") || "unknown"}`,
    `Tool names used: ${tools.length > 0 ? tools.join(", ") : lastActions.filter((a) => !a.startsWith("text:")).join(", ") || "none"}`,
    `Missed tool names (not args, not IDs): ${missed.length > 0 ? missed.join(", ") : "none"}`,
    traces ? `Traces:\n${traces}` : "",
  ]
    .filter((line) => line.length > 0)
    .join("\n");
}

export function parseSelfObsJson(raw: string): SelfObsDecision | undefined {
  const fence = raw.match(/```(?:json)?\s*([\s\S]*?)```/);
  const text = fence?.[1] ?? raw;
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start < 0 || end < 0) return undefined;
  try {
    const obj = JSON.parse(text.slice(start, end + 1)) as Record<string, unknown>;
    if (obj.action !== "wait" && obj.action !== "I_loop") return undefined;
    const patch = obj.graphPatch;
    const graphPatch =
      patch && typeof patch === "object" && !Array.isArray(patch) ? (patch as GraphPatch) : undefined;
    if (graphPatch && !graphPatch.taskIds) {
      const topIds = obj.taskIds;
      if (Array.isArray(topIds)) {
        graphPatch.taskIds = topIds.filter((x): x is string => typeof x === "string");
      }
    }
    if (graphPatch?.taskIds) {
      graphPatch.taskIds = graphPatch.taskIds.filter((x) => typeof x === "string" && x.length > 0);
    }
    return {
      action: obj.action,
      graphPatch,
      rationale: typeof obj.rationale === "string" ? obj.rationale : undefined,
    };
  } catch {
    return undefined;
  }
}

function allowedPatchNode(spec: GraphPatchNode): boolean {
  if (typeof spec.key !== "string" || spec.key.length === 0) return false;
  const role = (spec.role ?? spec.key).toLowerCase();
  const kind = (spec.kind ?? "").toLowerCase();
  return PATCH_ROLES.has(role) || kind === "policy" || spec.key === "policy-checklist";
}

function inferTechnique(g: AgentGraph): Tau2Technique {
  const nodes = flatten(g).map((f) => f.node);
  if (
    nodes.some((n) => n.key === "policy-checklist" || n.kind === "policy" || n.role === "policy")
  ) {
    return "policy-checklist";
  }
  if (nodes.some((n) => n.key === "validator" || n.role === "validator")) return "validator";
  if (nodes.some((n) => n.key === "critic" || n.role === "critic" || n.role === "refine")) {
    return "self-refine";
  }
  return techniqueOfGraph(g);
}

/** Apply a self-Obs graphPatch. Host gate: no gold IDs; only critic/refine/validator/policy nodes. */
export function applyGraphPatch(g: AgentGraph, patch?: GraphPatch): AgentGraph | undefined {
  if (!patch || !Array.isArray(patch.nodes) || patch.nodes.length === 0) return undefined;
  const next = cloneGraph(g);
  let changed = false;
  for (const spec of patch.nodes) {
    if (!allowedPatchNode(spec)) continue;
    const objective = stripGoldIds(spec.objective ?? spec.prompt ?? "");
    const prompt = spec.prompt != null ? stripGoldIds(spec.prompt) : undefined;
    const existing = findNode(next, spec.key);
    if (existing) {
      if (spec.objective != null) existing.objective = objective;
      if (prompt != null) existing.prompt = prompt;
      if (spec.role) existing.role = spec.role;
      if (spec.kind) existing.kind = spec.kind as AgentNode["kind"];
      changed = true;
      continue;
    }
    const role = spec.role ?? spec.key;
    const kind =
      (spec.kind as AgentNode["kind"] | undefined) ??
      (role === "policy" || spec.key === "policy-checklist" ? "policy" : "agent");
    const created = node({
      key: spec.key,
      role,
      kind,
      objective: objective || spec.key,
      ...(prompt != null ? { prompt } : {}),
      technique: patch.technique,
    });
    const parent = spec.parentKey ? findNode(next, spec.parentKey) : undefined;
    const host = parent ?? next.root;
    host.children = [...(host.children ?? []), created];
    changed = true;
  }
  if (!changed) return undefined;
  sanitizeGraphText(next);
  const technique = patch.technique ?? inferTechnique(next);
  next.version = g.version + 1;
  next.id = `${g.id}-${technique}`;
  next.meta = { ...(next.meta ?? {}), technique, intervention: "I_loop", selfEdit: true };
  return next;
}

function waitResult(
  graphBefore: AgentGraph,
  path: "self" | "fallback",
  rationale?: string,
  raw?: string,
  applyScope?: ApplyScope,
): SelfObsResult {
  const technique = techniqueOfGraph(graphBefore);
  return {
    arm: "I_loop",
    applied: false,
    techniqueBefore: technique,
    techniqueAfter: technique,
    graphBefore,
    graphAfter: graphBefore,
    graphDiff: [],
    path,
    action: "wait",
    rationale,
    servingPaused: false,
    raw,
    applyScope,
  };
}

function withFallback(
  graphBefore: AgentGraph,
  input: SelfObsInput,
  rationale: string,
  raw?: string,
): SelfObsResult {
  const fallback = applyILoop(graphBefore, episodesFromInput(input));
  return {
    ...fallback,
    path: "fallback",
    action: fallback.applied ? "I_loop" : fallback.action === "wait" ? "wait" : "I_loop",
    rationale: fallback.rationale === REFUSED_GLOBAL_ILOOP ? REFUSED_GLOBAL_ILOOP : rationale,
    servingPaused: false,
    raw,
    applyScope: fallback.applyScope,
  };
}

/**
 * Real LLM Obs (same 0731 provider as serving). Host applies only if JSON parses.
 * Invalid JSON → canned host ladder (AIRLINE_POLICY_CHECKLIST / self-refine / validator).
 */
export async function runSelfObs(input: SelfObsInput): Promise<SelfObsResult> {
  const provider = input.provider ?? createProvider();
  const graphBefore = input.graph;
  const user = formatSelfObsUser(input);
  let raw = "";
  try {
    raw = await provider.complete(
      [
        { role: "system", content: SELF_OBS_SYSTEM },
        { role: "user", content: user },
      ],
      { role: "self-obs", model: input.model },
    );
  } catch (err) {
    return withFallback(
      graphBefore,
      input,
      `self-obs provider error; fallback (${err instanceof Error ? err.message : "error"})`,
    );
  }

  const parsed = parseSelfObsJson(raw);
  if (!parsed) {
    return withFallback(graphBefore, input, "invalid self-obs JSON; fallback", raw);
  }
  const episodes = episodesFromInput(input);
  const applyScope = computeApplyScope(episodes, {
    patchTaskIds: parsed.graphPatch?.taskIds,
    taskIds: input.taskIds,
  });
  if (parsed.action === "wait") {
    return waitResult(graphBefore, "self", parsed.rationale, raw, applyScope);
  }

  if (applyScope.looped.length === 0 && episodes.length > 0) {
    return waitResult(
      graphBefore,
      "self",
      applyScope.waitKept.length > 0 ? REFUSED_GLOBAL_ILOOP : "incomplete licenses I_sku; no I_loop",
      raw,
      applyScope,
    );
  }

  const patched = applyGraphPatch(graphBefore, parsed.graphPatch);
  if (!patched) {
    return withFallback(graphBefore, input, parsed.rationale ?? "unusable graphPatch; fallback", raw);
  }

  const rec = reconcile(graphBefore, patched);
  return {
    arm: "I_loop",
    applied: true,
    techniqueBefore: techniqueOfGraph(graphBefore),
    techniqueAfter: techniqueOfGraph(patched),
    graphBefore,
    graphAfter: patched,
    graphDiff: diffOps(rec.ops),
    path: "self",
    action: "I_loop",
    rationale: parsed.rationale,
    servingPaused: false,
    raw,
    applyScope,
  };
}
