import { cloneGraph, findNode, flatten, node, type AgentGraph, type AgentNode, type Trace } from "../ir.js";
import { createProvider, type Provider } from "../providers.js";
import { reconcile } from "../reconciler.js";
import {
  applyILoop,
  diffOps,
  techniqueOfGraph,
  type ILoopResult,
} from "./tau2-improve.js";
import { serializeKernelC, sanitizeGraphText, stripGoldIds, missedToolNamesOnly } from "./tau2-kernel.js";
import { type Tau2Obs, type Tau2Technique } from "./tau2-types.js";

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
};

const PATCH_ROLES = new Set(["critic", "refine", "validator", "policy", "policy-checklist"]);

/**
 * Slow-clock self-Obs. The agent may change its own AgentGraph when the path
 * measure misses; wait when it hits. Do not invent reservation IDs. Do not
 * transfer the rewrite to a hidden host script.
 */
export const SELF_OBS_SYSTEM = `You are observing your own AgentGraph — kernel C — on the slow clock.
You may change your own AgentGraph when the path measure misses; wait when it hits.
Do not invent reservation IDs.
Do not transfer the rewrite to a hidden host script.
Return only JSON of the form {"action":"wait"|"I_loop","graphPatch":{"technique":"self-refine"|"validator"|"policy-checklist","nodes":[{"key":"...","role":"critic"|"refine"|"validator"|"policy","kind":"policy","objective":"...","prompt":"...","parentKey":"..."}]},"rationale":"..."}.
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

export function formatSelfObsUser(input: SelfObsInput): string {
  const rewards = rewardBits(input);
  const terms = (input.terminations ?? []).map(normalizeTermination);
  const fromObs = obsList(input.obs).flatMap((o) => missedToolNamesOnly(o.missedActions));
  const missed = [...new Set([...(input.missedToolNames ?? []), ...fromObs])];
  const tools = [...new Set(input.toolNames ?? [])];
  const traces = compactTraces(input.traces);
  const lastActions = obsList(input.obs).flatMap((o) => o.lastActions ?? []);
  return [
    "Current kernel C:",
    serializeKernelC(input.graph),
    "",
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
  };
}

function withFallback(graphBefore: AgentGraph, obs: SelfObsInput["obs"], rationale: string, raw?: string): SelfObsResult {
  const fallback = applyILoop(graphBefore, obs);
  return {
    ...fallback,
    path: "fallback",
    action: fallback.applied ? "I_loop" : "I_loop",
    rationale,
    servingPaused: false,
    raw,
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
      input.obs,
      `self-obs provider error; fallback (${err instanceof Error ? err.message : "error"})`,
    );
  }

  const parsed = parseSelfObsJson(raw);
  if (!parsed) {
    return withFallback(graphBefore, input.obs, "invalid self-obs JSON; fallback", raw);
  }
  if (parsed.action === "wait") {
    return waitResult(graphBefore, "self", parsed.rationale, raw);
  }

  const patched = applyGraphPatch(graphBefore, parsed.graphPatch);
  if (!patched) {
    return withFallback(graphBefore, input.obs, parsed.rationale ?? "unusable graphPatch; fallback", raw);
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
  };
}
