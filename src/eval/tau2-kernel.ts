import { flatten, type AgentGraph, type AgentNode } from "../ir.js";

/** Gold airline reservation IDs from τ² tasks. Never put these in prompts or kernel C. */
export const GOLD_RESERVATION_IDS = [
  "MSJ4OA",
  "S61CZX",
  "8C8K4E",
  "LU15PA",
  "UDMOP1",
  "XAZ3C0",
  "I6M8JQ",
  "4XGCCM",
  "NM1VX1",
  "H8Q05L",
  "KC18K6",
] as const;

export const GOLD_RESERVATION_RE = new RegExp(`\\b(?:${GOLD_RESERVATION_IDS.join("|")})\\b`, "gi");

/** Strip known gold reservation IDs and reservation_id write args from text. */
export function stripGoldIds(text: string): string {
  if (!text) return text;
  return text
    .replace(GOLD_RESERVATION_RE, "[redacted]")
    .replace(/reservation_id\s*[:=]\s*["']?[A-Z0-9_-]+["']?/gi, "reservation_id");
}

export function hasGoldReservationId(text: string): boolean {
  GOLD_RESERVATION_RE.lastIndex = 0;
  return GOLD_RESERVATION_RE.test(text);
}

export function missedToolNamesOnly(
  missed?: Array<{ name?: string; arguments?: Record<string, unknown> }> | null,
): string[] {
  if (!missed) return [];
  const names: string[] = [];
  for (const a of missed) {
    if (typeof a?.name === "string" && a.name.length > 0) names.push(a.name);
  }
  return names;
}

function compactObjective(n: AgentNode): string {
  const raw = stripGoldIds(n.objective ?? "").replace(/\s+/g, " ").trim();
  return raw.slice(0, 180);
}

function walkKernel(n: AgentNode, depth: number, lines: string[]): void {
  const pad = "  ".repeat(depth);
  const kind = n.kind && n.kind !== "agent" ? ` kind=${n.kind}` : "";
  const obj = compactObjective(n);
  lines.push(`${pad}${n.key} role=${n.role}${kind}${obj ? ` — ${obj}` : ""}`);
  for (const child of n.children ?? []) walkKernel(child, depth + 1, lines);
}

/**
 * Compact live AgentGraph for the serving system prompt.
 * Keys, roles, objectives, children. No gold reservation IDs, no gold write args.
 */
export function serializeKernelC(g: AgentGraph): string {
  const lines: string[] = [];
  walkKernel(g.root, 0, lines);
  return [
    `You are this AgentGraph (id=${g.id}); this is your current kernel C.`,
    "This turn is serving: follow the policy and tools. You do not mutate C here.",
    "<kernel>",
    lines.join("\n"),
    "</kernel>",
  ].join("\n");
}

/** Redact gold IDs from every node objective/prompt after a self-edit. */
export function sanitizeGraphText(g: AgentGraph): void {
  for (const { node: n } of flatten(g)) {
    n.objective = stripGoldIds(n.objective ?? "");
    if (n.prompt) n.prompt = stripGoldIds(n.prompt);
  }
}
