import { appendFileSync } from "node:fs";

/**
 * Agent-side debug trace. Set VDOM_TRACE=<file> to append one JSON line per
 * model request/response, tool execution, permission decision, compaction,
 * and MCP event. VDOM_TRACE_FULL=1 also records full message arrays.
 */
const file = () => process.env.VDOM_TRACE?.trim() || undefined;

export const traceFull = (): boolean => process.env.VDOM_TRACE_FULL === "1";

export function tracing(): boolean {
  return Boolean(file());
}

export function trace(event: string, data: Record<string, unknown>): void {
  const f = file();
  if (!f) return;
  try {
    appendFileSync(f, `${JSON.stringify({ t: new Date().toISOString(), event, ...data })}\n`);
  } catch {
    /* tracing must never break the agent */
  }
}

export function clip(s: string | null | undefined, max = 2000): string {
  if (!s) return "";
  return s.length > max ? `${s.slice(0, max)}…[+${s.length - max}]` : s;
}
