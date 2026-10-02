import { createHash, randomBytes } from "node:crypto";
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

/**
 * Canonical session history: one append-only, uncompressed JSONL file per
 * session (`events.jsonl`). Pi-style id/parentId tree + DSH-style turn/step
 * brackets and log-only events. Field names map onto OTel GenAI / ATIF for
 * export. The resume snapshot, transcripts, and findings are derived from it.
 */

export type EventType =
  | "session"
  | "config.change"
  | "turn.start"
  | "turn.end"
  | "user.message"
  | "llm.request"
  | "assistant.message"
  | "llm.error"
  | "tool.call"
  | "permission.request"
  | "permission.decision"
  | "tool.result"
  | "compaction.start"
  | "compaction.end"
  | "subagent.start"
  | "subagent.end"
  | "feedback"
  | "issue"
  | "lesson"
  | "guard"
  | "error";

export type HistoryEvent = {
  v: 1;
  seq: number;
  id: string;
  parentId: string | null;
  ts: string;
  type: EventType;
  turn: number;
  step?: number;
  /** "main" or "subagent:<label>" */
  who?: string;
  data: Record<string, unknown>;
};

const BLOB_THRESHOLD = 16 * 1024;

export class EventLog {
  private seq = 0;
  private last: string | null = null;
  turn = 0;
  readonly file: string;

  constructor(readonly dir: string) {
    mkdirSync(dir, { recursive: true });
    this.file = join(dir, "events.jsonl");
    if (existsSync(this.file)) {
      const evs = readEvents(dir);
      const tail = evs[evs.length - 1];
      if (tail) {
        this.seq = tail.seq;
        this.last = tail.id;
        this.turn = Math.max(...evs.map((e) => e.turn));
      }
    }
  }

  append(type: EventType, data: Record<string, unknown>, extra: { step?: number; who?: string } = {}): string {
    const id = `e_${randomBytes(4).toString("hex")}`;
    const ev: HistoryEvent = {
      v: 1,
      seq: ++this.seq,
      id,
      parentId: this.last,
      ts: new Date().toISOString(),
      type,
      turn: this.turn,
      ...(extra.step !== undefined ? { step: extra.step } : {}),
      ...(extra.who && extra.who !== "main" ? { who: extra.who } : {}),
      data,
    };
    try {
      appendFileSync(this.file, `${JSON.stringify(ev)}\n`);
    } catch (err) {
      process.stderr.write(`vdom: history append failed: ${String(err)}\n`);
    }
    this.last = id;
    return id;
  }

  /** Large outputs go to content-addressed blobs; the event keeps a preview + ref. */
  blob(text: string): { preview: string; blobRef?: string; bytes: number } {
    const bytes = Buffer.byteLength(text);
    if (bytes <= BLOB_THRESHOLD) return { preview: text, bytes };
    const ref = createHash("sha256").update(text).digest("hex").slice(0, 32);
    const dir = join(this.dir, "blobs");
    mkdirSync(dir, { recursive: true });
    const p = join(dir, ref);
    if (!existsSync(p)) writeFileSync(p, text);
    return { preview: `${text.slice(0, 4000)}\n…\n${text.slice(-4000)}`, blobRef: ref, bytes };
  }
}

export function readEvents(dir: string): HistoryEvent[] {
  const file = join(dir, "events.jsonl");
  if (!existsSync(file)) return [];
  const out: HistoryEvent[] = [];
  for (const line of readFileSync(file, "utf8").split("\n")) {
    if (!line.trim()) continue;
    try {
      out.push(JSON.parse(line) as HistoryEvent);
    } catch {
      /* torn last line after a crash */
    }
  }
  return out;
}

// ------------------------------------------------------------------ findings

export type Finding = {
  kind:
    | "tool_error"
    | "unknown_tool"
    | "unknown_argument"
    | "declined"
    | "loop"
    | "overflow"
    | "compaction_ineffective"
    | "compaction_thrash"
    | "orphan_tool_call"
    | "llm_error"
    | "harness_error"
    | "user_frustration"
    | "unverified_claim"
    | "truncated_output";
  severity: "high" | "medium" | "low";
  /** harness = likely a vdom bug; model = agent behaviour; env = provider/network/user setup. */
  blame: "harness" | "model" | "env" | "unknown";
  turn: number;
  message: string;
  events: string[];
};

const s = (v: unknown): string => (typeof v === "string" ? v : v === undefined || v === null ? "" : JSON.stringify(v));

/** Fold a session log into findings. Pure and deterministic. */
export function analyze(all: HistoryEvent[]): Finding[] {
  // The diagnostician's own activity is not part of the session's behaviour.
  const events = all.filter((e) => e.who !== "diagnosis");
  const out: Finding[] = [];
  const calls = new Map<string, HistoryEvent>();
  const results = new Set<string>();
  const recent: { sig: string; id: string; turn: number }[] = [];
  const compactions: HistoryEvent[] = [];
  let lastEditTurnStep: { turn: number; seq: number } | undefined;
  let lastVerifySeq = 0;

  for (const e of events) {
    const d = e.data;
    switch (e.type) {
      case "tool.call": {
        calls.set(s(d.toolCallId), e);
        const sig = `${s(d.name)}:${s(d.arguments)}`;
        recent.push({ sig, id: e.id, turn: e.turn });
        const same = recent.slice(-3);
        if (same.length === 3 && same.every((x) => x.sig === sig)) {
          out.push({ kind: "loop", severity: "medium", blame: "model", turn: e.turn, message: `3 identical ${s(d.name)} calls in a row`, events: same.map((x) => x.id) });
        }
        if (Array.isArray(d.ignoredArgs) && d.ignoredArgs.length) {
          out.push({
            kind: "unknown_argument",
            severity: "medium",
            blame: "harness",
            turn: e.turn,
            message: `${s(d.name)} received arguments it does not support and ignored them: ${(d.ignoredArgs as string[]).join(", ")}`,
            events: [e.id],
          });
        }
        break;
      }
      case "tool.result": {
        const id = s(d.toolCallId);
        results.add(id);
        const call = calls.get(id);
        const name = s(call?.data.name ?? d.name);
        const status = s(d.status);
        if (status === "error") {
          const err = (d.error ?? {}) as Record<string, unknown>;
          const type = s(err.type);
          const blame: Finding["blame"] = type === "unknown_tool" || type === "invalid_args" || type === "input" ? "model" : type === "exception" ? "harness" : "unknown";
          out.push({
            kind: type === "unknown_tool" ? "unknown_tool" : "tool_error",
            severity: type === "exception" ? "high" : "low",
            blame,
            turn: e.turn,
            message: `${name}: ${s(err.message).slice(0, 300)}`,
            events: [call?.id ?? e.id, e.id],
          });
        } else if (status === "denied") {
          out.push({ kind: "declined", severity: "low", blame: "model", turn: e.turn, message: `${name} was declined by the user`, events: [e.id] });
        }
        if (d.truncated) out.push({ kind: "truncated_output", severity: "low", blame: "unknown", turn: e.turn, message: `${name} output truncated (${s(d.outputBytes)} bytes)`, events: [e.id] });
        if ((name === "edit" || name === "write") && status === "ok") lastEditTurnStep = { turn: e.turn, seq: e.seq };
        if (name === "bash" && status === "ok" && /\b(test|tsc|build|lint|check|pytest|cargo|go test|vitest|jest)\b/.test(s(call?.data.arguments))) lastVerifySeq = e.seq;
        break;
      }
      case "llm.error": {
        const type = s(d.errorType);
        if (type === "context_overflow") out.push({ kind: "overflow", severity: "medium", blame: "harness", turn: e.turn, message: `context overflow before compaction: ${s(d.message).slice(0, 200)}`, events: [e.id] });
        else if (!d.willRetry) out.push({ kind: "llm_error", severity: "high", blame: type === "auth" || type === "billing" ? "env" : "unknown", turn: e.turn, message: s(d.message).slice(0, 300), events: [e.id] });
        break;
      }
      case "compaction.end": {
        compactions.push(e);
        const before = Number(d.tokensBefore ?? 0);
        const after = Number(d.tokensAfter ?? 0);
        if (d.ok && before > 0 && after / before > 0.7) {
          out.push({ kind: "compaction_ineffective", severity: "medium", blame: "harness", turn: e.turn, message: `compaction only reduced ${before} → ${after} tokens`, events: [e.id] });
        }
        const prev = compactions[compactions.length - 2];
        if (prev && e.turn - prev.turn <= 1 && d.trigger !== "manual") {
          out.push({ kind: "compaction_thrash", severity: "medium", blame: "harness", turn: e.turn, message: "two automatic compactions within one turn", events: [prev.id, e.id] });
        }
        break;
      }
      case "error":
        out.push({ kind: "harness_error", severity: "high", blame: "harness", turn: e.turn, message: `${s(d.scope)}: ${s(d.message).slice(0, 300)}`, events: [e.id] });
        break;
      case "feedback":
        out.push({ kind: "user_frustration", severity: "high", blame: "unknown", turn: Number(d.targetTurn ?? e.turn), message: `user signalled a bad turn ${s(d.targetTurn)}: "${s(d.text).slice(0, 120)}" (${s(d.signal)})`, events: [e.id] });
        break;
      case "turn.end": {
        if (lastEditTurnStep && lastEditTurnStep.turn === e.turn && lastVerifySeq < lastEditTurnStep.seq && s(d.stopReason) === "end_turn") {
          const text = s(d.finalText);
          if (/\b(pass(es|ed|ing)?|green|succeed(s|ed)?|works|fixed)\b/i.test(text)) {
            out.push({ kind: "unverified_claim", severity: "high", blame: "model", turn: e.turn, message: "claimed success after editing without running a check afterwards", events: [e.id] });
          }
        }
        break;
      }
      default:
        break;
    }
  }
  for (const [id, call] of calls) {
    if (!results.has(id)) out.push({ kind: "orphan_tool_call", severity: "medium", blame: "harness", turn: call.turn, message: `${s(call.data.name)} never produced a result`, events: [call.id] });
  }
  return groupRepeats(out);
}

/** A model that keeps calling the same nonexistent tool is a harness gap (missing alias / unclear tools), not noise. */
function groupRepeats(findings: Finding[]): Finding[] {
  const byTool = new Map<string, Finding[]>();
  for (const f of findings) {
    if (f.kind !== "unknown_tool") continue;
    const tool = f.message.split(":")[0]!;
    byTool.set(tool, [...(byTool.get(tool) ?? []), f]);
  }
  const out = findings.filter((f) => f.kind !== "unknown_tool" || (byTool.get(f.message.split(":")[0]!)?.length ?? 0) < 3);
  for (const [tool, fs] of byTool) {
    if (fs.length < 3) continue;
    out.push({
      kind: "unknown_tool",
      severity: "high",
      blame: "harness",
      turn: fs[0]!.turn,
      message: `${tool}: model called this nonexistent tool ${fs.length} times — add an alias or make the real tool obvious`,
      events: fs.flatMap((f) => f.events).slice(0, 20),
    });
  }
  return out;
}

// ------------------------------------------------------------------ rendering

function clipLines(t: string, maxLines: number, maxChars: number): string {
  let out = t.length > maxChars ? `${t.slice(0, maxChars)}…` : t;
  const lines = out.split("\n");
  if (lines.length > maxLines) out = `${lines.slice(0, maxLines).join("\n")}\n… (${lines.length - maxLines} more lines)`;
  return out;
}

function fence(t: string): string {
  const ticks = t.includes("```") ? "````" : "```";
  return `${ticks}\n${t}\n${ticks}`;
}

export type RenderOptions = {
  /** Only these turns (inclusive). */
  turns?: [number, number];
  /** Only errors, declines, compactions, feedback, issues. */
  faultsOnly?: boolean;
  /** Collapse outputs to this many lines (default 12; 0 = omit outputs). */
  outputLines?: number;
  includeFindings?: boolean;
};

/** Human- and agent-readable Markdown transcript, derived from events. */
export function renderMarkdown(events: HistoryEvent[], opts: RenderOptions = {}): string {
  const head = events.find((e) => e.type === "session");
  const lines: string[] = [];
  const outLines = opts.outputLines ?? 12;
  if (head) {
    const d = head.data;
    lines.push(`# Session ${s(d.sessionId)}`, "", `cwd: \`${s(d.cwd)}\` · started ${s(d.createdAt ?? head.ts)} · harness ${s((d.harness as Record<string, unknown>)?.version)}`, "");
  }
  const calls = new Map<string, HistoryEvent>();
  const inRange = (t: number) => !opts.turns || (t >= opts.turns[0] && t <= opts.turns[1]);
  for (const e of events) {
    if (e.type === "tool.call") calls.set(s(e.data.toolCallId), e);
    if (!inRange(e.turn)) continue;
    const d = e.data;
    const sub = e.who ? `  _(${e.who})_` : "";
    switch (e.type) {
      case "turn.start":
        if (!opts.faultsOnly) lines.push("", `## Turn ${e.turn}`, "");
        break;
      case "user.message":
        if (!opts.faultsOnly) lines.push(...s(d.text).split("\n").map((l) => `> ${l}`), "");
        break;
      case "assistant.message": {
        if (opts.faultsOnly) break;
        const text = s(d.text).trim();
        const usage = (d.usage ?? {}) as Record<string, unknown>;
        if (text) lines.push(text + sub, "");
        if (d.reasoningChars) lines.push(`<details><summary>thinking (${s(d.reasoningChars)} chars)</summary>\n\n${clipLines(s(d.reasoning), 20, 3000)}\n</details>`, "");
        lines.push(`<sub>model ${s(d.model)} · ${s(d.finishReason)} · ${s(d.durationMs)}ms${usage.input ? ` · ${s(usage.input)} in / ${s(usage.output)} out` : ""}</sub>`, "");
        break;
      }
      case "tool.result": {
        const call = calls.get(s(d.toolCallId));
        const status = s(d.status);
        if (opts.faultsOnly && status === "ok") break;
        const mark = status === "ok" ? "✓" : status === "denied" ? "⊘ denied" : status === "cancelled" ? "⊘ cancelled" : "✗ " + status;
        const args = s(call?.data.arguments);
        const ignored = call && Array.isArray(call.data.ignoredArgs) && call.data.ignoredArgs.length ? ` · ⚠ ignored args: ${(call.data.ignoredArgs as string[]).join(", ")}` : "";
        lines.push(`- **▸ ${s(call?.data.name ?? d.name)}** \`${clipLines(args, 1, 160)}\` — ${mark} (${s(d.durationMs)}ms)${ignored} \`[${call?.id ?? ""}→${e.id}]\`${sub}`);
        const err = d.error as Record<string, unknown> | undefined;
        if (err) lines.push(`  - error (${s(err.type)}): ${clipLines(s(err.message), 6, 800)}`);
        if (outLines > 0 && d.output) lines.push("", "<details><summary>output</summary>", "", fence(clipLines(s(d.output), outLines, 4000)), "</details>", "");
        break;
      }
      case "permission.decision":
        if (s(d.outcome) !== "allow_once" && s(d.outcome) !== "auto_allowed") lines.push(`- ⚠ permission for \`${s(d.tool)}\`: ${s(d.outcome)} (${s(d.decidedBy)})`);
        break;
      case "llm.error":
        lines.push(`- ✗ model error (${s(d.errorType)}${d.willRetry ? ", retrying" : ""}): ${clipLines(s(d.message), 3, 400)} \`[${e.id}]\``);
        break;
      case "compaction.end":
        lines.push("", `---`, `**Compacted** (${s(d.trigger)}): ${s(d.tokensBefore)} → ${s(d.tokensAfter)} tokens${d.ok ? "" : " — FAILED"} \`[${e.id}]\``, "", "---", "");
        break;
      case "feedback":
        lines.push(`- 🔥 **user feedback** on turn ${s(d.targetTurn)} (${s(d.signal)}): "${s(d.text).slice(0, 200)}" \`[${e.id}]\``);
        break;
      case "issue":
        lines.push(`- 🐞 **issue filed** ${s(d.issueId)}: ${s(d.title)} (${s(d.blame)}) \`[${e.id}]\``);
        break;
      case "error":
        lines.push(`- ✗ **harness error** (${s(d.scope)}): ${clipLines(s(d.message), 4, 600)} \`[${e.id}]\``);
        break;
      case "turn.end":
        if (!opts.faultsOnly) lines.push("", `<sub>turn ${e.turn} → ${s(d.stopReason)} in ${s(d.durationMs)}ms</sub>`);
        break;
      default:
        break;
    }
  }
  if (opts.includeFindings !== false) {
    const findings = analyze(events).filter((f) => inRange(f.turn));
    if (findings.length) {
      lines.push("", "## Findings", "");
      for (const f of findings) lines.push(`- [${f.severity}/${f.blame}] turn ${f.turn} ${f.kind}: ${f.message} \`${f.events.join(",")}\``);
    }
  }
  return lines.join("\n").replace(/\n{3,}/g, "\n\n").trim() + "\n";
}
