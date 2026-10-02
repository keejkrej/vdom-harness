import type { AgentConfig } from "./config.js";
import { streamChat, type ChatMessage } from "./llm.js";

/**
 * Pi-style compaction: keep the most recent ~keepRecentTokens verbatim, cut at
 * a user/assistant boundary (never a tool result), summarize the rest into a
 * structured checkpoint, and fold any previous summary in iteratively.
 */

export const RESERVE_TOKENS = 16_384;
export const KEEP_RECENT_TOKENS = 20_000;
const TOOL_RESULT_CHARS = 2000;

export type CompactionState = {
  summary?: string;
  readFiles: string[];
  modifiedFiles: string[];
};

const SYSTEM =
  "You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.\n\nDo NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.";

const FORMAT = `## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages.`;

const INITIAL = `The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

${FORMAT}`;

const UPDATE = `The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

${FORMAT}`;

export const SUMMARY_PREFIX = "The conversation history before this point was compacted into the following summary:";

export function estimateTokens(msgs: ChatMessage[]): number {
  let chars = 0;
  for (const m of msgs) {
    if (typeof m.content === "string") chars += m.content.length;
    else if (Array.isArray(m.content)) chars += m.content.reduce((n, p) => n + (p.type === "text" ? p.text.length : 4800), 0);
    if (m.role === "assistant") {
      chars += m.reasoning_content?.length ?? 0;
      for (const c of m.tool_calls ?? []) chars += c.function.name.length + c.function.arguments.length;
    }
  }
  return Math.ceil(chars / 4);
}

function msgText(m: ChatMessage): string {
  if (typeof m.content === "string") return m.content;
  if (Array.isArray(m.content)) return m.content.map((p) => (p.type === "text" ? p.text : "[image]")).join("\n");
  return "";
}

export function serialize(msgs: ChatMessage[]): string {
  const out: string[] = [];
  for (const m of msgs) {
    if (m.role === "user") out.push(`[User]: ${msgText(m)}`);
    else if (m.role === "assistant") {
      if (m.reasoning_content) out.push(`[Assistant thinking]: ${m.reasoning_content}`);
      if (m.content) out.push(`[Assistant]: ${m.content}`);
      if (m.tool_calls?.length) out.push(`[Assistant tool calls]: ${m.tool_calls.map((c) => `${c.function.name}(${c.function.arguments})`).join("; ")}`);
    } else if (m.role === "tool") {
      const t = m.content.length > TOOL_RESULT_CHARS ? `${m.content.slice(0, TOOL_RESULT_CHARS)}[... ${m.content.length - TOOL_RESULT_CHARS} more characters truncated]` : m.content;
      out.push(`[Tool result]: ${t}`);
    }
  }
  return out.join("\n\n");
}

/** Index of the first kept message, or -1 if nothing can be cut. */
export function findCut(msgs: ChatMessage[], keepTokens: number): number {
  let acc = 0;
  let i = msgs.length - 1;
  for (; i >= 0; i--) {
    acc += estimateTokens([msgs[i]!]);
    if (acc >= keepTokens) break;
  }
  if (i <= 0) {
    // Everything fits in the keep window; cut at the latest user message instead so a manual compact still does something.
    for (let j = msgs.length - 1; j > 0; j--) if (msgs[j]!.role === "user") return j;
    return -1;
  }
  for (let j = i; j < msgs.length; j++) if (msgs[j]!.role !== "tool") return j > 0 ? j : -1;
  return -1;
}

function trackFiles(msgs: ChatMessage[], state: CompactionState): void {
  const read = new Set(state.readFiles);
  const modified = new Set(state.modifiedFiles);
  for (const m of msgs) {
    if (m.role !== "assistant") continue;
    for (const c of m.tool_calls ?? []) {
      let path: unknown;
      try {
        path = (JSON.parse(c.function.arguments) as { path?: unknown }).path;
      } catch {
        continue;
      }
      if (typeof path !== "string") continue;
      if (c.function.name === "read") read.add(path);
      else if (c.function.name === "write" || c.function.name === "edit") modified.add(path);
    }
  }
  state.modifiedFiles = [...modified].sort();
  state.readFiles = [...read].filter((p) => !modified.has(p)).sort();
}

export function isSummaryMessage(m: ChatMessage | undefined): boolean {
  return m?.role === "user" && typeof m.content === "string" && m.content.startsWith(SUMMARY_PREFIX);
}

export async function compact(
  cfg: AgentConfig,
  model: string,
  msgs: ChatMessage[],
  state: CompactionState,
  signal: AbortSignal,
  instructions?: string,
): Promise<{ messages: ChatMessage[]; tokensBefore: number; tokensAfter: number } | undefined> {
  // A leading summary message is folded in through state.summary, not re-summarized.
  const body = isSummaryMessage(msgs[0]) ? msgs.slice(1) : msgs;
  const cut = findCut(body, KEEP_RECENT_TOKENS);
  if (cut <= 0) return undefined;
  const head = body.slice(0, cut);
  const kept = body.slice(cut);
  trackFiles(head, state);

  let prompt = `<conversation>\n${serialize(head)}\n</conversation>\n\n`;
  if (state.summary) prompt = `<previous-summary>\n${state.summary}\n</previous-summary>\n\n${prompt}${UPDATE}`;
  else prompt += INITIAL;
  if (instructions?.trim()) prompt += `\n\nAdditional focus: ${instructions.trim()}`;

  const res = await streamChat(cfg, {
    model,
    tools: [],
    signal,
    messages: [
      { role: "system", content: SYSTEM },
      { role: "user", content: prompt },
    ],
  });
  if (res.finishReason === "length" || !res.content.trim()) throw new Error("compaction summary was empty or truncated");
  state.summary = res.content.trim();

  const files: string[] = [];
  if (state.readFiles.length) files.push(`<read-files>\n${state.readFiles.join("\n")}\n</read-files>`);
  if (state.modifiedFiles.length) files.push(`<modified-files>\n${state.modifiedFiles.join("\n")}\n</modified-files>`);
  const summaryMsg: ChatMessage = {
    role: "user",
    content: `${SUMMARY_PREFIX}\n\n<summary>\n${state.summary}${files.length ? `\n\n${files.join("\n\n")}` : ""}\n</summary>`,
  };
  const messages = [summaryMsg, ...kept];
  return { messages, tokensBefore: estimateTokens(msgs), tokensAfter: estimateTokens(messages) };
}

const OVERFLOW_PATTERNS = [
  /prompt is too long/i,
  /context[ _-]?length/i,
  /context window/i,
  /maximum context/i,
  /too many tokens/i,
  /exceeds? (the )?(model'?s? )?(maximum|max|context)/i,
  /input (is )?too long/i,
  /reduce the length/i,
  /token limit/i,
  /request too large/i,
  /num_ctx/i,
];

export function isContextOverflow(message: string): boolean {
  return OVERFLOW_PATTERNS.some((re) => re.test(message));
}
