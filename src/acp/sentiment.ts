import type { AgentConfig } from "./config.js";
import { streamChat } from "./llm.js";

/**
 * Stage 2 of bad-turn detection: a small, fast model reads the user's new
 * message against a digest of the previous agent turn and says whether the
 * user is unhappy with it, and why. Runs concurrently with the reply; never
 * blocks serving.
 */

export type TurnDigest = {
  turn: number;
  userText: string;
  assistantText: string;
  /** One line per tool call: name, short args, status, error. */
  tools: string[];
  stopReason: string;
};

export type Interpretation = {
  unhappy: boolean;
  /** 0..1 */
  frustration: number;
  target: "previous_turn" | "harness" | "other" | "none";
  category:
    | "ignored_instruction"
    | "wrong_change"
    | "false_success_claim"
    | "tool_failure"
    | "hallucination"
    | "too_slow"
    | "too_verbose"
    | "gave_up"
    | "broke_something"
    | "other"
    | "none";
  reason: string;
};

const SYSTEM = `You watch a coding-agent session and judge, in real time, whether the user's NEW message signals that the agent's PREVIOUS turn went badly. Consider tone (profanity, sarcasm, "no", "I said", repetition, caps), content (complaints, corrections, re-asking), and any language.
Reply with ONLY a JSON object:
{"unhappy": boolean, "frustration": number 0..1, "target": "previous_turn"|"harness"|"other"|"none", "category": "ignored_instruction"|"wrong_change"|"false_success_claim"|"tool_failure"|"hallucination"|"too_slow"|"too_verbose"|"gave_up"|"broke_something"|"other"|"none", "reason": "one short sentence citing what in the previous turn the user is reacting to"}
A neutral follow-up request or new task is not unhappiness. Mild corrections count with frustration ~0.3-0.5.`;

export function digestText(d: TurnDigest): string {
  return [
    `PREVIOUS TURN ${d.turn} (stop: ${d.stopReason})`,
    `User asked: ${d.userText.slice(0, 1500)}`,
    d.tools.length ? `Agent tool calls:\n${d.tools.slice(-25).join("\n")}` : "Agent made no tool calls.",
    `Agent replied: ${d.assistantText.slice(-2500) || "(nothing)"}`,
  ].join("\n");
}

export function parseInterpretation(raw: string): Interpretation | undefined {
  const m = raw.match(/\{[\s\S]*\}/);
  if (!m) return undefined;
  try {
    const o = JSON.parse(m[0]) as Partial<Interpretation>;
    const frustration = Math.max(0, Math.min(1, Number(o.frustration ?? 0)));
    return {
      unhappy: o.unhappy === true || frustration >= 0.5,
      frustration,
      target: (o.target as Interpretation["target"]) ?? "none",
      category: (o.category as Interpretation["category"]) ?? "other",
      reason: String(o.reason ?? "").slice(0, 400),
    };
  } catch {
    return undefined;
  }
}

export async function interpret(cfg: AgentConfig, model: string, userText: string, prev: TurnDigest, signal: AbortSignal): Promise<Interpretation | undefined> {
  const res = await streamChat(cfg, {
    model,
    tools: [],
    signal,
    thought: "low",
    messages: [
      { role: "system", content: SYSTEM },
      { role: "user", content: `${digestText(prev)}\n\nNEW USER MESSAGE:\n${userText.slice(0, 2000)}` },
    ],
  });
  return parseInterpretation(res.content);
}
