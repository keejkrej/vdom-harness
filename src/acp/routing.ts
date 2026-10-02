import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

/**
 * Adaptive model routing: every session starts on the cheapest rung of the
 * ladder and steps up when it struggles (model error, empty responses,
 * repeated tool failures, user frustration, a verify_claims nudge, or a
 * learned-hot request bucket). The position is not a ratchet: an escalated
 * turn stays up for itself and the next one, then each clean turn steps back
 * down one rung, never below rung 0. Per-bucket outcomes (did the rung this
 * turn started on end badly?) persist under ~/.vdom/routing so `vdom doctor`
 * can show them and hot buckets can start on rung 1. Picking a model
 * manually stands routing down for that session.
 */

/** Why the ladder moved. */
export type EscalateReason = "llm_error" | "empty_response" | "tool_errors" | "guard_nudge" | "feedback" | "learned" | "decay";

/** The user-facing failure signals (everything except housekeeping moves). */
export const TRIGGER_REASONS: ReadonlySet<EscalateReason> = new Set(["llm_error", "empty_response", "tool_errors", "guard_nudge", "feedback"]);

export type RouteStep = { from: string; to: string; reason: EscalateReason; detail?: string };

/** Per-session routing state. Persisted subset: position + hold (SessionRecord.route). */
export type RouteState = {
  /** Routing active for this session (a manual model pick stands it down). */
  active: boolean;
  /** Index into the ladder. */
  position: number;
  /** True once this turn already stepped up mid-turn; at most one mid-turn step per turn. */
  stepUpDone: boolean;
  /** Failed tool results since the last model call. */
  toolErrors: number;
  /** This session's step-ups so far, for the turn-end summary. */
  steps: RouteStep[];
  /** Completed turns the elevated position is protected for after its own: the next one. */
  hold: number;
  /** Whether the previous completed turn was clean (no trigger, ended normally). */
  prevClean: boolean;
  /** User-signal step-up waiting to apply before the next model call. */
  pendingUp?: EscalateReason;
  /** A model error or empty response marked this turn bad (recorded even when no step was possible). */
  turnBad: boolean;
  /** Whether this turn made a routed model call (bucket outcomes only count these). */
  routedCall: boolean;
  /** Ladder index this turn started on, for bucket stats. */
  startRung: number;
};

/** Cross-session routing bookkeeping, persisted as ~/.vdom/routing/stats.json. */
export type RouteStats = {
  calls: Record<string, number>;
  escalations: Record<string, number>;
  /** Outcomes per bucket ("<kind>[/harness]") per starting rung ("<index>"). */
  buckets: Record<string, Record<string, { turns: number; bad: number }>>;
};

/** Tool failures within one turn before the ladder steps up. */
const TOOL_ERROR_THRESHOLD = 3;
/** A bucket whose rung-0 bad-rate is strictly above this over at least MIN_BUCKET_TURNS turns starts on rung 1. */
const BAD_RATE = 0.4;
const MIN_BUCKET_TURNS = 5;
/** Completed turns an elevated position is protected for after its own (bad) turn: the next one. */
const HOLD_TURNS = 1;

/** Cheap → strong model ids, strongest last; duplicates and blanks dropped. */
export function parseLadder(spec: string): string[] {
  return [...new Set(spec.split(",").map((s) => s.trim()).filter(Boolean))];
}

/** Display reason for a step, e.g. "learned: fix/harness". */
export function stepReason(step: RouteStep): string {
  return step.detail ? `${step.reason}: ${step.detail}` : step.reason;
}

const FIX_RE = /\b(fix|bug|broken|fail(s|ed|ing)?|crash(es|ed|ing)?|error|regression|debug|diagnose)\b/i;
const FEATURE_RE = /\b(add|implement|create|build|write|make|support)\b/i;
const REFACTOR_RE = /\b(refactor|rename|clean ?up|simplify|restructure|reorganize)\b/i;
const QUESTION_RE = /\?\s*$/;

/** Coarse request kind; checked in this order: fix > feature > refactor > question > other. */
export function classifyRequest(text: string): "question" | "fix" | "feature" | "refactor" | "other" {
  const t = text.trim().toLowerCase();
  if (FIX_RE.test(t)) return "fix";
  if (FEATURE_RE.test(t)) return "feature";
  if (REFACTOR_RE.test(t)) return "refactor";
  if (QUESTION_RE.test(t) || /^\s*(what|why|how|who|when|where|which|is|are|can|does|do|did|should|would)\b/.test(t)) return "question";
  return "other";
}

/** The harness itself is a checkout: routing learns its own buckets separately. */
export function isHarnessCheckout(cwd: string): boolean {
  return existsSync(join(cwd, "src", "acp", "agent.ts"));
}

/** Learning bucket: request kind, plus the harness flag for vdom checkouts. */
export function bucketKey(userText: string, cwd: string): string {
  return `${classifyRequest(userText)}${isHarnessCheckout(cwd) ? "/harness" : ""}`;
}

export class ModelRouter {
  readonly ladder: string[];
  private readonly statsPath: string;
  private stats: RouteStats;

  constructor(ladder: string[], private readonly statsDir: string) {
    if (ladder.length === 0) throw new Error("routing ladder must not be empty");
    this.ladder = ladder;
    this.statsPath = join(statsDir, "stats.json");
    this.stats = loadStats(this.statsPath);
  }

  /**
   * Session state. A resumed session continues at its saved position and decay
   * counters; a start model off the ladder stands routing down.
   */
  newState(startModel?: string, saved?: { position: number; hold: number; prevClean: boolean }): RouteState {
    const fresh = { stepUpDone: false, toolErrors: 0, steps: [] as RouteStep[], pendingUp: undefined, turnBad: false, routedCall: false };
    if (saved && startModel && this.ladder[saved.position] === startModel) {
      return { active: true, position: saved.position, hold: saved.hold, prevClean: saved.prevClean, startRung: saved.position, ...fresh };
    }
    const at = startModel ? this.ladder.indexOf(startModel) : 0;
    return { active: at >= 0, position: at >= 0 ? at : 0, hold: 0, prevClean: false, startRung: at >= 0 ? at : 0, ...fresh };
  }

  /** The model to use for the next call. */
  current(st: RouteState): string {
    return this.ladder[st.position]!;
  }

  /**
   * New turn: per-turn flags reset, then decay. The protection window is
   * consumed first; past it, each turn that follows a clean turn steps back
   * down one rung (never below 0). So an escalated turn stays up for itself
   * and the next one, then clean turns step back down. Returns the decay step,
   * if any. A queued user-signal step survives.
   */
  beginTurn(st: RouteState): RouteStep | undefined {
    if (!st.active) return undefined;
    st.stepUpDone = false;
    st.toolErrors = 0;
    st.turnBad = false;
    st.routedCall = false;
    st.startRung = st.position;
    if (st.hold > 0) {
      st.hold--;
      return undefined;
    }
    if (st.prevClean && st.position > 0) {
      const from = this.ladder[st.position]!;
      st.position -= 1;
      return { from, to: this.ladder[st.position]!, reason: "decay" };
    }
    return undefined;
  }

  /** A bad-turn signal (keyword detector or interpreter) queues a step-up for the next model call. */
  notePending(st: RouteState): void {
    if (!st.active || st.pendingUp || st.position >= this.ladder.length - 1) return;
    st.pendingUp = "feedback";
  }

  /** Apply a queued user-signal step; independent of the mid-turn escalation budget. */
  applyPending(st: RouteState): RouteStep | undefined {
    const reason = st.pendingUp;
    st.pendingUp = undefined;
    if (!reason || !st.active || st.stepUpDone) return undefined;
    const up = this.stepUp(st, reason, HOLD_TURNS);
    if (up) st.startRung = st.position;
    return up;
  }

  /** When rung 0 keeps failing for a request bucket, new turns there start one rung up. */
  learnedStart(st: RouteState, userText: string, cwd: string): RouteStep | undefined {
    if (!st.active || st.position !== 0 || this.ladder.length < 2) return undefined;
    const bucket = bucketKey(userText, cwd);
    if (!this.isHot(bucket)) return undefined;
    // No hold: a learned start decays at the same turn's end and re-applies every turn in the bucket.
    const up = this.stepUp(st, "learned", 0, bucket);
    if (up) st.startRung = st.position;
    return up;
  }

  /** Feed each tool result status; failures accumulate towards a step up. */
  observeToolStatus(st: RouteState, status: string): void {
    if (!st.active) return;
    if (status === "error" || status === "denied") st.toolErrors++;
  }

  /** Enough tool failures this turn to justify a step up (even when none is possible). */
  toolFailuresAtThreshold(st: RouteState): boolean {
    return st.active && st.toolErrors >= TOOL_ERROR_THRESHOLD;
  }

  /** Step up after repeated tool failures, if this turn has not stepped up yet. */
  maybeEscalateFromTools(st: RouteState): RouteStep | undefined {
    if (st.toolErrors < TOOL_ERROR_THRESHOLD) return undefined;
    return this.escalate(st, "tool_errors");
  }

  /** Mid-turn struggle: at most one step up per turn; undefined at the top of the ladder. */
  escalate(st: RouteState, reason: EscalateReason): RouteStep | undefined {
    if (!st.active || st.stepUpDone || st.position >= this.ladder.length - 1) return undefined;
    st.stepUpDone = true;
    return this.stepUp(st, reason, HOLD_TURNS);
  }

  /**
   * Turn-end bookkeeping: remember whether this turn was clean (ended normally
   * with no escalation trigger). Decay happens at the next beginTurn.
   */
  endTurn(st: RouteState, clean: boolean): void {
    if (!st.active) return;
    st.prevClean = clean;
  }

  /** Learning: how did a turn that started on a rung end, per request bucket? */
  recordTurn(st: RouteState, userText: string, cwd: string, bad: boolean): void {
    if (!st.active || !st.routedCall) return;
    const rungs = (this.stats.buckets[bucketKey(userText, cwd)] ??= {});
    const e = (rungs[String(st.startRung)] ??= { turns: 0, bad: 0 });
    e.turns++;
    if (bad) e.bad++;
    this.persist();
  }

  /** Count a served model call (the learning signal of the ladder). */
  recordCall(model: string): void {
    this.stats.calls[model] = (this.stats.calls[model] ?? 0) + 1;
    this.persist();
  }

  snapshot(): RouteStats {
    return structuredClone(this.stats);
  }

  /** A bucket is hot when rung 0's bad-rate is strictly above BAD_RATE over enough turns. */
  private isHot(bucket: string): boolean {
    const e = this.stats.buckets[bucket]?.["0"];
    return !!e && e.turns >= MIN_BUCKET_TURNS && e.bad / e.turns > BAD_RATE;
  }

  /** Move up one rung and record it. */
  private stepUp(st: RouteState, reason: EscalateReason, hold: number, detail?: string): RouteStep | undefined {
    if (!st.active || st.position >= this.ladder.length - 1) return undefined;
    const from = this.ladder[st.position]!;
    st.position += 1;
    const to = this.ladder[st.position]!;
    st.hold = hold;
    const step: RouteStep = { from, to, reason, ...(detail ? { detail } : {}) };
    st.steps.push(step);
    this.stats.escalations[to] = (this.stats.escalations[to] ?? 0) + 1;
    this.persist();
    return step;
  }

  private persist(): void {
    try {
      mkdirSync(dirname(this.statsPath), { recursive: true });
      writeFileSync(this.statsPath, JSON.stringify(this.stats));
    } catch (err) {
      process.stderr.write(`vdom: routing stats write failed: ${String(err)}\n`);
    }
  }
}

function loadStats(path: string): RouteStats {
  try {
    if (existsSync(path)) {
      const parsed = JSON.parse(readFileSync(path, "utf8")) as Partial<RouteStats>;
      if (parsed && typeof parsed === "object") return { calls: parsed.calls ?? {}, escalations: parsed.escalations ?? {}, buckets: parsed.buckets ?? {} };
    }
  } catch {
    /* corrupt stats file: start over */
  }
  return { calls: {}, escalations: {}, buckets: {} };
}

/** Human-readable routing stats for `vdom doctor`. */
export function describeStats(stats: RouteStats): string {
  const lines = ["routing stats:"];
  const calls = Object.entries(stats.calls);
  for (const [model, n] of calls) lines.push(`  ${model}: ${n} call${n === 1 ? "" : "s"}`);
  if (!calls.length) lines.push("  (no routed calls yet)");
  for (const [model, n] of Object.entries(stats.escalations)) lines.push(`  escalated to ${model}: ${n} time${n === 1 ? "" : "s"}`);
  const buckets = Object.entries(stats.buckets);
  if (buckets.length) {
    lines.push("routing buckets (turns per starting rung, bad-rate):");
    for (const [bucket, rungs] of buckets) {
      for (const [rung, e] of Object.entries(rungs).sort(([a], [b]) => Number(a) - Number(b))) {
        if (!e.turns) continue;
        lines.push(`  ${bucket} @ rung ${rung}: ${e.turns} turn${e.turns === 1 ? "" : "s"}, ${e.bad} bad (${Math.round((e.bad / e.turns) * 100)}%)`);
      }
    }
  }
  return lines.join("\n");
}