import { createHash, randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import type { AgentGraph } from "../ir.js";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import {
  PROTOCOL_VERSION,
  RequestError,
  type Agent,
  type AgentSideConnection,
  type AuthenticateRequest,
  type AvailableCommand,
  type CancelNotification,
  type CloseSessionRequest,
  type ContentBlock,
  type ForkSessionRequest,
  type ForkSessionResponse,
  type InitializeRequest,
  type InitializeResponse,
  type ListSessionsRequest,
  type ListSessionsResponse,
  type LoadSessionRequest,
  type LoadSessionResponse,
  type McpServer,
  type NewSessionRequest,
  type NewSessionResponse,
  type PermissionOption,
  type PlanEntry,
  type PromptRequest,
  type PromptResponse,
  type ResumeSessionRequest,
  type ResumeSessionResponse,
  type SessionConfigOption,
  type SessionModeState,
  type SessionUpdate,
  type SetSessionConfigOptionRequest,
  type SetSessionConfigOptionResponse,
  type SetSessionModeRequest,
  type StopReason,
  type ToolCallContent,
  type ToolKind,
} from "@agentclientprotocol/sdk";
import { vdomHome, type AgentConfig } from "./config.js";
import { compact, estimateTokens, isContextOverflow, RESERVE_TOKENS, type CompactionState } from "./compaction.js";
import { applyGraph, codingGraph, compileSystemPrompt, describeGraph } from "./graph.js";
import { listModels, LlmError, streamChat, THOUGHT_LEVELS, type ChatMessage, type ChatPart, type ChatToolCall, type ThoughtLevel, type ToolSchema } from "./llm.js";
import { collectSpecs, McpHub } from "./mcp.js";
import {
  expandSkill,
  loadContextFiles,
  loadPromptFile,
  loadSkills,
  loadTemplates,
  skillsPromptSection,
  substituteArgs,
  type ContextFile,
  type Skill,
  type Template,
} from "./resources.js";
import { SessionStore, type Mode, type SessionRecord } from "./store.js";
import { clip, trace, traceFull } from "./trace.js";
import { analyze, EventLog, readEvents, renderMarkdown, type HistoryEvent } from "./history.js";
import { detectFeedback, type FeedbackSignal } from "./feedback.js";
import { interpret, type Interpretation, type TurnDigest } from "./sentiment.js";
import { fileIssue, type Issue } from "./issues.js";
import { JobManager, resolveToolAlias, TOOLS, TOOLS_BY_NAME, ToolInputError, type Access, type Prepared, type ToolContext, type ToolDef, type ToolImage } from "./tools.js";

export const AGENT_NAME = "vdom";
export const AGENT_VERSION = "0.3.0";

const MODES: { id: Mode; name: string; description: string }[] = [
  { id: "agent", name: "Agent", description: "Edits inside the workspace run without asking; shell commands ask." },
  { id: "ask", name: "Ask", description: "Every edit, shell command, and mutating MCP call asks for approval." },
  { id: "plan", name: "Plan", description: "Read-only: investigate and propose a plan; no edits." },
];

const THOUGHT_NAMES: Record<ThoughtLevel, string> = { off: "Off", low: "Low", medium: "Medium", high: "High" };
const REPEAT_THRESHOLDS = new Set([3, 5, 8]);
/** Tools a subagent may not use. */
const CHILD_EXCLUDED = new Set(["subagent", "todo_write", "get_agent_graph", "set_agent_graph"]);

type Resources = { contextFiles: ContextFile[]; skills: Skill[]; templates: Template[]; system?: string; append?: string };

type LiveSession = {
  rec: SessionRecord;
  abort?: AbortController;
  /** Resolves with the stop reason of the turn in flight (steering waits on it). */
  running?: Promise<StopReason>;
  /** Prompts that arrived mid-turn; injected before the next model call. */
  pending: ChatMessage[];
  /** Permission keys the user chose "always allow" for. Not persisted. */
  alwaysAllow: Set<string>;
  lastPromptTokens?: number;
  jobs: JobManager;
  mcp: McpHub;
  res: Resources;
  repeat: { sig: string; count: number };
  /** Canonical append-only history (events.jsonl). */
  log: EventLog;
  /** Digest of the last finished turn, for the sentiment interpreter. */
  lastTurn?: TurnDigest;
  /** Tool lines of the turn in flight (becomes lastTurn.tools). */
  turnTools: string[];
  /** Messages for the user from background work (e.g. filed issues), shown at the next opportunity. */
  notices: string[];
  /** Background diagnosis runs; aborted on shutdown. */
  background: Set<AbortController>;
  /** Turns already sent to diagnosis (one diagnosis per bad turn). */
  diagnosed: Set<number>;
  /** Runtime guards switched on for this session ("verify_claims"). */
  guards: Set<string>;
  /** Successful edits / checks this turn, for the verify_claims guard. */
  claimState: { edits: number; verifiedEdits: number; nudged: boolean };
};

/** A conversation the loop drives: the session itself, or a subagent's scratch context. */
type Conv = { messages: ChatMessage[]; compaction: CompactionState };

type ChildOpts = {
  label: string;
  onActivity(line: string): void;
  /** Restrict to these tools (e.g. read-only for the diagnostician). */
  allow?: Set<string>;
  /** Extra system-prompt note for this child. */
  note?: string;
  /** Model override (e.g. a stronger diagnosis model). */
  model?: string;
};

const DIAGNOSIS_TOOLS = new Set(["read", "grep", "find", "ls", "history"]);

type ToolRun = { output: string; images?: ToolImage[] };

class Cancelled extends Error {}

export class VdomAgent implements Agent {
  private readonly sessions = new Map<string, LiveSession>();
  private readonly store: SessionStore;
  private modelCache?: { at: number; ids: string[] };

  constructor(
    private readonly conn: AgentSideConnection,
    private readonly cfg: AgentConfig,
  ) {
    this.store = new SessionStore(cfg.dataDir);
    void this.store.pruneEmpty().catch(() => {});
  }

  /** Stop background jobs and MCP servers (process exit). */
  async shutdown(): Promise<void> {
    for (const live of this.sessions.values()) {
      live.abort?.abort(new Cancelled());
      for (const bg of live.background) bg.abort(new Cancelled());
      live.jobs.killAll();
      await live.mcp.close();
    }
  }

  /** Wait for background diagnoses (tests, `vdom run`). */
  async settle(timeoutMs = 120_000): Promise<void> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline && [...this.sessions.values()].some((l) => l.background.size > 0)) {
      await new Promise((r) => setTimeout(r, 100));
    }
  }

  // ------------------------------------------------------------- lifecycle

  async initialize(_params: InitializeRequest): Promise<InitializeResponse> {
    return {
      protocolVersion: PROTOCOL_VERSION,
      agentCapabilities: {
        loadSession: true,
        promptCapabilities: { image: true, embeddedContext: true, audio: false },
        mcpCapabilities: { http: true, sse: false },
        sessionCapabilities: { resume: {}, list: {}, close: {}, fork: {} },
      },
      authMethods: [],
      agentInfo: { name: AGENT_NAME, title: "vdom coding agent", version: AGENT_VERSION },
    };
  }

  /** Credentials come from env / ~/.vdom/config.json; any method id is accepted. */
  async authenticate(_params: AuthenticateRequest): Promise<Record<string, never>> {
    return {};
  }

  async newSession(params: NewSessionRequest): Promise<NewSessionResponse> {
    const now = new Date().toISOString();
    const rec: SessionRecord = {
      version: 1,
      id: randomUUID(),
      cwd: params.cwd,
      roots: [params.cwd, ...(params.additionalDirectories ?? [])],
      model: this.cfg.model,
      mode: "agent",
      thought: "medium",
      graph: loadRepoGraph(params.cwd) ?? codingGraph(),
      messages: [],
      compaction: { readFiles: [], modifiedFiles: [] },
      createdAt: now,
      updatedAt: now,
    };
    const live = this.activate(rec, params.mcpServers);
    await this.store.save(rec);
    this.announceCommands(live);
    return { sessionId: rec.id, modes: this.modeState(rec), configOptions: await this.configOptions(rec) };
  }

  async loadSession(params: LoadSessionRequest): Promise<LoadSessionResponse> {
    const live = await this.attach(params.sessionId, params.cwd, params.additionalDirectories, params.mcpServers);
    await this.replay(live.rec);
    this.announceCommands(live);
    return { modes: this.modeState(live.rec), configOptions: await this.configOptions(live.rec) };
  }

  async resumeSession(params: ResumeSessionRequest): Promise<ResumeSessionResponse> {
    const live = await this.attach(params.sessionId, params.cwd, params.additionalDirectories, params.mcpServers);
    this.announceCommands(live);
    return { modes: this.modeState(live.rec), configOptions: await this.configOptions(live.rec) };
  }

  async unstable_forkSession(params: ForkSessionRequest): Promise<ForkSessionResponse> {
    const src = this.sessions.get(params.sessionId)?.rec ?? (await this.store.load(params.sessionId));
    if (!src) throw RequestError.resourceNotFound(`session ${params.sessionId}`);
    const now = new Date().toISOString();
    const rec: SessionRecord = {
      ...structuredClone(src),
      id: randomUUID(),
      cwd: params.cwd || src.cwd,
      roots: [params.cwd || src.cwd, ...(params.additionalDirectories ?? src.roots.slice(1))],
      title: src.title ? `${src.title} (fork)` : undefined,
      forkedFrom: src.id,
      createdAt: now,
      updatedAt: now,
    };
    repairToolPairs(rec.messages);
    const live = this.activate(rec, params.mcpServers);
    await this.store.save(rec);
    this.announceCommands(live);
    return { sessionId: rec.id, modes: this.modeState(rec), configOptions: await this.configOptions(rec) };
  }

  async listSessions(params: ListSessionsRequest): Promise<ListSessionsResponse> {
    const all = await this.store.list();
    return {
      sessions: all
        .filter((r) => r.messages.length > 0 && (!params.cwd || r.cwd === params.cwd))
        .slice(0, 200)
        .map((r) => ({ sessionId: r.id, cwd: r.cwd, title: r.title ?? null, updatedAt: r.updatedAt })),
    };
  }

  async closeSession(params: CloseSessionRequest): Promise<Record<string, never>> {
    const live = this.sessions.get(params.sessionId);
    if (live) {
      live.abort?.abort(new Cancelled());
      live.jobs.killAll();
      await live.mcp.close();
      this.sessions.delete(params.sessionId);
    }
    return {};
  }

  async setSessionMode(params: SetSessionModeRequest): Promise<Record<string, never>> {
    const live = this.live(params.sessionId);
    await this.applyMode(live, params.modeId);
    await this.store.save(live.rec);
    return {};
  }

  async setSessionConfigOption(params: SetSessionConfigOptionRequest): Promise<SetSessionConfigOptionResponse> {
    const live = this.live(params.sessionId);
    const value = String(params.value).trim();
    if (params.configId === "model") {
      if (!value) throw RequestError.invalidParams(undefined, "model must be non-empty");
      live.rec.model = value;
      live.lastPromptTokens = undefined;
    } else if (params.configId === "mode") {
      await this.applyMode(live, value);
    } else if (params.configId === "thought_level") {
      if (!THOUGHT_LEVELS.includes(value as ThoughtLevel)) throw RequestError.invalidParams(undefined, `thought_level must be one of ${THOUGHT_LEVELS.join(", ")}`);
      live.rec.thought = value as ThoughtLevel;
    } else {
      throw RequestError.invalidParams(undefined, `unknown config option ${params.configId}`);
    }
    if (params.configId !== "mode") live.log.append("config.change", { reason: "user", [params.configId === "thought_level" ? "thought" : params.configId]: value });
    await this.store.save(live.rec);
    return { configOptions: await this.configOptions(live.rec) };
  }

  async cancel(params: CancelNotification): Promise<void> {
    const live = this.sessions.get(params.sessionId);
    if (!live) return;
    live.pending.length = 0;
    live.abort?.abort(new Cancelled());
  }

  async extMethod(method: string, params: Record<string, unknown>): Promise<Record<string, unknown>> {
    switch (method) {
      // T3 Code's Cursor driver discovers models with this; vdom/list_models is the native name.
      case "cursor/list_available_models":
      case "vdom/list_models":
        return { models: (await this.availableModels()).map((m) => ({ value: m, name: m })) };
      // Unstable ACP model selector.
      case "session/set_model": {
        const live = this.live(String(params.sessionId));
        const modelId = String(params.modelId ?? "").trim();
        if (!modelId) throw RequestError.invalidParams(undefined, "modelId required");
        live.rec.model = modelId;
        await this.store.save(live.rec);
        return {};
      }
      default:
        throw RequestError.methodNotFound(method);
    }
  }

  // ------------------------------------------------------------- prompt turn

  async prompt(params: PromptRequest): Promise<PromptResponse> {
    const live = this.live(params.sessionId);
    const rec = live.rec;

    const userText = blocksText(params.prompt);
    const blocks = await this.expandSlash(live, params.prompt);
    if (blocks === null) {
      live.log.append("user.message", { text: userText, source: "command" });
      return { stopReason: "end_turn" };
    }
    const userMsg: ChatMessage = { role: "user", content: promptToChat(blocks) };

    if (live.running) {
      // Steering: delivered after the current tool batch, before the next model call.
      live.log.append("user.message", { text: userText, source: "steer" });
      const fb = detectFeedback(userText);
      if (fb) this.onBadTurn(live, live.log.turn, { source: "keyword", text: userText, detail: `${fb.signal}: ${fb.matched}` });
      live.pending.push(userMsg);
      return { stopReason: await live.running };
    }

    await this.flushNotices(live);

    // Bad-turn detection on the previous turn. Never blocks serving.
    const prev = live.lastTurn;
    const fb = prev ? detectFeedback(userText, prev.userText) : undefined;
    live.log.turn++;
    const turn = live.log.turn;
    const t0 = Date.now();
    live.log.append("turn.start", { turn });
    live.log.append("user.message", { text: userText, source: userText.startsWith("/") ? "command" : "human", ...(blocksText(blocks) !== userText ? { expanded: clip(blocksText(blocks), 4000) } : {}) });
    if (prev && fb) {
      // Serving still fixes the task in-session; diagnosis of the harness happens separately.
      userMsg.content = appendText(
        userMsg.content,
        `\n\n[vdom: this message reads as dissatisfaction with your previous turn (${fb.signal}: "${fb.matched}"). In one sentence, say what went wrong — check the history tool for turn ${prev.turn} if unsure — then fix it properly. Do not just apologize.]`,
      );
      this.onBadTurn(live, prev.turn, { source: "keyword", text: userText, detail: `${fb.signal}: ${fb.matched}` }, prev);
    } else if (prev && userText.trim() && !userText.startsWith("/")) {
      this.interpretInBackground(live, userText, prev);
    }

    rec.messages.push(userMsg);
    const turnStartIndex = rec.messages.length - 1;
    if (!rec.title) {
      rec.title = titleFrom(blocks);
      await this.update(rec.id, { sessionUpdate: "session_info_update", title: rec.title, updatedAt: new Date().toISOString() });
    }

    const abort = new AbortController();
    live.abort = abort;
    live.turnTools = [];
    live.claimState = { edits: 0, verifiedEdits: 0, nudged: false };
    const run = (async (): Promise<StopReason> => {
      let stop: StopReason = "end_turn";
      try {
        await live.mcp.whenReady();
        stop = await this.runLoop(live, rec, abort.signal);
        return stop;
      } catch (err) {
        if (abort.signal.aborted || err instanceof Cancelled) {
          stop = "cancelled";
          return stop;
        }
        live.log.append("error", { scope: "agent", type: err instanceof LlmError ? "llm" : "exception", message: err instanceof Error ? err.message : String(err), stack: err instanceof Error ? clip(err.stack, 3000) : undefined });
        stop = "end_turn";
        throw err;
      } finally {
        repairToolPairs(rec.messages);
        // Steering that arrived after the last model call becomes the next user turn's context.
        if (live.pending.length) rec.messages.push(...live.pending.splice(0));
        const finalText = [...rec.messages.slice(turnStartIndex)].reverse().find((m) => m.role === "assistant" && typeof m.content === "string" && m.content.trim());
        const assistantText = finalText && finalText.role === "assistant" ? (finalText.content ?? "") : "";
        live.log.append("turn.end", { turn, stopReason: stop, durationMs: Date.now() - t0, finalText: clip(assistantText, 4000) });
        live.lastTurn = { turn, userText, assistantText, tools: live.turnTools, stopReason: stop };
        if (live.abort === abort) live.abort = undefined;
        live.running = undefined;
        await this.store.save(rec).catch((e) => process.stderr.write(`vdom: save failed: ${String(e)}\n`));
        await this.flushNotices(live);
      }
    })();
    live.running = run.catch(() => "end_turn" as StopReason);
    try {
      return { stopReason: await run };
    } catch (err) {
      throw RequestError.internalError(undefined, err instanceof Error ? err.message : String(err));
    }
  }

  /**
   * Built-in commands (/compact, /reload), /skill:name, and prompt templates.
   * Returns null when the command was fully handled without a model turn.
   */
  // ------------------------------------------------------------- bad-turn pipeline

  /**
   * A turn went badly (user signal or fault). Log it and diagnose in the
   * background; the diagnosis yields an immediate in-session correction and,
   * if the harness is at fault, a long-term issue. Never blocks serving.
   */
  private onBadTurn(live: LiveSession, targetTurn: number, signal: Issue["signal"], digest?: TurnDigest, interp?: Interpretation): void {
    live.log.append("feedback", { targetTurn, signal: signal.source, text: signal.text, detail: signal.detail, ...(interp ? { interpretation: interp } : {}) });
    if (live.diagnosed.has(targetTurn) || process.env.VDOM_DIAGNOSE === "off") return;
    live.diagnosed.add(targetTurn);
    const ac = new AbortController();
    live.background.add(ac);
    void this.runDiagnosis(live, targetTurn, signal, digest, interp, ac.signal)
      .catch((err) => {
        if (!ac.signal.aborted) live.log.append("error", { scope: "diagnosis", type: "exception", message: err instanceof Error ? err.message : String(err) });
      })
      .finally(() => live.background.delete(ac));
  }

  /** Stage 2: the sentiment interpreter, concurrent with the reply. */
  private interpretInBackground(live: LiveSession, userText: string, prev: TurnDigest): void {
    const model = this.cfg.sentimentModel;
    if (!model) return;
    const ac = new AbortController();
    live.background.add(ac);
    void interpret(this.cfg, model, userText, prev, ac.signal)
      .then((r) => {
        trace("interpretation", { session: live.rec.id, turn: prev.turn, result: r });
        if (r?.unhappy && r.target !== "other") {
          this.onBadTurn(live, prev.turn, { source: "interpreter", text: userText, detail: `${r.category} (${r.frustration.toFixed(2)}): ${r.reason}` }, prev, r);
        }
      })
      .catch((err) => trace("interpretation_error", { session: live.rec.id, error: String(err) }))
      .finally(() => live.background.delete(ac));
  }

  private async runDiagnosis(live: LiveSession, targetTurn: number, signal: Issue["signal"], digest: TurnDigest | undefined, interp: Interpretation | undefined, abort: AbortSignal): Promise<void> {
    const events = readEvents(live.log.dir).filter((e) => e.who !== "diagnosis");
    const transcript = renderMarkdown(events, { turns: [Math.max(1, targetTurn - 1), targetTurn], outputLines: 25 });
    const brief = [
      `A user of the vdom coding agent reacted badly to turn ${targetTurn} of session ${live.rec.id}.`,
      `User reaction: "${(signal.text ?? "").slice(0, 600)}"${signal.detail ? ` (${signal.detail})` : ""}`,
      interp ? `Interpreter: ${interp.category}, frustration ${interp.frustration.toFixed(2)} — ${interp.reason}` : "",
      "",
      "Recorded transcript of the turn(s), from the harness log (exact tool calls, arguments, results, errors, compactions):",
      transcript.slice(0, 40_000),
      "",
      `Tools the agent had: ${this.toolSchemas(live).map((t) => t.name).join(", ")}. Mode: ${live.rec.mode}. Model: ${live.rec.model}.`,
      `The harness source is at ${HARNESS_ROOT.replace(/\\/g, "/")} (src/acp/agent.ts loop, tools.ts tools, graph.ts prompt). You may read it, and use history for more turns.`,
      "",
      "Answer two questions:",
      "1. IN-SESSION: what single, concrete rule should the agent follow for the rest of this session so this never happens again in it? Phrase it as an imperative the agent can obey (not 'be careful'). If the failure is a false or unverified claim of success, also request guard \"verify_claims\".",
      "2. LONG-TERM: what change to the harness (prompt, tool schema/behaviour, guard, policy, or code bug) would have prevented it for everyone? Blame the harness whenever a harness change would have prevented the mistake, even if the model made it.",
      "",
      "Finish with ONLY this JSON (no prose after it):",
      '{"what_happened": "...", "blame": "harness|model|env|unclear", "category": "...", "session_rule": {"rule": "...", "guard": "verify_claims|none"} or null, "issue": {"title": "...", "root_cause": "...", "proposed_fix": "...", "files": ["src/acp/..."], "repro": "...", "severity": "high|medium|low"} or null, "evidence": ["e_..."]}',
    ]
      .filter((l) => l !== "")
      .join("\n");
    const conv: Conv = { messages: [{ role: "user", content: brief }], compaction: { readFiles: [], modifiedFiles: [] } };
    await this.runLoop(live, conv, abort, {
      label: "diagnosis",
      onActivity: () => {},
      allow: DIAGNOSIS_TOOLS,
      note: "You are vdom's diagnostician. You inspect a failed turn of another agent session and its harness; you never change files. Be concrete and evidence-based; cite event ids from the transcript.",
      ...(this.cfg.diagnosisModel ? { model: this.cfg.diagnosisModel } : {}),
    });
    const last = [...conv.messages].reverse().find((m) => m.role === "assistant" && typeof m.content === "string" && m.content.includes("{"));
    const parsed = parseJsonObject(last && last.role === "assistant" ? (last.content ?? "") : "");
    if (!parsed) {
      live.log.append("error", { scope: "diagnosis", type: "unparseable", message: "diagnostician did not return JSON" });
      return;
    }
    const str = (v: unknown) => (typeof v === "string" ? v : "");
    const rule = parsed.session_rule as { rule?: unknown; guard?: unknown } | null | undefined;
    if (rule && str(rule.rule).trim()) this.applyLesson(live, targetTurn, str(rule.rule).trim(), str(parsed.what_happened), str(rule.guard));
    const iss = parsed.issue as Record<string, unknown> | null | undefined;
    if (iss && str(iss.title)) {
      const blame = ["harness", "model", "env"].includes(str(parsed.blame)) ? (str(parsed.blame) as Issue["blame"]) : "unclear";
      const issue = fileIssue({
        sessionId: live.rec.id,
        sessionDir: live.log.dir,
        turn: targetTurn,
        signal,
        title: str(iss.title).slice(0, 200),
        blame,
        category: str(parsed.category) || "other",
        whatHappened: str(parsed.what_happened),
        rootCause: str(iss.root_cause),
        proposedFix: str(iss.proposed_fix),
        files: Array.isArray(iss.files) ? iss.files.map(String) : [],
        repro: str(iss.repro),
        evidence: Array.isArray(parsed.evidence) ? parsed.evidence.map(String) : [],
        severity: ["high", "medium", "low"].includes(str(iss.severity)) ? (str(iss.severity) as Issue["severity"]) : "medium",
      });
      live.log.append("issue", { issueId: issue.id, title: issue.title, blame: issue.blame, targetTurn });
      live.notices.push(`🐞 Filed ${issue.blame} issue ${issue.id}: ${issue.title} (\`vdom issues show ${issue.id}\`)`);
    }
    if (!live.running) await this.flushNotices(live);
  }

  /** Real-time improvement: reconcile a lesson into the live session graph (session-scoped). */
  private applyLesson(live: LiveSession, turn: number, rule: string, whatHappened: string, guard: string): void {
    const rec = live.rec;
    const n = (rec.graph.root.children ?? []).filter((c) => c.key.startsWith("lesson-")).length + 1;
    const next = structuredClone(rec.graph);
    next.root.children = [
      ...(next.root.children ?? []),
      { key: `lesson-${n}`, kind: "policy", role: "lesson", objective: rule, prompt: `Learned in this session after the user objected to turn ${turn}: ${whatHappened.slice(0, 400)}`, persistence: "session" },
    ];
    const r = applyGraph(rec.graph, next);
    if (!r.ok) {
      live.log.append("error", { scope: "lesson", type: "rejected", message: r.reason });
      return;
    }
    rec.graph = r.graph;
    if (guard === "verify_claims") live.guards.add("verify_claims");
    live.log.append("lesson", { turn, key: `lesson-${n}`, rule, guard: guard || "none", diff: r.diff });
    live.notices.push(`📌 Correction for the rest of this session: ${rule}${guard === "verify_claims" ? " (claims are now checked before I finish a turn)" : ""}`);
  }

  private async historyFor(live: LiveSession, q: { session?: string; turns?: string; faultsOnly?: boolean; list?: boolean }): Promise<string> {
    if (q.list) {
      const all = await this.store.list();
      return all
        .filter((r) => r.messages.length > 0)
        .slice(0, 25)
        .map((r) => `${r.id}  ${r.updatedAt.slice(0, 16)}  ${r.title ?? "(untitled)"}  [${r.cwd}]`)
        .join("\n") || "(no sessions)";
    }
    const current = !q.session || q.session === "current" || q.session === live.rec.id;
    const dir = current ? live.log.dir : await this.store.locate(q.session!);
    if (!dir) return `No session ${q.session}`;
    const events = readEvents(dir).filter((e) => e.who !== "diagnosis");
    const maxTurn = Math.max(0, ...events.map((e) => e.turn));
    let range: [number, number] = [Math.max(1, maxTurn - 1), maxTurn];
    const t = (q.turns ?? "last").trim();
    if (t === "all") range = [0, maxTurn];
    else if (/^\d+$/.test(t)) range = [Number(t), Number(t)];
    else if (/^\d+-\d+$/.test(t)) {
      const [a, b] = t.split("-").map(Number) as [number, number];
      range = [a, b];
    }
    const md = renderMarkdown(events, { turns: range, faultsOnly: q.faultsOnly === true, outputLines: 15 });
    return md.length > 60_000 ? `${md.slice(0, 60_000)}\n… (truncated; request fewer turns)` : md;
  }

  private async flushNotices(live: LiveSession): Promise<void> {
    if (!live.notices.length) return;
    const text = live.notices.splice(0).join("\n");
    await this.say(live.rec.id, `\n\n${text}\n`);
  }

  private async expandSlash(live: LiveSession, blocks: ContentBlock[]): Promise<ContentBlock[] | null> {
    const first = blocks[0];
    if (!first || first.type !== "text") return blocks;
    const m = first.text.match(/^\/([^\s]+)(?:\s+([\s\S]*))?$/);
    if (!m) return blocks;
    const name = m[1]!;
    const rest = m[2] ?? "";
    const rec = live.rec;
    const replaceFirst = (text: string): ContentBlock[] => [{ type: "text", text }, ...blocks.slice(1)];
    if (name === "compact") {
      if (live.running) throw RequestError.invalidRequest(undefined, "cannot compact while a turn is running");
      const abort = new AbortController();
      live.abort = abort;
      try {
        const r = await this.compactNow(live, rec, abort.signal, rest);
        await this.say(rec.id, r ? `Compacted conversation: ~${r.tokensBefore} → ~${r.tokensAfter} tokens.` : "Nothing to compact yet.");
      } finally {
        live.abort = undefined;
        await this.store.save(rec);
      }
      return null;
    }
    if (name === "reload") {
      live.res = loadResources(rec.cwd);
      this.announceCommands(live);
      await this.say(rec.id, `Reloaded ${live.res.contextFiles.length} context files, ${live.res.skills.length} skills, ${live.res.templates.length} prompt templates.`);
      return null;
    }
    if (name.startsWith("skill:")) {
      const skill = live.res.skills.find((s) => s.name === name.slice(6));
      return skill ? replaceFirst(expandSkill(skill, rest)) : blocks;
    }
    const tpl = live.res.templates.find((t) => t.name === name);
    return tpl ? replaceFirst(substituteArgs(tpl.body, rest)) : blocks;
  }

  /** The agent loop, shared by the session and its subagents. */
  private async runLoop(live: LiveSession, conv: Conv, signal: AbortSignal, child?: ChildOpts): Promise<StopReason> {
    const rec = live.rec;
    let overflowRetried = false;
    let emptyRetries = 0;
    for (let step = 0; step < this.cfg.maxSteps; step++) {
      if (signal.aborted) return "cancelled";
      if (!child && live.pending.length) conv.messages.push(...live.pending.splice(0));
      const system = this.systemPrompt(live, child);
      await this.maybeCompact(live, conv, signal, system, child);

      const tools = this.toolSchemas(live, child);
      const messageId = randomUUID();
      const who = whoOf(child);
      const model = child?.model ?? rec.model;
      const t0 = Date.now();
      const estTokens = estimateTokens([{ role: "system", content: system }, ...conv.messages]);
      live.log.append("llm.request", { model, thought: rec.thought, messages: conv.messages.length, tools: tools.length, contextTokensEst: estTokens, contextWindow: this.cfg.contextTokens }, { step, who });
      const lastMsg = conv.messages[conv.messages.length - 1];
      trace("llm_request", {
        session: rec.id,
        who,
        step,
        model: rec.model,
        thought: rec.thought,
        messages: conv.messages.length,
        tools: tools.length,
        estTokens: estimateTokens([{ role: "system", content: system }, ...conv.messages]),
        last: lastMsg ? { role: lastMsg.role, content: clip(typeof lastMsg.content === "string" ? lastMsg.content : JSON.stringify(lastMsg.content), 1500) } : undefined,
        ...(traceFull() ? { system, all: conv.messages } : {}),
      });
      let res;
      try {
        res = await streamChat(
          this.cfg,
          { model, messages: [{ role: "system", content: system }, ...conv.messages], tools, signal, thought: rec.thought },
          child
            ? {}
            : {
                onText: (t) => this.update(rec.id, { sessionUpdate: "agent_message_chunk", messageId, content: { type: "text", text: t } }),
                onReasoning: (t) => this.update(rec.id, { sessionUpdate: "agent_thought_chunk", messageId, content: { type: "text", text: t } }),
              },
        );
      } catch (err) {
        trace("llm_error", { session: rec.id, who, step, ms: Date.now() - t0, aborted: signal.aborted, error: err instanceof Error ? err.message : String(err) });
        const overflow = !signal.aborted && err instanceof LlmError && isContextOverflow(err.message);
        live.log.append(
          "llm.error",
          { model, errorType: signal.aborted ? "aborted" : classifyLlmError(err), status: err instanceof LlmError ? err.status : undefined, message: clip(err instanceof Error ? err.message : String(err), 2000), willRetry: overflow && !overflowRetried, durationMs: Date.now() - t0 },
          { step, who },
        );
        if (overflow && !overflowRetried) {
          overflowRetried = true;
          const r = await this.compactNow(live, conv, signal, undefined, child, "overflow_error");
          if (r) continue;
          throw new Error(`Context overflow recovery failed: nothing left to compact. Try a model with a larger context window. (${err.message})`);
        }
        throw err;
      }
      trace("llm_response", {
        session: rec.id,
        who,
        step,
        ms: Date.now() - t0,
        finish: res.finishReason,
        usage: res.usage,
        content: clip(res.content, 3000),
        reasoningChars: res.reasoning.length,
        ...(traceFull() ? { reasoning: res.reasoning } : {}),
        toolCalls: res.toolCalls.map((c) => ({ id: c.id, name: c.function.name, args: clip(c.function.arguments, 3000) })),
      });
      const reasoning = res.reasoning ? live.log.blob(res.reasoning) : undefined;
      live.log.append(
        "assistant.message",
        {
          model: res.model ?? model,
          text: res.content,
          reasoningChars: res.reasoning.length,
          ...(reasoning ? { reasoning: reasoning.preview, ...(reasoning.blobRef ? { reasoningRef: reasoning.blobRef } : {}) } : {}),
          toolCalls: res.toolCalls.map((c) => ({ id: c.id, name: c.function.name, arguments: clip(c.function.arguments, 4000) })),
          finishReason: res.finishReason,
          usage: res.usage ? { input: res.usage.promptTokens, output: res.usage.completionTokens, total: res.usage.totalTokens } : undefined,
          durationMs: Date.now() - t0,
        },
        { step, who },
      );
      if (res.usage && !child) {
        live.lastPromptTokens = res.usage.promptTokens;
        await this.update(rec.id, { sessionUpdate: "usage_update", used: res.usage.totalTokens, size: this.cfg.contextTokens });
      }

      conv.messages.push({ role: "assistant", content: res.content || null, ...(res.toolCalls.length ? { tool_calls: res.toolCalls } : {}) });

      if (res.toolCalls.length === 0 && !res.content.trim() && res.finishReason !== "length" && res.finishReason !== "content_filter") {
        // An empty answer is a stalled model, not a finished task (seen in a recorded fixer session that "ended" with nothing).
        emptyRetries++;
        conv.messages.pop();
        live.log.append("llm.error", { model, errorType: "empty_response", message: `no text and no tool calls (finish_reason ${res.finishReason ?? "missing"})`, willRetry: emptyRetries <= 2 }, { step, who });
        if (emptyRetries <= 2) {
          if (emptyRetries === 2) conv.messages.push({ role: "user", content: "[vdom: your last two responses were empty. Continue the task: call a tool, or give your final answer with what you found and what is left.]" });
          continue;
        }
        const msg = "The model returned empty responses three times in a row; stopping this turn.";
        conv.messages.push({ role: "assistant", content: msg });
        if (!child) await this.say(rec.id, msg);
        return "end_turn";
      }
      if (res.toolCalls.length === 0) {
        if (!child && live.pending.length) continue;
        if (!child && this.claimGuard(live, conv, res.content)) continue;
        if (!child) await this.store.save(rec);
        if (res.finishReason === "length") return "max_tokens";
        if (res.finishReason === "content_filter") return "refusal";
        return "end_turn";
      }

      if (res.finishReason === "length") {
        // Arguments may be cut off; never execute them.
        for (const c of res.toolCalls) {
          conv.messages.push({
            role: "tool",
            tool_call_id: c.id,
            content: "Error: your response hit the output token limit, so this tool call's arguments may be truncated and it was not executed. Retry with smaller steps (e.g. several small edits, or write the file in parts).",
          });
        }
        continue;
      }

      await this.runToolCalls(live, conv, res.toolCalls, signal, child);
      if (!child) await this.store.save(rec);
    }
    return "max_turn_requests";
  }

  /** Concurrency-safe calls (reads, subagents, read-only MCP) run in parallel; others alone, in order. */
  private async runToolCalls(live: LiveSession, conv: Conv, calls: ChatToolCall[], signal: AbortSignal, child?: ChildOpts): Promise<void> {
    const results = new Map<string, ToolRun>();
    let cancelled = false;
    const safe = (c: ChatToolCall) => {
      const name = c.function.name;
      const def = TOOLS_BY_NAME.get(name);
      if (def) return def.access === "read" || name === "subagent";
      return live.mcp.find(name)?.readOnly === true;
    };
    let i = 0;
    try {
      while (i < calls.length && !signal.aborted) {
        const batch: ChatToolCall[] = [];
        for (let j = i; j < calls.length && safe(calls[j]!); j++) batch.push(calls[j]!);
        if (batch.length > 1) {
          const outs = await Promise.all(batch.map((c) => this.runOneTool(live, c, signal, child)));
          batch.forEach((c, k) => results.set(c.id, outs[k]!));
          i += batch.length;
          continue;
        }
        const call = calls[i]!;
        results.set(call.id, await this.runOneTool(live, call, signal, child));
        i++;
      }
    } catch (err) {
      if (!(err instanceof Cancelled) && !signal.aborted) throw err;
      cancelled = true;
    }
    const images: ToolImage[] = [];
    for (const c of calls) {
      const r = results.get(c.id);
      let output = r?.output ?? "Cancelled by user.";
      if (r) output += this.repeatReminder(live, c);
      conv.messages.push({ role: "tool", tool_call_id: c.id, content: output });
      if (r?.images) images.push(...r.images);
    }
    if (images.length) {
      // Chat-completions tool messages are text-only; attach images as a follow-up user message.
      conv.messages.push({
        role: "user",
        content: [{ type: "text", text: "[Images returned by the tool calls above]" }, ...images.map((im): ChatPart => ({ type: "image_url", image_url: { url: `data:${im.mimeType};base64,${im.data}` } }))],
      });
    }
    if (cancelled || signal.aborted) throw new Cancelled();
  }

  private repeatReminder(live: LiveSession, c: ChatToolCall): string {
    const sig = `${c.function.name}\u0000${c.function.arguments}`;
    if (live.repeat.sig === sig) live.repeat.count++;
    else live.repeat = { sig, count: 1 };
    return REPEAT_THRESHOLDS.has(live.repeat.count)
      ? `\n\n[Reminder: this is identical ${c.function.name} call #${live.repeat.count} in a row. Its result will not change — try a different approach or explain what is blocking you.]`
      : "";
  }

  private async runOneTool(live: LiveSession, call: ChatToolCall, signal: AbortSignal, child?: ChildOpts): Promise<ToolRun> {
    const rec = live.rec;
    const id = child ? `${call.id}-sub-${randomUUID().slice(0, 8)}` : call.id;
    let name = call.function.name;
    let aliasOf: string | undefined;
    const who = whoOf(child);
    const emit = (u: SessionUpdate) => (child ? Promise.resolve() : this.update(rec.id, u));
    const tStart = Date.now();
    const record = (status: "ok" | "error" | "denied" | "cancelled", output: string, error?: { type: string; message: string; stack?: string }) => {
      const b = live.log.blob(output);
      live.log.append(
        "tool.result",
        { toolCallId: id, name, status, durationMs: Date.now() - tStart, output: b.preview, outputBytes: b.bytes, ...(b.blobRef ? { outputRef: b.blobRef } : {}), ...(error ? { error } : {}) },
        { who },
      );
      if (!child) live.turnTools.push(`${name} ${clip(call.function.arguments, 140)} → ${status}${error ? `: ${clip(error.message, 160)}` : ""}`);
    };

    let args: Record<string, unknown> = {};
    try {
      const parsed = call.function.arguments.trim() ? (JSON.parse(call.function.arguments) as unknown) : {};
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) args = parsed as Record<string, unknown>;
      else throw new Error("arguments must be a JSON object");
    } catch (err) {
      live.log.append("tool.call", { toolCallId: id, name, kind: "other", argumentsRaw: clip(call.function.arguments, 4000) }, { who });
      await emit({ sessionUpdate: "tool_call", toolCallId: id, title: name, kind: "other", status: "failed", rawInput: call.function.arguments });
      const msg = `Error: invalid JSON arguments (${String(err)}). Send a JSON object matching the tool schema.`;
      record("error", msg, { type: "invalid_args", message: String(err) });
      return { output: msg };
    }
    const alias = resolveToolAlias(name, args);
    if (alias && this.toolSchemas(live, child).some((t) => t.name === alias.name)) {
      aliasOf = alias.aliasOf;
      name = alias.name;
      args = alias.args;
    }

    let def: { kind: ToolKind; access: Access | "mcp"; name: string } | undefined;
    let prepared: Prepared;
    try {
      const mcpTool = name.startsWith("mcp__") ? live.mcp.find(name) : undefined;
      const builtin = TOOLS_BY_NAME.get(name);
      if (mcpTool) {
        def = { kind: mcpTool.readOnly ? "read" : "other", access: mcpTool.readOnly ? "read" : "mcp", name };
        prepared = {
          title: `${mcpTool.server}: ${mcpTool.tool}`,
          execute: async () => {
            const r = await live.mcp.call(mcpTool, args, signal);
            return { output: r.text, isError: r.isError, ...(r.images.length ? { images: r.images } : {}) };
          },
        };
      } else if (builtin && this.toolSchemas(live, child).some((t) => t.name === name)) {
        def = { kind: builtin.kind, access: builtin.access, name };
        live.log.append("tool.call", { toolCallId: id, name, kind: builtin.kind, arguments: args, ignoredArgs: ignoredArgs(builtin, args), ...(aliasOf ? { aliasOf } : {}) }, { who });
        prepared = await builtin.prepare(args, this.toolContext(live, id, signal, child));
      } else {
        live.log.append("tool.call", { toolCallId: id, name, kind: "other", arguments: args }, { who });
        await emit({ sessionUpdate: "tool_call", toolCallId: id, title: name, kind: "other", status: "failed", rawInput: args });
        const msg = `Error: unknown or unavailable tool ${name}. Available: ${this.toolSchemas(live, child).map((t) => t.name).join(", ")}`;
        record("error", msg, { type: "unknown_tool", message: `unknown tool ${name}` });
        return { output: msg };
      }
    } catch (err) {
      const msg = errMessage(err);
      await emit({ sessionUpdate: "tool_call", toolCallId: id, title: `${name} failed`, kind: def?.kind ?? "other", status: "failed", rawInput: args, content: [textContent(msg)] });
      record("error", `Error: ${msg}`, errorInfo(err));
      return { output: `Error: ${msg}` };
    }
    if (def.access === "mcp" || name.startsWith("mcp__")) live.log.append("tool.call", { toolCallId: id, name, kind: def.kind, arguments: args }, { who });

    child?.onActivity(prepared.title);
    await emit({
      sessionUpdate: "tool_call",
      toolCallId: id,
      title: prepared.title,
      kind: def.kind,
      status: "pending",
      rawInput: args,
      ...(prepared.locations ? { locations: prepared.locations } : {}),
      ...(prepared.preview ? { content: prepared.preview } : {}),
    });

    const gate = await this.authorize(live, def, args, prepared, id);
    trace("permission", { session: rec.id, tool: name, id, gate, title: prepared.title });
    if (gate !== "allow") {
      await emit({ sessionUpdate: "tool_call_update", toolCallId: id, status: "failed", content: [textContent(gate === "plan" ? "Blocked in plan mode." : "Declined by user.")] });
      const output =
        gate === "plan"
          ? "Error: plan mode is read-only. Describe the change in your plan instead."
          : "The user declined this tool call. Do not retry it; adjust your approach or ask the user.";
      record(gate === "cancelled" ? "cancelled" : "denied", output);
      if (gate === "cancelled") throw new Cancelled();
      return { output };
    }

    await emit({ sessionUpdate: "tool_call_update", toolCallId: id, status: "in_progress" });
    try {
      const t0 = Date.now();
      const out = await prepared.execute();
      record(out.isError ? "error" : "ok", out.output, out.isError ? { type: name === "bash" ? "nonzero_exit" : "tool_reported", message: out.output.split("\n").slice(-3).join(" ").slice(0, 500) } : undefined);
      if (!child && !out.isError) {
        const cs = live.claimState;
        if (name === "edit" || name === "write") cs.edits++;
        else if (name === "bash" && CHECK_RE.test(String(args.command ?? ""))) cs.verifiedEdits = cs.edits;
      }
      trace("tool", { session: rec.id, who, tool: name, id, ms: Date.now() - t0, isError: out.isError === true, args: clip(JSON.stringify(args), 2000), output: clip(out.output, 4000) });
      await emit({
        sessionUpdate: "tool_call_update",
        toolCallId: id,
        status: out.isError ? "failed" : "completed",
        content: out.content ?? [textContent(out.output.slice(0, 8000))],
        rawOutput: { output: out.output.length > 20_000 ? `${out.output.slice(0, 20_000)}…` : out.output },
      });
      const aliasNote = aliasOf ? `\n\n[note: there is no \`${aliasOf}\` tool; this ran \`${name}\`. Call \`${name}\` directly.]` : "";
      return { output: out.output + aliasNote, ...(out.images ? { images: out.images } : {}) };
    } catch (err) {
      const msg = errMessage(err);
      await emit({ sessionUpdate: "tool_call_update", toolCallId: id, status: "failed", content: [textContent(msg)] });
      record(signal.aborted ? "cancelled" : "error", `Error: ${msg}`, signal.aborted ? undefined : errorInfo(err));
      if (signal.aborted) throw new Cancelled();
      return { output: `Error: ${msg}` };
    }
  }

  /**
   * verify_claims guard: the agent may not end a turn claiming success after
   * editing without a successful check since the last edit. One nudge per turn.
   */
  private claimGuard(live: LiveSession, conv: Conv, text: string): boolean {
    const cs = live.claimState;
    if (!live.guards.has("verify_claims") || cs.nudged || cs.edits === 0 || cs.verifiedEdits >= cs.edits) return false;
    if (!SUCCESS_CLAIM_RE.test(text)) return false;
    cs.nudged = true;
    live.log.append("guard", { guard: "verify_claims", action: "nudge", edits: cs.edits, verifiedEdits: cs.verifiedEdits, claim: clip(text, 400) });
    conv.messages.push({
      role: "user",
      content:
        "[vdom guard: verify_claims] You are claiming success, but no check (test/build/typecheck/lint) has succeeded since your last edit. Run the relevant check now and report its real result — or explicitly retract the claim. Do not claim anything you have not observed.",
    });
    return true;
  }

  private toolContext(live: LiveSession, toolCallId: string, signal: AbortSignal, child?: ChildOpts): ToolContext {
    const rec = live.rec;
    return {
      cfg: this.cfg,
      cwd: rec.cwd,
      roots: rec.roots,
      signal,
      env: { VDOM_SESSION_ID: rec.id, VDOM_MODEL: rec.model, VDOM_MODE: rec.mode },
      jobs: live.jobs,
      progress: (content) => {
        if (!child) void this.update(rec.id, { sessionUpdate: "tool_call_update", toolCallId, content });
      },
      setPlan: async (entries: PlanEntry[]) => {
        await this.update(rec.id, { sessionUpdate: "plan", entries });
      },
      getGraph: () => describeGraph(rec.graph),
      setGraph: (raw) => {
        const r = applyGraph(rec.graph, raw);
        if (!r.ok) return { ok: false, message: `Rejected: ${r.reason}` };
        rec.graph = r.graph;
        saveRepoGraph(rec.cwd, r.graph);
        return { ok: true, message: `Reconciled AgentGraph ${r.graph.id} v${r.graph.version}:\n${r.diff}\nThe new rules apply from your next step.` };
      },
      ...(child ? {} : { runSubagent: (description: string, prompt: string) => this.runSubagent(live, description, prompt, signal, toolCallId) }),
      history: (q) => this.historyFor(live, q),
    };
  }

  private async runSubagent(live: LiveSession, description: string, prompt: string, signal: AbortSignal, parentToolCallId: string): Promise<string> {
    const conv: Conv = { messages: [{ role: "user", content: prompt }], compaction: { readFiles: [], modifiedFiles: [] } };
    const lines: string[] = [];
    const onActivity = (line: string) => {
      lines.push(line);
      void this.update(live.rec.id, {
        sessionUpdate: "tool_call_update",
        toolCallId: parentToolCallId,
        content: [textContent(lines.slice(-12).map((l) => `→ ${l}`).join("\n"))],
      });
    };
    const stop = await this.runLoop(live, conv, signal, { label: description, onActivity });
    const last = [...conv.messages].reverse().find((m) => m.role === "assistant" && typeof m.content === "string" && m.content.trim());
    const report = last && last.role === "assistant" ? (last.content ?? "") : "";
    return stop === "end_turn" ? report : `${report}\n\n[subagent stopped: ${stop}]`;
  }

  /** allow | declined | cancelled | plan */
  private async authorize(
    live: LiveSession,
    def: { kind: ToolKind; access: Access | "mcp"; name: string },
    args: Record<string, unknown>,
    prepared: Prepared,
    toolCallId: string,
  ): Promise<"allow" | "declined" | "cancelled" | "plan"> {
    const mode = live.rec.mode;
    if (def.access === "read" || def.access === "meta") return "allow";
    if (mode === "plan" && (def.access === "edit" || def.access === "mcp")) return "plan";
    if (this.cfg.fullAccess) return "allow";
    const key = permissionKey(def.name, def.access, args);
    if (live.alwaysAllow.has(key)) return "allow";
    const ask = mode === "ask" || mode === "plan" || def.access === "exec" || prepared.outsideWorkspace === true;
    if (!ask) return "allow";

    const options: PermissionOption[] = [
      { optionId: "allow_once", name: "Allow", kind: "allow_once" },
      { optionId: "allow_always", name: alwaysLabel(def.name, key), kind: "allow_always" },
      { optionId: "reject_once", name: "Reject", kind: "reject_once" },
    ];
    const tAsk = Date.now();
    live.log.append("permission.request", { toolCallId, tool: def.name, title: prepared.title, options: options.map((o) => o.kind) });
    const res = await this.conn.requestPermission({
      sessionId: live.rec.id,
      toolCall: {
        toolCallId,
        title: prepared.title,
        kind: def.kind,
        status: "pending",
        rawInput: args,
        ...(prepared.locations ? { locations: prepared.locations } : {}),
        ...(prepared.preview ? { content: prepared.preview } : {}),
      },
      options,
    });
    live.log.append("permission.decision", {
      toolCallId,
      tool: def.name,
      outcome: res.outcome.outcome === "cancelled" ? "cancelled" : res.outcome.optionId,
      decidedBy: "user",
      latencyMs: Date.now() - tAsk,
    });
    if (res.outcome.outcome === "cancelled") return "cancelled";
    if (res.outcome.optionId === "allow_always") {
      live.alwaysAllow.add(key);
      return "allow";
    }
    return res.outcome.optionId === "allow_once" ? "allow" : "declined";
  }

  // ------------------------------------------------------------- prompt & context

  private systemPrompt(live: LiveSession, child?: ChildOpts): string {
    const rec = live.rec;
    const tools = this.toolSchemas(live, child)
      .map((s) => TOOLS_BY_NAME.get(s.name))
      .filter((t): t is ToolDef => Boolean(t))
      .map((t) => ({ name: t.schema.name, snippet: t.snippet, guidelines: t.guidelines }));
    const childNote = child?.note
      ? child.note
      : child
      ? "You are a subagent: a parent agent delegated one self-contained task to you with a fresh context. Work autonomously — you cannot ask the user anything. Finish with a concise report for the parent: findings, files changed (paths), and anything unresolved."
      : undefined;
    return compileSystemPrompt(
      rec.graph,
      {
        cwd: rec.cwd,
        roots: rec.roots,
        platform: process.platform,
        shell: this.cfg.shell,
        model: rec.model,
        mode: rec.mode,
        date: new Date().toISOString().slice(0, 10),
        gitBranch: gitBranch(rec.cwd),
      },
      {
        ...(live.res.system ? { system: live.res.system } : {}),
        ...(live.res.append || childNote ? { append: [live.res.append, childNote].filter(Boolean).join("\n\n") } : {}),
        contextFiles: live.res.contextFiles,
        skills: skillsPromptSection(live.res.skills),
        tools,
        mcp: live.mcp.describe(),
        self: selfNote(rec.cwd),
      },
    );
  }

  private toolSchemas(live: LiveSession, child?: ChildOpts): ToolSchema[] {
    const mode = live.rec.mode;
    const builtins = TOOLS.filter((t) => !(mode === "plan" && t.access === "edit") && !(child && CHILD_EXCLUDED.has(t.schema.name)) && !(child?.allow && !child.allow.has(t.schema.name))).map((t) => t.schema);
    const mcp = child?.allow ? [] : live.mcp.tools().filter((t) => !(mode === "plan" && !t.readOnly)).map((t) => t.schema);
    return [...builtins, ...mcp];
  }

  private async maybeCompact(live: LiveSession, conv: Conv, signal: AbortSignal, system: string, child?: ChildOpts): Promise<void> {
    const estimate = estimateTokens([{ role: "system", content: system }, ...conv.messages]);
    const used = Math.max(child ? 0 : (live.lastPromptTokens ?? 0), estimate);
    if (used <= this.cfg.contextTokens - RESERVE_TOKENS) return;
    await this.compactNow(live, conv, signal, undefined, child, "threshold");
  }

  private async compactNow(live: LiveSession, conv: Conv, signal: AbortSignal, instructions?: string, child?: ChildOpts, trigger: "threshold" | "overflow_error" | "manual" = "manual") {
    const rec = live.rec;
    const tC = Date.now();
    const tokensBefore = estimateTokens(conv.messages);
    live.log.append("compaction.start", { trigger, tokensBefore, contextWindow: this.cfg.contextTokens, lastPromptTokens: child ? undefined : live.lastPromptTokens }, { who: whoOf(child) });
    if (!child) await this.update(rec.id, { sessionUpdate: "agent_thought_chunk", content: { type: "text", text: "Compacting conversation history…\n" } });
    const r = await compact(this.cfg, rec.model, conv.messages, conv.compaction, signal, instructions);
    trace("compaction", { session: rec.id, child: Boolean(child), instructions, result: r ? { before: r.tokensBefore, after: r.tokensAfter } : null });
    live.log.append("compaction.end", { trigger, ok: Boolean(r), tokensBefore: r?.tokensBefore ?? tokensBefore, tokensAfter: r?.tokensAfter, summary: r ? clip(conv.compaction.summary, 6000) : undefined, files: { read: conv.compaction.readFiles, modified: conv.compaction.modifiedFiles }, durationMs: Date.now() - tC, ...(instructions ? { instructions } : {}) }, { who: whoOf(child) });
    if (!r) return undefined;
    conv.messages = r.messages;
    if (!child) {
      live.lastPromptTokens = undefined;
      await this.store.save(rec);
    }
    return r;
  }

  // ------------------------------------------------------------- helpers

  private activate(rec: SessionRecord, mcpServers?: McpServer[]): LiveSession {
    const mcp = new McpHub();
    mcp.start(collectSpecs(vdomHome(), rec.cwd, mcpServers), rec.cwd);
    const log = new EventLog(this.store.dirFor(rec));
    if (log.turn === 0 && !readEvents(log.dir).length) {
      log.append("session", {
        format: "vdom-session",
        version: 1,
        sessionId: rec.id,
        createdAt: rec.createdAt,
        cwd: rec.cwd,
        roots: rec.roots,
        harness: { name: AGENT_NAME, version: AGENT_VERSION, root: HARNESS_ROOT, gitSha: gitHead(HARNESS_ROOT) },
        ...(rec.forkedFrom ? { parent: { sessionId: rec.forkedFrom, kind: "fork" } } : {}),
      });
      log.append("config.change", { reason: "init", model: rec.model, mode: rec.mode, thought: rec.thought, contextWindow: this.cfg.contextTokens, provider: this.cfg.providerName });
    }
    const live: LiveSession = {
      rec,
      pending: [],
      alwaysAllow: new Set(),
      jobs: new JobManager(),
      mcp,
      res: loadResources(rec.cwd),
      repeat: { sig: "", count: 0 },
      log,
      turnTools: [],
      notices: [],
      background: new Set(),
      diagnosed: new Set(),
      guards: new Set(this.cfg.guards),
      claimState: { edits: 0, verifiedEdits: 0, nudged: false },
    };
    this.sessions.set(rec.id, live);
    return live;
  }

  private live(sessionId: string): LiveSession {
    const live = this.sessions.get(sessionId);
    if (!live) throw RequestError.resourceNotFound(`session ${sessionId}`);
    return live;
  }

  private async attach(sessionId: string, cwd: string, extra?: string[], mcpServers?: McpServer[]): Promise<LiveSession> {
    const existing = this.sessions.get(sessionId);
    if (existing) return existing;
    const rec = await this.store.load(sessionId);
    if (!rec) throw RequestError.resourceNotFound(`session ${sessionId}`);
    rec.cwd = cwd || rec.cwd;
    rec.roots = [rec.cwd, ...(extra ?? rec.roots.slice(1))];
    rec.thought ??= "medium";
    rec.compaction ??= { readFiles: [], modifiedFiles: [] };
    repairToolPairs(rec.messages);
    return this.activate(rec, mcpServers);
  }

  /** Re-emit history so the client can rebuild the transcript (session/load). */
  private async replay(rec: SessionRecord): Promise<void> {
    for (const m of rec.messages) {
      if (m.role === "user") {
        const text = typeof m.content === "string" ? m.content : m.content.map((p) => (p.type === "text" ? p.text : "[image]")).join("\n");
        await this.update(rec.id, { sessionUpdate: "user_message_chunk", content: { type: "text", text } });
      } else if (m.role === "assistant") {
        if (m.content) await this.update(rec.id, { sessionUpdate: "agent_message_chunk", content: { type: "text", text: m.content } });
        for (const c of m.tool_calls ?? []) {
          const def = TOOLS_BY_NAME.get(c.function.name);
          await this.update(rec.id, {
            sessionUpdate: "tool_call",
            toolCallId: c.id,
            title: c.function.name,
            kind: def?.kind ?? "other",
            status: "completed",
            rawInput: safeJson(c.function.arguments),
          });
        }
      }
    }
  }

  /** Slash commands for the client's command palette. Sent after the session response. */
  private announceCommands(live: LiveSession): void {
    const commands: AvailableCommand[] = [
      { name: "compact", description: "Summarize older conversation to free context", input: { hint: "optional focus for the summary" } },
      { name: "reload", description: "Reload AGENTS.md, skills, and prompt templates" },
      ...live.res.skills.map((s) => ({ name: `skill:${s.name}`, description: s.description.slice(0, 200), input: { hint: "optional arguments" } })),
      ...live.res.templates.map((t) => ({ name: t.name, description: t.description, ...(t.argumentHint ? { input: { hint: t.argumentHint } } : {}) })),
    ];
    setTimeout(() => void this.update(live.rec.id, { sessionUpdate: "available_commands_update", availableCommands: commands }), 20);
  }

  private async applyMode(live: LiveSession, modeId: string): Promise<void> {
    const mode = MODES.find((m) => m.id === modeId);
    if (!mode) throw RequestError.invalidParams(undefined, `unknown mode ${modeId}`);
    live.rec.mode = mode.id;
    live.log.append("config.change", { reason: "user", mode: mode.id });
    await this.update(live.rec.id, { sessionUpdate: "current_mode_update", currentModeId: mode.id });
  }

  private modeState(rec: SessionRecord): SessionModeState {
    return { currentModeId: rec.mode, availableModes: MODES.map((m) => ({ id: m.id, name: m.name, description: m.description })) };
  }

  private async availableModels(): Promise<string[]> {
    if (this.cfg.models.length > 0) return this.cfg.models;
    if (!this.modelCache || Date.now() - this.modelCache.at > 10 * 60_000) {
      const ids = await listModels(this.cfg);
      if (ids.length > 0 || !this.modelCache) this.modelCache = { at: Date.now(), ids };
    }
    const ids = this.modelCache.ids;
    return ids.includes(this.cfg.model) ? ids : [this.cfg.model, ...ids];
  }

  private async configOptions(rec: SessionRecord): Promise<SessionConfigOption[]> {
    const models = await this.availableModels();
    const list = models.includes(rec.model) ? models : [rec.model, ...models];
    return [
      { id: "model", name: "Model", category: "model", type: "select", currentValue: rec.model, options: list.map((m) => ({ value: m, name: m })) },
      {
        id: "mode",
        name: "Mode",
        category: "mode",
        type: "select",
        currentValue: rec.mode,
        options: MODES.map((m) => ({ value: m.id, name: m.name, description: m.description })),
      },
      {
        id: "thought_level",
        name: "Thinking",
        category: "thought_level",
        type: "select",
        currentValue: rec.thought ?? "medium",
        options: THOUGHT_LEVELS.map((l) => ({ value: l, name: THOUGHT_NAMES[l] })),
      },
    ];
  }

  private async say(sessionId: string, text: string): Promise<void> {
    await this.update(sessionId, { sessionUpdate: "agent_message_chunk", content: { type: "text", text } });
  }

  private async update(sessionId: string, update: SessionUpdate): Promise<void> {
    try {
      await this.conn.sessionUpdate({ sessionId, update });
    } catch (err) {
      process.stderr.write(`vdom: session/update failed: ${String(err)}\n`);
    }
  }
}

// ------------------------------------------------------------- pure helpers

/** Commands that count as verification for the verify_claims guard. */
const CHECK_RE = /\b(test|tests|tsc|typecheck|type-check|build|lint|check|pytest|vitest|jest|mocha|cargo (test|check|build)|go (test|build|vet)|make|ruff|mypy|eslint|selftest)\b/i;
const SUCCESS_CLAIM_RE =
  /\b(all )?(tests?|checks?|suite|build|typecheck)( now)? (pass(es|ed|ing)?|succeed(s|ed)?|green)\b|\bpasses\b|\bnow (works|passes)\b|\b(is|are) (now )?(fixed|working)\b|\bfixed (it|the)\b|\bverified\b/i;

function whoOf(child?: ChildOpts): string {
  if (!child) return "main";
  return child.label === "diagnosis" ? "diagnosis" : `subagent:${child.label}`;
}

function blocksText(blocks: ContentBlock[]): string {
  return blocks
    .map((b) => (b.type === "text" ? b.text : b.type === "resource_link" ? `[${b.uri}]` : b.type === "resource" ? `[${b.resource.uri}]` : `[${b.type}]`))
    .join("\n");
}

function appendText(content: string | ChatPart[], extra: string): string | ChatPart[] {
  return typeof content === "string" ? content + extra : [...content, { type: "text", text: extra }];
}

function gitHead(dir: string): string | undefined {
  try {
    const r = spawnSync("git", ["rev-parse", "--short=12", "HEAD"], { cwd: dir, encoding: "utf8", timeout: 3000, windowsHide: true });
    return r.status === 0 ? r.stdout.trim() : undefined;
  } catch {
    return undefined;
  }
}

/** Last balanced {...} object in model output. */
export function parseJsonObject(text: string): Record<string, unknown> | undefined {
  const fenced = text.match(/```(?:json)?\s*(\{[\s\S]*?\})\s*```\s*$/);
  const candidates = fenced ? [fenced[1]!] : [];
  for (let start = text.lastIndexOf("{"); start >= 0; start = text.lastIndexOf("{", start - 1)) {
    let depth = 0;
    let inStr = false;
    for (let i = start; i < text.length; i++) {
      const c = text[i];
      if (inStr) {
        if (c === "\\") i++;
        else if (c === '"') inStr = false;
        continue;
      }
      if (c === '"') inStr = true;
      else if (c === "{") depth++;
      else if (c === "}" && --depth === 0) {
        candidates.push(text.slice(start, i + 1));
        break;
      }
    }
    if (candidates.length > (fenced ? 1 : 0)) {
      // prefer the outermost object that parses and mentions a known key
      for (const cand of candidates) {
        try {
          const o = JSON.parse(cand) as Record<string, unknown>;
          if (o && typeof o === "object" && ("issue" in o || "session_rule" in o || "what_happened" in o)) return o;
        } catch {
          /* keep scanning */
        }
      }
    }
    if (start === 0) break;
  }
  return undefined;
}

function classifyLlmError(err: unknown): string {
  const status = err instanceof LlmError ? err.status : undefined;
  const msg = err instanceof Error ? err.message : String(err);
  if (isContextOverflow(msg)) return "context_overflow";
  if (status === 401 || status === 403) return "auth";
  if (status === 402) return "billing";
  if (status === 429) return "rate_limit";
  if (status && status >= 500) return "server";
  if (status && status >= 400) return "bad_request";
  if (/timeout|timed out/i.test(msg)) return "timeout";
  return "network";
}

function errorInfo(err: unknown): { type: string; message: string; stack?: string } {
  const message = errMessage(err);
  if (err instanceof ToolInputError || (err as NodeJS.ErrnoException)?.code === "ENOENT") return { type: "input", message };
  return { type: "exception", message, ...(err instanceof Error && err.stack ? { stack: clip(err.stack, 3000) } : {}) };
}

/** Argument aliases tools accept on purpose (not "ignored"). */
const ARG_ALIASES: Record<string, string[]> = {
  edit: ["oldText", "newText", "old_string", "new_string"],
  grep: ["ignore_case", "max_results"],
  find: ["max_results"],
  bash: ["timeout_ms", "description"],
};

/** Arguments the model sent that the tool's schema does not define (silently dropped by the tool). */
function ignoredArgs(def: ToolDef, args: Record<string, unknown>): string[] {
  const props = (def.schema.parameters as { properties?: Record<string, unknown> }).properties ?? {};
  const ok = new Set([...Object.keys(props), ...(ARG_ALIASES[def.schema.name] ?? [])]);
  return Object.keys(args).filter((k) => !ok.has(k));
}

/** Repository root of the running vdom harness (src/acp or dist/acp → ../..). */
export const HARNESS_ROOT = fileURLToPath(new URL("../..", import.meta.url)).replace(/[\\/]$/, "");

const samePath = (a: string, b: string) => (process.platform === "win32" ? resolve(a).toLowerCase() === resolve(b).toLowerCase() : resolve(a) === resolve(b));

/** Where "yourself" is: the running copy vs. a checkout of the harness in the working directory. */
function selfNote(cwd: string): string {
  const running = HARNESS_ROOT.replace(/\\/g, "/");
  const lines = [`You are running from the vdom harness at ${running} (agent loop src/acp/agent.ts, tools src/acp/tools.ts, prompt src/acp/graph.ts, session log src/acp/history.ts).`];
  const root = repoRootOf(cwd);
  const isHarnessCheckout = root !== undefined && existsSync(join(root, "src", "acp", "agent.ts"));
  if (isHarnessCheckout && !samePath(root, HARNESS_ROOT)) {
    lines.push(
      `The working directory is a separate checkout of your own harness (${root.replace(/\\/g, "/")}). When asked to improve or fix yourself, change code THERE — never edit the running copy at ${running}. Changes reach the running agent only through the gated staging → prod promotion.`,
    );
  } else {
    lines.push("If the user asks you to debug or improve yourself, that is the code to change.");
  }
  lines.push(
    'To reproduce harness behaviour, drive a dev build over ACP from the checkout root: `node --import tsx src/acp/cli.ts client --cwd <dir> -V "<prompt>"`, and read recorded sessions with the history tool or `vdom sessions show <id>`.',
    "Verify with the harness tests (`npm run build`, `npm test`) and report only output you observed.",
  );
  return lines.join("\n");
}

/** Graph rules written with set_agent_graph persist per repository across sessions. */
function repoGraphPath(cwd: string): string {
  const root = repoRootOf(cwd) ?? cwd;
  const key = createHash("sha1").update(process.platform === "win32" ? root.toLowerCase() : root).digest("hex").slice(0, 16);
  return join(vdomHome(), "graphs", `${key}.json`);
}

function repoRootOf(cwd: string): string | undefined {
  let d = resolve(cwd);
  for (;;) {
    if (existsSync(join(d, ".git"))) return d;
    const up = dirname(d);
    if (up === d) return undefined;
    d = up;
  }
}

function loadRepoGraph(cwd: string): AgentGraph | undefined {
  try {
    const raw = JSON.parse(readFileSync(repoGraphPath(cwd), "utf8")) as AgentGraph;
    return raw?.root?.key === "coder" ? raw : undefined;
  } catch {
    return undefined;
  }
}

function saveRepoGraph(cwd: string, g: AgentGraph): void {
  try {
    const p = repoGraphPath(cwd);
    mkdirSync(dirname(p), { recursive: true });
    writeFileSync(p, JSON.stringify({ ...g, meta: { ...(g.meta ?? {}), repo: repoRootOf(cwd) ?? cwd } }, null, 2));
  } catch (err) {
    process.stderr.write(`vdom: could not persist agent graph: ${String(err)}\n`);
  }
}

function loadResources(cwd: string): Resources {
  const home = vdomHome();
  return {
    contextFiles: loadContextFiles(home, cwd),
    skills: loadSkills(home, cwd),
    templates: loadTemplates(home, cwd),
    system: loadPromptFile(home, cwd, "SYSTEM.md"),
    append: loadPromptFile(home, cwd, "APPEND_SYSTEM.md"),
  };
}

function textContent(t: string): ToolCallContent {
  return { type: "content", content: { type: "text", text: t } };
}

function errMessage(err: unknown): string {
  if (err instanceof ToolInputError) return err.message;
  const e = err as NodeJS.ErrnoException;
  if (e?.code === "ENOENT") return `no such file or directory: ${e.path ?? ""}`;
  return err instanceof Error ? err.message : String(err);
}

function safeJson(s: string): unknown {
  try {
    return JSON.parse(s);
  } catch {
    return s;
  }
}

function permissionKey(name: string, access: Access | "mcp", args: Record<string, unknown>): string {
  if (access === "edit") return "edit";
  if (name === "bash") {
    const words = String(args.command ?? "").trim().split(/\s+/).slice(0, 2);
    return `bash:${words.join(" ")}`;
  }
  return name;
}

function alwaysLabel(name: string, key: string): string {
  if (key === "edit") return "Allow all edits this session";
  if (key.startsWith("bash:")) return `Always allow \`${key.slice(5)}\` this session`;
  return `Always allow ${name} this session`;
}

export function promptToChat(blocks: ContentBlock[]): string | ChatPart[] {
  const parts: ChatPart[] = [];
  for (const b of blocks) {
    switch (b.type) {
      case "text":
        parts.push({ type: "text", text: b.text });
        break;
      case "image":
        parts.push({ type: "image_url", image_url: { url: `data:${b.mimeType};base64,${b.data}` } });
        break;
      case "resource_link":
        parts.push({ type: "text", text: `[Referenced: ${uriToPath(b.uri)}${b.description ? ` — ${b.description}` : ""}]` });
        break;
      case "resource": {
        const r = b.resource;
        if ("text" in r) parts.push({ type: "text", text: `<file path="${uriToPath(r.uri)}">\n${r.text}\n</file>` });
        else if (r.mimeType?.startsWith("image/")) parts.push({ type: "image_url", image_url: { url: `data:${r.mimeType};base64,${r.blob}` } });
        else parts.push({ type: "text", text: `[Attached binary resource: ${uriToPath(r.uri)}]` });
        break;
      }
      default:
        break;
    }
  }
  if (parts.every((p) => p.type === "text")) return parts.map((p) => (p as { text: string }).text).join("\n\n");
  return parts;
}

function uriToPath(uri: string): string {
  if (uri.startsWith("file://")) {
    try {
      return fileURLToPath(uri);
    } catch {
      return uri;
    }
  }
  return uri;
}

function titleFrom(blocks: ContentBlock[]): string {
  const t = blocks.find((b) => b.type === "text");
  const s = t && t.type === "text" ? t.text.trim().split("\n")[0]! : "New session";
  return s.length > 80 ? `${s.slice(0, 77)}…` : s;
}

/**
 * OpenAI-compatible APIs reject an assistant tool_call without a matching tool
 * message. After a crash or cancel, fill gaps so the session stays usable.
 */
export function repairToolPairs(msgs: ChatMessage[]): void {
  for (let i = 0; i < msgs.length; i++) {
    const m = msgs[i]!;
    if (m.role !== "assistant" || !m.tool_calls?.length) continue;
    const answered = new Set<string>();
    let j = i + 1;
    while (j < msgs.length && msgs[j]!.role === "tool") {
      answered.add((msgs[j] as { tool_call_id: string }).tool_call_id);
      j++;
    }
    const missing = m.tool_calls.filter((c) => !answered.has(c.id));
    if (missing.length) {
      msgs.splice(j, 0, ...missing.map((c) => ({ role: "tool" as const, tool_call_id: c.id, content: "Cancelled by user." })));
      i = j + missing.length - 1;
    }
  }
}

const branchCache = new Map<string, { at: number; branch?: string }>();
function gitBranch(cwd: string): string | undefined {
  const hit = branchCache.get(cwd);
  if (hit && Date.now() - hit.at < 5000) return hit.branch;
  let branch: string | undefined;
  try {
    const r = spawnSync("git", ["rev-parse", "--abbrev-ref", "HEAD"], { cwd, encoding: "utf8", timeout: 3000, windowsHide: true });
    if (r.status === 0) branch = r.stdout.trim() || undefined;
  } catch {
    /* not a repo / no git */
  }
  branchCache.set(cwd, { at: Date.now(), branch });
  return branch;
}
