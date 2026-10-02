import { spawn, spawnSync } from "node:child_process";
import { appendFileSync, writeFileSync } from "node:fs";
import { createInterface } from "node:readline/promises";
import { Readable, Writable } from "node:stream";
import {
  ClientSideConnection,
  PROTOCOL_VERSION,
  ndJsonStream,
  type AnyMessage,
  type Client,
  type McpServer,
  type RequestPermissionRequest,
  type SessionNotification,
  type Stream,
  type ToolCallContent,
} from "@agentclientprotocol/sdk";

/**
 * A scriptable ACP client: drive any ACP agent (vdom itself, a dev build, or
 * another agent) and see exactly what it does. Used for debugging and for
 * letting one agent drive another.
 */
export type ApprovePolicy = "allow" | "always" | "reject" | "prompt";

export type DriveOptions = {
  cwd: string;
  /** Prompts sent one after another in the same session. */
  prompts: string[];
  /** Load an existing session (replays history) instead of creating one. */
  session?: string;
  /** Resume the most recent session for cwd. */
  continueLast?: boolean;
  model?: string;
  mode?: string;
  thought?: string;
  approve: ApprovePolicy;
  /** Machine-readable ndjson events on stdout instead of a transcript. */
  json: boolean;
  /** Append every JSON-RPC message (both directions) to this file. */
  trace?: string;
  showThinking: boolean;
  /** Show tool arguments and longer results. */
  verbose: boolean;
  /** Cancel a prompt that runs longer than this. */
  timeoutSec?: number;
  mcpServers?: McpServer[];
  /** Called once the session is established and the first prompt is about to go out. */
  onReady?: () => void;
};

const DIM = "\x1b[2m";
const RESET = "\x1b[0m";

function tee(stream: Stream, file: string): Stream {
  const log = (dir: string) =>
    new TransformStream<AnyMessage, AnyMessage>({
      transform(msg, ctl) {
        appendFileSync(file, `${JSON.stringify({ t: new Date().toISOString(), dir, msg })}\n`);
        ctl.enqueue(msg);
      },
    });
  const outbound = log("client→agent");
  void outbound.readable.pipeTo(stream.writable).catch(() => {});
  return { writable: outbound.writable, readable: stream.readable.pipeThrough(log("agent→client")) };
}

/** Spawn an ACP agent command and connect to its stdio. */
export function spawnAgent(cmd: string[], cwd: string, env: NodeJS.ProcessEnv): { stream: Stream; kill(): void; exited: Promise<number | null> } {
  const [bin, ...args] = cmd;
  if (!bin) throw new Error("empty agent command");
  const child = spawn(bin, args, { cwd, env, stdio: ["pipe", "pipe", "inherit"], detached: process.platform !== "win32", shell: process.platform === "win32" && /\.(cmd|bat)$/i.test(bin), windowsHide: true });
  const exited = new Promise<number | null>((r) => child.on("close", (code) => r(code)));
  const stream = ndJsonStream(Writable.toWeb(child.stdin!) as WritableStream<Uint8Array>, Readable.toWeb(child.stdout!) as ReadableStream<Uint8Array>);
  // Killing only the direct child leaves the agent's own children (shells it
  // spawned, background jobs) running to edit files after a stopped run
  // (I-20261002-7e86). Take the whole process tree down instead: taskkill /T
  // on Windows, the process group on POSIX (see tools.ts killTree).
  const kill = () => {
    if (child.exitCode !== null) return;
    try {
      // Synchronous on purpose: the client may exit the moment kill() returns
      // (e.g. its own stdin closed), and a fire-and-forget taskkill is itself a
      // child of this dying process — it gets torn down before finishing its
      // sweep. tools.ts can spawn; here the caller is usually exiting.
      if (process.platform === "win32") spawnSync("taskkill", ["/pid", String(child.pid), "/T", "/F"], { stdio: "ignore", windowsHide: true });
      else process.kill(-child.pid!, "SIGKILL");
    } catch {
      /* already gone */
    }
    child.kill();
  };
  return { stream, exited, kill };
}

function contentText(c: ToolCallContent[] | null | undefined, max: number): string {
  if (!c) return "";
  const parts = c.map((x) => {
    if (x.type === "content" && x.content.type === "text") return x.content.text;
    if (x.type === "diff") {
      const before = x.oldText?.split("\n").length ?? 0;
      const after = x.newText.split("\n").length;
      return `diff ${x.path} (${before} → ${after} lines)`;
    }
    if (x.type === "terminal") return `[terminal ${x.terminalId}]`;
    return `[${x.type}]`;
  });
  const s = parts.join("\n").trim();
  return s.length > max ? `${s.slice(0, max)}… [${s.length - max} more chars]` : s;
}

function indent(s: string, pad = "    "): string {
  return s.split("\n").map((l) => pad + l).join("\n");
}

export async function drive(stream: Stream, opts: DriveOptions): Promise<{ stopReasons: string[]; sessionId: string }> {
  if (opts.trace) {
    writeFileSync(opts.trace, "");
    stream = tee(stream, opts.trace);
  }
  const started = Date.now();
  const emit = (type: string, data: Record<string, unknown>) => {
    if (opts.json) process.stdout.write(`${JSON.stringify({ t: Date.now() - started, type, ...data })}\n`);
  };
  let lastKind = "";
  const out = (kind: string, s: string) => {
    if (opts.json) return;
    if (kind !== lastKind && lastKind && !s.startsWith("\n")) process.stdout.write("\n");
    lastKind = kind;
    process.stdout.write(s);
  };
  const resultMax = opts.verbose ? 4000 : 400;
  const titles = new Map<string, string>();

  const decide = async (p: RequestPermissionRequest): Promise<string | undefined> => {
    const byKind = (k: string) => p.options.find((o) => o.kind === k)?.optionId;
    switch (opts.approve) {
      case "allow":
        return byKind("allow_once") ?? byKind("allow_always");
      case "always":
        return byKind("allow_always") ?? byKind("allow_once");
      case "reject":
        return byKind("reject_once") ?? byKind("reject_always");
      case "prompt": {
        const rl = createInterface({ input: process.stdin, output: process.stderr });
        const menu = p.options.map((o, i) => `${i + 1}) ${o.name}`).join("  ");
        const ans = await rl.question(`\n? ${p.toolCall.title}\n  ${menu}\n  > `);
        rl.close();
        return p.options[Math.max(0, Number(ans) - 1)]?.optionId ?? byKind("reject_once");
      }
    }
  };

  const client: Client = {
    async requestPermission(p) {
      const optionId = await decide(p);
      emit("permission", { toolCallId: p.toolCall.toolCallId, title: p.toolCall.title, kind: p.toolCall.kind, rawInput: p.toolCall.rawInput, decision: optionId ?? "cancelled" });
      out("perm", `\n  ? permission: ${p.toolCall.title} → ${optionId ?? "cancelled"}\n`);
      return optionId ? { outcome: { outcome: "selected", optionId } } : { outcome: { outcome: "cancelled" } };
    },
    async sessionUpdate({ update: u }: SessionNotification) {
      switch (u.sessionUpdate) {
        case "agent_message_chunk":
          if (u.content.type === "text") {
            emit("text", { text: u.content.text });
            out("text", u.content.text);
          }
          break;
        case "agent_thought_chunk":
          if (u.content.type === "text") {
            emit("thought", { text: u.content.text });
            if (opts.showThinking) out("thought", `${DIM}${u.content.text}${RESET}`);
          }
          break;
        case "user_message_chunk":
          if (u.content.type === "text") {
            emit("user_replay", { text: u.content.text });
            out("user", `\n▶ ${u.content.text.split("\n")[0]}\n`);
          }
          break;
        case "tool_call":
          titles.set(u.toolCallId, u.title);
          emit("tool_call", { id: u.toolCallId, title: u.title, kind: u.kind, status: u.status, rawInput: u.rawInput });
          out("tool", `\n⚙ [${u.kind ?? "other"}] ${u.title}${u.status === "failed" ? `  ✗ ${contentText(u.content, resultMax)}` : ""}\n`);
          if (opts.verbose && u.rawInput !== undefined) out("tool", `${DIM}${indent(JSON.stringify(u.rawInput, null, 2).slice(0, 1500))}${RESET}\n`);
          break;
        case "tool_call_update":
          emit("tool_update", { id: u.toolCallId, status: u.status, content: contentText(u.content, 20_000) });
          if (u.status === "completed" || u.status === "failed") {
            const mark = u.status === "completed" ? "✓" : "✗";
            const body = contentText(u.content, resultMax);
            out("tool", `  ${mark} ${titles.get(u.toolCallId) ?? u.toolCallId}${body ? `\n${DIM}${indent(body)}${RESET}` : ""}\n`);
          }
          break;
        case "plan":
          emit("plan", { entries: u.entries });
          out("plan", `\nPlan:\n${u.entries.map((e) => `  [${e.status === "completed" ? "x" : e.status === "in_progress" ? "~" : " "}] ${e.content}`).join("\n")}\n`);
          break;
        case "usage_update":
          emit("usage", { used: u.used, size: u.size });
          if (opts.verbose) out("usage", `${DIM}  [context ${u.used}/${u.size}]${RESET}\n`);
          break;
        case "current_mode_update":
          emit("mode", { mode: u.currentModeId });
          break;
        case "session_info_update":
          emit("session_info", { title: u.title });
          break;
        case "available_commands_update":
          emit("commands", { commands: u.availableCommands.map((c) => c.name) });
          break;
        default:
          emit(u.sessionUpdate, {});
      }
    },
  };

  const conn = new ClientSideConnection(() => client, stream);
  const init = await conn.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} });
  emit("initialized", { agent: init.agentInfo });
  await conn.authenticate({ methodId: "vdom" }).catch(() => {});

  let sessionId = opts.session;
  if (!sessionId && opts.continueLast && conn.listSessions) {
    const list = await conn.listSessions({ cwd: opts.cwd }).catch(() => undefined);
    sessionId = list?.sessions[0]?.sessionId;
  }
  const mcpServers = opts.mcpServers ?? [];
  if (sessionId) {
    await conn.loadSession({ sessionId, cwd: opts.cwd, mcpServers });
    out("info", `\n${DIM}[loaded session ${sessionId}]${RESET}\n`);
  } else {
    sessionId = (await conn.newSession({ cwd: opts.cwd, mcpServers })).sessionId;
  }
  emit("session", { sessionId });
  const sid = sessionId;

  if (opts.model) await conn.setSessionConfigOption({ sessionId: sid, configId: "model", value: opts.model });
  if (opts.mode) await conn.setSessionConfigOption({ sessionId: sid, configId: "mode", value: opts.mode });
  if (opts.thought) await conn.setSessionConfigOption({ sessionId: sid, configId: "thought_level", value: opts.thought });
  opts.onReady?.();

  const stopReasons: string[] = [];
  const onSigint = () => void conn.cancel({ sessionId: sid });
  process.on("SIGINT", onSigint);
  try {
    for (const prompt of opts.prompts) {
      out("user", `\n▶ ${prompt}\n`);
      emit("prompt", { text: prompt });
      const timer = opts.timeoutSec ? setTimeout(() => void conn.cancel({ sessionId: sid }), opts.timeoutSec * 1000) : undefined;
      try {
        const res = await conn.prompt({ sessionId: sid, prompt: [{ type: "text", text: prompt }] });
        stopReasons.push(res.stopReason);
        emit("stop", { stopReason: res.stopReason });
        out("stop", `\n${DIM}[${res.stopReason}]${RESET}\n`);
      } catch (err) {
        const msg = err instanceof Error ? err.message : JSON.stringify(err);
        stopReasons.push("error");
        emit("error", { message: msg });
        out("stop", `\n[error] ${msg}\n`);
        break;
      } finally {
        if (timer) clearTimeout(timer);
      }
    }
  } finally {
    process.off("SIGINT", onSigint);
  }
  out("info", `${DIM}session ${sid} · ${Math.round((Date.now() - started) / 1000)}s${RESET}\n`);
  emit("done", { sessionId: sid, stopReasons });
  return { stopReasons, sessionId: sid };
}
