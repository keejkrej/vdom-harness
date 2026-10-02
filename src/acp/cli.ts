#!/usr/bin/env node
import { Readable, Writable } from "node:stream";
import { resolve } from "node:path";
import {
  AgentSideConnection,
  ClientSideConnection,
  PROTOCOL_VERSION,
  ndJsonStream,
  type Client,
  type Stream,
} from "@agentclientprotocol/sdk";
import { loadConfig, type AgentConfig } from "./config.js";
import { parseArgs, splitCommand, type Parsed } from "./cli-args.js";
import { drive, spawnAgent, type DriveOptions } from "./client.js";
import { AGENT_VERSION, VdomAgent } from "./agent.js";
import { listModels, streamChat } from "./llm.js";
import { SessionStore } from "./store.js";
import { fixIssue } from "./fix.js";
import { analyze, readEvents, renderMarkdown } from "./history.js";
import { fileIssue, listIssues, loadIssue, renderIssue, saveIssue } from "./issues.js";
import { createStaging, describeEnv, dropStaging, EnvError, gateStaging, initEnv, promote, rollback } from "./envs.js";

async function sessionsCommand(p: Parsed, cfg: AgentConfig): Promise<number> {
  const store = new SessionStore(cfg.dataDir);
  const [sub, id, range] = p.positional;
  if (!sub || sub === "list") {
    for (const r of (await store.list()).filter((x) => x.messages.length > 0).slice(0, 50)) {
      process.stdout.write(`${r.id}  ${r.updatedAt.slice(0, 16)}  ${r.title ?? "(untitled)"}  [${r.cwd}]\n`);
    }
    return 0;
  }
  if (!id) {
    process.stderr.write(`usage: vdom sessions ${sub} <id> [turns A-B]\n`);
    return 2;
  }
  const dir = await store.locate(id);
  if (!dir) {
    process.stderr.write(`no session ${id}\n`);
    return 1;
  }
  const events = readEvents(dir);
  if (sub === "show") {
    const turns = range && /^\d+(-\d+)?$/.test(range) ? (range.split("-").map(Number) as number[]) : undefined;
    process.stdout.write(renderMarkdown(events, { ...(turns ? { turns: [turns[0]!, turns[1] ?? turns[0]!] as [number, number] } : {}), faultsOnly: p.ops.faults }));
    return 0;
  }
  if (sub === "analyze") {
    const findings = analyze(events);
    if (p.json) process.stdout.write(`${JSON.stringify(findings, null, 2)}\n`);
    else for (const f of findings) process.stdout.write(`[${f.severity}/${f.blame}] turn ${f.turn} ${f.kind}: ${f.message}  (${f.events.join(",")})\n`);
    return 0;
  }
  if (sub === "path") {
    process.stdout.write(`${dir}\n`);
    return 0;
  }
  process.stderr.write("usage: vdom sessions [list|show <id> [A-B]|analyze <id>|path <id>]\n");
  return 2;
}

function issuesCommand(p: Parsed): number {
  const [sub, id] = p.positional;
  if (!sub || sub === "list") {
    for (const i of listIssues()) process.stdout.write(`${i.id}  ${i.status.padEnd(7)}  ${i.severity.padEnd(6)}  ${i.blame.padEnd(7)}  ${i.title}\n`);
    return 0;
  }
  if (sub === "show" && id) {
    const i = loadIssue(id);
    if (!i) {
      process.stderr.write(`no issue ${id}\n`);
      return 1;
    }
    process.stdout.write(`${p.json ? JSON.stringify(i, null, 2) : renderIssue(i)}\n`);
    return 0;
  }
  if (sub === "new") {
    const whatHappened = p.positional.slice(1).join(" ").trim();
    if (!p.issue.title || !whatHappened) {
      process.stderr.write('usage: vdom issues new --title "..." [--blame harness|model|env|unclear] [--severity high|medium|low] [--files a,b] [--repro "..."] [--fix "..."] [--session <id> --turn <n>] "what happened"...\n');
      return 2;
    }
    const issue = fileIssue({
      sessionId: p.issue.session ?? "",
      sessionDir: "",
      turn: p.issue.turn ?? 0,
      signal: { source: "manual" },
      title: p.issue.title.slice(0, 200),
      blame: p.issue.blame ?? "unclear",
      category: "manual",
      whatHappened,
      rootCause: "",
      proposedFix: p.issue.fix ?? "",
      files: p.issue.files,
      repro: p.issue.repro ?? "",
      evidence: [],
      severity: p.issue.severity ?? "medium",
    });
    process.stdout.write(`${issue.id}\n`);
    return 0;
  }
  if (sub === "close" && id) {
    const i = loadIssue(id);
    if (!i) {
      process.stderr.write(`no issue ${id}\n`);
      return 1;
    }
    saveIssue({ ...i, status: p.issue.status ?? "fixed" });
    process.stdout.write(`${id} ${p.issue.status ?? "fixed"}\n`);
    return 0;
  }
  process.stderr.write("usage: vdom issues [list|show <id>|new --title \"...\" [options] <what happened>|close <id> [--status fixed|wontfix|duplicate]]\n");
  return 2;
}

function envCommand(p: Parsed): number {
  const [sub, name] = p.positional;
  try {
    switch (sub) {
      case "init":
        initEnv({ repo: p.ops.repo ?? process.cwd(), ...(p.ops.ref ? { ref: p.ops.ref } : {}) });
        break;
      case "status":
      case undefined:
        break;
      case "stage":
        if (!name) throw new EnvError("usage: vdom env stage <name>");
        process.stdout.write(`${createStaging(name).path}\n`);
        return 0;
      case "gate": {
        if (!name) throw new EnvError("usage: vdom env gate <name> [--repro <cmd>]…");
        const g = gateStaging(name, p.ops.repro);
        for (const s of g.steps) process.stdout.write(`${s.ok ? "✓" : "✗"} ${s.cmd} (${Math.round(s.ms / 1000)}s)${s.ok ? "" : `\n${s.tail.slice(-1500)}`}\n`);
        process.stdout.write(`gate ${g.passed ? "PASSED" : "FAILED"} at ${g.commit.slice(0, 10)}\n`);
        return g.passed ? 0 : 1;
      }
      case "promote": {
        if (!name) throw new EnvError("usage: vdom env promote <name>");
        const r = promote(name, { force: p.ops.force });
        process.stdout.write(`prod ${r.from.slice(0, 10)} → ${r.to.slice(0, 10)}\n${r.diffstat}\n`);
        return 0;
      }
      case "rollback": {
        const r = rollback();
        process.stdout.write(`prod ${r.from.slice(0, 10)} → ${r.to.slice(0, 10)} (rolled back)\n`);
        return 0;
      }
      case "drop":
        if (!name) throw new EnvError("usage: vdom env drop <name>");
        dropStaging(name);
        return 0;
      default:
        throw new EnvError(`unknown env command ${sub}; use init|status|stage|gate|promote|rollback|drop`);
    }
    process.stdout.write(`${describeEnv()}\n`);
    return 0;
  } catch (err) {
    process.stderr.write(`vdom env: ${err instanceof Error ? err.message : String(err)}\n`);
    return 1;
  }
}

/** Cursor-style date version; T3's Cursor slot gates on YYYY.MM.DD >= 2026.04.08. */
const CLI_VERSION = `2026.10.02-vdom-${AGENT_VERSION}`;

const HELP = `vdom ${AGENT_VERSION} — ACP coding agent

Usage:
  vdom [acp]                 Serve the Agent Client Protocol on stdio (default)
  vdom run "<prompt>"        Run prompt(s) on an in-process agent and print the transcript
  vdom client "<prompt>"     Same, but spawn the agent (--agent "<cmd>" for any ACP agent)
  vdom models                List models from the configured endpoint (exit 1 if unavailable)
  vdom doctor                Check config and make a test completion
  vdom about [--format json] Version / account / endpoint info
  vdom --version             Print the version

Slash commands (send as the prompt): /compact [focus], /reload, /skill:<name> [args], /<template> [args]

Options:
  --model <id>          Default model
  --base-url <url>      OpenAI-compatible endpoint (alias: -e, --endpoint)
  --cwd <dir>           Working directory for \`run\`
  --force               Never ask for permission (aliases: --yolo, --always-approve)
  --config <path>       Config file (default ~/.vdom/config.json)

Environment:
  OLLAMA_API_KEY        → https://ollama.com/v1 (Ollama Cloud)
  VDOM_BASE_URL / VDOM_API_KEY / VDOM_MODEL / VDOM_MODELS
  OPENROUTER_API_KEY, OPENAI_API_KEY (+ OPENAI_BASE_URL)
`;

function serveStdio(cfg: AgentConfig): void {
  // stdout is the protocol channel; route stray logging to stderr.
  console.log = (...args: unknown[]) => process.stderr.write(`${args.map(String).join(" ")}\n`);
  console.info = console.log;
  const stream = ndJsonStream(Writable.toWeb(process.stdout) as WritableStream<Uint8Array>, Readable.toWeb(process.stdin) as ReadableStream<Uint8Array>);
  let agent: VdomAgent | undefined;
  new AgentSideConnection((conn) => (agent = new VdomAgent(conn, cfg)), stream);
  const exit = () => {
    const done = agent?.shutdown() ?? Promise.resolve();
    void Promise.race([done, new Promise((r) => setTimeout(r, 3000))]).finally(() => process.exit(0));
  };
  process.stdin.on("end", exit);
  process.on("SIGTERM", exit);
}

/** Two in-memory ACP streams wired back to back. */
function pipePair(): [Stream, Stream] {
  const a = new TransformStream<Uint8Array, Uint8Array>();
  const b = new TransformStream<Uint8Array, Uint8Array>();
  return [ndJsonStream(a.writable, b.readable), ndJsonStream(b.writable, a.readable)];
}

function driveOptions(p: Parsed, cwd: string, prompts: string[]): DriveOptions {
  const d = p.drive;
  return {
    cwd,
    prompts,
    ...(d.session ? { session: d.session } : {}),
    continueLast: d.continueLast,
    ...(p.overrides.model ? { model: p.overrides.model } : {}),
    ...(d.mode ? { mode: d.mode } : {}),
    ...(d.thought ? { thought: d.thought } : {}),
    approve: d.approve ?? (process.stdin.isTTY && !p.json ? "prompt" : "allow"),
    json: p.json,
    ...(d.trace ? { trace: d.trace } : {}),
    showThinking: d.showThinking,
    verbose: d.verbose,
    ...(d.timeoutSec ? { timeoutSec: d.timeoutSec } : {}),
  };
}

/** `vdom run`: drive an in-process agent through a real ACP connection. */
async function runInProcess(cfg: AgentConfig, opts: DriveOptions): Promise<number> {
  if (opts.trace) process.env.VDOM_TRACE ??= `${opts.trace}.agent.ndjson`;
  const [agentSide, clientSide] = pipePair();
  let agent: VdomAgent | undefined;
  new AgentSideConnection((conn) => (agent = new VdomAgent(conn, cfg)), agentSide);
  try {
    const r = await drive(clientSide, opts);
    // Let background diagnoses (bad-turn analysis) finish before exiting.
    await agent?.settle();
    return r.stopReasons.every((x) => x === "end_turn") ? 0 : 1;
  } finally {
    await agent?.shutdown();
  }
}

/** `vdom client`: spawn an ACP agent (default: this vdom) and drive it over stdio. */
async function runClient(p: Parsed, opts: DriveOptions): Promise<number> {
  let cmd: string[];
  if (p.drive.agent) cmd = splitCommand(p.drive.agent);
  else {
    // Re-run this same entry point (works for dist/ and for tsx dev runs).
    cmd = [process.execPath, ...process.execArgv, process.argv[1]!, "acp"];
    if (p.overrides.baseUrl) cmd.push("-e", p.overrides.baseUrl);
    if (p.overrides.configPath) cmd.push("--config", p.overrides.configPath);
    if (p.overrides.fullAccess) cmd.push("--force");
  }
  const env = { ...process.env, ...(opts.trace ? { VDOM_TRACE: `${opts.trace}.agent.ndjson` } : {}) };
  const agent = spawnAgent(cmd, opts.cwd, env);
  try {
    const r = await drive(agent.stream, opts);
    return r.stopReasons.every((x) => x === "end_turn") ? 0 : 1;
  } finally {
    agent.kill();
  }
}

async function doctor(cfg: AgentConfig): Promise<number> {
  const mask = cfg.apiKey ? `${cfg.apiKey.slice(0, 4)}…${cfg.apiKey.slice(-4)}` : "(none)";
  process.stdout.write(
    [
      `vdom ${AGENT_VERSION}`,
      `provider   ${cfg.providerName}`,
      `base URL   ${cfg.baseUrl}`,
      `API key    ${mask}`,
      `model      ${cfg.model}`,
      `shell      ${cfg.shell}`,
      `sessions   ${cfg.dataDir}`,
      `fullAccess ${cfg.fullAccess}`,
      "",
    ].join("\n"),
  );
  const models = await listModels(cfg);
  process.stdout.write(`models     ${models.length ? `${models.length} available` : "listing unavailable"}\n`);
  try {
    const r = await streamChat(cfg, {
      model: cfg.model,
      tools: [],
      signal: AbortSignal.timeout(60_000),
      messages: [{ role: "user", content: "Reply with exactly: ok" }],
    });
    process.stdout.write(`completion ok (${r.model ?? cfg.model}): ${JSON.stringify(r.content.trim().slice(0, 40))}\n`);
    return 0;
  } catch (err) {
    process.stdout.write(`completion FAILED: ${err instanceof Error ? err.message : String(err)}\n`);
    return 1;
  }
}

export async function main(argv = process.argv.slice(2)): Promise<number | undefined> {
  const p = parseArgs(argv);
  const cfg = loadConfig(p.overrides);
  switch (p.command) {
    case "acp":
      serveStdio(cfg);
      return undefined;
    case "run":
    case "client": {
      const prompts = [...(p.positional.length ? [p.positional.join(" ")] : []), ...p.drive.prompts].filter((x) => x.trim());
      if (!prompts.length) {
        process.stderr.write(`usage: vdom ${p.command} "<prompt>" [-p "<next prompt>"…] [--cwd dir] [--session id | -c]\n`);
        return 2;
      }
      const opts = driveOptions(p, resolve(p.cwd ?? process.cwd()), prompts);
      return p.command === "run" ? runInProcess(cfg, opts) : runClient(p, opts);
    }
    case "env":
      return envCommand(p);
    case "sessions":
      return sessionsCommand(p, cfg);
    case "fix": {
      const issueId = p.positional[0];
      if (!issueId) {
        process.stderr.write("usage: vdom fix <issue-id> [--promote]\n");
        return 2;
      }
      try {
        const r = await fixIssue({
          issueId,
          cfg,
          promote: p.ops.promote,
          drive: async (cwd, prompt) => {
            // Staging is a disposable worktree: the fixer runs without permission prompts.
            const fixCfg = { ...cfg, fullAccess: true };
            const [agentSide, clientSide] = pipePair();
            let agent: VdomAgent | undefined;
            new AgentSideConnection((conn) => (agent = new VdomAgent(conn, fixCfg)), agentSide);
            try {
              const res = await drive(clientSide, { ...driveOptions(p, cwd, [prompt]), approve: "always", verbose: p.drive.verbose });
              return res.stopReasons;
            } finally {
              await agent?.shutdown();
            }
          },
        });
        return r.gate?.passed && r.red?.ok ? 0 : 1;
      } catch (err) {
        process.stderr.write(`vdom fix: ${err instanceof Error ? err.message : String(err)}\n`);
        return 1;
      }
    }
    case "issues":
      return issuesCommand(p);
    case "models": {
      const ids = await listModels(cfg);
      for (const m of ids) process.stdout.write(`${m}\n`);
      return ids.length ? 0 : 1;
    }
    case "doctor":
      return doctor(cfg);
    case "about": {
      const authed = Boolean(cfg.apiKey) || !cfg.baseUrl.includes("ollama.com");
      const info = {
        cliVersion: CLI_VERSION,
        version: AGENT_VERSION,
        userEmail: authed ? `${cfg.providerName} · ${cfg.model}` : null,
        authenticated: authed,
        provider: cfg.providerName,
        model: cfg.model,
        baseUrl: cfg.baseUrl,
      };
      process.stdout.write(p.json ? `${JSON.stringify(info)}\n` : `About vdom\n\nCLI Version         ${CLI_VERSION}\nUser Email          ${info.userEmail ?? "Not logged in"}\nModel               ${cfg.model}\n`);
      return 0;
    }
    case "version":
      process.stdout.write(`${AGENT_VERSION}\n`);
      return 0;
    case "help":
      process.stdout.write(HELP);
      return 0;
    default:
      process.stderr.write(`vdom: unknown command ${p.command}\n\n${HELP}`);
      return 2;
  }
}

main().then(
  (code) => {
    if (code !== undefined) process.exit(code);
  },
  (err) => {
    process.stderr.write(`vdom: ${err instanceof Error ? (err.stack ?? err.message) : String(err)}\n`);
    process.exit(1);
  },
);
