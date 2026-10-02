/**
 * End-to-end: a real `vdom acp` child process over stdio, driven by the SDK's
 * ClientSideConnection, against a scripted OpenAI-compatible streaming server.
 * No network, no API key.
 */
import assert from "node:assert/strict";
import { spawn, type ChildProcess } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, existsSync, writeFileSync } from "node:fs";
import { createServer, type IncomingMessage, type Server } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Readable, Writable } from "node:stream";
import { fileURLToPath } from "node:url";
import {
  ClientSideConnection,
  PROTOCOL_VERSION,
  ndJsonStream,
  type Client,
  type RequestPermissionRequest,
  type SessionNotification,
} from "@agentclientprotocol/sdk";
import { repairToolPairs } from "./agent.js";
import { parseArgs } from "./cli-args.js";
import { SUMMARY_PREFIX } from "./compaction.js";
import type { ChatMessage } from "./llm.js";

type Step =
  | { text: string; reasoning?: string; delayMs?: number }
  | { calls: { name: string; args: Record<string, unknown> }[]; delayMs?: number; finish?: string }
  | { status: number; body: string };

type Req = { model: string; messages: ChatMessage[]; tools?: { function: { name: string } }[]; reasoning_effort?: string };

const script: Step[] = [];
const sentimentScript: Step[] = [];
const diagnosisScript: Step[] = [];
const sentimentRequests: Req[] = [];
const diagnosisRequests: Req[] = [];
const requests: Req[] = [];

function sse(res: import("node:http").ServerResponse, chunks: unknown[]): void {
  res.writeHead(200, { "Content-Type": "text/event-stream" });
  for (const c of chunks) res.write(`data: ${JSON.stringify(c)}\n\n`);
  res.write("data: [DONE]\n\n");
  res.end();
}

async function body(req: IncomingMessage): Promise<string> {
  let s = "";
  for await (const c of req) s += c;
  return s;
}

function startMock(): Promise<{ server: Server; url: string }> {
  const server = createServer(async (req, res) => {
    if (req.url?.endsWith("/models")) {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ data: [{ id: "mock-coder" }, { id: "mock-big" }] }));
      return;
    }
    const parsed = JSON.parse(await body(req)) as Req;
    // Route by role: the interpreter and diagnostician run concurrently with serving.
    const sys = typeof parsed.messages[0]?.content === "string" ? parsed.messages[0].content : "";
    let step: Step;
    if (sys.includes("You watch a coding-agent session")) {
      sentimentRequests.push(parsed);
      step = sentimentScript.shift() ?? { text: '{"unhappy": false, "frustration": 0, "target": "none", "category": "none", "reason": ""}' };
    } else if (sys.includes("vdom's diagnostician")) {
      diagnosisRequests.push(parsed);
      step = diagnosisScript.shift() ?? { text: '{"what_happened": "n/a", "blame": "unclear", "category": "none", "session_rule": null, "issue": null, "evidence": []}' };
    } else {
      requests.push(parsed);
      step = script.shift() ?? { text: "(script exhausted)" };
    }
    if ("delayMs" in step && step.delayMs) await new Promise((r) => setTimeout(r, step.delayMs));
    if ("status" in step) {
      res.writeHead(step.status, { "Content-Type": "application/json" });
      res.end(step.body);
      return;
    }
    if ("calls" in step) {
      sse(res, [
        ...step.calls.flatMap((c, i) => {
          const args = JSON.stringify(c.args);
          // Split arguments across two deltas like real providers do.
          return [
            { choices: [{ delta: { tool_calls: [{ index: i, id: `call_${requests.length}_${i}`, function: { name: c.name, arguments: args.slice(0, 5) } }] } }] },
            { choices: [{ delta: { tool_calls: [{ index: i, function: { arguments: args.slice(5) } }] } }] },
          ];
        }),
        { choices: [{ delta: {}, finish_reason: step.finish ?? "tool_calls" }], usage: { prompt_tokens: 100, completion_tokens: 10, total_tokens: 110 } },
      ]);
      return;
    }
    const half = Math.ceil(step.text.length / 2);
    sse(res, [
      ...(step.reasoning ? [{ choices: [{ delta: { reasoning_content: step.reasoning } }] }] : []),
      { model: "mock-coder", choices: [{ delta: { content: step.text.slice(0, half) } }] },
      { choices: [{ delta: { content: step.text.slice(half) } }] },
      { choices: [{ delta: {}, finish_reason: "stop" }] },
    ]);
  });
  return new Promise((resolve) =>
    server.listen(0, "127.0.0.1", () => {
      const addr = server.address() as { port: number };
      resolve({ server, url: `http://127.0.0.1:${addr.port}/v1` });
    }),
  );
}

type Harness = {
  child: ChildProcess;
  conn: ClientSideConnection;
  updates: SessionNotification[];
  permissions: RequestPermissionRequest[];
  answer: { next: "allow_once" | "allow_always" | "reject_once" | "cancel" };
};

const here = fileURLToPath(new URL(".", import.meta.url));

function startAgent(env: NodeJS.ProcessEnv, extraArgs: string[] = []): Harness {
  const child = spawn(process.execPath, ["--import", "tsx", join(here, "cli.ts"), "acp", ...extraArgs], {
    env,
    stdio: ["pipe", "pipe", "inherit"],
  });
  const updates: SessionNotification[] = [];
  const permissions: RequestPermissionRequest[] = [];
  const answer: Harness["answer"] = { next: "allow_once" };
  let conn!: ClientSideConnection;
  const client: Client = {
    async requestPermission(p) {
      permissions.push(p);
      if (answer.next === "cancel") {
        await conn.cancel({ sessionId: p.sessionId });
        return { outcome: { outcome: "cancelled" } };
      }
      return { outcome: { outcome: "selected", optionId: answer.next } };
    },
    async sessionUpdate(n) {
      updates.push(n);
    },
  };
  const stream = ndJsonStream(Writable.toWeb(child.stdin!) as WritableStream<Uint8Array>, Readable.toWeb(child.stdout!) as ReadableStream<Uint8Array>);
  conn = new ClientSideConnection(() => client, stream);
  return { child, conn, updates, permissions, answer };
}

function texts(updates: SessionNotification[], kind: "agent_message_chunk" | "agent_thought_chunk" | "user_message_chunk"): string {
  return updates.map((u) => u.update).map((u) => (u.sessionUpdate === kind && u.content.type === "text" ? u.content.text : "")).join("");
}

const last = (): Req => requests[requests.length - 1]!;
const toolResults = (r: Req = last()): string[] => r.messages.filter((m) => m.role === "tool").map((m) => (m as { content: string }).content);
const systemOf = (r: Req = last()): string => (r.messages[0] as { content: string }).content;
const toolNames = (r: Req = last()): string[] => (r.tools ?? []).map((t) => t.function.name);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function main(): Promise<void> {
  // ---- pure helpers
  {
    const msgs: ChatMessage[] = [
      { role: "user", content: "x" },
      { role: "assistant", content: null, tool_calls: [{ id: "a", type: "function", function: { name: "bash", arguments: "{}" } }, { id: "b", type: "function", function: { name: "bash", arguments: "{}" } }] },
      { role: "tool", tool_call_id: "a", content: "ok" },
      { role: "user", content: "y" },
    ];
    repairToolPairs(msgs);
    assert.deepEqual(msgs.map((m) => m.role), ["user", "assistant", "tool", "tool", "user"]);
    assert.equal(parseArgs(["-e", "http://x", "--force", "acp"]).overrides.fullAccess, true);
    assert.equal(parseArgs(["agent", "--always-approve", "stdio"]).command, "acp");
    console.log("ok helpers");
  }

  const { server, url } = await startMock();
  const home = mkdtempSync(join(tmpdir(), "vdom-home-"));
  const ws = mkdtempSync(join(tmpdir(), "vdom-ws-"));
  writeFileSync(join(ws, "app.txt"), "alpha\nbeta\ngamma\n");
  writeFileSync(join(ws, "AGENTS.md"), "Project rule: always say please.");
  mkdirSync(join(ws, ".agents", "skills", "deploy"), { recursive: true });
  writeFileSync(join(ws, ".agents", "skills", "deploy", "SKILL.md"), "---\nname: deploy\ndescription: Deploy the app\n---\nRun ./deploy.sh then verify.");
  mkdirSync(join(ws, ".vdom", "prompts"), { recursive: true });
  writeFileSync(join(ws, ".vdom", "prompts", "review.md"), "---\ndescription: Review a file\nargument-hint: <file>\n---\nReview $1 carefully.");
  // 1x1 transparent PNG
  writeFileSync(join(ws, "pixel.png"), Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==", "base64"));
  const env = {
    ...process.env,
    VDOM_HOME: home,
    VDOM_BASE_URL: url,
    VDOM_API_KEY: "test",
    VDOM_MODEL: "mock-coder",
    VDOM_SENTIMENT_MODEL: "mock-small",
    VDOM_TRACE: "",
    OLLAMA_API_KEY: "",
    OPENROUTER_API_KEY: "",
    OPENAI_API_KEY: "",
  };
  const mcpEcho = { name: "echo", command: process.execPath, args: ["--import", import.meta.resolve("tsx"), join(here, "fixtures", "mcp-echo.ts")], env: [] };

  let h = startAgent(env);
  try {
    // ---- handshake (T3 always authenticates, with its own method id)
    const init = await h.conn.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} });
    assert.equal(init.agentCapabilities?.loadSession, true);
    assert.ok(init.agentCapabilities?.sessionCapabilities?.resume);
    assert.equal(init.agentCapabilities?.mcpCapabilities?.http, true);
    await h.conn.authenticate({ methodId: "cursor_login" });
    const avail = (await h.conn.extMethod("vdom/list_models", {})) as { models: { value: string }[] };
    assert.deepEqual(avail.models.map((m) => m.value), ["mock-big", "mock-coder"]);
    console.log("ok initialize/authenticate/list_models");

    const s = await h.conn.newSession({ cwd: ws, mcpServers: [mcpEcho] });
    const sid = s.sessionId;
    assert.equal(s.configOptions?.find((o) => o.id === "model")?.currentValue, "mock-coder");
    assert.equal(s.configOptions?.find((o) => o.id === "thought_level")?.currentValue, "medium");
    assert.equal(s.modes?.currentModeId, "agent");
    await sleep(200);
    const cmds = h.updates.map((u) => u.update).find((u) => u.sessionUpdate === "available_commands_update");
    const cmdNames = cmds && cmds.sessionUpdate === "available_commands_update" ? cmds.availableCommands.map((c) => c.name) : [];
    assert.ok(cmdNames.includes("compact") && cmdNames.includes("skill:deploy") && cmdNames.includes("review"), `commands: ${cmdNames}`);
    console.log("ok session/new + available commands (compact, skills, templates)");

    // ---- agent mode: read + edit run without asking; bash asks
    script.push(
      { calls: [{ name: "read", args: { path: "app.txt" } }, { name: "grep", args: { pattern: "beta" } }, { name: "read", args: { path: "pixel.png" } }] },
      { calls: [{ name: "edit", args: { path: "app.txt", edits: [{ oldText: "alpha", newText: "ALPHA" }, { oldText: "gamma", newText: "GAMMA" }] } }] },
      { calls: [{ name: "write", args: { path: "sub/new.txt", content: "fresh\n" } }] },
      { calls: [{ name: "todo_write", args: { todos: [{ content: "edit app", status: "completed" }, { content: "verify", status: "in_progress" }] } }] },
      { calls: [{ name: "bash", args: { command: "echo shell-$((40+2))" } }] },
      { text: "All done.", reasoning: "thinking about it" },
    );
    let r = await h.conn.prompt({ sessionId: sid, prompt: [{ type: "text", text: "Uppercase the ends" }] });
    assert.equal(r.stopReason, "end_turn");
    assert.equal(readFileSync(join(ws, "app.txt"), "utf8"), "ALPHA\nbeta\nGAMMA\n");
    assert.equal(readFileSync(join(ws, "sub", "new.txt"), "utf8"), "fresh\n");
    assert.equal(h.permissions.length, 1, "only bash should ask in agent mode");
    assert.equal(h.permissions[0]!.toolCall.kind, "execute");
    const tr = toolResults();
    assert.equal(tr[0], "alpha\nbeta\ngamma");
    assert.match(tr[1]!, /app\.txt:2: beta/);
    assert.match(tr[2]!, /Read image file \[image\/png\]/);
    assert.match(tr.join("\n"), /Successfully replaced 2 block\(s\)/);
    assert.match(tr.join("\n"), /shell-42/);
    const imgMsg = last().messages.find((m) => m.role === "user" && Array.isArray(m.content));
    assert.ok(imgMsg && Array.isArray(imgMsg.content) && imgMsg.content.some((p) => p.type === "image_url"), "image tool result attached as user image");
    assert.equal(texts(h.updates, "agent_message_chunk"), "All done.");
    assert.equal(texts(h.updates, "agent_thought_chunk"), "thinking about it");
    const ups = h.updates.map((u) => u.update);
    assert.ok(ups.some((u) => u.sessionUpdate === "tool_call_update" && u.content?.some((c) => c.type === "diff" && c.newText.includes("GAMMA"))));
    assert.ok(ups.some((u) => u.sessionUpdate === "plan" && u.entries.length === 2));
    assert.ok(ups.some((u) => u.sessionUpdate === "session_info_update" && u.title === "Uppercase the ends"));
    const sys = systemOf(requests[0]);
    assert.match(sys, /Project rule: always say please\./);
    assert.match(sys, /<available_skills>[\s\S]*deploy/);
    assert.match(sys, /<mcp_servers>[\s\S]*echo/);
    assert.ok(toolNames(requests[0]).includes("mcp__echo__echo"));
    assert.equal(requests[0]!.reasoning_effort, "medium");
    console.log("ok agent turn: parallel read/grep/image, multi-edit, write, plan, bash permission, context files, skills, MCP listing");

    // ---- tool errors flow back to the model, not thrown
    script.push({ calls: [{ name: "edit", args: { path: "app.txt", edits: [{ oldText: "nope", newText: "x" }] } }] }, { text: "ok" });
    r = await h.conn.prompt({ sessionId: sid, prompt: [{ type: "text", text: "bad edit" }] });
    assert.equal(r.stopReason, "end_turn");
    assert.match(toolResults().at(-1)!, /Could not find the exact text/);
    console.log("ok tool errors flow back");

    // ---- habitual tool name: `search` runs grep instead of failing
    script.push({ calls: [{ name: "search", args: { query: "beta", path: "." } }] }, { text: "found it" });
    await h.conn.prompt({ sessionId: sid, prompt: [{ type: "text", text: "where is beta" }] });
    assert.match(toolResults().at(-1)!, /app\.txt:2: beta/);
    assert.match(toolResults().at(-1)!, /there is no `search` tool; this ran `grep`/);
    console.log("ok tool alias (search → grep)");

    // ---- an empty response is retried, not taken as "done"
    script.push({ text: "" }, { text: "recovered from empty" });
    r = await h.conn.prompt({ sessionId: sid, prompt: [{ type: "text", text: "go on" }] });
    assert.equal(r.stopReason, "end_turn");
    assert.match(texts(h.updates, "agent_message_chunk"), /recovered from empty/);
    console.log("ok empty response retried");

    // ---- MCP tool call
    script.push({ calls: [{ name: "mcp__echo__echo", args: { text: "hi" } }] }, { text: "echoed" });
    await h.conn.prompt({ sessionId: sid, prompt: [{ type: "text", text: "echo" }] });
    assert.equal(toolResults().at(-1), "echo: hi");
    console.log("ok MCP tool call");

    // ---- template + skill expansion
    script.push({ text: "reviewed" });
    await h.conn.prompt({ sessionId: sid, prompt: [{ type: "text", text: "/review app.txt" }] });
    const lu = last().messages.filter((m) => m.role === "user").at(-1)!;
    assert.equal(lu.content, "Review app.txt carefully.");
    script.push({ text: "deploying" });
    await h.conn.prompt({ sessionId: sid, prompt: [{ type: "text", text: "/skill:deploy staging" }] });
    assert.match(String(last().messages.filter((m) => m.role === "user").at(-1)!.content), /<skill name="deploy"[\s\S]*Run \.\/deploy\.sh[\s\S]*staging$/);
    console.log("ok prompt templates + /skill:");

    // ---- thought level → reasoning_effort
    await h.conn.setSessionConfigOption({ sessionId: sid, configId: "thought_level", value: "high" });
    script.push({ text: "thought hard" });
    await h.conn.prompt({ sessionId: sid, prompt: [{ type: "text", text: "think" }] });
    assert.equal(last().reasoning_effort, "high");
    console.log("ok thought_level");

    // ---- truncated (length) tool calls are never executed
    script.push({ calls: [{ name: "write", args: { path: "trunc.txt", content: "half" } }], finish: "length" }, { text: "retrying smaller" });
    await h.conn.prompt({ sessionId: sid, prompt: [{ type: "text", text: "big write" }] });
    assert.ok(!existsSync(join(ws, "trunc.txt")));
    assert.match(toolResults().at(-1)!, /output token limit/);
    console.log("ok length-truncated tool calls skipped");

    // ---- repeat-call reminder
    script.push(
      { calls: [{ name: "ls", args: { path: "." } }] },
      { calls: [{ name: "ls", args: { path: "." } }] },
      { calls: [{ name: "ls", args: { path: "." } }] },
      { text: "stop looping" },
    );
    await h.conn.prompt({ sessionId: sid, prompt: [{ type: "text", text: "loop" }] });
    assert.match(toolResults().at(-1)!, /Reminder: this is identical ls call #3 in a row/);
    console.log("ok repeat reminder");

    // ---- subagent: fresh context, no nested subagent tool, report returned
    script.push(
      { calls: [{ name: "subagent", args: { description: "find gamma", prompt: "Find which line has GAMMA in app.txt and report." } }] },
      { calls: [{ name: "grep", args: { pattern: "GAMMA" } }] },
      { text: "GAMMA is on line 3" },
      { text: "Subagent says line 3." },
    );
    r = await h.conn.prompt({ sessionId: sid, prompt: [{ type: "text", text: "delegate" }] });
    assert.equal(r.stopReason, "end_turn");
    const childReq = requests.at(-3)!;
    assert.match(systemOf(childReq), /You are a subagent/);
    assert.ok(!toolNames(childReq).includes("subagent"));
    assert.equal(childReq.messages.length, 2, "child starts with a fresh context");
    assert.equal(toolResults().at(-1), "GAMMA is on line 3");
    console.log("ok subagent");

    // ---- background job
    script.push(
      { calls: [{ name: "bash", args: { command: "sleep 1; echo bg-done", run_in_background: true } }] },
      { calls: [{ name: "job_output", args: { job_id: "job_1", wait: 10 } }] },
      { text: "job finished" },
    );
    await h.conn.prompt({ sessionId: sid, prompt: [{ type: "text", text: "background" }] });
    assert.match(toolResults().at(-2)!, /Started job_1/);
    assert.match(toolResults().at(-1)!, /exited with code 0[\s\S]*bg-done/);
    console.log("ok background jobs");

    // ---- manual /compact with focus, then the summary leads the next request
    script.push({ text: "## Goal\nUppercase things.\n\n## Next Steps\n1. Continue" });
    r = await h.conn.prompt({ sessionId: sid, prompt: [{ type: "text", text: "/compact keep file names" }] });
    assert.equal(r.stopReason, "end_turn");
    assert.match(String(last().messages[1]!.content), /Additional focus: keep file names/);
    assert.equal(toolNames(last()).length, 0, "summary call has no tools");
    script.push({ text: "after compact" });
    await h.conn.prompt({ sessionId: sid, prompt: [{ type: "text", text: "continue" }] });
    const first = last().messages[1]!;
    assert.ok(typeof first.content === "string" && first.content.startsWith(SUMMARY_PREFIX));
    assert.match(String(first.content), /<modified-files>[\s\S]*app\.txt/);
    console.log("ok /compact");

    // ---- context overflow → compact → retry once
    script.push({ status: 400, body: '{"error":{"message":"This model\'s maximum context length is 8192 tokens"}}' }, { text: "## Goal\nsummary" }, { text: "recovered after overflow" });
    r = await h.conn.prompt({ sessionId: sid, prompt: [{ type: "text", text: "overflow" }] });
    assert.equal(r.stopReason, "end_turn");
    assert.match(texts(h.updates, "agent_message_chunk"), /recovered after overflow/);
    console.log("ok overflow recovery");

    // ---- steering: a prompt sent mid-turn is folded into the running turn
    script.push({ calls: [{ name: "ls", args: {} }], delayMs: 800 }, { text: "saw your steer" });
    const p1 = h.conn.prompt({ sessionId: sid, prompt: [{ type: "text", text: "slow task" }] });
    await sleep(300);
    const p2 = h.conn.prompt({ sessionId: sid, prompt: [{ type: "text", text: "also check README" }] });
    const [r1, r2] = await Promise.all([p1, p2]);
    assert.equal(r1.stopReason, "end_turn");
    assert.equal(r2.stopReason, "end_turn");
    assert.ok(last().messages.some((m) => m.role === "user" && m.content === "also check README"));
    console.log("ok steering");

    // ---- ask mode: rejecting an edit leaves the file alone; plan mode hides edit + mutating MCP
    await h.conn.setSessionConfigOption({ sessionId: sid, configId: "mode", value: "ask" });
    h.answer.next = "reject_once";
    script.push({ calls: [{ name: "write", args: { path: "app.txt", content: "clobbered" } }] }, { text: "understood" });
    await h.conn.prompt({ sessionId: sid, prompt: [{ type: "text", text: "overwrite" }] });
    assert.equal(readFileSync(join(ws, "app.txt"), "utf8"), "ALPHA\nbeta\nGAMMA\n");
    assert.match(toolResults().at(-1)!, /declined/);
    assert.ok(h.permissions.at(-1)!.toolCall.content?.some((c) => c.type === "diff"), "permission prompt carries the diff");
    h.answer.next = "allow_once";
    await h.conn.setSessionMode({ sessionId: sid, modeId: "plan" });
    script.push({ text: "Here is the plan." });
    await h.conn.prompt({ sessionId: sid, prompt: [{ type: "text", text: "plan it" }] });
    const pt = toolNames();
    assert.ok(!pt.includes("edit") && !pt.includes("write") && !pt.includes("mcp__echo__write_note") && pt.includes("mcp__echo__echo"));
    assert.match(systemOf(), /<plan_mode>/);
    await h.conn.setSessionMode({ sessionId: sid, modeId: "agent" });
    console.log("ok ask + plan modes");

    // ---- cancel mid-stream and during a permission prompt; history stays valid
    script.push({ text: "slow", delayMs: 3000 });
    const pc = h.conn.prompt({ sessionId: sid, prompt: [{ type: "text", text: "slow one" }] });
    await sleep(300);
    await h.conn.cancel({ sessionId: sid });
    assert.equal((await pc).stopReason, "cancelled");
    h.answer.next = "cancel";
    script.push({ calls: [{ name: "bash", args: { command: "echo never" } }] });
    r = await h.conn.prompt({ sessionId: sid, prompt: [{ type: "text", text: "cancel at prompt" }] });
    assert.equal(r.stopReason, "cancelled");
    h.answer.next = "allow_once";
    script.push({ text: "still healthy" });
    r = await h.conn.prompt({ sessionId: sid, prompt: [{ type: "text", text: "after cancel" }] });
    assert.equal(r.stopReason, "end_turn");
    const msgs = last().messages;
    msgs.forEach((m, i) => {
      if (m.role === "assistant" && m.tool_calls?.length) assert.equal(msgs[i + 1]!.role, "tool", "tool_calls always answered");
    });
    console.log("ok cancel");

    // ---- fork
    const forked = await h.conn.unstable_forkSession({ sessionId: sid, cwd: ws });
    assert.notEqual(forked.sessionId, sid);
    script.push({ text: "forked reply" });
    await h.conn.prompt({ sessionId: forked.sessionId, prompt: [{ type: "text", text: "in fork" }] });
    assert.ok(last().messages.some((m) => m.role === "user" && m.content === "after cancel"));
    console.log("ok fork");

    // ---- real-time improvement: guard, keyword → diagnosis → in-session lesson + issue, interpreter
    {
      const s3 = await h.conn.newSession({ cwd: ws, mcpServers: [] });
      const sid3 = s3.sessionId;
      writeFileSync(join(ws, "calc.js"), "export const add = (a, b) => a - b;\n");
      // Turn 1: edits, then claims success without running anything → verify_claims guard nudges once.
      script.push(
        { calls: [{ name: "edit", args: { path: "calc.js", edits: [{ oldText: "a - b", newText: "a + b" }] } }] },
        { text: "Fixed it, all tests pass." },
        { calls: [{ name: "bash", args: { command: "echo running tests && exit 0" } }] },
        { text: "Ran the tests; they pass." },
      );
      r = await h.conn.prompt({ sessionId: sid3, prompt: [{ type: "text", text: "Fix the add bug in calc.js" }] });
      assert.equal(r.stopReason, "end_turn");
      assert.ok(requests.some((q) => q.messages.some((m) => m.role === "user" && String(m.content).includes("[vdom guard: verify_claims]"))), "guard nudged");
      console.log("ok verify_claims guard");

      // Turn 2: the user is furious → keyword stage nudges serving AND diagnosis runs in the background.
      diagnosisScript.push(
        { calls: [{ name: "history", args: { turns: "1" } }] },
        {
          text:
            'Looking at e_x… {"what_happened": "claimed tests pass before running any", "blame": "harness", "category": "false_success_claim", "session_rule": {"rule": "Never state that tests pass unless a test command succeeded after your last edit in this turn; quote its output.", "guard": "verify_claims"}, "issue": {"title": "Agent can claim success before verifying", "root_cause": "no default prompt rule tying claims to observed checks", "proposed_fix": "add a working rule and keep verify_claims guard on", "files": ["src/acp/graph.ts"], "repro": "edit then claim success without running tests", "severity": "high"}, "evidence": ["e_1"]}',
        },
      );
      const updatesBefore = h.updates.length;
      script.push({ text: "You're right — I claimed the tests passed before running them. They pass now (ran after the edit)." });
      r = await h.conn.prompt({ sessionId: sid3, prompt: [{ type: "text", text: "wtf, you lied about the tests" }] });
      assert.equal(r.stopReason, "end_turn");
      const lastUserMsg = String(last().messages.filter((m) => m.role === "user").at(-1)!.content);
      assert.match(lastUserMsg, /\[vdom: this message reads as dissatisfaction/);
      let issueFile: string | undefined;
      for (let i = 0; i < 100 && !issueFile; i++) {
        await sleep(100);
        const dir = join(home, "issues");
        issueFile = existsSync(dir) ? (await import("node:fs")).readdirSync(dir).find((f) => f.endsWith(".json")) : undefined;
      }
      assert.ok(issueFile, "issue filed");
      const issue = JSON.parse(readFileSync(join(home, "issues", issueFile!), "utf8"));
      assert.equal(issue.blame, "harness");
      assert.equal(issue.signal.source, "keyword");
      assert.match(diagnosisRequests[0]!.messages[1]!.content as string, /wtf, you lied/);
      assert.ok(toolResults(diagnosisRequests[1]!).some((t) => /Fix the add bug/.test(t)), "diagnostician read the bad turn via history");
      assert.ok(!toolNames(diagnosisRequests[0]!).includes("edit") && !toolNames(diagnosisRequests[0]!).includes("bash"), "diagnostician is read-only");
      for (let i = 0; i < 50 && !texts(h.updates.slice(updatesBefore), "agent_message_chunk").includes("📌"); i++) await sleep(100);
      const notice = texts(h.updates.slice(updatesBefore), "agent_message_chunk");
      assert.match(notice, /📌 Correction for the rest of this session: Never state that tests pass/);
      assert.match(notice, /🐞 Filed harness issue I-/);
      console.log("ok keyword → background diagnosis → issue + in-session correction");

      // Turn 3: the lesson is live in the reconciled graph (system prompt), immediately.
      script.push({ text: "ok" });
      await h.conn.prompt({ sessionId: sid3, prompt: [{ type: "text", text: "now add a subtract function" }] });
      assert.match(systemOf(), /lesson \(lesson-1\)[\s\S]*Never state that tests pass/);
      console.log("ok lesson applied to the live graph");

      // Turn 4: polite but unhappy — no keyword; the interpreter catches it.
      sentimentScript.push({
        text: '{"unhappy": true, "frustration": 0.6, "target": "previous_turn", "category": "ignored_instruction", "reason": "user wanted subtract to be exported"}',
      });
      diagnosisScript.push({
        text: '{"what_happened": "did not add the function", "blame": "model", "category": "ignored_instruction", "session_rule": {"rule": "When asked to add a function, add and export it in the same turn, then show the diff.", "guard": "none"}, "issue": null, "evidence": []}',
      });
      script.push({ text: "Adding it now." });
      const before4 = sentimentRequests.length;
      await h.conn.prompt({ sessionId: sid3, prompt: [{ type: "text", text: "hmm, where is the subtract function though?" }] });
      for (let i = 0; i < 100 && !h.updates.slice(updatesBefore).some((u) => u.update.sessionUpdate === "agent_message_chunk" && u.update.content.type === "text" && u.update.content.text.includes("add and export it")); i++) await sleep(100);
      assert.ok(sentimentRequests.length > before4, "interpreter consulted");
      assert.match(sentimentRequests.at(-1)!.messages[1]!.content as string, /PREVIOUS TURN 3[\s\S]*now add a subtract function/);
      script.push({ text: "done" });
      await h.conn.prompt({ sessionId: sid3, prompt: [{ type: "text", text: "thanks" }] });
      assert.match(systemOf(), /lesson-2[\s\S]*add and export it/);
      console.log("ok interpreter → diagnosis → second lesson");

      // The canonical log records all of it, and the analyzer sees it.
      const { analyze: analyzeLog, readEvents: readLog } = await import("./history.js");
      const sessRoot = join(home, "sessions");
      const fs = await import("node:fs");
      const sdir = fs.readdirSync(sessRoot).map((d) => join(sessRoot, d, sid3)).find((p) => fs.existsSync(join(p, "events.jsonl")))!;
      const evs = readLog(sdir);
      const types = new Set(evs.map((e) => e.type));
      for (const t of ["session", "turn.start", "user.message", "llm.request", "assistant.message", "tool.call", "tool.result", "guard", "feedback", "issue", "lesson", "turn.end"]) {
        assert.ok(types.has(t as never), `log has ${t}`);
      }
      assert.ok(analyzeLog(evs).some((f) => f.kind === "user_frustration" && f.turn === 1));
      console.log("ok canonical event log");
    }

    // ---- persistence: a new process loads and replays the session
    h.child.kill();
    h = startAgent(env);
    await h.conn.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} });
    const listed = await h.conn.listSessions({ cwd: ws });
    assert.ok(listed.sessions.some((x) => x.sessionId === sid));
    const loaded = await h.conn.loadSession({ sessionId: sid, cwd: ws, mcpServers: [] });
    assert.equal(loaded.configOptions?.find((o) => o.id === "thought_level")?.currentValue, "high");
    assert.match(texts(h.updates, "agent_message_chunk"), /after compact|still healthy/);
    script.push({ text: "resumed" });
    r = await h.conn.prompt({ sessionId: sid, prompt: [{ type: "text", text: "continue" }] });
    assert.equal(r.stopReason, "end_turn");
    assert.ok(last().messages.some((m) => m.role === "user" && m.content === "after cancel"));
    console.log("ok session/load replay + continue");

    // ---- --force (T3 full-access) never asks
    h.child.kill();
    h = startAgent(env, ["--force"]);
    await h.conn.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} });
    const s2 = await h.conn.newSession({ cwd: ws, mcpServers: [] });
    await h.conn.setSessionMode({ sessionId: s2.sessionId, modeId: "ask" });
    script.push({ calls: [{ name: "bash", args: { command: "echo forced" } }] }, { text: "done" });
    await h.conn.prompt({ sessionId: s2.sessionId, prompt: [{ type: "text", text: "force" }] });
    assert.equal(h.permissions.length, 0);
    assert.match(toolResults().at(-1)!, /forced/);
    console.log("ok --force");
  } finally {
    h.child.kill();
    server.close();
  }
  console.log("acp selftest passed");
  process.exit(0);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
