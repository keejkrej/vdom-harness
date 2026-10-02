/** Pure unit tests: edit engine, truncation, resources, compaction cut points. */
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { applyEdits, coerceEdits, EditError, withFileLock } from "./edit.js";
import { findCut, isContextOverflow, serialize } from "./compaction.js";
import { loadContextFiles, loadSkills, loadTemplates, substituteArgs, splitArgs } from "./resources.js";
import { resolveToolAlias, TOOLS_BY_NAME, truncateHead, truncateTail, unsupportedArgs, type ToolContext } from "./tools.js";
import type { ChatMessage } from "./llm.js";
import { detectFeedback } from "./feedback.js";
import { parseInterpretation } from "./sentiment.js";
import { ModelRouter, parseLadder, bucketKey, classifyRequest, describeStats, isHarnessCheckout } from "./routing.js";
import { looksTruncated, parseArgs } from "./cli-args.js";
import { loadConfig } from "./config.js"; // regression I-20261002-2be2: list values as strings
import { parseJsonObject } from "./agent.js";
import { analyze, EventLog, readEvents, renderMarkdown } from "./history.js";

// ---- edit: exact multi-edit against the original
{
  const r = applyEdits("a\nb\nc\nd\n", [{ oldText: "b", newText: "B" }, { oldText: "d", newText: "D" }], "f");
  assert.equal(r.after, "a\nB\nc\nD\n");
  assert.equal(r.firstChangedLine, 2);
  assert.equal(r.replaced, 2);
  assert.equal(r.fuzzy, false);
}
// ---- edit: CRLF + BOM preserved, LF oldText matches
{
  const r = applyEdits("﻿one\r\ntwo\r\nthree\r\n", [{ oldText: "two\nthree", newText: "2\n3" }], "f");
  assert.equal(r.after, "﻿one\r\n2\r\n3\r\n");
}
// ---- edit: fuzzy (trailing spaces, smart quotes, nbsp) rewrites only touched lines
{
  const src = "keep   \nconst s = “hi”;  \nlast line\n";
  const r = applyEdits(src, [{ oldText: 'const s = "hi";', newText: 'const s = "bye";' }], "f");
  assert.equal(r.fuzzy, true);
  assert.equal(r.after, 'keep   \nconst s = "bye";\nlast line\n');
}
// ---- edit: trailing newline in oldText under fuzzy matching does not duplicate newlines
{
  const r = applyEdits("a  \nb\nc\n", [{ oldText: "a\nb\n", newText: "X\n" }], "f");
  assert.equal(r.after, "X\nc\n");
}
// ---- edit: errors
assert.throws(() => applyEdits("x x", [{ oldText: "x", newText: "y" }], "f"), /Found 2 occurrences/);
assert.throws(() => applyEdits("abc", [{ oldText: "zzz", newText: "y" }], "f"), /Could not find the exact text/);
assert.throws(() => applyEdits("abcdef", [{ oldText: "abcd", newText: "1" }, { oldText: "cdef", newText: "2" }], "f"), /overlap/);
assert.throws(() => applyEdits("abc", [{ oldText: "b", newText: "b" }], "f"), /No changes made/);
assert.throws(() => applyEdits("abc", [{ oldText: "", newText: "b" }], "f"), EditError);
// ---- coerceEdits accepts the shapes models send
assert.deepEqual(coerceEdits({ edits: '[{"oldText":"a","newText":"b"}]' }), [{ oldText: "a", newText: "b" }]);
assert.deepEqual(coerceEdits({ edits: { oldText: "a", newText: "b" } }), [{ oldText: "a", newText: "b" }]);
assert.deepEqual(coerceEdits({ old_string: "a", new_string: "b" }), [{ oldText: "a", newText: "b" }]);
console.log("ok edit engine");

// ---- file lock serializes per path
{
  const order: string[] = [];
  await Promise.all([
    withFileLock("/x", async () => {
      await new Promise((r) => setTimeout(r, 30));
      order.push("first");
    }),
    withFileLock("/x", async () => {
      order.push("second");
    }),
  ]);
  assert.deepEqual(order, ["first", "second"]);
  console.log("ok file lock");
}

// ---- truncation
{
  const lines = Array.from({ length: 3000 }, (_, i) => `line ${i + 1}`);
  assert.equal(truncateHead(lines).kept.length, 2000);
  const t = truncateTail(lines.join("\n"));
  assert.equal(t.firstLine, 1001);
  assert.ok(t.text.endsWith("line 3000"));
  const big = truncateTail("x".repeat(200_000));
  assert.ok(Buffer.byteLength(big.text) <= 50 * 1024);
  console.log("ok truncation");
}

// ---- templates
assert.deepEqual(splitArgs(`a "b c" 'd e' f`), ["a", "b c", "d e", "f"]);
assert.equal(substituteArgs("fix $1 in $2; all=$@; args=$ARGUMENTS", "bug src/x.ts"), "fix bug in src/x.ts; all=bug src/x.ts; args=bug src/x.ts");
assert.equal(substituteArgs("${1:-main} ${4:-none} ${@:2}", "a b c"), "a none b c");
assert.equal(substituteArgs("${@:2:1}|${ARGUMENTS:-x}", ""), "|x");
console.log("ok templates");

// ---- resources: context chain, skills, templates
{
  const home = mkdtempSync(join(tmpdir(), "vdom-res-home-"));
  const repo = mkdtempSync(join(tmpdir(), "vdom-res-repo-"));
  mkdirSync(join(repo, ".git"));
  const sub = join(repo, "pkg", "app");
  mkdirSync(sub, { recursive: true });
  writeFileSync(join(home, "AGENTS.md"), "global rule");
  writeFileSync(join(repo, "AGENTS.md"), "repo rule");
  writeFileSync(join(repo, "pkg", "CLAUDE.md"), "pkg rule");
  writeFileSync(join(sub, "AGENTS.override.md"), "app override");
  writeFileSync(join(sub, "AGENTS.md"), "ignored because override wins");
  const ctx = loadContextFiles(home, sub).map((f) => f.content);
  assert.deepEqual(ctx.slice(0, 1), ["global rule"]);
  assert.deepEqual(ctx.slice(-3), ["repo rule", "pkg rule", "app override"]);

  mkdirSync(join(repo, ".agents", "skills", "release"), { recursive: true });
  writeFileSync(join(repo, ".agents", "skills", "release", "SKILL.md"), "---\nname: release\ndescription: Cut a release\n---\nRun the release steps.");
  mkdirSync(join(home, "skills", "nodesc"), { recursive: true });
  writeFileSync(join(home, "skills", "nodesc", "SKILL.md"), "no frontmatter");
  const skills = loadSkills(home, sub);
  assert.ok(skills.some((s) => s.name === "release" && s.description === "Cut a release"));
  assert.ok(!skills.some((s) => s.name === "nodesc"));

  mkdirSync(join(sub, ".vdom", "prompts"), { recursive: true });
  writeFileSync(join(sub, ".vdom", "prompts", "review.md"), "---\ndescription: Review a file\nargument-hint: <file>\n---\nReview $1 carefully.");
  const tpls = loadTemplates(home, sub);
  const review = tpls.find((t) => t.name === "review");
  assert.equal(review?.description, "Review a file");
  assert.equal(review?.argumentHint, "<file>");
  console.log("ok resources");
}

// ---- compaction helpers
{
  const big = "x".repeat(40_000); // ~10k tokens each
  const msgs: ChatMessage[] = [
    { role: "user", content: big },
    { role: "assistant", content: null, tool_calls: [{ id: "1", type: "function", function: { name: "read", arguments: '{"path":"a.ts"}' } }] },
    { role: "tool", tool_call_id: "1", content: big },
    { role: "assistant", content: big },
    { role: "user", content: big },
    { role: "assistant", content: "done" },
  ];
  const cut = findCut(msgs, 20_000);
  assert.ok(cut > 0);
  assert.notEqual(msgs[cut]!.role, "tool");
  assert.match(serialize(msgs.slice(0, 3)), /\[Assistant tool calls\]: read\(\{"path":"a.ts"\}\)/);
  assert.ok(isContextOverflow("This model's maximum context length is 8192 tokens"));
  assert.ok(isContextOverflow("prompt is too long: 250000 tokens > 200000 maximum"));
  assert.ok(!isContextOverflow("rate limit exceeded"));
  console.log("ok compaction helpers");
}

// ---- bad-turn detection: keyword stage
{
  const hit = (t: string, prev?: string) => detectFeedback(t, prev)?.signal;
  assert.equal(hit("wtf is this"), "frustration");
  assert.equal(hit("fu"), "frustration");
  assert.equal(hit("WHY DID YOU DELETE THE FILE"), "frustration");
  assert.equal(hit("no, I said lines 120-200"), "correction");
  assert.equal(hit("you ignored my instruction"), "correction");
  assert.equal(hit("add a test for the parser", "add a test for the parser"), "repeat");
  assert.equal(hit("Add a farewell export to g.js"), undefined);
  assert.equal(hit("please refactor the store so that it does not write the whole file on every save; it should append"), undefined);
  console.log("ok feedback keywords");
}

// ---- interpreter + diagnostician output parsing
{
  const r = parseInterpretation('Sure.\n{"unhappy": true, "frustration": 0.8, "target": "previous_turn", "category": "false_success_claim", "reason": "claimed tests pass"}');
  assert.equal(r?.unhappy, true);
  assert.equal(r?.category, "false_success_claim");
  assert.equal(parseInterpretation("no json here"), undefined);
  const d = parseJsonObject('Analysis… {"a": {"b": 1}} more text\n```json\n{"what_happened": "x {y}", "blame": "harness", "session_rule": {"rule": "r", "guard": "none"}, "issue": null}\n```');
  assert.equal(d?.blame, "harness");
  assert.equal((d?.session_rule as { rule: string }).rule, "r");
  console.log("ok interpretation/diagnosis parsing");
}

// ---- event log + analyzer + renderer
{
  const dir = mkdtempSync(join(tmpdir(), "vdom-log-"));
  const log = new EventLog(dir);
  log.append("session", { sessionId: "s1", cwd: "/repo" });
  log.turn = 1;
  log.append("turn.start", { turn: 1 });
  log.append("user.message", { text: "show lines 120-200 of a.ts" });
  log.append("tool.call", { toolCallId: "c1", name: "read", arguments: { path: "a.ts", line_start: 120, line_end: 200 }, ignoredArgs: ["line_start", "line_end"] });
  log.append("tool.result", { toolCallId: "c1", name: "read", status: "ok", durationMs: 3, output: "…" });
  log.append("tool.call", { toolCallId: "c2", name: "edit", arguments: { path: "a.ts" } });
  log.append("tool.result", { toolCallId: "c2", name: "edit", status: "ok", durationMs: 2, output: "ok" });
  log.append("tool.call", { toolCallId: "c3", name: "search", arguments: {} });
  log.append("tool.result", { toolCallId: "c3", name: "search", status: "error", durationMs: 0, error: { type: "unknown_tool", message: "unknown tool search" } });
  log.append("tool.call", { toolCallId: "c4", name: "bash", arguments: { command: "sleep 9" } });
  log.append("compaction.end", { trigger: "threshold", ok: true, tokensBefore: 1000, tokensAfter: 900 });
  log.append("turn.end", { turn: 1, stopReason: "end_turn", finalText: "Done — all tests pass." });
  log.turn = 2;
  log.append("feedback", { targetTurn: 1, signal: "keyword", text: "wtf you lied" });
  // A reopened log continues the sequence.
  const again = new EventLog(dir);
  assert.equal(again.turn, 2);
  const evs = readEvents(dir);
  assert.equal(evs[1]!.parentId, evs[0]!.id);
  const kinds = analyze(evs).map((f) => f.kind).sort();
  assert.deepEqual(kinds, ["compaction_ineffective", "orphan_tool_call", "unknown_argument", "unknown_tool", "unverified_claim", "user_frustration"].sort());
  const md = renderMarkdown(evs);
  assert.match(md, /ignored args: line_start, line_end/);
  assert.match(md, /user feedback/);
  assert.match(md, /## Findings/);
  const big = log.blob("x".repeat(40_000));
  assert.ok(big.blobRef && big.preview.length < 10_000);
  console.log("ok event log / analyzer / renderer");
}

// ---- read maps line_start/line_end to offset/limit; unknown args are not silently ignored
{
  const readTool = TOOLS_BY_NAME.get("read")!;
  // line_start/line_end are documented aliases, so they are not "unsupported".
  assert.deepEqual(unsupportedArgs(readTool, { path: "x.txt", line_start: 2, line_end: 3 }), []);
  // genuinely unknown arguments are reported, not dropped.
  assert.deepEqual(unsupportedArgs(readTool, { path: "x.txt", bogus: 1 }), ["bogus"]);
  const dir = mkdtempSync(join(tmpdir(), "vdom-read-"));
  writeFileSync(join(dir, "x.txt"), "one\ntwo\nthree\n");
  const ctx = { cwd: dir } as ToolContext;
  const prepared = await readTool.prepare({ path: "x.txt", line_start: 2, line_end: 3 }, ctx);
  const out = await prepared.execute();
  assert.equal(out.output, "two\nthree");
  console.log("ok read line_start/line_end aliases");
}

// ---- habitual tool names map onto real tools (gpt-oss reaches for `search`)
{
  assert.deepEqual(resolveToolAlias("search", { path: "src", query: "ignoredArgs", max_results: 20 }), {
    name: "grep",
    args: { path: "src", pattern: "ignoredArgs", limit: 20 },
    aliasOf: "search",
  });
  assert.equal(resolveToolAlias("read_file", { file_path: "a.ts" })?.args.path, "a.ts");
  assert.equal(resolveToolAlias("functions.shell", { cmd: "ls" })?.name, "bash");
  assert.equal(resolveToolAlias("grep", { pattern: "x" }), undefined, "real tools are not aliased");
  assert.equal(resolveToolAlias("frobnicate", {}), undefined);
  console.log("ok tool aliases");
}

// ---- repeated unknown tools become one harness finding
{
  const dir = mkdtempSync(join(tmpdir(), "vdom-log-"));
  const log = new EventLog(dir);
  log.turn = 1;
  for (let i = 0; i < 4; i++) {
    log.append("tool.call", { toolCallId: `s${i}`, name: "search", arguments: {} });
    log.append("tool.result", { toolCallId: `s${i}`, name: "search", status: "error", error: { type: "unknown_tool", message: "unknown tool search" } });
  }
  const f = analyze(readEvents(dir)).filter((x) => x.kind === "unknown_tool");
  assert.equal(f.length, 1);
  assert.equal(f[0]!.blame, "harness");
  assert.equal(f[0]!.severity, "high");
  assert.match(f[0]!.message, /4 times/);
  console.log("ok repeated unknown tool grouped");
}

// ---- adaptive routing: cheap-first, at most one step up per turn, position persists
{
  const stats = mkdtempSync(join(tmpdir(), "vdom-routing-"));
  const r = new ModelRouter(["cheap", "mid", "big"], stats);
  const st = r.newState();
  assert.equal(st.active, true);
  assert.equal(r.current(st), "cheap", "sessions start on the cheapest rung");
  r.beginTurn(st);
  assert.equal(r.escalate(st, "llm_error")?.to, "mid");
  assert.equal(r.escalate(st, "empty_response"), undefined, "at most one step up per turn");
  assert.equal(r.current(st), "mid", "position persists within the session");
  r.beginTurn(st);
  assert.equal(r.current(st), "mid", "a new turn keeps the earned position");
  assert.equal(r.escalate(st, "tool_errors")?.to, "big");
  r.beginTurn(st);
  assert.equal(r.escalate(st, "llm_error"), undefined, "no step up from the top rung");
  // Tool failures accumulate within a turn, reset after a step, reset each turn.
  const st2 = r.newState();
  r.observeToolStatus(st2, "error");
  r.observeToolStatus(st2, "ok");
  r.observeToolStatus(st2, "denied");
  assert.equal(r.maybeEscalateFromTools(st2), undefined, "two failures are not enough");
  r.observeToolStatus(st2, "error");
  assert.equal(r.maybeEscalateFromTools(st2)?.reason, "tool_errors");
  assert.equal(st2.position, 1);
  r.observeToolStatus(st2, "error");
  r.observeToolStatus(st2, "error");
  r.observeToolStatus(st2, "error");
  assert.equal(r.maybeEscalateFromTools(st2), undefined, "no second step up in the same turn");
  r.beginTurn(st2);
  assert.equal(st2.toolErrors, 0, "tool failures reset each turn");
  // Stats persist across processes (the learning signal).
  r.recordCall("cheap");
  r.recordCall("mid");
  const r2 = new ModelRouter(["cheap", "mid", "big"], stats);
  assert.deepEqual(r2.snapshot().calls, { cheap: 1, mid: 1 });
  assert.deepEqual(r2.snapshot().escalations, { mid: 2, big: 1 });
  // A start model off the ladder stands routing down.
  const st3 = r.newState("gpt-9");
  assert.equal(st3.active, false);
  r.beginTurn(st3);
  r.observeToolStatus(st3, "error");
  assert.equal(r.maybeEscalateFromTools(st3), undefined, "inactive routing never steps up");
  assert.deepEqual(parseLadder(" a , b ,, a "), ["a", "b"]);
  console.log("ok adaptive routing");
}

// ---- routing gap 1: user signals step the ladder up (feedback detector, interpreter)
{
  const stats = mkdtempSync(join(tmpdir(), "vdom-routing-"));
  const r = new ModelRouter(["cheap", "mid", "big"], stats);
  const st = r.newState("cheap");
  r.beginTurn(st);
  r.notePending(st);
  r.notePending(st);
  assert.equal(st.pendingUp, "feedback", "one queued step, not more");
  r.beginTurn(st); // a pending step survives the turn boundary
  const up = r.applyPending(st);
  assert.equal(up?.from, "cheap");
  assert.equal(up?.to, "mid");
  assert.equal(up?.reason, "feedback");
  assert.equal(st.stepUpDone, false, "the mid-turn escalation budget is untouched");
  assert.equal(r.applyPending(st), undefined, "the queue is consumed once");
  assert.equal(r.escalate(st, "llm_error")?.to, "big", "mid-turn escalation still available");
  const top = r.newState("big");
  r.beginTurn(top);
  r.notePending(top);
  assert.equal(top.pendingUp, undefined, "no queued step at the top rung");
  console.log("ok routing: user-signal step-up");
}

// ---- routing gap 2: decay — escalated turns keep the rung for themselves and the next, then step down
{
  const stats = mkdtempSync(join(tmpdir(), "vdom-routing-"));
  const r = new ModelRouter(["cheap", "mid", "big"], stats);
  const st = r.newState();
  r.beginTurn(st);
  r.escalate(st, "llm_error");
  r.endTurn(st, false); // the escalated turn itself was bad
  assert.equal(r.beginTurn(st), undefined, "the next turn keeps the earned rung");
  assert.equal(r.current(st), "mid");
  r.endTurn(st, true); // clean
  assert.equal(r.current(st), "mid", "... and serves it");
  const down = r.beginTurn(st);
  assert.equal(down?.reason, "decay");
  assert.equal(down?.to, "cheap", "the turn after the protected one steps back to rung 0");
  assert.equal(r.current(st), "cheap");
  // A bad turn stops decay: no step down while turns keep going badly.
  r.endTurn(st, false);
  const stMid = r.newState("mid");
  const st2 = stMid;
  assert.equal(st2.position, 1);
  r.endTurn(st2, true);
  const down2 = r.beginTurn(st2);
  assert.equal(down2?.reason, "decay", "decay resumes after a clean turn");
  assert.equal(r.current(st2), "cheap");
  // Never below rung 0.
  r.endTurn(st2, true);
  assert.equal(r.beginTurn(st2), undefined, "rung 0 never decays");
  console.log("ok routing: decay");
}

// ---- routing gap 3: bucket learning — hot buckets start on rung 1
{
  const stats = mkdtempSync(join(tmpdir(), "vdom-routing-"));
  const r = new ModelRouter(["cheap", "mid"], stats);
  const cwd = process.cwd();
  const key = bucketKey("fix the failing test", cwd);
  assert.equal(classifyRequest("fix the failing test"), "fix");
  assert.equal(classifyRequest("what does this do?"), "question");
  assert.equal(classifyRequest("add a login page"), "feature");
  assert.equal(classifyRequest("refactor the store"), "refactor");
  assert.equal(classifyRequest("hi"), "other");
  assert.equal(key, `fix${isHarnessCheckout(cwd) ? "/harness" : ""}`, "kind plus harness flag");
  // Five turns on rung 0, three bad → bad-rate 0.6 > 0.4.
  const st = r.newState();
  for (const bad of [true, false, true, false, true]) {
    r.beginTurn(st);
    st.routedCall = true;
    r.recordTurn(st, "fix the failing test", cwd, bad);
  }
  const fresh = r.newState();
  r.beginTurn(fresh);
  const up = r.learnedStart(fresh, "please fix the bug", cwd);
  assert.equal(up?.reason, "learned");
  assert.equal(up?.detail, key);
  assert.equal(r.current(fresh), "mid", "hot buckets start one rung up");
  r.endTurn(fresh, true);
  assert.equal(r.beginTurn(fresh)?.to, "cheap", "a learned start decays at the same turn's end");
  // A cold bucket stays on rung 0.
  const cold = r.newState();
  r.beginTurn(cold);
  assert.equal(r.learnedStart(cold, "what is this file?", cwd), undefined, "cold buckets stay cheap");
  // Four turns are not enough to flag a bucket hot.
  const r3 = new ModelRouter(["cheap", "mid"], stats);
  const st2 = r3.newState();
  for (const bad of [true, true, true, true]) {
    r3.beginTurn(st2);
    st2.routedCall = true;
    r3.recordTurn(st2, "add a feature", cwd, bad);
  }
  const fresh2 = r3.newState();
  r3.beginTurn(fresh2);
  assert.equal(r3.learnedStart(fresh2, "add a feature", cwd), undefined, "under MIN_BUCKET_TURNS stays cheap");
  // Stats persist: a fresh router in the same home re-learns.
  const r4 = new ModelRouter(["cheap", "mid"], stats);
  const fresh3 = r4.newState();
  r4.beginTurn(fresh3);
  assert.equal(r4.learnedStart(fresh3, "fix the build", cwd)?.detail, key, "bucket stats survive restarts");
  const desc = describeStats(r4.snapshot());
  assert.match(desc, new RegExp(`${key.replace("/", "\\/")} @ rung 0: 5 turns, 3 bad`));
  console.log("ok routing: bucket learning");
}

// ---- routing: a saved position + hold survives session resume
{
  const stats = mkdtempSync(join(tmpdir(), "vdom-routing-"));
  const r = new ModelRouter(["cheap", "mid", "big"], stats);
  const saved = r.newState("mid", { position: 1, hold: 0, prevClean: true });
  assert.equal(saved.position, 1, "a resumed session keeps its rung");
  assert.equal(saved.hold, 0, "... and its decay counters");
  assert.equal(r.newState("mid", { position: 2, hold: 0, prevClean: true }).position, 1, "a stale saved position falls back to the model's rung");
  assert.equal(r.newState("gpt-9", { position: 1, hold: 0, prevClean: true }).active, false, "off-ladder model stands routing down even with saved state");
  console.log("ok routing: resume persistence");
}

// ---- cli-args: --prompt-file sources and truncation heuristic (vdom.cmd cuts multi-line prompts)
{
  const p1 = parseArgs(["client", "--cwd", "/x", "--prompt-file", "brief.txt"]);
  assert.deepEqual(p1.drive.promptSources, ["brief.txt"]);
  assert.deepEqual(p1.drive.prompts, []);
  const p2 = parseArgs(["-p", "first", "--prompt-file", "-"]);
  assert.deepEqual(p2.drive.promptSources, ["-"], "a second --prompt-file - is not deduped away");
  assert.deepEqual(p2.drive.prompts, ["first"]);
  // A prompt that arrives as its first line only (the cmd.exe truncation shape) is flagged.
  assert.equal(looksTruncated('Build the {"path": "a.json spec'), true, "odd quote count = truncated shape");
  assert.equal(looksTruncated('Line one\nLine two ends with a period.'), false);
  assert.equal(looksTruncated("all fine, no quotes"), false);
  assert.equal(parseArgs(["-p", "multi\nline"]).drive.prompts[0], "multi\nline", "parseArgs itself never cuts argv");
  console.log("ok cli-args: --prompt-file + looksTruncated");
}

// ---- config: model lists accept strings like the CLI flag does (I-20261002-2be2)
{
  const dir = mkdtempSync(join(tmpdir(), "vdom-cfg-"));
  // Same shape as --ladder / VDOM_ROUTE_LADDER, the natural way to write it.
  writeFileSync(join(dir, "config.json"), JSON.stringify({ routeLadder: "glm-5.3-flash,glm-5.3", models: "a,b" }));
  const cfg = loadConfig({ configPath: join(dir, "config.json") });
  assert.deepEqual(cfg.routeLadder, ["glm-5.3-flash", "glm-5.3"], "routeLadder string is parsed like the CLI flag");
  assert.deepEqual(cfg.models, ["a", "b"], "models string is parsed like VDOM_MODELS");
  // Arrays pass through, with blanks and duplicates dropped.
  writeFileSync(join(dir, "config.json"), JSON.stringify({ routeLadder: ["cheap", "cheap", "", "big"] }));
  const cfg2 = loadConfig({ configPath: join(dir, "config.json") });
  assert.deepEqual(cfg2.routeLadder, ["cheap", "big"]);
  // A non-list type is reported with the key name, not silently kept.
  writeFileSync(join(dir, "config.json"), JSON.stringify({ routeLadder: 42 }));
  const origWrite = process.stderr.write.bind(process.stderr);
  let warned = "";
  process.stderr.write = (chunk: unknown) => {
    warned += String(chunk);
    return true;
  };
  try {
    const cfg3 = loadConfig({ configPath: join(dir, "config.json") });
    assert.deepEqual(cfg3.routeLadder, [], "a wrongly-typed value falls back to routing off, not a crash");
  } finally {
    process.stderr.write = origWrite;
  }
  assert.match(warned, /routeLadder/, "the warning names the offending key");
  console.log("ok config: list values accept strings");
}

console.log("unit selftest passed");
