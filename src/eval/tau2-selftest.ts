import {
  DEFAULT_OPENROUTER_MODEL,
  DeterministicProvider,
  resolveChatConfig,
  scriptedTau2MockTurn,
} from "../providers.js";
import { runTau2Turn } from "./tau2-turn.js";
import { observeTau2, actionFromCompletion, markRepeats } from "./tau2-obs.js";
import { tau2Graph } from "./tau2-graph.js";

let passed = 0;
let failed = 0;

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) {
    failed += 1;
    throw new Error(`FAIL: ${msg}`);
  }
  passed += 1;
}

function assertEq<T>(actual: T, expected: T, msg: string): void {
  if (actual !== expected) {
    failed += 1;
    throw new Error(`FAIL: ${msg}\n  expected: ${JSON.stringify(expected)}\n  actual:   ${JSON.stringify(actual)}`);
  }
  passed += 1;
}

async function testScriptedMock(): Promise<void> {
  const tools = [
    { name: "create_task", description: "Create a task" },
    { name: "update_task_status", description: "Update status" },
  ];
  const first = scriptedTau2MockTurn(
    [{ role: "user", content: "Create a new task called Important Meeting for user_1." }],
    tools,
  );
  assert(first?.toolCalls?.[0]?.name === "create_task", "first turn calls create_task");
  assertEq(first?.toolCalls?.[0]?.arguments.user_id, "user_1", "user_id user_1");
  assertEq(first?.toolCalls?.[0]?.arguments.title, "Important Meeting", "title Important Meeting");

  const confirm = scriptedTau2MockTurn(
    [
      { role: "user", content: "Create a task called Important Meeting" },
      {
        role: "assistant",
        content: "",
        tool_calls: first!.toolCalls,
      },
      {
        role: "tool",
        name: "create_task",
        tool_call_id: "call_create_task_1",
        content: '{"task_id":"task_2","title":"Important Meeting","status":"pending"}',
      },
    ],
    tools,
  );
  assert(confirm?.content.toLowerCase().includes("success"), "confirms after tool result");
  assert(!confirm?.toolCalls, "confirmation is text, not a tool call");
}

async function testTurnNoKey(): Promise<void> {
  const provider = new DeterministicProvider();
  const first = await runTau2Turn({
    policy: "Create tasks when asked.",
    tools: [{ name: "create_task", parameters: { type: "object" } }],
    messages: [
      { role: "assistant", content: "Hi! How can I help you today?" },
      { role: "user", content: "Please create a task called Important Meeting for user_1." },
    ],
    provider,
    technique: "one-shot",
  });
  assertEq(first.toolCalls?.[0]?.name, "create_task", "one-shot turn emits create_task");
  assert(first.traces.length >= 1, "traces recorded");
  assertEq(first.traces[0]!.role, "solve", "trace role is solve");

  const second = await runTau2Turn({
    policy: "Create tasks when asked.",
    tools: [{ name: "create_task" }],
    messages: [
      { role: "user", content: "Please create a task called Important Meeting for user_1." },
      { role: "assistant", content: "", tool_calls: first.toolCalls },
      {
        role: "tool",
        name: "create_task",
        tool_call_id: "call_create_task_1",
        content: '{"task_id":"task_2","title":"Important Meeting","status":"pending"}',
      },
    ],
    provider,
  });
  assert(second.content.toLowerCase().includes("success"), "second turn confirms");
  assert(!second.toolCalls, "no tool on confirm");
}

async function testSelfRefineStillScriptsMock(): Promise<void> {
  const provider = new DeterministicProvider();
  const turn = await runTau2Turn({
    policy: "policy",
    tools: [{ name: "create_task" }],
    messages: [{ role: "user", content: "Create a new task called Important Meeting for user_1." }],
    provider,
    technique: "self-refine",
  });
  assertEq(turn.toolCalls?.[0]?.name, "create_task", "self-refine still scripts mock tool");
  assert(turn.traces.some((t) => t.role === "critic"), "critic trace present");
  assert(turn.traces.some((t) => t.role === "refine"), "refine trace present");
}

async function testObs(): Promise<void> {
  const actions = markRepeats([
    { kind: "tool", text: "create_task", toolName: "create_task", toolArgs: { title: "A" }, ok: true },
    { kind: "tool", text: "create_task", toolName: "create_task", toolArgs: { title: "A" }, ok: true },
    { kind: "text", text: "ok" },
  ]);
  assertEq(actions[1]?.repeat, true, "repeat flagged");
  const obs = observeTau2({
    traces: [],
    actions,
    reward: 1,
    toolFailures: 0,
  });
  assertEq(obs.nSuccessProxy, 1, "p_hit proxy 1 on reward 1");
  assertEq(obs.repeatActions, 1, "one repeat");
  assert(obs.channels.includes("env"), "env channel when tools used");
  assertEq(actionFromCompletion({ content: "hi" }).kind, "text", "text completion");
}

async function testGraphAndConfig(): Promise<void> {
  assertEq(tau2Graph("one-shot").meta?.technique, "one-shot", "oneshot graph");
  assertEq(tau2Graph("self-refine").root.children?.[0]?.role, "critic", "self-refine critic");
  assert(tau2Graph("reflexion").root.children?.some((c) => c.role === "memory"), "reflexion memory");
  assertEq(DEFAULT_OPENROUTER_MODEL, "deepseek/deepseek-v4-flash-0731", "0731 not 0424");
  const cfg = resolveChatConfig();
  assertEq(cfg, null, "no key → no live chat config (DeterministicProvider)");
}

async function testWordReverseUntouched(): Promise<void> {
  const p = new DeterministicProvider();
  const out = await p.complete([
    { role: "system", content: "Role: solve" },
    { role: "user", content: 'Input: dom virtual' },
  ], { role: "solve" });
  assertEq(out, "lautriv mod", "complete() still naive-reverses without lesson");
}

async function main(): Promise<void> {
  const tests: Array<[string, () => Promise<void>]> = [
    ["scripted mock create_task", testScriptedMock],
    ["runTau2Turn no key", testTurnNoKey],
    ["self-refine mock scripts", testSelfRefineStillScriptsMock],
    ["obs / repeats / p_hit", testObs],
    ["graph + 0731 default + no-key config", testGraphAndConfig],
    ["DeterministicProvider word-reverse intact", testWordReverseUntouched],
  ];
  for (const [name, fn] of tests) {
    try {
      await fn();
      console.log(`ok  ${name}`);
    } catch (err) {
      console.error(`not ok  ${name}`);
      console.error(err instanceof Error ? err.message : err);
      process.exitCode = 1;
    }
  }
  console.log("");
  console.log(`${passed} passed, ${failed} failed`);
  if (failed > 0) process.exitCode = 1;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
