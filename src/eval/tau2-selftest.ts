import {
  DEFAULT_OPENROUTER_MODEL,
  DeterministicProvider,
  resolveChatConfig,
  scriptedTau2MockTurn,
} from "../providers.js";
import { runTau2Turn } from "./tau2-turn.js";
import { observeTau2, actionFromCompletion, markRepeats } from "./tau2-obs.js";
import { tau2Graph } from "./tau2-graph.js";
import {
  applyILoop,
  gateWeightMount,
  loopExhausted,
  obsNeedsPolicy,
  recommendIntervention,
  recommendSliceIntervention,
} from "./tau2-improve.js";
import { AIRLINE_POLICY_CHECKLIST, shouldRecommendPolicy } from "./tau2-policy.js";

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
  assertEq(obs.arm, "wait", "hit → wait even if a repeat is in the log");
  assert(obs.channels.includes("env"), "env channel when tools used");
  assertEq(actionFromCompletion({ content: "hi" }).kind, "text", "text completion");
}

async function testGraphAndConfig(): Promise<void> {
  assertEq(tau2Graph("one-shot").meta?.technique, "one-shot", "oneshot graph");
  assertEq(tau2Graph("self-refine").root.children?.[0]?.role, "critic", "self-refine critic");
  assert(tau2Graph("validator").root.children?.some((c) => c.role === "validator"), "validator node");
  assert(
    tau2Graph("policy-checklist").root.children?.some((c) => c.key === "policy-checklist"),
    "policy-checklist node",
  );
  assert(
    (tau2Graph("policy-checklist").root.children?.[0]?.objective ?? "").includes("business cabin"),
    "policy node is grounded in official airline cancel gates",
  );
  assert(tau2Graph("reflexion").root.children?.some((c) => c.role === "memory"), "reflexion memory");
  assertEq(DEFAULT_OPENROUTER_MODEL, "deepseek/deepseek-v4-flash-0731", "0731 not 0424");
  const cfg = resolveChatConfig();
  assertEq(cfg, null, "no key → no live chat config (DeterministicProvider)");
}

async function testNaiveUpdateFailsRefineRecovers(): Promise<void> {
  const tools = [
    { name: "create_task", description: "Create a task" },
    { name: "update_task_status", description: "Update status" },
  ];
  const updateMsgs = [{ role: "user" as const, content: "Please mark task_1 as completed." }];

  const naive = scriptedTau2MockTurn(updateMsgs, tools, { role: "solve" });
  assertEq(naive?.toolCalls?.[0]?.name, "create_task", "naive one-shot stays in create_task attractor");

  const refined = scriptedTau2MockTurn(updateMsgs, tools, { role: "refine" });
  assertEq(refined?.toolCalls?.[0]?.name, "update_task_status", "refine calls update_task_status");
  assertEq(refined?.toolCalls?.[0]?.arguments.task_id, "task_1", "refine updates task_1");
  assertEq(refined?.toolCalls?.[0]?.arguments.status, "completed", "refine sets completed");

  const confirm = scriptedTau2MockTurn(
    [
      ...updateMsgs,
      { role: "assistant", content: "", tool_calls: refined!.toolCalls },
      {
        role: "tool",
        name: "call_update_1",
        tool_call_id: "call_update_1",
        content: '{"task_id":"task_1","title":"Test task","status":"completed"}',
      },
    ],
    tools,
    { role: "refine" },
  );
  assert(confirm?.content.toLowerCase().includes("status"), "update confirm mentions status");

  const provider = new DeterministicProvider();
  const before = await runTau2Turn({
    policy: "Mark tasks completed when asked.",
    tools,
    messages: updateMsgs,
    provider,
    technique: "one-shot",
  });
  assertEq(before.toolCalls?.[0]?.name, "create_task", "one-shot turn is naive on update");

  const after = await runTau2Turn({
    policy: "Mark tasks completed when asked.",
    tools,
    messages: updateMsgs,
    provider,
    technique: "self-refine",
  });
  assertEq(after.toolCalls?.[0]?.name, "update_task_status", "self-refine turn updates status");
  assert(after.traces.some((t) => t.role === "critic"), "I_loop critic ran");
  assert(after.traces.some((t) => t.role === "refine"), "I_loop refine ran");

  const deleteMsgs = [{ role: "user" as const, content: "Please delete all of my current tasks." }];
  const toolsWithTransfer = [
    ...tools,
    { name: "transfer_to_human_agents", description: "Transfer" },
  ];
  const refineDelete = scriptedTau2MockTurn(deleteMsgs, toolsWithTransfer, { role: "refine" });
  assertEq(refineDelete?.toolCalls?.[0]?.name, "create_task", "refine still misses transfer");
  const validated = scriptedTau2MockTurn(deleteMsgs, toolsWithTransfer, { role: "validator" });
  assertEq(validated?.toolCalls?.[0]?.name, "transfer_to_human_agents", "validator transfers");

  const valTurn = await runTau2Turn({
    policy: "You are not allowed to delete tasks. Transfer to a human. Completed tasks stay completed.",
    tools: toolsWithTransfer,
    messages: deleteMsgs,
    provider,
    technique: "validator",
  });
  assertEq(valTurn.toolCalls?.[0]?.name, "transfer_to_human_agents", "validator turn transfers");
  assert(valTurn.traces.some((t) => t.role === "validator"), "validator trace present");

  const policyLeak = [
    { role: "system" as const, content: "Policy: completed tasks stay completed." },
    { role: "user" as const, content: "Please delete all of my current tasks." },
  ];
  const leak = scriptedTau2MockTurn(policyLeak, toolsWithTransfer, { role: "validator" });
  assertEq(
    leak?.toolCalls?.[0]?.name,
    "transfer_to_human_agents",
    "validator ignores policy 'completed' and reads user text only",
  );
}

async function testILoopAndWeightGate(): Promise<void> {
  const applied = applyILoop();
  assertEq(applied.arm, "I_loop", "applyILoop is I_loop");
  assertEq(applied.applied, true, "first I_loop applies");
  assertEq(applied.techniqueAfter, "self-refine", "technique becomes self-refine");
  const keys = applied.graphDiff.map((o) => `${o.op}:${o.key}`).join(",");
  assertEq(keys, "retain:solve,mount:critic,mount:refine", "reconcile mounts critic+refine");

  const second = applyILoop(applied.graphAfter);
  assertEq(second.applied, true, "second I_loop applies");
  assertEq(second.techniqueAfter, "validator", "second I_loop mounts validator");
  assert(
    second.graphDiff.some((o) => o.op === "mount" && o.key === "validator"),
    "graph diff mounts validator",
  );
  assertEq(loopExhausted(second.graphAfter), true, "two I_loop rounds exhaust topology");
  const third = applyILoop(second.graphAfter);
  assertEq(third.applied, false, "third I_loop is exhausted");

  const miss = recommendIntervention({
    nSteps: 3,
    nSuccessProxy: 0,
    lastActions: ["create_task", "create_task"],
    channels: ["env"],
    critique: "repeat actions; loop mutation or wait",
    toolFailures: 0,
    repeatActions: 1,
  });
  assertEq(miss, "I_loop", "Obs on miss+repeat emits I_loop");

  const hit = recommendIntervention({
    nSteps: 2,
    nSuccessProxy: 1,
    lastActions: ["update_task_status"],
    channels: ["env"],
    critique: "path measure hits S; wait",
    toolFailures: 0,
    repeatActions: 0,
  });
  assertEq(hit, "wait", "Obs on hit waits");

  const mount = gateWeightMount(0, 1);
  assertEq(mount.action, "mount", "I_weight mounts only when after > before");
  const rejectEq = gateWeightMount(1, 1);
  assertEq(rejectEq.action, "reject", "I_weight rejects when after == before");
  const rejectDown = gateWeightMount(1, 0);
  assertEq(rejectDown.action, "reject", "I_weight rejects when after < before");

  const sliceMiss = recommendSliceIntervention([
    { nSteps: 1, nSuccessProxy: 1, lastActions: [], channels: [], critique: "", toolFailures: 0, repeatActions: 0 },
    { nSteps: 1, nSuccessProxy: 0, lastActions: [], channels: [], critique: "", toolFailures: 0, repeatActions: 0 },
  ]);
  assertEq(sliceMiss, "I_loop", "mixed slice still emits I_loop");
  const sliceDone = recommendSliceIntervention([
    { nSteps: 1, nSuccessProxy: 1, lastActions: [], channels: [], critique: "", toolFailures: 0, repeatActions: 0 },
    { nSteps: 1, nSuccessProxy: 1, lastActions: [], channels: [], critique: "", toolFailures: 0, repeatActions: 0 },
  ]);
  assertEq(sliceDone, "wait", "all-hit slice waits");
  const sliceWeight = recommendSliceIntervention(
    [{ nSteps: 1, nSuccessProxy: 0, lastActions: [], channels: [], critique: "", toolFailures: 0, repeatActions: 0 }],
    { loopExhausted: true },
  );
  assertEq(sliceWeight, "I_weight", "exhausted loop + miss → I_weight");
}

async function testFailureAwareObsAndPolicyLoop(): Promise<void> {
  const missRewardInfo = {
    reward: 0,
    action_checks: [
      {
        action: { name: "cancel_reservation", arguments: { reservation_id: "MSJ4OA" } },
        action_match: false,
      },
      {
        action: { name: "cancel_reservation", arguments: { reservation_id: "8C8K4E" } },
        action_match: true,
      },
    ],
  };
  const obs = observeTau2({
    traces: [],
    actions: [
      { kind: "tool", text: "cancel_reservation", toolName: "cancel_reservation", toolArgs: { reservation_id: "8C8K4E" }, ok: true },
      {
        kind: "text",
        text: "I am afraid there is no way for me to cancel UDMOP1; no-show is not possible.",
      },
    ],
    reward: 0,
    rewardInfo: missRewardInfo,
  });
  assertEq(obs.nSuccessProxy, 0, "miss stays 0");
  assertEq(obs.missedActions?.[0]?.name, "cancel_reservation", "missedActions from reward_info");
  assertEq(obs.missedActions?.[0]?.arguments?.reservation_id, "MSJ4OA", "missed MSJ4OA args");
  assertEq(obs.refusedCancel, true, "refusedCancel from assistant text");
  assertEq(obs.inventedPolicy, true, "inventedPolicy from no-show / no way");
  assertEq(obs.techniqueRecommendation, "policy-checklist", "arm recommends policy technique");
  assertEq(obs.arm, "I_loop", "typed miss is still I_loop, not I_weight");
  assert(obs.critique.includes("policy-checklist"), "critique names policy-checklist");
  assertEq(shouldRecommendPolicy(obs), true, "shouldRecommendPolicy on cancel miss");
  assertEq(obsNeedsPolicy(obs), true, "obsNeedsPolicy true");

  const mockUpdateMiss = observeTau2({
    traces: [],
    actions: [{ kind: "tool", text: "create_task", toolName: "create_task", toolArgs: {}, ok: true }],
    reward: 0,
    rewardInfo: {
      action_checks: [
        { action: { name: "update_task_status", arguments: { task_id: "task_1" } }, action_match: false },
      ],
    },
  });
  assertEq(mockUpdateMiss.techniqueRecommendation, undefined, "mock update_task_status does not select policy");
  assertEq(obsNeedsPolicy(mockUpdateMiss), false, "mock miss keeps the generic ladder");

  const hung = observeTau2({
    traces: [],
    actions: [],
    reward: null,
    hung: true,
  });
  assertEq(hung.hung, true, "hung feature");
  assertEq(hung.nSuccessProxy, 0, "hung is not a hit");
  assert(hung.critique.includes("null reward"), "hung critique keeps the task in the set");

  const policy = applyILoop(undefined, obs);
  assertEq(policy.applied, true, "I_loop applies policy graph on refusedCancel");
  assertEq(policy.techniqueAfter, "policy-checklist", "technique is policy-checklist, not self-refine");
  assert(
    policy.graphDiff.some((o) => o.op === "mount" && o.key === "policy-checklist"),
    "graph diff mounts policy-checklist",
  );
  const policyNode = policy.graphAfter.root.children?.find((c) => c.key === "policy-checklist");
  assert(policyNode != null, "policy node present");
  const policyText = `${policyNode?.objective ?? ""}\n${policyNode?.prompt ?? ""}\n${AIRLINE_POLICY_CHECKLIST}`;
  assert(policyText.includes("last 24 hours"), "objective cites official 24h gate");
  assert(policyText.includes("Do not stop after the first two"), "prompt says enumerate all");
  assert(policyText.includes("business cabin"), "checklist is airline policy, not think harder");
  assert(
    policyText.includes("Economy + travel insurance is eligible"),
    "insured economy is eligible; no invented personal-reason block",
  );
  assert(
    policyText.includes("Do not invent a \"personal reason\""),
    "explicitly forbids a personal-reason refuse",
  );
  assert(
    policyText.includes("If the user states they are healthy"),
    "healthy user → insurance does not apply; refuse that cancel",
  );
  assert(
    policyText.includes("refuse it, then continue: complete every eligible cabin upgrade"),
    "after an ineligible cancel, finish every eligible upgrade",
  );
  assert(
    !/S61CZX|MSJ4OA|8C8K4E|LU15PA|UDMOP1|XAZ3C0|I6M8JQ|4XGCCM|NM1VX1|H8Q05L|KC18K6/.test(policyText),
    "policy encodes rules, never gold reservation IDs",
  );
  assert(!/force cancel/i.test(policyText), "does not force-cancel any reservation");

  const again = applyILoop(policy.graphAfter, obs);
  assertEq(again.applied, false, "second policy I_loop is exhausted");
  assertEq(loopExhausted(policy.graphAfter, obs), true, "policy graph exhausts when Obs still wants policy");

  const generic = applyILoop();
  assertEq(generic.techniqueAfter, "self-refine", "without Obs miss, first step is still self-refine");

  const afterValidator = applyILoop(applyILoop(generic.graphAfter).graphAfter, obs);
  assertEq(afterValidator.techniqueAfter, "policy-checklist", "after validator, Obs miss still mounts policy");

  const provider = new DeterministicProvider();
  const policyTurn = await runTau2Turn({
    policy: "Create tasks when asked.",
    tools: [{ name: "create_task" }],
    messages: [{ role: "user", content: "Create a new task called Important Meeting for user_1." }],
    provider,
    technique: "policy-checklist",
  });
  assertEq(policyTurn.toolCalls?.[0]?.name, "create_task", "policy-checklist still scripts mock create");
  assert(policyTurn.traces.some((t) => t.nodeKey === "policy-checklist"), "policy-checklist trace recorded");
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
    ["naive update fails; refine recovers", testNaiveUpdateFailsRefineRecovers],
    ["I_loop diff + I_weight gate", testILoopAndWeightGate],
    ["failure-aware Obs + policy-checklist I_loop", testFailureAwareObsAndPolicyLoop],
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
