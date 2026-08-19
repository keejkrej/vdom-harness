import {
  DEFAULT_OPENROUTER_MODEL,
  DeterministicProvider,
  resolveChatConfig,
  scriptedTau2MockTurn,
  type Message,
  type Provider,
} from "../providers.js";
import { runTau2Turn } from "./tau2-turn.js";
import { observeTau2, actionFromCompletion, markRepeats } from "./tau2-obs.js";
import { tau2Graph } from "./tau2-graph.js";
import {
  applyILoop,
  gateWeightMount,
  graphHas,
  loopExhausted,
  obsNeedsPolicy,
  recommendIntervention,
  recommendSliceIntervention,
} from "./tau2-improve.js";
import { AIRLINE_POLICY_CHECKLIST, shouldRecommendPolicy } from "./tau2-policy.js";
import { GOLD_RESERVATION_IDS, hasGoldReservationId, serializeKernelC } from "./tau2-kernel.js";
import { runSelfObs, SELF_OBS_SYSTEM } from "./tau2-self-obs.js";

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

function scriptedProvider(body: string): Provider {
  return {
    name: "scripted-self-obs",
    async complete() {
      return body;
    },
    async completeTurn() {
      return { content: body };
    },
  };
}

async function testVisibleKernelC(): Promise<void> {
  const seen: Message[][] = [];
  const inner = new DeterministicProvider();
  const provider: Provider = {
    name: "capture",
    async complete(msgs, opts) {
      seen.push(msgs);
      return inner.complete(msgs, opts);
    },
    async completeTurn(msgs, opts) {
      seen.push(msgs);
      return inner.completeTurn(msgs, opts);
    },
  };
  const graph = tau2Graph("one-shot");
  const turn = await runTau2Turn({
    policy: "Create tasks when asked.",
    tools: [{ name: "create_task", parameters: { type: "object" } }],
    messages: [
      { role: "assistant", content: "Hi! How can I help you today?" },
      { role: "user", content: "Please create a task called Important Meeting for user_1." },
    ],
    provider,
    technique: "one-shot",
    graph,
  });
  const systems = seen.flatMap((msgs) => msgs.filter((m) => m.role === "system").map((m) => m.content));
  assert(systems.length > 0, "system message sent");
  const sys = turn.system;
  assert(sys.includes(graph.id), "runTau2Turn system contains graph id");
  assert(sys.includes("solve"), "runTau2Turn system contains oneshot key solve");
  assert(/kernel C/i.test(sys), "system names kernel C");
  assert(sys.includes("You are this AgentGraph"), "system says you are this graph");
  assert(sys.includes("get_agent_graph"), "system names get_agent_graph");
  assert(sys.includes("set_agent_graph"), "system names set_agent_graph");
  assert(sys.includes("You may read and rewrite your own graph"), "willingness: may rewrite C");
  assert(sys.includes("get_agent_graph before set_agent_graph"), "willingness: get before set");
  assert(systems.every((s) => s.includes("solve") && s.includes(graph.id)), "every system dump includes C");
  assertEq(turn.toolCalls?.[0]?.name, "create_task", "oneshot still scripts create_task");
}

async function testSelfObsPatchChangesGraph(): Promise<void> {
  const start = tau2Graph("one-shot");
  const result = await runSelfObs({
    graph: start,
    rewards: [0],
    terminations: ["user_stop"],
    missedToolNames: ["update_task_status"],
    provider: scriptedProvider(
      JSON.stringify({
        action: "I_loop",
        graphPatch: {
          technique: "self-refine",
          nodes: [
            {
              key: "new-critic",
              role: "critic",
              objective: "Look at this graph and these traces",
              prompt: "Write a short critique from this miss. Do not invent reservation IDs.",
            },
          ],
        },
        rationale: "path missed; rewrite C",
      }),
    ),
  });
  assertEq(result.path, "self", "scripted patch is the self path");
  assertEq(result.action, "I_loop", "action is I_loop");
  assertEq(result.applied, true, "patch applied");
  assertEq(result.servingPaused, false, "servingPaused stays false");
  assert(graphHas(result.graphAfter, "new-critic"), "graph gained new node key");
  assert(!graphHas(start, "new-critic"), "original graph unchanged");
}

async function testSelfObsWaitDoesNotChangeGraph(): Promise<void> {
  const start = tau2Graph("one-shot");
  const keysBefore = start.root.key;
  const result = await runSelfObs({
    graph: start,
    rewards: [1],
    terminations: ["user_stop"],
    provider: scriptedProvider(JSON.stringify({ action: "wait", rationale: "path measure hits S" })),
  });
  assertEq(result.path, "self", "wait is a valid self-Obs decision");
  assertEq(result.action, "wait", "action is wait");
  assertEq(result.applied, false, "wait does not apply");
  assertEq(result.graphAfter.root.key, keysBefore, "root key unchanged");
  assertEq(result.graphAfter.version, start.version, "version unchanged");
  assert(!graphHas(result.graphAfter, "critic"), "wait did not mount critic");
  assertEq(result.graphDiff.length, 0, "no graph diff on wait");
}

async function testNoGoldIdsInNewPrompts(): Promise<void> {
  const texts = [
    SELF_OBS_SYSTEM,
    serializeKernelC(tau2Graph("one-shot")),
    serializeKernelC(tau2Graph("self-refine")),
    serializeKernelC(tau2Graph("validator")),
    serializeKernelC(tau2Graph("policy-checklist")),
  ];
  const captured: string[] = [];
  const inner = new DeterministicProvider();
  const provider: Provider = {
    name: "capture-gold",
    async complete(msgs, opts) {
      captured.push(...msgs.filter((m) => m.role === "system").map((m) => m.content));
      return inner.complete(msgs, opts);
    },
    async completeTurn(msgs, opts) {
      captured.push(...msgs.filter((m) => m.role === "system").map((m) => m.content));
      return inner.completeTurn(msgs, opts);
    },
  };
  const turn = await runTau2Turn({
    policy: "policy",
    tools: [{ name: "create_task" }],
    messages: [{ role: "user", content: "Create a new task called Important Meeting for user_1." }],
    provider,
    technique: "one-shot",
  });
  texts.push(turn.system, ...captured, SELF_OBS_SYSTEM);
  for (const t of texts) {
    assert(!hasGoldReservationId(t), `no gold reservation IDs in prompt (${GOLD_RESERVATION_IDS[0]}…)`);
  }
  assert(SELF_OBS_SYSTEM.includes("You may change your own AgentGraph when the path measure misses"), "willingness: may change C on miss");
  assert(SELF_OBS_SYSTEM.includes("wait when it hits"), "willingness: wait on hit");
  assert(SELF_OBS_SYSTEM.includes("Do not invent reservation IDs"), "willingness: no invented IDs");
  assert(
    SELF_OBS_SYSTEM.includes("Do not transfer the rewrite to a hidden host script"),
    "willingness: do not hide the rewrite",
  );
}

async function testSelfObsFallbackInvalidJson(): Promise<void> {
  const miss = observeTau2({
    traces: [],
    actions: [
      {
        kind: "text",
        text: "I am afraid there is no way for me to cancel; no-show is not possible.",
      },
    ],
    reward: 0,
    rewardInfo: {
      action_checks: [
        { action: { name: "cancel_reservation", arguments: { reservation_id: "MSJ4OA" } }, action_match: false },
      ],
    },
  });
  const result = await runSelfObs({
    graph: tau2Graph("one-shot"),
    obs: miss,
    rewards: [0],
    missedToolNames: ["cancel_reservation"],
    provider: scriptedProvider("this is not json at all"),
  });
  assertEq(result.path, "fallback", "invalid JSON uses host fallback");
  assertEq(result.applied, true, "fallback still applies I_loop");
  assert(graphHas(result.graphAfter, "policy-checklist"), "fallback mounts canned policy-checklist");
  const node = result.graphAfter.root.children?.find((c) => c.key === "policy-checklist");
  assert((node?.prompt ?? "").includes("last 24 hours"), "fallback text is the canned checklist");
}

async function testMockSelfObsLadder(): Promise<void> {
  const provider = new DeterministicProvider();
  const r0 = await runSelfObs({
    graph: tau2Graph("one-shot"),
    rewards: [0, 0],
    terminations: ["user_stop", "transfer"],
    missedToolNames: ["update_task_status"],
    provider,
  });
  assertEq(r0.path, "self", "mock self-Obs is scripted, not fallback");
  assertEq(r0.applied, true, "round 0 miss applies");
  assert(graphHas(r0.graphAfter, "critic"), "0 → mount critic");
  assert(graphHas(r0.graphAfter, "refine"), "0 → mount refine");

  const r1 = await runSelfObs({
    graph: r0.graphAfter,
    rewards: [1, 0],
    terminations: ["user_stop", "transfer"],
    missedToolNames: ["transfer_to_human_agents"],
    provider,
  });
  assertEq(r1.path, "self", "round 1 still self");
  assert(graphHas(r1.graphAfter, "validator"), "0.5 → mount validator");

  const r2 = await runSelfObs({
    graph: r1.graphAfter,
    rewards: [1, 1],
    terminations: ["user_stop", "user_stop"],
    provider,
  });
  assertEq(r2.action, "wait", "1.0 → wait");
  assertEq(r2.applied, false, "hit does not change the graph");
  assertEq(r2.graphAfter.version, r1.graphAfter.version, "wait keeps version");
}

function fakeEnvExecutor(calls?: Array<{ name: string }>): string[] {
  const executed: string[] = [];
  for (const c of calls ?? []) {
    if (c.name === "get_agent_graph" || c.name === "set_agent_graph") {
      throw new Error(`leaked ${c.name} to env`);
    }
    executed.push(c.name);
  }
  return executed;
}

async function testGetThenSetChangesGraph(): Promise<void> {
  const listed: string[][] = [];
  const provider: Provider = {
    name: "get-then-set",
    async complete() {
      return "";
    },
    async completeTurn(msgs, opts) {
      listed.push((opts?.tools ?? []).map((t) => t.name));
      const lastTool = [...msgs].reverse().find((m) => m.role === "tool");
      if (!lastTool) {
        return { content: "", toolCalls: [{ id: "g1", name: "get_agent_graph", arguments: {} }] };
      }
      if (lastTool.name === "get_agent_graph") {
        assert(lastTool.content.includes("solve"), "get_agent_graph returns live key solve");
        assert(!hasGoldReservationId(lastTool.content), "get payload has no gold IDs");
        return {
          content: "",
          toolCalls: [
            {
              id: "s1",
              name: "set_agent_graph",
              arguments: {
                graphPatch: {
                  technique: "self-refine",
                  nodes: [
                    {
                      key: "live-critic",
                      role: "critic",
                      objective: "Critique the next tool from this miss",
                      prompt: "Name the correct write tool. Do not invent reservation IDs.",
                    },
                  ],
                },
              },
            },
          ],
        };
      }
      if (lastTool.name === "set_agent_graph") {
        const body = lastTool.content;
        assert(body.includes("applied") || body.includes("live-critic"), "set result mentions apply/key");
        return {
          content: "",
          toolCalls: [
            {
              id: "c1",
              name: "create_task",
              arguments: { user_id: "user_1", title: "Important Meeting" },
            },
          ],
        };
      }
      return { content: "ok" };
    },
  };

  const turn = await runTau2Turn({
    policy: "Create tasks when asked.",
    tools: [{ name: "create_task", parameters: { type: "object" } }],
    messages: [{ role: "user", content: "Please create a task called Important Meeting for user_1." }],
    provider,
    technique: "one-shot",
    graph: tau2Graph("one-shot"),
  });
  assert(listed.some((ts) => ts.includes("get_agent_graph") && ts.includes("set_agent_graph")), "kernel tools on the list");
  assert(graphHas(turn.graph, "live-critic"), "set_agent_graph mounted new node key");
  assertEq(turn.servingPaused, false, "servingPaused stays false after set");
  const setEdit = turn.graphEdits.find((e) => e.tool === "set_agent_graph");
  assert(setEdit?.applied === true, "set logged applied");
  assert(setEdit?.rejected === false, "set not rejected");
  const executed = fakeEnvExecutor(turn.toolCalls);
  assertEq(executed[0], "create_task", "gym tool after get/set; kernel names never reach env");
  assert(!(turn.toolCalls ?? []).some((t) => t.name === "get_agent_graph" || t.name === "set_agent_graph"), "kernel tools stripped from result");
}

async function testSetRejectsGoldIds(): Promise<void> {
  const provider: Provider = {
    name: "set-gold",
    async complete() {
      return "";
    },
    async completeTurn(msgs) {
      const lastTool = [...msgs].reverse().find((m) => m.role === "tool");
      if (!lastTool) {
        return {
          content: "",
          toolCalls: [
            {
              id: "s-gold",
              name: "set_agent_graph",
              arguments: {
                graphPatch: {
                  nodes: [
                    {
                      key: "policy-checklist",
                      role: "critic",
                      prompt: "Cancel reservation MSJ4OA then S61CZX.",
                    },
                  ],
                },
              },
            },
          ],
        };
      }
      return { content: "I will follow the policy." };
    },
  };
  const start = tau2Graph("one-shot");
  const turn = await runTau2Turn({
    policy: "policy",
    tools: [{ name: "create_task" }],
    messages: [{ role: "user", content: "hello" }],
    provider,
    technique: "one-shot",
    graph: start,
  });
  const setEdit = turn.graphEdits.find((e) => e.tool === "set_agent_graph");
  assert(setEdit?.rejected === true, "gold ID payload rejected");
  assert(setEdit?.applied === false, "gold ID set not applied");
  assert(!graphHas(turn.graph, "policy-checklist"), "rejected set does not mount a node");
  fakeEnvExecutor(turn.toolCalls);
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
    ["runTau2Turn system contains live graph C", testVisibleKernelC],
    ["self-Obs patch changes graph (new node key)", testSelfObsPatchChangesGraph],
    ["self-Obs wait does not change graph", testSelfObsWaitDoesNotChangeGraph],
    ["no gold IDs in new prompts", testNoGoldIdsInNewPrompts],
    ["invalid self-Obs JSON uses fallback checklist", testSelfObsFallbackInvalidJson],
    ["mock scripted self-Obs still 0 → 0.5 → 1.0", testMockSelfObsLadder],
    ["get then set changes graph; env never sees kernel tools", testGetThenSetChangesGraph],
    ["set_agent_graph rejects gold reservation IDs", testSetRejectsGoldIds],
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
