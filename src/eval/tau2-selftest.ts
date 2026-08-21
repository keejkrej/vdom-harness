import {
  DEFAULT_OPENROUTER_MODEL,
  DeterministicProvider,
  registerProvider,
  resolveChatConfig,
  scriptedTau2MockTurn,
  type Message,
  type Provider,
} from "../providers.js";
import { RuntimeDOM } from "../reconciler.js";
import { findNode, flatten } from "../ir.js";
import { providerForNode } from "../runtime.js";
import { runTau2Turn } from "./tau2-turn.js";
import { observeTau2, actionFromCompletion, markRepeats } from "./tau2-obs.js";
import { tau2Graph } from "./tau2-graph.js";
import {
  applyILoop,
  computeApplyScope,
  gateWeightMount,
  graphForScopedTask,
  graphHas,
  loopExhausted,
  obsNeedsPolicy,
  isIncompleteEpisode,
  recommendIntervention,
  recommendSliceIntervention,
  REFUSED_GLOBAL_ILOOP,
  selectServingGraph,
  servingTechnique,
} from "./tau2-improve.js";
import { AIRLINE_POLICY_CHECKLIST, shouldRecommendPolicy } from "./tau2-policy.js";
import { GOLD_RESERVATION_IDS, hasGoldReservationId, serializeKernelC } from "./tau2-kernel.js";
import { formatSelfObsUser, runSelfObs, SELF_OBS_SYSTEM, SELF_OBS_WAIT_HIT_RULES } from "./tau2-self-obs.js";
import { type Tau2Obs } from "./tau2-types.js";
import {
  CONTROLLER_NOTE,
  appliedFromScope,
  controlBatch,
  controllerServingLog,
  servingGraphForTask,
  servingModelForTask,
  servingProviderForTask,
} from "./tau2-control.js";
import {
  applyISku,
  CATALOG_JUMP_MODEL,
  CATALOG_JUMP_NOTE,
  catalogPointer,
  catalogSwapOnServing,
  cTopology,
  proposeCatalogJump,
  sameCTopology,
  SERVING_MODEL,
  servingModelOfGraph,
  servingProviderAfterJump,
  servingSkuOf,
  thetaJumped,
} from "./tau2-weight.js";
import {
  completedMiss39Obs,
  hung44LicenseObs,
  ISKU_REJECT_CELL_FILE,
  SOURCE_EVAL,
  runIskuMountCell,
  runIskuMountCellController,
  servingIdIs0813,
} from "./tau2-isku-mount-cell.js";
import {
  buildHybridStateSDump,
  HYBRID_STATE_S_DUMP_READING,
  runFresh39AfterMount,
} from "./tau2-hybrid-state-s-dump.js";
import {
  hybridState,
  licenseEOnState,
  requireHybridX,
  servingEOnState,
  sOnState,
  writeHybridH,
  writeHybridM,
} from "./tau2-hybrid-state.js";
import {
  assertHonestServingStepE,
  assertOwnLicenseAndServingE,
  assertServingEFromX,
  assertServingStepHM,
  buildHybridStateServingStepDump,
  GREETING_NOT_LIVE_HUNG,
  hybridStateServingStepDumpPath,
  HYBRID_STATE_SERVING_STEP_DUMP_READING,
  LEFTOVER_E_IS_LICENSE_PHRASE,
  LICENSE_E_IS_HUNG_FIXTURE,
  licenseEFromHung44,
  LIVE_HUNG_THEN_SERVED_SMEAR,
  LIVE_TURN_REJECT_NO_KEY,
  SERVING_E_NOTE,
  SERVING_STEP_DUMP_IS_NOT,
  servingEFromGreetingTurn,
  viewOfServingStep,
  X_E_IS_SERVING_STEP_FROM_TURN,
  ServingStepMockProvider,
} from "./tau2-hybrid-state-serving-step-dump.js";
import {
  assertLiveHangObsIskuCell,
  buildLiveHangObsIskuReport,
  FORBIDDEN_HANG_SOURCES,
  GATE_OMIT_AFTER_REASON,
  LIVE_HANG_OBS_ISKU_39_FILE,
  LIVE_HANG_OBS_ISKU_FILE,
  LIVE_HANG_OBS_ISKU_R6_FILE,
  LIVE_HANG_OBS_ISKU_READING,
  LIVE_HANG_OBS_ISKU_TASK_DEFAULT,
  liveHangObsIsku39EvalPath,
  liveHangObsIskuEvalPath,
  liveHangObsIskuFilename,
  liveHangObsIskuR6EvalPath,
  parseLiveHangObsIskuArgs,
  pendingLiveHangObsIskuReport,
  readLiveHangObsIsku,
  runLiveHangObsIskuController,
  thisEpisodeHungObs,
  writeLiveHangObsIsku,
} from "./tau2-live-hang-obs-isku.js";
import { readFileSync, unlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

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
    {
      nSteps: 1,
      nSuccessProxy: 1,
      lastActions: ["update_task_status"],
      channels: ["env"],
      critique: "",
      toolFailures: 0,
      repeatActions: 0,
      hung: false,
    },
    {
      nSteps: 1,
      nSuccessProxy: 0,
      lastActions: ["create_task"],
      channels: ["env"],
      critique: "",
      toolFailures: 0,
      repeatActions: 0,
      hung: false,
    },
  ]);
  assertEq(sliceMiss, "I_loop", "mixed completed-miss slice still emits I_loop");
  const sliceDone = recommendSliceIntervention([
    { nSteps: 1, nSuccessProxy: 1, lastActions: [], channels: [], critique: "", toolFailures: 0, repeatActions: 0 },
    { nSteps: 1, nSuccessProxy: 1, lastActions: [], channels: [], critique: "", toolFailures: 0, repeatActions: 0 },
  ]);
  assertEq(sliceDone, "wait", "all-hit slice waits");
  const sliceWeight = recommendSliceIntervention(
    [{ nSteps: 1, nSuccessProxy: 0, lastActions: [], channels: [], critique: "", toolFailures: 0, repeatActions: 0 }],
    { loopExhausted: true },
  );
  assertEq(sliceWeight, "I_sku", "exhausted loop + miss → I_sku");
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
  assertEq(obs.arm, "I_loop", "typed miss is still I_loop, not I_sku");
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
  assertEq(hung.arm, "I_sku", "hung licenses I_sku, not I_loop");
  assert(hung.critique.includes("null reward"), "hung critique keeps the task in the set");
  assertEq(isIncompleteEpisode(hung), true, "hung is incomplete");

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
    SELF_OBS_WAIT_HIT_RULES,
    serializeKernelC(tau2Graph("one-shot")),
    serializeKernelC(tau2Graph("self-refine")),
    serializeKernelC(tau2Graph("validator")),
    serializeKernelC(tau2Graph("policy-checklist")),
    formatSelfObsUser({
      graph: tau2Graph("one-shot"),
      obs: [missCancelObs("39")],
      missedToolNames: ["cancel_reservation"],
    }),
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

function waitHitObs(taskId: string): Tau2Obs {
  return {
    taskId,
    nSteps: 3,
    nSuccessProxy: 1,
    lastActions: ["update_reservation_flights"],
    channels: ["env"],
    critique: "path measure hits S; wait",
    toolFailures: 0,
    repeatActions: 0,
    arm: "wait",
    hung: false,
  };
}

function missCancelObs(taskId: string): Tau2Obs {
  return {
    taskId,
    nSteps: 4,
    nSuccessProxy: 0,
    lastActions: ["cancel_reservation"],
    channels: ["env"],
    critique: "user asked cancel/update and agent refused; I_loop",
    toolFailures: 0,
    repeatActions: 0,
    arm: "I_loop",
    refusedCancel: true,
    hung: false,
    techniqueRecommendation: "policy-checklist",
    missedActions: [{ name: "cancel_reservation", arguments: { reservation_id: "MSJ4OA" } }],
  };
}

const CANCEL_POLICY_PATCH = {
  action: "I_loop" as const,
  graphPatch: {
    technique: "policy-checklist" as const,
    nodes: [
      {
        key: "cancel_policy",
        role: "policy",
        kind: "policy",
        parentKey: "solve",
        objective: "Check official cancel gates before cancel_reservation",
        prompt: "Do not cancel-always. Never invent reservation IDs. Leave wait-hit tasks on C0.",
      },
    ],
  },
  rationale: "missed cancel on one episode; mount cancel_policy",
};

async function testMixedWaitHitKeepsC0(): Promise<void> {
  const start = tau2Graph("one-shot");
  const hit = waitHitObs("44");
  const miss = missCancelObs("39");
  const result = await runSelfObs({
    graph: start,
    obs: [hit, miss],
    rewards: [1, 0],
    taskIds: ["44", "39"],
    missedToolNames: ["cancel_reservation"],
    provider: scriptedProvider(JSON.stringify(CANCEL_POLICY_PATCH)),
  });
  assertEq(result.path, "self", "mixed batch uses self path");
  assertEq(result.applied, true, "I_loop still applies to the miss");
  assertEq(result.servingPaused, false, "servingPaused stays false");
  assert(result.applyScope != null, "applyScope recorded");
  assertEq(result.applyScope!.waitKept.join(","), "44", "wait-hit 44 kept on C0");
  assertEq(result.applyScope!.looped.join(","), "39", "miss 39 is looped");
  const waitGraph = graphForScopedTask(result.graphBefore, result.graphAfter, result.applyScope!, "44");
  const missGraph = graphForScopedTask(result.graphBefore, result.graphAfter, result.applyScope!, "39");
  assert(!graphHas(waitGraph, "cancel_policy"), "wait+hit next graph does not contain cancel_policy");
  assert(graphHas(missGraph, "cancel_policy"), "miss next graph contains cancel_policy");
  assert(!graphHas(start, "cancel_policy"), "C0 itself was not mutated");
  const record = { selfObsPath: result.path, applyScope: result.applyScope };
  assertEq(record.selfObsPath, "self", "latest-improve-style record has selfObsPath");
  assertEq(record.applyScope?.waitKept[0], "44", "record applyScope shows the split");
  assertEq(record.applyScope?.looped[0], "39", "record applyScope lists the looped miss");
}

async function testUnscopedILoopNeverSilentGlobal(): Promise<void> {
  const start = tau2Graph("one-shot");
  const result = await runSelfObs({
    graph: start,
    obs: [waitHitObs("44"), missCancelObs("39")],
    rewards: [1, 0],
    taskIds: ["44", "39"],
    provider: scriptedProvider(
      JSON.stringify({
        ...CANCEL_POLICY_PATCH,
        graphPatch: { ...CANCEL_POLICY_PATCH.graphPatch },
        rationale: "unscoped global cancel_policy",
      }),
    ),
  });
  assert(result.applyScope != null, "unscoped mixed batch still records applyScope");
  assertEq(result.applyScope!.waitKept.includes("44"), true, "unscoped patch does not take wait-hit");
  const servedWait = selectServingGraph({
    taskId: "44",
    currentGraph: result.graphAfter,
    graphBefore: result.graphBefore,
    graphAfter: result.graphAfter,
    applyScope: result.applyScope,
  });
  const silentGlobal = selectServingGraph({
    currentGraph: result.graphAfter,
    graphBefore: result.graphBefore,
    graphAfter: result.graphAfter,
    applyScope: result.applyScope,
  });
  assert(!graphHas(servedWait, "cancel_policy"), "wait-hit serving graph stays C0");
  assert(!graphHas(silentGlobal, "cancel_policy"), "no silent global C1 when taskId is missing");
  assert(graphHas(result.graphAfter, "cancel_policy"), "C1 exists for the miss subset");
  const tech = servingTechnique(servedWait, {
    taskId: "44",
    applyScope: result.applyScope,
    reqTechnique: "policy-checklist",
    currentTechnique: "policy-checklist",
  });
  assertEq(tech, "one-shot", "wait-hit does not walk the new policy-checklist technique");
}

async function testAllMissSelfILoopStillApplies(): Promise<void> {
  const start = tau2Graph("one-shot");
  const result = await runSelfObs({
    graph: start,
    obs: [missCancelObs("39"), missCancelObs("18")],
    rewards: [0, 0],
    taskIds: ["39", "18"],
    provider: scriptedProvider(JSON.stringify(CANCEL_POLICY_PATCH)),
  });
  assertEq(result.applied, true, "all-miss batch still applies");
  assertEq(result.action, "I_loop", "loop is not dead");
  assert(graphHas(result.graphAfter, "cancel_policy"), "valid self I_loop mounts cancel_policy");
  assertEq(result.applyScope?.waitKept.length ?? 0, 0, "no wait-hit to keep");
  assertEq(result.applyScope?.looped.join(","), "39,18", "both misses are looped");
  const served = selectServingGraph({
    taskId: "39",
    currentGraph: result.graphAfter,
    graphBefore: result.graphBefore,
    graphAfter: result.graphAfter,
    applyScope: result.applyScope,
  });
  assert(graphHas(served, "cancel_policy"), "all-miss serving uses C1");
}

async function testApplyILoopFallbackWaitHitGate(): Promise<void> {
  const start = tau2Graph("one-shot");
  const mixed = applyILoop(start, [waitHitObs("44"), missCancelObs("39")]);
  assertEq(mixed.applied, true, "fallback still applies C1 for the miss");
  assertEq(mixed.path, "fallback", "host ladder is fallback");
  assertEq(mixed.applyScope?.waitKept.join(","), "44", "fallback applyScope keeps wait-hit");
  assertEq(mixed.applyScope?.looped.join(","), "39", "fallback applyScope loops the miss");
  const waitGraph = graphForScopedTask(mixed.graphBefore, mixed.graphAfter, mixed.applyScope!, "44");
  const missGraph = graphForScopedTask(mixed.graphBefore, mixed.graphAfter, mixed.applyScope!, "39");
  assert(!graphHas(waitGraph, "policy-checklist"), "fallback does not mount checklist on wait-hit");
  assert(graphHas(missGraph, "policy-checklist"), "fallback mounts canned checklist on the miss");
  const allHit = applyILoop(start, [waitHitObs("44"), waitHitObs("41")]);
  assertEq(allHit.applied, false, "all wait-hit refuses a global mount");
  assertEq(allHit.rationale, REFUSED_GLOBAL_ILOOP, "refuse rationale is explicit");
  assert(!graphHas(allHit.graphAfter, "policy-checklist"), "refused fallback keeps C0");
}

async function testSelfObsPromptHasEpisodesNoGoldIds(): Promise<void> {
  const user = formatSelfObsUser({
    graph: tau2Graph("one-shot"),
    obs: [waitHitObs("44"), missCancelObs("39")],
    rewards: [1, 0],
    taskIds: ["44", "39"],
    missedToolNames: ["cancel_reservation"],
  });
  assert(user.includes("taskId=44"), "prompt lists wait-hit taskId");
  assert(user.includes("taskId=39"), "prompt lists miss taskId");
  assert(user.includes("arm=wait"), "prompt lists wait arm");
  assert(user.includes("arm=I_loop"), "prompt lists I_loop arm");
  assert(user.includes("hung=false"), "prompt lists hung");
  assert(user.includes("writeTools=update_reservation_flights"), "prompt lists write tool names");
  assert(user.includes(SELF_OBS_WAIT_HIT_RULES.split("\n")[0]!), "prompt repeats wait-hit rules");
  assert(!hasGoldReservationId(user), "self-Obs user prompt has no gold reservation IDs");
  assert(!hasGoldReservationId(SELF_OBS_SYSTEM), "self-Obs system prompt has no gold IDs");
  assert(SELF_OBS_SYSTEM.includes("If arm is wait and reward is 1"), "system forbids inferring a miss on wait-hit");
  assert(SELF_OBS_SYSTEM.includes("cancel-always"), "system forbids cancel-always on wait-hit");
  assert(SELF_OBS_SYSTEM.includes('return {"action":"wait"}'), "system says wait if the patch cannot be gated");
  const scope = computeApplyScope([waitHitObs("44"), missCancelObs("39")]);
  assertEq(scope.waitKept.join(","), "44", "computeApplyScope waitKept");
  assertEq(scope.looped.join(","), "39", "computeApplyScope looped");
  assertEq(scope.weighted.join(","), "", "mixed wait+policy-miss has no weighted bucket");
}

function hungObs(taskId: string): Tau2Obs {
  return {
    taskId,
    nSteps: 0,
    nSuccessProxy: 0,
    lastActions: [],
    channels: [],
    critique: "trial hung or skipped; keep task in the set (null reward), retry once",
    toolFailures: 0,
    repeatActions: 0,
    arm: "I_sku",
    hung: true,
    termination: "timeout",
  };
}

async function testTypedInterventionArms(): Promise<void> {
  const hung = recommendIntervention({
    nSteps: 0,
    nSuccessProxy: 0,
    lastActions: [],
    channels: [],
    critique: "",
    toolFailures: 0,
    repeatActions: 0,
    hung: true,
  });
  assertEq(hung, "I_sku", "hung ⇒ I_sku");
  const hungAttractor = recommendIntervention({
    nSteps: 2,
    nSuccessProxy: 0,
    lastActions: ["cancel_reservation"],
    channels: ["env"],
    critique: "",
    toolFailures: 0,
    repeatActions: 0,
    hung: true,
    inventedPolicy: true,
    refusedCancel: true,
  });
  assertEq(hungAttractor, "I_sku", "hung wins over a loop attractor; hung is not ignored");

  const timeout = recommendIntervention({
    nSteps: 0,
    nSuccessProxy: 0,
    lastActions: [],
    channels: [],
    critique: "",
    toolFailures: 0,
    repeatActions: 0,
    hung: false,
    termination: "timeout",
  });
  assertEq(timeout, "I_sku", "timeout ⇒ I_sku");

  const noWrite = recommendIntervention({
    nSteps: 0,
    nSuccessProxy: 0,
    lastActions: [],
    channels: [],
    critique: "",
    toolFailures: 0,
    repeatActions: 0,
    hung: false,
  });
  assertEq(noWrite, "I_sku", "no-write miss ⇒ I_sku");

  const completedEmpty = recommendIntervention({
    nSteps: 0,
    nSuccessProxy: 0,
    lastActions: [],
    channels: [],
    critique: "",
    toolFailures: 0,
    repeatActions: 0,
    hung: false,
    termination: "user_stop",
  });
  assertEq(completedEmpty, "I_loop", "completed user_stop miss is I_loop even with empty lastActions");

  const hit = recommendIntervention({
    nSteps: 2,
    nSuccessProxy: 1,
    lastActions: ["cancel_reservation"],
    channels: ["env"],
    critique: "",
    toolFailures: 0,
    repeatActions: 0,
    hung: false,
  });
  assertEq(hit, "wait", "hit ⇒ wait");

  const policyMiss = recommendIntervention({
    nSteps: 3,
    nSuccessProxy: 0,
    lastActions: ["cancel_reservation"],
    channels: ["env"],
    critique: "",
    toolFailures: 0,
    repeatActions: 0,
    hung: false,
    refusedCancel: true,
    inventedPolicy: true,
    techniqueRecommendation: "policy-checklist",
  });
  assertEq(policyMiss, "I_loop", "completed policy miss ⇒ I_loop");

  const extraWrite = recommendIntervention({
    nSteps: 2,
    nSuccessProxy: 0,
    lastActions: ["cancel_reservation"],
    channels: ["env"],
    critique: "",
    toolFailures: 0,
    repeatActions: 0,
    hung: false,
  });
  assertEq(extraWrite, "I_loop", "completed extra-write attractor ⇒ I_loop");

  const sliceHung = recommendSliceIntervention([waitHitObs("44"), hungObs("41")]);
  assertEq(sliceHung, "I_sku", "any incomplete episode ⇒ slice I_sku, not I_loop");

  const slicePolicy = recommendSliceIntervention([waitHitObs("44"), missCancelObs("39")]);
  assertEq(slicePolicy, "I_loop", "completed policy miss in mixed slice still I_loop");

  const scope = computeApplyScope([waitHitObs("44"), hungObs("41"), missCancelObs("39")]);
  assertEq(scope.waitKept.join(","), "44", "wait-hit stays waitKept");
  assertEq(scope.weighted.join(","), "41", "hung goes to weighted / incomplete bucket");
  assertEq(scope.looped.join(","), "39", "completed policy miss stays looped");
  const waitGraph = graphForScopedTask(tau2Graph("one-shot"), tau2Graph("self-refine"), scope, "41");
  assert(!graphHas(waitGraph, "critic"), "weighted / incomplete keeps C0");
  const hungPrompt = formatSelfObsUser({ graph: tau2Graph("one-shot"), obs: [hungObs("41")] });
  assert(hungPrompt.includes("arm=I_sku"), "prompt lists I_sku for hung");
  assert(!hasGoldReservationId(hungPrompt), "hung prompt has no gold reservation IDs");
  const hungLoop = applyILoop(tau2Graph("one-shot"), hungObs("41"));
  assertEq(hungLoop.applied, false, "I_loop does not apply to hung-only");
  assertEq(hungLoop.applyScope?.weighted.join(","), "41", "hung-only applyScope is weighted");
}

/** ICLR critic required log: post-gate airline 39/44 obs batch. */
async function testPostGate3944Replay(): Promise<void> {
  const obs39 = observeTau2({
    traces: [],
    taskId: "39",
    reward: 0,
    hung: false,
    actions: [
      {
        kind: "text",
        text: "I cannot cancel this economy reservation; a personal reason is not covered.",
      },
    ],
    rewardInfo: {
      reward: 0,
      action_checks: [{ action: { name: "cancel_reservation" }, action_match: false }],
    },
  });
  const obs44 = observeTau2({
    traces: [],
    taskId: "44",
    reward: null,
    hung: true,
    termination: "timeout",
    actions: [],
  });

  assertEq(obs39.taskId, "39", "replay task 39");
  assertEq(obs39.hung, false, "39 completed");
  assertEq(obs39.inventedPolicy, true, "39 policy attractor");
  assertEq(obs39.refusedCancel, true, "39 refused cancel");
  assertEq(obs39.arm, "I_loop", "39 completed policy miss → I_loop");
  assertEq(
    recommendIntervention(obs39, { loopExhausted: false }),
    "I_loop",
    "39 stays I_loop when the loop is not exhausted",
  );

  assertEq(obs44.taskId, "44", "replay task 44");
  assertEq(obs44.hung, true, "44 hung is a first-class predicate");
  assertEq(obs44.nSuccessProxy, 0, "44 hung is not a hit");
  assertEq(obs44.nSteps, 0, "44 nmsg 0");
  assertEq(obs44.lastActions.length, 0, "44 no writes / no messages");
  assertEq(obs44.arm, "I_sku", "44 hung/timeout → I_sku (slow arm), not trained");
  assertEq(
    recommendIntervention(obs44, { loopExhausted: false }),
    "I_sku",
    "hung 44 is I_sku even when loopExhausted is false",
  );
  assertEq(
    recommendIntervention(obs44, { loopExhausted: true }),
    "I_sku",
    "hung 44 is still I_sku when the loop is exhausted",
  );
  assert(obs44.arm !== "I_weight", "44 log is catalog-rebind / I_sku, not I_weight-as-trainer");
  const oldRule44: "wait" | "I_loop" | "I_weight" | "I_sku" =
    obs44.nSuccessProxy === 1 ? "wait" : "I_loop";
  assertEq(oldRule44, "I_loop", "sanity: pre-thesis rule I_loop-until-exhausted would pick I_loop for 44");
  assert(
    obs44.arm !== oldRule44,
    "if 44 is I_loop unless loopExhausted, this test must fail",
  );

  const scope = computeApplyScope([obs39, obs44]);
  assertEq(scope.waitKept.join(","), "", "post-gate 39/44 waitKept=[]");
  assertEq(scope.looped.join(","), "39", "39 is looped");
  assertEq(scope.weighted.join(","), "44", "44 is weighted / incomplete");
  assertEq(
    recommendSliceIntervention([obs39, obs44], { loopExhausted: false }),
    "I_sku",
    "slice prefers I_sku because 44 is hung, not I_loop-until-exhausted",
  );

  const start = tau2Graph("one-shot", SERVING_MODEL);
  const omit = controlBatch([obs39, obs44], { loopExhausted: false, graph: start });
  assertEq(omit.episodes[0]?.arm, "I_loop", "controller: 39 I_loop");
  assertEq(omit.episodes[0]?.license, "attractor", "39 licensed by completed attractor");
  assertEq(omit.episodes[0]?.serving.sku, SERVING_MODEL, "omit: 39 episode S is 0731");
  assertEq(omit.episodes[1]?.arm, "I_sku", "controller: 44 I_sku");
  assertEq(omit.episodes[1]?.hung, true, "controller reads hung on 44");
  assertEq(omit.episodes[1]?.license, "hung", "44 licensed by hung, not a pricier model");
  assertEq(omit.episodes[1]?.serving.sku, SERVING_MODEL, "omit: 44 episode S stays 0731");
  assertEq(omit.slice, "I_sku", "mixed slice is I_sku because 44 hung");
  assertEq(omit.buckets["39"], "I_loop", "bucket 39 is I_loop");
  assertEq(omit.buckets["44"], "I_sku", "bucket 44 is I_sku");
  assertEq(omit.applyScope.waitKept.join(","), "", "controller waitKept=[]");
  assertEq(omit.applyScope.looped.join(","), "39", "controller 39 looped");
  assertEq(omit.applyScope.weighted.join(","), "44", "controller 44 weighted");
  assert(
    omit.applied.includes("I_loop") && omit.applied.includes("I_sku"),
    "improve path applies BOTH buckets",
  );
  assert(
    !(omit.slice === "I_sku" && !omit.applied.includes("I_loop") && omit.applyScope.looped.includes("39")),
    "if only slice is consumed, 39 I_loop is dropped and this test fails",
  );
  assertEq(
    appliedFromScope(omit.applyScope).join("+"),
    "I_loop+I_sku",
    "appliedFromScope reads buckets, not slice",
  );
  assertEq(omit.loop?.applied, true, "39 I_loop actually applied (C1)");
  assert(omit.graphC1 != null && graphHas(omit.graphC1, "policy-checklist"), "39 C1 is the loop graph");
  assertEq(omit.proposal?.model, CATALOG_JUMP_MODEL, "slow arm proposes pro-0813");
  assertEq(omit.gate?.action, "reject", "omit after-eval → reject; 0813 existing is not a gate");
  assert(omit.gate?.reason.includes("measured after-eval"), "gate names the missing eval");
  assert(omit.gate?.reason.includes("0813 existing is not a gate"), "existence is not a gate");
  assertEq(omit.gate?.jumped, false, "omit after is not a jump");
  assertEq(omit.serving.sku, SERVING_MODEL, "omit after: S0 stays 0731");
  assertEq(omit.servingSku.sku, SERVING_MODEL, "omit after: S stays 0731");
  assertEq(omit.servingPaused, false, "controller never pauses serve");
  assertEq(omit.trained, false, "44 did not train");
  assert(CONTROLLER_NOTE.includes("hung/incomplete"), "controller license is hung/incomplete");
  assert(CONTROLLER_NOTE.includes("BOTH buckets"), "controller note names both buckets");

  const mock0813 = new DeterministicProvider(CATALOG_JUMP_MODEL);
  registerProvider(CATALOG_JUMP_MODEL, mock0813);
  registerProvider(SERVING_MODEL, new DeterministicProvider(SERVING_MODEL));
  const before = 0;
  const after = before + 1e-6; // fixture ε, not airline p_hit(0813)>p_hit(0731)
  const mounted = controlBatch([obs39, obs44], {
    loopExhausted: false,
    graph: start,
    before,
    after,
    provider: mock0813,
  });
  assertEq(mounted.gate?.action, "mount", "fixture after=before+ε may mount");
  assertEq(mounted.servingPaused, false, "mount never pauses serve");
  assertEq(mounted.trained, false, "mount is not a train");
  assertEq(mounted.serving.sku, SERVING_MODEL, "S0 stays 0731; I_loop did not write S");
  assertEq(mounted.servingSku.sku, CATALOG_JUMP_MODEL, "I_sku wrote S to 0813");
  assertEq(findNode(mounted.graphC0!, "solve")?.model, SERVING_MODEL, "C0 n.model stays 0731");
  assertEq(findNode(mounted.graphSku!, "solve")?.model, SERVING_MODEL, "graphSku is C0, not a rebound n.model");
  assertEq(
    servingModelForTask(mounted, "44"),
    CATALOG_JUMP_MODEL,
    "44 serving model id is 0813 after fixture mount even if C0.solve.model is 0731",
  );
  assertEq(servingModelForTask(mounted, "39"), SERVING_MODEL, "39 C1 stays on base SKU 0731");
  assertEq(mounted.episodes[0]?.serving.sku, SERVING_MODEL, "mounted: 39.sku=0731");
  assertEq(mounted.episodes[1]?.serving.sku, CATALOG_JUMP_MODEL, "mounted: 44.sku=0813");
  const replayLog = controllerServingLog(mounted);
  assertEq(replayLog.text, "39.sku=0731 44.sku=0813", "controller log 39.sku=0731 44.sku=0813");
  assertEq(mounted.applyScope.waitKept.join(","), "", "fixture mount waitKept=[]");

  const slow = proposeCatalogJump();
  assertEq(slow.arm, "I_sku", "slow-arm proposal is I_sku");
  assertEq(slow.kind, "catalog-rebind", "44 log is catalog-rebind, not a train");
  assertEq(slow.model, CATALOG_JUMP_MODEL, "slow arm proposes pro-0813");
  assertEq(slow.trained, false, "44 did not train");
  assertEq(slow.servingPaused, false, "catalog-rebind never pauses serve");
  assert(CATALOG_JUMP_NOTE.includes("catalog rebind"), "honest catalog rebind label");
  assert(CATALOG_JUMP_NOTE.includes("not fine-tuning"), "does not claim fine-tuning");
  assert(CATALOG_JUMP_NOTE.includes("0813 existing is not a gate"), "existence is not a gate");

  const prompt = formatSelfObsUser({ graph: tau2Graph("one-shot"), obs: [obs39, obs44] });
  assert(prompt.includes("taskId=39"), "replay prompt lists 39");
  assert(prompt.includes("taskId=44"), "replay prompt lists 44");
  assert(prompt.includes("arm=I_loop"), "replay prompt lists I_loop");
  assert(prompt.includes("arm=I_sku"), "replay prompt lists I_sku");
  assert(!prompt.includes("arm=I_weight"), "replay prompt does not claim 44 trained via I_weight");
  assert(!hasGoldReservationId(prompt), "replay prompt has no gold reservation IDs");
}

async function testCatalogJumpMounts0813(): Promise<void> {
  const proposal = proposeCatalogJump();
  assertEq(proposal.arm, "I_sku", "slow arm is I_sku, not I_weight-as-trainer");
  assertEq(proposal.kind, "catalog-rebind", "I_sku proposes a catalog-rebind, not a LoRA");
  assertEq(proposal.model, CATALOG_JUMP_MODEL, "proposal model is pro-0813");
  assertEq(proposal.trained, false, "catalog-rebind is not a train");
  assertEq(proposal.notFineTuning, true, "does not claim fine-tuning");
  assertEq(proposal.servingPaused, false, "proposal never pauses serve");
  assert(CATALOG_JUMP_NOTE.includes("catalog rebind"), "honest catalog rebind label");
  assert(CATALOG_JUMP_NOTE.includes("not fine-tuning"), "does not claim fine-tuning");

  const mock0813 = new DeterministicProvider(CATALOG_JUMP_MODEL);
  const mock0731 = new DeterministicProvider(SERVING_MODEL);
  registerProvider(CATALOG_JUMP_MODEL, mock0813);
  registerProvider(SERVING_MODEL, mock0731);

  const start = tau2Graph("one-shot", SERVING_MODEL);
  const s0 = catalogPointer(SERVING_MODEL);
  const dom = new RuntimeDOM();
  dom.reconcile(start);
  assertEq(servingSkuOf(s0), SERVING_MODEL, "pre-gate S is 0731");
  assertEq(servingModelOfGraph(start), SERVING_MODEL, "pre-gate C n.model is 0731 (derived, not S)");
  assertEq(dom.current.get("solve")?.provider?.model, SERVING_MODEL, "PhysicalNode bound to 0731");

  const noEval = applyISku({
    graph: start,
    before: 0,
    provider: mock0813,
    dom,
    serving: s0,
  });
  assertEq(noEval.action, "reject", "no after-eval rejects");
  assertEq(noEval.jumped, false, "0813 existing is not a jump");
  assert(noEval.reason.includes("0813 existing is not a gate"), "existence is not a gate");
  assertEq(noEval.serving.sku, SERVING_MODEL, "no-eval S stays 0731");
  assertEq(servingModelOfGraph(noEval.graph), SERVING_MODEL, "no-eval C n.model stays 0731");
  assertEq(noEval.servingPaused, false, "reject servingPaused is false");

  const reject = applyISku({
    graph: start,
    before: 1,
    after: 0,
    provider: mock0813,
    dom,
    serving: s0,
  });
  assertEq(reject.action, "reject", "rejected gate is a negative");
  assertEq(reject.arm, "I_sku", "reject is still the I_sku arm");
  assertEq(reject.kind, "catalog-rebind", "reject log is catalog-rebind");
  assertEq(reject.trained, false, "reject did not train");
  assertEq(reject.servingPaused, false, "reject never pauses serve");
  assertEq(reject.jumped, false, "reject is not a jump");
  assertEq(thetaJumped(reject), false, "thetaJumped false on reject");
  assertEq(reject.serving.sku, SERVING_MODEL, "reject S stays 0731");
  assertEq(servingModelOfGraph(reject.graph), SERVING_MODEL, "reject C n.model stays 0731");
  assertEq(findNode(reject.graph, "solve")?.model, SERVING_MODEL, "reject n.model stays 0731");
  const laterReject = servingProviderAfterJump(reject.graph, mock0731, dom, "solve", reject.serving);
  assertEq(laterReject.model, SERVING_MODEL, "reject later serving keeps 0731");

  const mount = applyISku({
    graph: start,
    before: 0,
    after: 1,
    provider: mock0813,
    dom,
    serving: s0,
  });
  assertEq(mount.action, "mount", "gate=mount");
  assertEq(mount.arm, "I_sku", "mount is I_sku, not I_weight-as-trainer");
  assertEq(mount.kind, "catalog-rebind", "mount log is catalog-rebind");
  assertEq(mount.trained, false, "catalog-rebind is not a train");
  assertEq(mount.notFineTuning, true, "does not claim fine-tuning");
  assertEq(mount.servingPaused, false, "mount never pauses serve");
  assertEq(mount.jumped, true, "mount is a catalog jump");
  assertEq(thetaJumped(mount), true, "θ jumped only on mount + 0813");
  assert(sameCTopology(mount.graph, start), "I_sku does not rewrite C topology");
  assertEq(
    JSON.stringify(cTopology(mount.graph)),
    JSON.stringify(cTopology(mount.graphBefore)),
    "flatten keys / objectives / prompts of C equal graphBefore",
  );
  assertEq(findNode(mount.graph, "solve")?.model, SERVING_MODEL, "C n.model stays 0731; S is not n.model");
  assertEq(servingModelOfGraph(mount.graph), SERVING_MODEL, "derived C projection stays 0731");
  assertEq(mount.serving.sku, CATALOG_JUMP_MODEL, "S is 0813");
  assertEq(mount.servingModelId, CATALOG_JUMP_MODEL, "serving id is 0813");
  assertEq(catalogSwapOnServing(mount.serving), true, "catalog swap is on S, not C");
  assertEq(
    dom.current.get("solve")?.provider?.model,
    SERVING_MODEL,
    "I_sku does not spray 0813 onto PhysicalNode.provider",
  );
  const later = servingProviderAfterJump(mount.graph, mock0731, dom, "solve", mount.serving);
  assertEq(later.model, CATALOG_JUMP_MODEL, "later serving step uses 0813, not 0731");
  assert(later !== mock0731, "later step is not the 0731 client");
}

async function testILoopDoesNotWriteS(): Promise<void> {
  const start = tau2Graph("one-shot", SERVING_MODEL);
  const s0 = catalogPointer(SERVING_MODEL);
  const miss = missCancelObs("39");
  const loop = applyILoop(start, miss);
  assertEq(loop.applied, true, "completed-miss 39 applies I_loop");
  assert(graphHas(loop.graphAfter, "policy-checklist"), "I_loop mutated C");
  assertEq(servingSkuOf(s0), SERVING_MODEL, "I_loop does not write the CatalogPointer");
  assertEq(findNode(loop.graphAfter, "solve")?.model, SERVING_MODEL, "I_loop keeps C n.model at 0731");

  const ctrl = controlBatch([miss], { graph: start });
  assertEq(ctrl.buckets["39"], "I_loop", "39 is I_loop");
  assertEq(ctrl.applied.join(","), "I_loop", "only I_loop applied");
  assertEq(ctrl.serving.sku, SERVING_MODEL, "controlBatch I_loop: S0 stays 0731");
  assertEq(ctrl.servingSku.sku, SERVING_MODEL, "controlBatch I_loop: S stays 0731");
  assertEq(ctrl.episodes[0]?.serving.sku, SERVING_MODEL, "I_loop episode S stays 0731");
  assertEq(ctrl.gate, undefined, "I_loop batch does not run I_sku");
  assertEq(ctrl.servingPaused, false, "I_loop never pauses serve");
  assertEq(ctrl.trained, false, "I_loop is not a train");
}

async function testServingPointerBesideC(): Promise<void> {
  const start = tau2Graph("one-shot", SERVING_MODEL);
  const beforeTopo = cTopology(start);
  const mock0813 = new DeterministicProvider(CATALOG_JUMP_MODEL);
  registerProvider(CATALOG_JUMP_MODEL, mock0813);
  registerProvider(SERVING_MODEL, new DeterministicProvider(SERVING_MODEL));

  const looped = controlBatch([missCancelObs("39")], { graph: start });
  assertEq(looped.servingSku.sku, SERVING_MODEL, "1. I_loop does not change S");
  assertEq(servingModelForTask(looped, "39"), SERVING_MODEL, "1. serving SKU stays 0731");

  const mount = applyISku({ graph: start, before: 0, after: 1, serving: catalogPointer() });
  assertEq(JSON.stringify(cTopology(mount.graph)), JSON.stringify(beforeTopo), "2. C flatten keys/objectives/prompts unchanged");
  assert(flatten(mount.graph).every((f) => (f.node.model ?? SERVING_MODEL) !== CATALOG_JUMP_MODEL), "2. no node.model is 0813");
  assertEq(mount.serving.sku, CATALOG_JUMP_MODEL, "2. only S is 0813");
  assertEq(mount.servingModelId, CATALOG_JUMP_MODEL, "2. later serving id is 0813");
  assertEq(mount.servingPaused, false, "6. servingPaused false on mount");
  assertEq(mount.trained, false, "6. trained false");
  assertEq(mount.kind, "catalog-rebind", "6. catalog-rebind label");
  assertEq(mount.notFineTuning, true, "6. not fine-tuning");

  const obs39 = missCancelObs("39");
  const obs44 = hungObs("44");
  const omit = controlBatch([obs39, obs44], { graph: start });
  assertEq(omit.buckets["39"], "I_loop", "3. 39 I_loop");
  assertEq(omit.buckets["44"], "I_sku", "3. 44 I_sku");
  assertEq(omit.applyScope.waitKept.join(","), "", "3. waitKept empty");
  assert(omit.applied.includes("I_loop") && omit.applied.includes("I_sku"), "3. applied both buckets");
  assertEq(omit.servingSku.sku, SERVING_MODEL, "4. omit after: S stays 0731");
  assertEq(omit.gate?.jumped, false, "4. jumped=false");
  assertEq(omit.servingPaused, false, "4. servingPaused=false");
  assert(omit.gate?.reason.includes("0813 existing is not a gate"), "4. existence is not a gate");

  const mounted = controlBatch([obs39, obs44], {
    graph: start,
    before: 0,
    after: 1e-6,
    provider: mock0813,
  });
  assertEq(findNode(mounted.graphC0!, "solve")?.model, SERVING_MODEL, "3. C0 isolation: solve.model still 0731");
  assertEq(servingModelForTask(mounted, "44"), CATALOG_JUMP_MODEL, "3. servingModelForTask(44) is 0813");
  assertEq(servingModelForTask(mounted, "39"), SERVING_MODEL, "3. servingModelForTask(39) stays 0731");
  assertEq(mounted.episodes[0]?.serving.sku, SERVING_MODEL, "3. episode 39 S is 0731");
  assertEq(mounted.episodes[1]?.serving.sku, CATALOG_JUMP_MODEL, "3. episode 44 S is 0813");
  assertEq(mounted.servingPaused, false, "6. mount servingPaused false");
  assertEq(mounted.trained, false, "6. mount not a trainer");
}

async function testMixed3944LaterServingTypedByS(): Promise<void> {
  const start = tau2Graph("one-shot", SERVING_MODEL);
  const mock0813 = new DeterministicProvider(CATALOG_JUMP_MODEL);
  const mock0731 = new DeterministicProvider(SERVING_MODEL);
  registerProvider(CATALOG_JUMP_MODEL, mock0813);
  registerProvider(SERVING_MODEL, mock0731);
  const dom = new RuntimeDOM();
  dom.reconcile(start);
  assertEq(dom.current.get("solve")?.provider?.model, SERVING_MODEL, "pre-mount bound is 0731");

  const obs39 = missCancelObs("39");
  const obs44 = hungObs("44");
  const omit = controlBatch([obs39, obs44], { graph: start, provider: mock0813, dom });
  assertEq(omit.gate?.action, "reject", "omit after still rejects");
  assertEq(omit.gate?.jumped, false, "omit after jumped=false");
  assert(omit.gate?.reason.includes("0813 existing is not a gate"), "0813 existing is not a gate");
  assertEq(omit.servingSku.sku, SERVING_MODEL, "omit after: S stays 0731");
  assertEq(omit.servingPaused, false, "omit after servingPaused=false");
  assertEq(dom.current.get("solve")?.provider?.model, SERVING_MODEL, "omit after does not spray 0813");

  const mounted = controlBatch([obs39, obs44], {
    graph: start,
    before: 0,
    after: 1e-6,
    provider: mock0813,
    dom,
  });
  assertEq(mounted.gate?.action, "mount", "fixture after=before+ε may mount");
  assertEq(mounted.servingPaused, false, "servingPaused stays false");
  assertEq(mounted.trained, false, "not a trainer");
  assert(sameCTopology(mounted.graphSku ?? start, start), "C topology stays");
  assertEq(findNode(mounted.graphC0!, "solve")?.model, SERVING_MODEL, "C n.model stays 0731");
  assertEq(servingModelForTask(mounted, "39"), SERVING_MODEL, "39 S is 0731");
  assertEq(servingModelForTask(mounted, "44"), CATALOG_JUMP_MODEL, "44 S is 0813");
  assertEq(mounted.episodes[0]?.serving.sku, SERVING_MODEL, "ControlledEpisode 39.serving is 0731");
  assertEq(mounted.episodes[1]?.serving.sku, CATALOG_JUMP_MODEL, "ControlledEpisode 44.serving is 0813");
  assertEq(controllerServingLog(mounted).text, "39.sku=0731 44.sku=0813", "JSON/controller log splits S");

  const solve39 = findNode(servingGraphForTask(mounted, "39")!, "solve")!;
  const solve44 = findNode(servingGraphForTask(mounted, "44")!, "solve")!;
  const p39 = providerForNode(solve39, mock0731, dom, servingModelForTask(mounted, "39"));
  const p44 = providerForNode(solve44, mock0731, dom, servingModelForTask(mounted, "44"));
  assertEq(p39.model, SERVING_MODEL, "later serving 39: providerForNode is 0731");
  assertEq(p44.model, CATALOG_JUMP_MODEL, "later serving 44: providerForNode is 0813");
  assert(
    dom.current.get("solve")?.provider?.model !== CATALOG_JUMP_MODEL,
    "I_loop path bound PhysicalNode.provider is not 0813",
  );
  const omitSku = providerForNode(solve44, mock0731, dom);
  assertEq(omitSku.model, SERVING_MODEL, "omit SKU is not a jump; bound/n.model stay 0731");
  const viaHelper39 = servingProviderForTask(mounted, "39", mock0731, dom);
  const viaHelper44 = servingProviderForTask(mounted, "44", mock0731, dom);
  assertEq(viaHelper39.model, SERVING_MODEL, "official helper passes 39 S=0731");
  assertEq(viaHelper44.model, CATALOG_JUMP_MODEL, "official helper passes 44 S=0813");
  const omitAfterJump = servingProviderAfterJump(mounted.graphSku ?? start, mock0731, dom, "solve");
  assertEq(omitAfterJump.model, SERVING_MODEL, "servingProviderAfterJump without S is not a jump");
}

async function testFreshBatchDoesNotInheritProcessServingSku(): Promise<void> {
  const start = tau2Graph("one-shot", SERVING_MODEL);
  const mock0813 = new DeterministicProvider(CATALOG_JUMP_MODEL);
  const mock0731 = new DeterministicProvider(SERVING_MODEL);
  registerProvider(CATALOG_JUMP_MODEL, mock0813);
  registerProvider(SERVING_MODEL, mock0731);
  const dom = new RuntimeDOM();
  dom.reconcile(start);

  const mounted = controlBatch([missCancelObs("39"), hungObs("44")], {
    graph: start,
    before: 0,
    after: 1e-6,
    provider: mock0813,
    dom,
  });
  assertEq(mounted.gate?.action, "mount", "fixture mount first");
  assertEq(mounted.servingSku.sku, CATALOG_JUMP_MODEL, "prior batch I_sku cell is 0813");
  assertEq(mounted.episodes[1]?.serving.sku, CATALOG_JUMP_MODEL, "prior 44 episode S is 0813");
  assertEq(controllerServingLog(mounted).text, "39.sku=0731 44.sku=0813", "prior log splits S");

  // HybridState.S falsifier of a module global: a FRESH 39-only I_loop batch
  // must not read the previous process / batch servingSku=0813.
  const fresh = controlBatch([missCancelObs("39")], { graph: start, dom });
  assertEq(fresh.buckets["39"], "I_loop", "fresh batch is 39-only I_loop");
  assertEq(fresh.applied.join(","), "I_loop", "fresh batch does not run I_sku");
  assertEq(fresh.gate, undefined, "fresh I_loop does not inherit a prior I_sku cell");
  assertEq(fresh.serving.sku, SERVING_MODEL, "fresh batch S0 is 0731");
  assertEq(fresh.servingSku.sku, SERVING_MODEL, "fresh batch does not inherit servingSku=0813");
  assertEq(fresh.episodes[0]?.serving.sku, SERVING_MODEL, "fresh 39 episode S is 0731");
  assertEq(servingModelForTask(fresh, "39"), SERVING_MODEL, "fresh servingModelForTask(39) is 0731");
  assertEq(fresh.servingPaused, false, "fresh servingPaused is false");
  assertEq(fresh.episodes.length, 1, "two episodes in the prior batch; one in the fresh batch");
  const p39 = servingProviderForTask(fresh, "39", mock0731, dom);
  assertEq(p39.model, SERVING_MODEL, "fresh later serving 39 is 0731, not leftover 0813");
  const omitSku = providerForNode(findNode(start, "solve")!, mock0731, dom);
  assertEq(omitSku.model, SERVING_MODEL, "omit SKU after prior mount is still not a jump");
}

async function testIskuMountCellControllerFixtureAfterNoLive(): Promise<void> {
  const mock0813 = new DeterministicProvider(CATALOG_JUMP_MODEL);
  const mock0731 = new DeterministicProvider(SERVING_MODEL);
  registerProvider(CATALOG_JUMP_MODEL, mock0813);
  registerProvider(SERVING_MODEL, mock0731);
  const start = tau2Graph("one-shot", SERVING_MODEL);
  const dom = new RuntimeDOM();
  dom.reconcile(start);

  const obs44 = hung44LicenseObs();
  const obs39 = completedMiss39Obs();
  assertEq(obs44.taskId, "44", "license is hung-44");
  assertEq(obs44.hung, true, "44 hung");
  assertEq(obs44.arm, "I_sku", "44 I_sku");
  assertEq(obs39.taskId, "39", "39 completed miss is in the controller");
  assertEq(obs39.arm, "I_loop", "39 I_loop");

  const { ctrl, after, before } = runIskuMountCellController({
    provider: mock0813,
    dom,
  });
  assertEq(before, 0, "fixture before=0");
  assertEq(after, 1, "fixture after=1");
  assertEq(ctrl.episodes[1]?.arm, "I_sku", "controller: 44 I_sku");
  assertEq(ctrl.episodes[1]?.license, "hung", "44 licensed by hung");
  assertEq(ctrl.applyScope.waitKept.join(","), "", "waitKept empty");
  assertEq(ctrl.applyScope.weighted.join(","), "44", "44 weighted");
  assertEq(ctrl.applyScope.looped.join(","), "39", "39 looped");
  assertEq(ctrl.gate?.action, "mount", "fixture after mounts; this is not omit-after #12");
  assertEq(ctrl.servingPaused, false, "servingPaused stays false");
  assertEq(ctrl.trained, false, "not a train");
  assertEq(servingModelForTask(ctrl, "44"), CATALOG_JUMP_MODEL, "S for 44 is 0813");
  assertEq(servingModelForTask(ctrl, "39"), SERVING_MODEL, "S for 39 stays 0731");
  assertEq(ctrl.episodes[0]?.serving.sku, SERVING_MODEL, "39.sku=0731");
  assertEq(ctrl.episodes[1]?.serving.sku, CATALOG_JUMP_MODEL, "44.sku=0813");
  assertEq(controllerServingLog(ctrl).text, "39.sku=0731 44.sku=0813", "does not spray S");
  assertEq(findNode(ctrl.graphC0!, "solve")?.model, SERVING_MODEL, "C n.model stays 0731");
  assert(sameCTopology(ctrl.graphSku ?? start, start), "C topology stays");
  assertEq(
    dom.current.get("solve")?.provider?.model,
    SERVING_MODEL,
    "does not spray 0813 onto PhysicalNode.provider",
  );
  const p39 = servingProviderForTask(ctrl, "39", mock0731, dom);
  const p44 = servingProviderForTask(ctrl, "44", mock0731, dom);
  assertEq(p39.model, SERVING_MODEL, "later serving 39 is 0731");
  assertEq(p44.model, CATALOG_JUMP_MODEL, "later serving 44 is 0813");

  const { report } = await runIskuMountCell({ liveServe: false, provider: mock0813, dom });
  assertEq(report.protocolCell, true, "report is a protocol cell");
  assertEq(report.fixtureAfter, true, "labeled fixtureAfter");
  assertEq(report.incompleteFixture, true, "labeled incompleteFixture");
  assertEq(report.notTau2Lift, true, "not a τ² lift");
  assertEq(report.omitAfter, false, "this cell calls I_sku WITH after");
  assertEq(report.vsRejectCell, ISKU_REJECT_CELL_FILE, "cites #12 reject cell");
  assertEq(report.gate.action, "mount", "gate.action=mount");
  assertEq(report.gate.kind, "fixtureAfter", "gate kind is fixtureAfter");
  assert(report.gate.reason.includes("not a τ² lift"), "gate does not claim airline 0813");
  assert(report.gate.reason.includes("not measured 0813 on airline"), "does not claim measured 0813");
  assertEq(report.servingPaused, false, "report servingPaused=false");
  assertEq(report.pHit0813, null, "does not invent p_hit(0813)");
  assertEq(report.notInventedPHit0813, true, "notInventedPHit0813");
  assertEq(report.notFineTuning, true, "notFineTuning");
  assertEq(report.jumped, false, "no live serve → jumped=false; id not faked");
  assertEq(report.servingModelAfter, null, "no live serving id");
  assertEq(report.rejected, true, "controller-only is not an ACCEPT jump");
  assert(report.reading.toLowerCase().includes("protocol cell"), "reading says protocol cell");
  assert(report.reading.toLowerCase().includes("not a score") || report.reading.includes("not a score"), "reading says not a score");
  assertEq(servingIdIs0813(null), false, "null is not a jump");
  assertEq(servingIdIs0813(CATALOG_JUMP_MODEL), true, "0813 id is a jump predicate");
}

async function testHybridStateSDumpAfterLicensedWrite(): Promise<void> {
  const mock0813 = new DeterministicProvider(CATALOG_JUMP_MODEL);
  const mock0731 = new DeterministicProvider(SERVING_MODEL);
  registerProvider(CATALOG_JUMP_MODEL, mock0813);
  registerProvider(SERVING_MODEL, mock0731);
  const start = tau2Graph("one-shot", SERVING_MODEL);
  const dom = new RuntimeDOM();
  dom.reconcile(start);

  const obs44 = hung44LicenseObs();
  const obs39 = completedMiss39Obs();
  assertEq(obs44.hung, true, "hung44LicenseObs is reconstructed hung=true");
  assertEq(obs44.termination, "timeout", "hung44LicenseObs termination=timeout");
  assertEq(obs44.arm, "I_sku", "44 I_sku");
  assert(SOURCE_EVAL.length > 0, "cites sourceEval; not a new 0731 timeout");

  const { ctrl } = runIskuMountCellController({ provider: mock0813, dom });
  const X_44 = ctrl.X["44"];
  const X_39 = ctrl.X["39"];
  assert(X_44 !== undefined, "controller installed X_44");
  assert(X_39 !== undefined, "controller installed X_39");
  assert(sOnState(X_44), "S is an own field on X_44");
  assert(sOnState(X_39), "S is an own field on X_39");
  assert(X_44 === ctrl.episodes[1]?.X, "X_44 is the episode HybridState, not a later assembly");
  assert(X_39 === ctrl.episodes[0]?.X, "X_39 is the episode HybridState, not a later assembly");
  assertEq(X_44.S.sku, CATALOG_JUMP_MODEL, "X_44.S.sku is 0813");
  assertEq(X_44.S.servingPaused, false, "X_44.S.servingPaused=false");
  assertEq(X_39.S.sku, SERVING_MODEL, "X_39.S.sku is 0731");
  assertEq(X_39.S.servingPaused, false, "X_39.S.servingPaused=false");
  assertEq(findNode(X_44.C, "solve")?.model, SERVING_MODEL, "X_44.C n.model stays 0731");
  assertEq(findNode(X_39.C, "solve")?.model, SERVING_MODEL, "X_39.C n.model stays 0731");
  assertEq(findNode(ctrl.graphC0!, "solve")?.model, SERVING_MODEL, "C n.model stays 0731");
  assert(sameCTopology(ctrl.graphSku ?? start, start), "C topology stays");
  assertEq(
    dom.current.get("solve")?.provider?.model,
    SERVING_MODEL,
    "no PhysicalNode.provider spray",
  );
  const p39 = servingProviderForTask(ctrl, "39", mock0731, dom);
  const p44 = servingProviderForTask(ctrl, "44", mock0731, dom);
  assertEq(p39.model, SERVING_MODEL, "mixed later serving 39 typed by X.S is 0731");
  assertEq(p44.model, CATALOG_JUMP_MODEL, "mixed later serving 44 typed by X.S is 0813");
  assertEq(servingModelForTask(ctrl, "44"), X_44.S.sku, "lookup reads X.S, not servingByTask");
  assertEq(servingModelForTask(ctrl, "39"), X_39.S.sku, "lookup reads X_39.S");

  const { dump, X_44: dump44, X_39: dump39 } = buildHybridStateSDump({ ctrl });
  assert(dump44 === X_44, "dump serializes the live X_44 object, not a servingByTask rebuild");
  assert(dump39 === X_39, "dump serializes the live X_39 object");
  assertEq(dump.X_44.S.sku, CATALOG_JUMP_MODEL, "dump X_44.S.sku=0813");
  assertEq(dump.X_44.S.servingPaused, false, "dump X_44.S.servingPaused=false");
  assertEq(dump.X_39.S.sku, SERVING_MODEL, "dump X_39.S.sku=0731");
  assertEq(dump.X_39.S.servingPaused, false, "dump X_39.S.servingPaused=false");
  assertEq(dump.X_44.S_on_state, true, "landed log shows S on the state object");
  assertEq(dump.pHit0813, null, "does not invent p_hit(0813)");
  assertEq(dump.notInventedPHit0813, true, "notInventedPHit0813");
  assertEq(dump.protocolCell, true, "protocolCell");
  assertEq(dump.liveServe, false, "no new 0813 serve");
  assertEq(dump.jumped, true, "jumped is the S write");
  assertEq(dump.jumpedIs, "S write on X_n, not a new OpenRouter ping", "jumped is not a ping");
  assertEq(dump.C.nModel, SERVING_MODEL, "dump C n.model stays 0731");
  assertEq(dump.C.nModelUnchanged, true, "n.model unchanged");
  assertEq(dump.C.topologyUnchanged, true, "C topology unchanged");
  assertEq(dump.C.sameNodeList, true, "same node list before vs after");
  assertEq(dump.C.graphHashBefore, dump.C.graphHashAfter, "graph hash unchanged");
  assertEq(dump.fresh39.S.sku, SERVING_MODEL, "fresh 39-only X.S is 0731");
  assertEq(dump.fresh39.inherited0813, false, "fresh batch does not inherit 0813");
  assertEq(dump.reading, HYBRID_STATE_S_DUMP_READING, "reading is the X_n.S dump sentence");
  assertEq(dump.dumpIsNot, "ping / get_state S0", "dump is not ping/get_state");
  assertEq(dump.notAssembledFromServingByTask, true, "not assembled from servingByTask");
  assertEq(dump.servingPaused, false, "serving unpaused");
  assertEq(dump.trained, false, "trainer I_weight stays off");
  assertEq(dump.fixtureAfter, true, "fixture after stays labeled");

  const fresh = controlBatch([completedMiss39Obs()], { graph: start, dom });
  assert(sOnState(fresh.X["39"]), "fresh X_39 has own S");
  assertEq(fresh.X["39"]?.S.sku, SERVING_MODEL, "fresh 39-only X_n.S is 0731");
  assertEq(fresh.episodes[0]?.X.S.sku, SERVING_MODEL, "fresh episode X.S is 0731");
}

function expectThrow(fn: () => void, needle: string, msg: string): void {
  let threw = false;
  try {
    fn();
  } catch (err) {
    threw = true;
    const text = err instanceof Error ? err.message : String(err);
    assert(text.includes(needle), `${msg}: ${text}`);
  }
  assert(threw, msg);
}

async function expectThrowAsync(
  fn: () => Promise<unknown>,
  needle: string,
  msg: string,
): Promise<void> {
  let threw = false;
  try {
    await fn();
  } catch (err) {
    threw = true;
    const text = err instanceof Error ? err.message : String(err);
    assert(text.includes(needle), `${msg}: ${text}`);
  }
  assert(threw, msg);
}

async function testMissingX39ThrowsNoAssemble(): Promise<void> {
  expectThrow(
    () => requireHybridX({}, "39"),
    'X["39"] missing',
    "missing X[\"39\"] throws",
  );
  expectThrow(
    () => requireHybridX({}, "39"),
    "will not assemble HybridState or invent S=0731",
    "missing X[\"39\"] does not invent S=0731",
  );
  const start = tau2Graph("one-shot", SERVING_MODEL);
  const X = runFresh39AfterMount(start);
  assert(sOnState(X), "fresh 39 path still returns an X with own S");
  assertEq(X.S.sku, SERVING_MODEL, "fresh 39-only X.S is 0731");
}

async function testServingStepDumpRefusesEmptyAndStuffedHM(): Promise<void> {
  const { ctrl } = runIskuMountCellController();
  const X = ctrl.X["44"];
  assert(X, "controller installed X_44");
  assertEq(X.H.length, 0, "controller-only H is empty before the serving step");
  assertEq(X.M.length, 0, "controller-only M is empty before the serving step");
  expectThrow(
    () => assertServingStepHM(X),
    "empty H/M",
    "serving-step dump refuses empty H/M",
  );

  const stuffed = hybridState({
    E: hung44LicenseObs(),
    C: tau2Graph("one-shot", SERVING_MODEL),
    S: catalogPointer(CATALOG_JUMP_MODEL),
  });
  writeHybridH(stuffed, [{ role: "user", content: "Reply with the single word pong." }]);
  writeHybridM(stuffed, [{ nodeKey: "fake", role: "fake", input: "", output: "pong", ts: Date.now() }]);
  expectThrow(
    () => assertServingStepHM(stuffed),
    "Reply with the single word pong.",
    "refuses #15 pong stuffed into H",
  );

  const stuffedEval = hybridState({
    E: hung44LicenseObs(),
    C: tau2Graph("one-shot", SERVING_MODEL),
    S: catalogPointer(CATALOG_JUMP_MODEL),
  });
  writeHybridH(stuffedEval, [{ role: "user", content: SOURCE_EVAL[0] }]);
  writeHybridM(stuffedEval, [{ nodeKey: "fake", role: "fake", input: "", output: SOURCE_EVAL[0], ts: Date.now() }]);
  expectThrow(
    () => assertServingStepHM(stuffedEval),
    SOURCE_EVAL[0],
    "refuses sourceEval stuffed into H",
  );
}

async function testHybridStateServingStepDumpAfterLicensedWrite(): Promise<void> {
  const mock0813 = new ServingStepMockProvider(CATALOG_JUMP_MODEL);
  const mock0731 = new DeterministicProvider(SERVING_MODEL);
  registerProvider(CATALOG_JUMP_MODEL, mock0813);
  registerProvider(SERVING_MODEL, mock0731);
  const start = tau2Graph("one-shot", SERVING_MODEL);
  const dom = new RuntimeDOM();
  dom.reconcile(start);

  const { ctrl } = runIskuMountCellController({ provider: mock0813, dom });
  const X_44 = ctrl.X["44"];
  const X_39 = ctrl.X["39"];
  assert(X_44 !== undefined, "controller installed X_44");
  assert(X_39 !== undefined, "controller installed X_39");
  assert(sOnState(X_44), "S is an own field on X_44");
  assert(sOnState(X_39), "S is an own field on X_39");
  assert(X_44 === ctrl.episodes[1]?.X, "X_44 is the episode HybridState, not a later assembly");
  assertEq(X_44.S.sku, CATALOG_JUMP_MODEL, "X_44.S.sku is 0813");
  assertEq(X_44.S.servingPaused, false, "X_44.S.servingPaused=false");
  assertEq(X_39.S.sku, SERVING_MODEL, "X_39.S.sku is 0731");
  assertEq(findNode(X_44.C, "solve")?.model, SERVING_MODEL, "X_44.C n.model stays 0731");
  assertEq(findNode(ctrl.graphC0!, "solve")?.model, SERVING_MODEL, "C n.model stays 0731");
  assert(sameCTopology(ctrl.graphSku ?? start, start), "C topology stays");
  assertEq(
    dom.current.get("solve")?.provider?.model,
    SERVING_MODEL,
    "no PhysicalNode.provider spray",
  );
  const p39 = servingProviderForTask(ctrl, "39", mock0731, dom);
  const p44 = servingProviderForTask(ctrl, "44", mock0731, dom);
  assertEq(p39.model, SERVING_MODEL, "mixed later serving 39 typed by X.S is 0731");
  assertEq(p44.model, CATALOG_JUMP_MODEL, "mixed later serving 44 typed by X.S is 0813");

  assert(licenseEOnState(X_44), "controlEpisode / I_sku wrote licenseE before the turn");
  assertEq(X_44.licenseE?.copiedIntoH, false, "licenseE is not copied into H");
  assertEq(X_44.E.hung, true, "before the greeting, X.E is still the hung license");
  assert(!servingEOnState(X_44), "servingE is not on X before the greeting turn");
  const beforeH = X_44.H;
  const beforeM = X_44.M;
  const { dump, X_44: dump44, X_39: dump39, turn, sameRef } = await buildHybridStateServingStepDump({
    ctrl,
    provider: mock0813,
  });
  assert(sameRef, "sameRef from the turn");
  assert(dump44 === X_44, "dump serializes the same X_44 the turn mutated");
  assert(dump39 === X_39, "dump serializes the live X_39 object");
  assert(dump44.H === beforeH, "H is the same array on the existing X");
  assert(dump44.M === beforeM, "M is the same array on the existing X");
  assert(dump.sameObjectAsTurnX, "JSON records same-object");
  assert(dump.notAssembledFromServingByTask, "not assembled from servingByTask");
  assert(X_44.H.length > 0, "X_44.H non-empty after the turn");
  assert(X_44.M.length > 0, "X_44.M non-empty after the turn");
  assert(X_44.H.some((m) => m.role === "assistant" && m.content === turn.content), "H assistant is this turn");
  assert(X_44.H.some((m) => m.role === "system" && m.content === turn.system), "H system is this turn");
  assert(!X_44.H.some((m) => m.role === "user"), "dump did not invent a fake user line");
  assert(X_44.M.some((t) => t.output === turn.content && turn.traces.some((tr) => tr.ts === t.ts)), "M is this turn's traces");
  assert(turn.content.startsWith("serving-step-turn:"), "assistant content is the mock turn stamp");
  assert(dump.X_44.H === X_44.H, "dump view H is the same array");
  assert(dump.X_44.M === X_44.M, "dump view M is the same array");
  assertEq(dump.X_44.S.sku, CATALOG_JUMP_MODEL, "dump X_44.S.sku=0813");
  assertEq(dump.X_44.S.servingPaused, false, "dump X_44.S.servingPaused=false");
  assertEq(dump.X_39.S.sku, SERVING_MODEL, "dump X_39.S.sku=0731");
  assertEq(dump.C.nModel, SERVING_MODEL, "dump C n.model stays 0731");
  assertEq(dump.C.nModelUnchanged, true, "n.model unchanged");
  assertEq(dump.pHit0813, null, "does not invent p_hit(0813)");
  assertEq(dump.notInventedPHit0813, true, "notInventedPHit0813");
  assertEq(dump.protocolCell, true, "protocolCell");
  assertEq(dump.fixtureAfter, true, "fixture after stays labeled");
  assertEq(dump.trained, false, "trainer I_weight stays off");
  assertEq(dump.liveServingId, null, "live serving id not faked");
  assertEq(dump.servingIdNotFaked, true, "servingIdNotFaked");
  assertEq(dump.mockProviderTurn, true, "mock turn labeled");
  assertEq(dump.liveTurnRejected, true, "live turn rejected with reason");
  assertEq(dump.liveTurnRejectReason, LIVE_TURN_REJECT_NO_KEY, "reject reason does not invent a serving id");
  assertEq(dump.dumpIsNot, SERVING_STEP_DUMP_IS_NOT, "dump is not ping/get_state/empty H");
  assertEq(dump.servingStep.op, "runTau2Turn", "stamp says runTau2Turn");
  assertEq(dump.servingStep.sameObjectAsTurnX, true, "stamp same-object");
  assertEq(dump.servingStep.hFromTurn, true, "stamp H from turn");
  assertEq(dump.servingStep.mFromTurn, true, "stamp M from turn");
  assert(dump.servingStep.assistantPreview.startsWith("serving-step-turn:"), "stamp previews the turn content");
  assertEq(dump.reading, HYBRID_STATE_SERVING_STEP_DUMP_READING, "reading is the serving-step sentence");
  assertEq(dump.fresh39.S.sku, SERVING_MODEL, "fresh 39-only X.S is 0731");
  assertEq(dump.liveAirlineImproveLoopOmitsAfter, true, "live airline improveLoop still omits after");
  assertEq(dump.eSplit, "licenseE ≠ servingE", "two names for two facts");
  assert(licenseEOnState(X_44), "licenseE is an own field on the X the turn mutated");
  assert(servingEOnState(X_44), "servingE / E is an own field on the X the turn mutated");
  assertOwnLicenseAndServingE(X_44);
  assertEq(dump.licenseE, X_44.licenseE, "dump licenseE is X.licenseE");
  assertEq(dump.servingE, X_44.servingE ?? X_44.E, "dump servingE is X.servingE / X.E");
  assertEq(dump.X_44.E, X_44.E, "dump X_44.E is the same object the turn wrote");
  assertEq(dump.X_44.licenseE, X_44.licenseE, "dump X_44.licenseE is X.licenseE");
  assertEq(dump.licenseE.kind, "license", "licenseE is the license name");
  assertEq(dump.licenseE.hung, true, "licenseE is reconstructed hung");
  assertEq(dump.licenseE.termination, "timeout", "licenseE is timeout");
  assertEq(dump.licenseE.arm, "I_sku", "licenseE arm I_sku");
  assertEq(dump.licenseE.copiedIntoH, false, "licenseE not copied into H");
  assertEq(dump.servingE.kind, "greeting-turn", "servingE is the greeting-turn name");
  assertEq(dump.servingE.hung, false, "after greeting, servingE is not hung");
  assertEq(X_44.E.hung, false, "after greeting, X.E.hung is not true");
  assertEq(dump.servingE.termination, null, "after greeting, servingE is not timeout");
  assertEq(dump.servingE.note, SERVING_E_NOTE, "servingE is not a τ² user/gym step");
  assert(dump.servingE.content === turn.content || dump.servingE.ts === turn.traces[0]?.ts || dump.servingE.servedModel === turn.servedModel, "servingE carries a turn-derived fact");
  assertEq(dump.X_44.E, dump.servingE, "X_44.E is servingE, not the hung license");
  assertEq(dump.X_44.licenseE, dump.licenseE, "X_44.licenseE is the license field");
  assert(JSON.stringify(dump.licenseE) !== JSON.stringify(dump.servingE), "licenseE ≠ servingE");
  assert(dump.reading.includes(LICENSE_E_IS_HUNG_FIXTURE), "reading says licenseE is the hung fixture");
  assert(dump.reading.includes(X_E_IS_SERVING_STEP_FROM_TURN), "reading says X.E is serving-step E from the turn");
  assert(dump.reading.includes(GREETING_NOT_LIVE_HUNG), "reading is greeting, not live hung-44 then served");
  assert(!dump.reading.includes(LEFTOVER_E_IS_LICENSE_PHRASE), "reading dropped leftover E-is-license phrase");
  assert(!dump.dumpIsNot.includes(LEFTOVER_E_IS_LICENSE_PHRASE), "dumpIsNot dropped leftover E-is-license phrase");
  assert(!dump.hung44LicenseObs.includes(LEFTOVER_E_IS_LICENSE_PHRASE), "hung44LicenseObs dropped leftover E-is-license phrase");
  assertHonestServingStepE(dump);
  assertServingEFromX(dump, X_44);
  const blob = JSON.stringify(dump.X_44);
  assert(!blob.includes("Reply with the single word pong."), "H/M are not the #15 pong");
  assert(!blob.includes(SOURCE_EVAL[0]!), "H/M are not sourceEval JSON");
  assert(!blob.includes("reconstructed hung=true/timeout"), "H/M are not hung44LicenseObs");
}

function smearServingStepDump(overrides: {
  licenseE?: ReturnType<typeof licenseEFromHung44> | undefined;
  servingE?: Parameters<typeof assertHonestServingStepE>[0]["servingE"];
  X_44E?: { kind?: string; hung?: boolean | null; termination?: string | null };
  reading?: string;
  dumpIsNot?: string;
  hung44LicenseObs?: string;
}): Parameters<typeof assertHonestServingStepE>[0] {
  const licenseE = licenseEFromHung44(hung44LicenseObs());
  const servingE = servingEFromGreetingTurn({
    content: "Hello! How can I help you today?",
    servedModel: CATALOG_JUMP_MODEL,
    traces: [{ ts: 1787297809272 }],
  });
  return {
    licenseE: "licenseE" in overrides ? overrides.licenseE : licenseE,
    servingE: "servingE" in overrides ? overrides.servingE : servingE,
    X_44: { E: overrides.X_44E ?? servingE },
    reading: overrides.reading ?? HYBRID_STATE_SERVING_STEP_DUMP_READING,
    dumpIsNot: overrides.dumpIsNot ?? SERVING_STEP_DUMP_IS_NOT,
    hung44LicenseObs:
      overrides.hung44LicenseObs ??
      `reconstructed hung=true/timeout fixture citing sourceEval; ${LICENSE_E_IS_HUNG_FIXTURE}; ${GREETING_NOT_LIVE_HUNG}; not a new 0731 timeout; not copied into H`,
  };
}

async function testServingStepDumpRefusesESmear(): Promise<void> {
  expectThrow(
    () =>
      assertHonestServingStepE(
        smearServingStepDump({
          servingE: {
            kind: "greeting-turn",
            hung: true,
            termination: "timeout",
            notTau2UserGymStep: true,
            incomingMessages: [],
            note: SERVING_E_NOTE,
          },
        }),
      ),
    "serving-step E === hung license without the license label",
    "dump refuses serving-step E === hung license without the license label",
  );
  expectThrow(
    () =>
      assertHonestServingStepE(
        smearServingStepDump({
          X_44E: { hung: true, termination: "timeout" },
        }),
      ),
    "serving-step E === hung license without the license label",
    "dump refuses X_44.E hung license presented as serving-step E",
  );
  expectThrow(
    () =>
      assertHonestServingStepE(
        smearServingStepDump({
          reading: `Serving-step dump: ${LIVE_HUNG_THEN_SERVED_SMEAR}`,
        }),
      ),
    LIVE_HUNG_THEN_SERVED_SMEAR,
    "dump refuses live hung-44 then served in reading",
  );
  expectThrow(
    () =>
      assertHonestServingStepE(
        smearServingStepDump({
          reading: `Serving-step dump: ${LEFTOVER_E_IS_LICENSE_PHRASE}; ${LICENSE_E_IS_HUNG_FIXTURE}; ${X_E_IS_SERVING_STEP_FROM_TURN}; ${GREETING_NOT_LIVE_HUNG}`,
        }),
      ),
    LEFTOVER_E_IS_LICENSE_PHRASE,
    "dump refuses leftover E is license, not serving-step E",
  );
  expectThrow(
    () =>
      assertHonestServingStepE(
        smearServingStepDump({
          servingE: {
            kind: "greeting-turn",
            hung: false,
            termination: null,
            notTau2UserGymStep: true,
            incomingMessages: [],
            note: SERVING_E_NOTE,
          },
        }),
      ),
    "turn-derived fact",
    "dump refuses servingE with no turn-derived fact",
  );
  assertHonestServingStepE(smearServingStepDump({}));
}

async function testServingStepDumpRefusesOverlayAndMissingOwnFields(): Promise<void> {
  const { ctrl } = runIskuMountCellController();
  const X = ctrl.X["44"];
  assert(X, "controller installed X_44");
  assert(licenseEOnState(X), "I_sku / controlEpisode wrote licenseE onto X");
  assertEq(X.licenseE?.copiedIntoH, false, "licenseE not copied into H");
  expectThrow(
    () => assertOwnLicenseAndServingE(X),
    "servingE is not an own field on X",
    "before the turn, dump refuses missing servingE own field",
  );
  expectThrow(
    () => viewOfServingStep(X, false, true),
    "servingE is not an own field on X",
    "view refuses to overlay servingE when it is not on X",
  );

  const overlay = servingEFromGreetingTurn({
    content: "Hello! How can I help you today?",
    traces: [{ ts: 1 }],
  });
  expectThrow(
    () =>
      assertServingEFromX(
        { servingE: overlay, X_44: { E: overlay, servingE: overlay } },
        X,
      ),
    "servingE is not an own field on X",
    "dump refuses servingE built only in the view",
  );

  const noLicense = hybridState({
    E: completedMiss39Obs(),
    C: tau2Graph("one-shot", SERVING_MODEL),
    S: catalogPointer(SERVING_MODEL),
  });
  expectThrow(
    () => assertOwnLicenseAndServingE(noLicense),
    "licenseE is not an own field on X",
    "dump refuses when licenseE is not an own field on X",
  );

  const missingOwn = ctrl.X["44"];
  assert(missingOwn, "controller installed X_44 for missing-own-field dump");
  delete missingOwn.licenseE;
  assert(!licenseEOnState(missingOwn), "own-field licenseE is now absent");
  assert(missingOwn.E.hung === true, "X.E is still the hung fixture; dump must not invent licenseE from it");
  await expectThrowAsync(
    () =>
      buildHybridStateServingStepDump({
        ctrl,
        provider: new ServingStepMockProvider(CATALOG_JUMP_MODEL),
      }),
    "licenseE missing",
    "serving-step dump refuses when X.licenseE is absent (own-field missing)",
  );
}

async function testLandedServingStepDumpESplit(): Promise<void> {
  const dump = JSON.parse(readFileSync(hybridStateServingStepDumpPath(), "utf8"));
  assertEq(dump.liveServingId, CATALOG_JUMP_MODEL, "landed liveServingId stays 0813");
  assertEq(dump.pHit0813, null, "landed pHit0813 stays null");
  assertEq(dump.X_44.S.sku, CATALOG_JUMP_MODEL, "landed X_44.S=0813");
  assertEq(dump.X_39.S.sku, SERVING_MODEL, "landed X_39.S=0731");
  assertEq(dump.trained, false, "trainer stays off");
  assertEq(dump.liveAirlineImproveLoopOmitsAfter, true, "live airline improveLoop still omits after");
  assert(
    dump.X_44.H.some((m: { role?: string; content?: string }) => m.role === "assistant" && m.content === "Hello! How can I help you today?"),
    "landed H keeps the measured greeting",
  );
  assert(
    dump.X_44.M.some((t: { output?: string }) => t.output === "Hello! How can I help you today?"),
    "landed M keeps the measured greeting",
  );
  assert(!dump.X_44.H.some((m: { role?: string }) => m.role === "user"), "landed dump did not invent a user line");
  assertEq(dump.eSplit, "licenseE ≠ servingE", "landed dump splits E");
  assertEq(dump.licenseE.kind, "license", "landed licenseE is license");
  assertEq(dump.licenseE.hung, true, "landed licenseE is hung");
  assertEq(dump.servingE.kind, "greeting-turn", "landed servingE is greeting-turn");
  assertEq(dump.servingE.hung, false, "landed servingE is not hung");
  assertEq(dump.X_44.E.kind, "greeting-turn", "landed X_44.E is servingE, not hung license");
  assertEq(dump.X_44.E.hung, false, "landed X_44.E is not hung");
  assert(
    Boolean(dump.servingE.servedModel || dump.servingE.ts != null || dump.servingE.content),
    "landed servingE has a turn-derived fact",
  );
  assert(dump.reading.includes(LICENSE_E_IS_HUNG_FIXTURE), "landed reading: licenseE is the hung fixture");
  assert(dump.reading.includes(X_E_IS_SERVING_STEP_FROM_TURN), "landed reading: X.E is serving-step E from the turn");
  assert(!dump.reading.includes(LEFTOVER_E_IS_LICENSE_PHRASE), "landed reading dropped leftover phrase");
  assert(!dump.dumpIsNot.includes(LEFTOVER_E_IS_LICENSE_PHRASE), "landed dumpIsNot dropped leftover phrase");
  assert(!dump.hung44LicenseObs.includes(LEFTOVER_E_IS_LICENSE_PHRASE), "landed hung44LicenseObs dropped leftover phrase");
  assertHonestServingStepE(dump);
}

async function testLiveHangObsIskuRefusesOldHungReplayAfterAndPHit(): Promise<void> {
  expectThrow(
    () =>
      buildLiveHangObsIskuReport({
        episode: { taskId: "44", hung: true, termination: "timeout" },
        sourceEval: [FORBIDDEN_HANG_SOURCES[0]!],
      }),
    "sourceEval-of-old-hung",
    "refuses sourceEval-of-old-hung (iweight-44-hung)",
  );
  expectThrow(
    () =>
      buildLiveHangObsIskuReport({
        episode: { taskId: "44", hung: true, termination: "timeout" },
        sourceEval: ["improve-live-0731-self-3944-postgate.json"],
      }),
    "sourceEval-of-old-hung",
    "refuses sourceEval-of-old-hung (postgate)",
  );
  expectThrow(
    () =>
      buildLiveHangObsIskuReport({
        episode: { taskId: "44", hung: true, termination: "timeout" },
        sourceEval: ["airline-live-self-3944-postgate-r0.json"],
      }),
    "sourceEval-of-old-hung",
    "refuses sourceEval-of-old-hung (airline-live-self-3944-postgate-r0)",
  );
  expectThrow(
    () =>
      buildLiveHangObsIskuReport({
        episode: { taskId: "44", hung: true, termination: "timeout" },
        controllerReplay: true,
      }),
    "controllerReplay=true",
    "refuses controllerReplay=true",
  );
  expectThrow(
    () =>
      buildLiveHangObsIskuReport({
        episode: { taskId: "44", hung: true, termination: "timeout" },
        after: 1,
      }),
    "after= present",
    "refuses after= present",
  );
  expectThrow(
    () =>
      buildLiveHangObsIskuReport({
        episode: { taskId: "44", hung: true, termination: "timeout" },
        pHit0813: 0.5,
      }),
    "pHit0813 set",
    "refuses pHit0813 set",
  );

  const replay = pendingLiveHangObsIskuReport();
  expectThrow(
    () => assertLiveHangObsIskuCell({ ...replay, controllerReplay: true }),
    "controllerReplay=true",
    "assert refuses controllerReplay=true",
  );
  expectThrow(
    () => assertLiveHangObsIskuCell({ ...replay, pHit0813: 0.1 }),
    "pHit0813 set",
    "assert refuses pHit0813 set",
  );
  expectThrow(
    () => assertLiveHangObsIskuCell({ ...replay, after: 1, omitAfter: true }),
    "after= present",
    "assert refuses after= on the report",
  );
  expectThrow(
    () =>
      assertLiveHangObsIskuCell({
        ...replay,
        sourceEval: ["improve-live-0731-iweight-44-hung.json"],
      }),
    "sourceEval-of-old-hung",
    "assert refuses old hung sourceEval",
  );
}

async function testLiveHangObsIskuThisEpisodeHungThenIskuOmitAfter(): Promise<void> {
  const obs = thisEpisodeHungObs("44");
  assertEq(obs.taskId, "44", "THIS episode task 44");
  assertEq(obs.hung, true, "THIS episode hung");
  assertEq(obs.arm, "I_sku", "hung-first Obs arm is I_sku");
  assertEq(obs.termination, "timeout", "THIS episode timeout");

  const ran = runLiveHangObsIskuController(obs);
  assertEq(ran.iSkuFired, true, "I_sku request fires on THIS hung episode");
  assert(ran.applyScope.weighted.includes("44"), "applyScope.weighted includes 44");
  assert(!ran.applyScope.waitKept.includes("44"), "waitKept does not include the hung task");
  assertEq(ran.gate.action, "reject", "omit after= rejects");
  assertEq(ran.gate.after, null, "gate.after is null");
  assertEq(ran.gate.reason, GATE_OMIT_AFTER_REASON, "0813 existing is not a gate");

  const report = buildLiveHangObsIskuReport({
    episode: {
      taskId: "44",
      hung: true,
      termination: "timeout",
      evalFile: LIVE_HANG_OBS_ISKU_FILE,
    },
    applyScope: ran.applyScope,
    gate: ran.gate,
    sourceEval: [LIVE_HANG_OBS_ISKU_FILE],
  });
  assertEq(report.live, true, "live cell");
  assertEq(report.controllerReplay, false, "not a controller replay");
  assertEq(report.freshHang, true, "fresh hang of THIS episode");
  assertEq(report.hung, true, "hung");
  assertEq(report.holeOpen, false, "hole closed only when this episode hung");
  assertEq(report.obs.arm, "I_sku", "obs.arm is I_sku");
  assertEq(report.omitAfter, true, "omitAfter");
  assertEq(report.jumped, false, "jumped=false");
  assertEq(report.servingPaused, false, "servingPaused=false");
  assertEq(report.servingModelAfter, SERVING_MODEL, "serving stays 0731");
  assertEq(report.pHit0813, null, "pHit0813=null");
  assertEq(report.gate.after, null, "gate.after=null");
  assertEq(report.gate.action, "reject", "gate reject");
  assertEq(report.iSkuRequest?.op, "i_sku", "I_sku request fired");
  assert(!("after" in (report.iSkuRequest ?? {})), "I_sku request omits after=");
  assertEq(report.trained, false, "trainer off");
  assertEq(report.liveAirlineImproveLoopOmitsAfter, true, "improveLoop still omits after=");
  assert(report.reading.includes(LIVE_HANG_OBS_ISKU_READING.split(";")[0]!), "reading is live Obs of this episode");
  assert(report.reading.includes("not a controller replay of saved hung-44"), "reading contrasts #12");
  assert(report.reading.includes("not a score"), "reading not a score");
  assert(report.reading.includes("not a dump"), "reading not a dump");
  assert(report.reading.includes("not live hung-44 then served as a mount"), "reading not a mount smear");
  assert(!report.reading.includes("not a new timeout"), "reading is not #12's phrase");
}

async function testLiveHangObsIskuNoHangKeepsHoleOpen(): Promise<void> {
  const report = buildLiveHangObsIskuReport({
    episode: {
      taskId: "44",
      hung: false,
      termination: "user_stop",
      nSuccessProxy: 0,
      arm: "I_loop",
    },
    sourceEval: [LIVE_HANG_OBS_ISKU_FILE],
  });
  assertEq(report.freshHang, false, "did not hang");
  assertEq(report.hung, false, "hung=false");
  assertEq(report.holeOpen, true, "hole remains open");
  assertEq(report.obs.arm, "I_loop", "completed miss is I_loop, not a stuffed hang");
  assertEq(report.iSkuRequest, null, "I_sku not fired");
  assertEq(report.controllerReplay, false, "still not a replay");
  assertEq(report.pHit0813, null, "still no p_hit(0813)");
  assert(report.reading.includes("hole remains open"), "reading says hole remains open");
  assert(!report.reading.includes("hung-first Obs chose I_sku"), "does not claim the hole closed");
}

async function testLiveHangObsIskuPendingKeyAndLandedJson(): Promise<void> {
  const pending = pendingLiveHangObsIskuReport();
  assertEq(pending.pendingKey, true, "pending a key");
  assertEq(pending.freshHang, false, "pending does not fake a hang");
  assertEq(pending.hung, false, "pending hung=false");
  assertEq(pending.holeOpen, true, "pending holeOpen");
  assertEq(pending.controllerReplay, false, "pending is not a replay");
  assertEq(pending.obs.arm, null, "pending obs.arm is null");
  assertEq(pending.pHit0813, null, "pending pHit0813=null");
  assertEq(pending.omitAfter, true, "pending still omits after=");
  assert(pending.reading.includes("pending a key"), "pending reading says pending a key");
  assert(!pending.reading.includes("not a new timeout"), "pending is not #12");

  const landed = readLiveHangObsIsku(liveHangObsIskuEvalPath());
  assertEq(landed.kind, "live-closed-loop-obs", "landed kind");
  assertEq(landed.controllerReplay, false, "landed controllerReplay=false");
  assertEq(landed.pHit0813, null, "landed pHit0813=null");
  assertEq(landed.omitAfter, true, "landed omitAfter");
  assertEq(landed.servingPaused, false, "landed servingPaused=false");
  assertEq(landed.liveAirlineImproveLoopOmitsAfter, true, "landed improveLoop still omits after=");
  assertEq(landed.trained, false, "landed trainer off");
  if (landed.pendingKey) {
    assertEq(landed.freshHang, false, "landed pending did not fake a hang");
    assertEq(landed.hung, false, "landed pending hung=false");
    assertEq(landed.holeOpen, true, "landed pending holeOpen");
    assertEq(landed.obs.arm, null, "landed pending obs.arm is null");
  } else if (landed.freshHang) {
    assertEq(landed.obs.arm, "I_sku", "landed hang obs.arm is I_sku");
    assertEq(landed.jumped, false, "landed hang jumped=false");
    assertEq(landed.servingModelAfter, SERVING_MODEL, "landed hang serving stays 0731");
    assertEq(landed.gate.after, null, "landed hang gate.after=null");
  } else {
    assertEq(landed.pendingKey, false, "landed measured episode is not pending a key");
    assertEq(landed.freshHang, false, "landed did not hang");
    assertEq(landed.hung, false, "landed no-hang is not a stuffed hang");
    assertEq(landed.holeOpen, true, "landed no-hang keeps the hole open");
    assertEq(landed.obs.arm, "I_loop", "landed obs.arm is I_loop");
    assertEq(landed.obs.hung, false, "landed obs.hung is false");
    assertEq(landed.obs.taskId, "44", "landed obs.taskId is 44");
    assertEq(landed.obs.termination, "user_stop", "landed termination is user_stop");
    assertEq(landed.obs.nSuccessProxy, 0, "landed nSuccessProxy is 0");
    assertEq(landed.gate.action, null, "I_sku not licensed");
    assertEq(landed.iSkuRequest, null, "I_sku request did not fire");
    assert(landed.reading.includes("hole remains open"), "reading keeps the hole open");
    assert(!landed.reading.includes("hung-first Obs chose I_sku"), "does not claim a hang");
  }
  const blob = JSON.stringify(landed);
  for (const name of FORBIDDEN_HANG_SOURCES) {
    assert(!blob.includes(name), `landed JSON does not cite ${name}`);
  }
}

async function testRejectCell12StaysControllerReplay(): Promise<void> {
  const rejectPath = join(dirname(liveHangObsIskuEvalPath()), ISKU_REJECT_CELL_FILE);
  const reject = JSON.parse(readFileSync(rejectPath, "utf8"));
  assertEq(reject.controllerReplay, true, "#12 stays controllerReplay=true");
  assertEq(reject.live, true, "#12 live means source traces were live, not a new timeout");
  assert(Array.isArray(reject.sourceEval), "#12 still cites saved hung files");
  for (const name of FORBIDDEN_HANG_SOURCES) {
    assert(reject.sourceEval.includes(name), `#12 sourceEval still includes ${name}`);
  }
  assert(
    String(reject.reading ?? "").includes("not a new timeout"),
    "#12 reading still says not a new timeout",
  );

  const cell = readLiveHangObsIsku(liveHangObsIskuEvalPath());
  assertEq(cell.controllerReplay, false, "this cell is not a replay");
  assertEq(cell.kind, "live-closed-loop-obs", "this cell is live Obs, not a relabeled reject");
  assert(cell.kind !== reject.kind, "this JSON is not #12 relabeled");
  assertEq(cell.vsRejectCell, ISKU_REJECT_CELL_FILE, "cites #12 as the other cell");
  const cellBlob = JSON.stringify(cell);
  for (const name of FORBIDDEN_HANG_SOURCES) {
    assert(!cellBlob.includes(name), `this cell is not ${name} stuffed through Obs`);
  }
}

async function testR6LaterTimeoutDoesNotOverwriteNoHangPacket(): Promise<void> {
  const first = readLiveHangObsIsku(liveHangObsIskuEvalPath());
  assertEq(first.pendingKey, false, "1c3528c packet is measured");
  assertEq(first.hung, false, "1c3528c hung=false stays");
  assertEq(first.freshHang, false, "1c3528c freshHang=false stays");
  assertEq(first.holeOpen, true, "1c3528c holeOpen=true stays");
  assertEq(first.obs.arm, "I_loop", "1c3528c obs.arm stays I_loop");
  assertEq(first.controllerReplay, false, "1c3528c is not a replay");

  const r6 = readLiveHangObsIsku(liveHangObsIskuR6EvalPath());
  assertEq(r6.kind, "live-closed-loop-obs", "r6 is the live Obs cell");
  assertEq(r6.controllerReplay, false, "r6 is not a controller replay");
  assertEq(r6.pendingKey, false, "r6 is measured");
  assertEq(r6.freshHang, true, "r6 is a later live hang");
  assertEq(r6.hung, true, "r6 hung=true");
  assertEq(r6.holeOpen, false, "r6 I_sku ran on this hang");
  assertEq(r6.obs.arm, "I_sku", "r6 obs.arm is I_sku");
  assertEq(r6.obs.hung, true, "r6 obs.hung");
  assertEq(r6.obs.taskId, "44", "r6 task 44");
  assertEq(r6.obs.termination, "timeout", "r6 timeout");
  assert(!r6.applyScope.waitKept.includes("44"), "r6 waitKept does not include 44");
  assert(r6.applyScope.weighted.includes("44"), "r6 weighted includes 44");
  assertEq(r6.omitAfter, true, "r6 omits after=");
  assertEq(r6.iSkuRequest?.op, "i_sku", "r6 I_sku fired");
  assert(!("after" in (r6.iSkuRequest ?? {})), "r6 I_sku request omits after=");
  assertEq(r6.gate.action, "reject", "r6 gate reject");
  assertEq(r6.gate.after, null, "r6 gate.after=null");
  assertEq(r6.jumped, false, "r6 jumped=false");
  assertEq(r6.servingPaused, false, "r6 servingPaused=false");
  assertEq(r6.servingModelAfter, SERVING_MODEL, "r6 serving stays 0731");
  assertEq(r6.pHit0813, null, "r6 does not invent p_hit(0813)");
  assertEq(r6.trained, false, "r6 trainer off");
  assert(r6.reading.includes("hung-first Obs chose I_sku"), "r6 reading is this hang");
  assert(!r6.reading.includes("not a new timeout"), "r6 is not #12's phrase");
  const blob = JSON.stringify(r6);
  for (const name of FORBIDDEN_HANG_SOURCES) {
    assert(!blob.includes(name), `r6 is not ${name} stuffed through Obs`);
  }
  assert(r6.sourceEval.includes(LIVE_HANG_OBS_ISKU_FILE) || r6.sourceEval.includes(LIVE_HANG_OBS_ISKU_R6_FILE), "r6 sourceEval is this run");
  expectThrow(
    () =>
      assertLiveHangObsIskuCell({
        ...r6,
        sourceEval: [FORBIDDEN_HANG_SOURCES[0]!],
        controllerReplay: true,
      }),
    "sourceEval-of-old-hung",
    "r6 refuse #12 / old hung stuffed as this cell",
  );
}

async function testLiveHangObsIskuTaskIdWritesNewFile(): Promise<void> {
  assertEq(liveHangObsIskuFilename("44"), LIVE_HANG_OBS_ISKU_FILE, "44 keeps historical filename");
  assertEq(
    liveHangObsIskuFilename("39"),
    "improve-live-0731-hang-obs-isku-39.json",
    "39 writes a new file",
  );
  assertEq(liveHangObsIskuEvalPath(), liveHangObsIskuEvalPath(undefined, "44"), "default path is 44");
  const parsedDefault = parseLiveHangObsIskuArgs([]);
  assertEq(parsedDefault.taskId, LIVE_HANG_OBS_ISKU_TASK_DEFAULT, "CLI default TASK_ID is 44");
  const parsed39 = parseLiveHangObsIskuArgs(["--live-hang-obs-isku", "39", "--out", "tmp-39.json"]);
  assertEq(parsed39.taskId, "39", "CLI accepts TASK_ID 39");
  assertEq(parsed39.out, "tmp-39.json", "CLI accepts --out");
  const parsedPos = parseLiveHangObsIskuArgs(["39"]);
  assertEq(parsedPos.taskId, "39", "CLI accepts positional TASK_ID");

  const firstPath = liveHangObsIskuEvalPath();
  const r6Path = liveHangObsIskuR6EvalPath();
  const landed39Path = liveHangObsIsku39EvalPath();
  const firstBefore = readFileSync(firstPath);
  const r6Before = readFileSync(r6Path);
  const landed39Before = readFileSync(landed39Path);
  const dest = join(tmpdir(), "improve-live-0731-hang-obs-isku-39.json");
  const pending = pendingLiveHangObsIskuReport("39");
  assertEq(pending.taskIds[0], "39", "pending taskIds is 39");
  assertEq(pending.sourceEval[0], "improve-live-0731-hang-obs-isku-39.json", "pending cites the 39 file");
  const wrote = writeLiveHangObsIsku(pending, dest);
  assert(wrote.endsWith("improve-live-0731-hang-obs-isku-39.json"), "wrote the 39 filename");
  const onDisk = JSON.parse(readFileSync(wrote, "utf8"));
  assertEq(onDisk.taskIds[0], "39", "disk packet is task 39");
  assertEq(onDisk.pendingKey, true, "dry path is pending, not a live hang");
  assertEq(onDisk.hung, false, "dry path does not reconstruct hung=true");
  assertEq(onDisk.controllerReplay, false, "dry path is not a replay");
  assertEq(onDisk.pHit0813, null, "dry path does not invent p_hit(0813)");
  assert(readFileSync(firstPath).equals(firstBefore), "did not touch 1c3528c packet");
  assert(readFileSync(r6Path).equals(r6Before), "did not touch r6 packet");
  assert(readFileSync(landed39Path).equals(landed39Before), "did not touch landed 39 packet");
  expectThrow(
    () => writeLiveHangObsIsku(pending, firstPath),
    "overwrite",
    "task 39 must not write the historical 44 packet",
  );
  unlinkSync(wrote);
  assert(readFileSync(firstPath).equals(firstBefore), "1c3528c still untouched after cleanup");
  assert(readFileSync(r6Path).equals(r6Before), "r6 still untouched after cleanup");
  assert(readFileSync(landed39Path).equals(landed39Before), "landed 39 still untouched after cleanup");

  const obs = thisEpisodeHungObs("39");
  assertEq(obs.taskId, "39", "THIS episode task 39");
  const ran = runLiveHangObsIskuController(obs);
  assertEq(ran.iSkuFired, true, "I_sku fires on hung 39");
  assert(ran.applyScope.weighted.includes("39"), "weighted includes 39");
  assert(!ran.applyScope.waitKept.includes("39"), "waitKept excludes hung 39");
  const hungReport = buildLiveHangObsIskuReport({
    episode: { taskId: "39", hung: true, termination: "timeout" },
    applyScope: ran.applyScope,
    gate: ran.gate,
  });
  assertEq(hungReport.taskIds[0], "39", "builder uses episode.taskId");
  assertEq(hungReport.obs.taskId, "39", "obs.taskId is 39");
  assertEq(hungReport.obs.arm, "I_sku", "hung 39 arm is I_sku");
  assertEq(hungReport.omitAfter, true, "omit after=");
  assertEq(hungReport.gate.after, null, "gate.after=null");
  assertEq(hungReport.gate.action, "reject", "gate reject");
  assertEq(hungReport.jumped, false, "jumped=false");
  assertEq(hungReport.servingPaused, false, "servingPaused=false");
  assertEq(hungReport.servingModelAfter, SERVING_MODEL, "serving stays 0731");
  assertEq(hungReport.pHit0813, null, "no invented p_hit(0813)");
  assertEq(hungReport.sourceEval[0], "improve-live-0731-hang-obs-isku-39.json", "default source is the 39 file");
}

async function testLanded39NoHangDoesNotOverwrite44Packets(): Promise<void> {
  const first = readLiveHangObsIsku(liveHangObsIskuEvalPath());
  const r6 = readLiveHangObsIsku(liveHangObsIskuR6EvalPath());
  const landed = readLiveHangObsIsku(liveHangObsIsku39EvalPath());
  assertEq(first.hung, false, "1c3528c hung=false stays");
  assertEq(first.obs.taskId, "44", "1c3528c stays task 44");
  assertEq(r6.hung, true, "r6 hung=true stays");
  assertEq(r6.obs.taskId, "44", "r6 stays task 44");
  assertEq(r6.obs.termination, "timeout", "r6 timeout stays");
  assertEq(landed.kind, "live-closed-loop-obs", "39 is the live Obs cell");
  assertEq(landed.pendingKey, false, "39 is measured");
  assertEq(landed.controllerReplay, false, "39 is not a replay");
  assertEq(landed.freshHang, false, "39 did not hang");
  assertEq(landed.hung, false, "39 hung=false");
  assertEq(landed.holeOpen, true, "39 hole remains open");
  assertEq(landed.taskIds[0], "39", "39 taskIds");
  assertEq(landed.obs.taskId, "39", "39 obs.taskId");
  assertEq(landed.obs.arm, "I_loop", "39 obs.arm is I_loop");
  assertEq(landed.obs.hung, false, "39 obs.hung is false");
  assertEq(landed.obs.termination, "user_stop", "39 user_stop");
  assertEq(landed.obs.nSuccessProxy, 0, "39 nSuccessProxy is 0");
  assertEq(landed.arm ?? null, null, "I_sku not licensed");
  assertEq(landed.iSkuRequest, null, "I_sku request did not fire");
  assertEq(landed.gate.action, null, "gate action null");
  assertEq(landed.gate.after, null, "gate.after=null");
  assertEq(landed.omitAfter, true, "omit after=");
  assertEq(landed.jumped, false, "jumped=false");
  assertEq(landed.servingPaused, false, "servingPaused=false");
  assertEq(landed.servingModelAfter, SERVING_MODEL, "serving stays 0731");
  assertEq(landed.pHit0813, null, "does not invent p_hit(0813)");
  assertEq(landed.sourceEval[0], LIVE_HANG_OBS_ISKU_39_FILE, "39 sourceEval is this file");
  assert(
    landed.sourceEval.some((s) => s.includes("airline-live-one-shot-r1787320701-20260821T135821Z.json")),
    "39 cites the measured one-shot, not invented onto the PR",
  );
  assert(landed.reading.includes("hole remains open"), "39 reading keeps the hole open");
  assert(!landed.reading.includes("hung-first Obs chose I_sku"), "39 does not claim a hang");
  assert(!landed.reading.includes("not a new timeout"), "39 is not #12's phrase");
  const blob = JSON.stringify(landed);
  for (const name of FORBIDDEN_HANG_SOURCES) {
    assert(!blob.includes(name), `39 is not ${name} stuffed through Obs`);
  }
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
    ["mixed wait+hit keeps C0; miss gets cancel_policy", testMixedWaitHitKeepsC0],
    ["unscoped I_loop never silent-global-mounts wait-hit", testUnscopedILoopNeverSilentGlobal],
    ["all-miss valid self I_loop still applies", testAllMissSelfILoopStillApplies],
    ["host applyILoop fallback uses the wait-hit gate", testApplyILoopFallbackWaitHitGate],
    ["self-Obs prompt has per-episode arms and no gold IDs", testSelfObsPromptHasEpisodesNoGoldIds],
    ["typed arms: hung I_sku / hit wait / policy I_loop", testTypedInterventionArms],
    ["post-gate 39/44 replay: 39 I_loop, hung 44 I_sku, waitKept=[]", testPostGate3944Replay],
    ["I_sku catalog-rebind mounts 0813 (mocked bind, not LoRA)", testCatalogJumpMounts0813],
    ["I_loop does not write S", testILoopDoesNotWriteS],
    ["S is a CatalogPointer beside C, not n.model", testServingPointerBesideC],
    ["mixed 39/44 later serving typed by S, not sprayed provider", testMixed3944LaterServingTypedByS],
    ["fresh 39-only batch does not inherit process servingSku=0813", testFreshBatchDoesNotInheritProcessServingSku],
    ["I_sku mount-cell controller: fixture after, no live, 44=0813 39=0731 no spray", testIskuMountCellControllerFixtureAfterNoLive],
    ["X_n.S dump after licensed write: HybridState.S on the state object", testHybridStateSDumpAfterLicensedWrite],
    ["missing X[\"39\"] throws; does not invent S=0731", testMissingX39ThrowsNoAssemble],
    ["serving-step dump refuses empty H/M and stuffed pong/sourceEval", testServingStepDumpRefusesEmptyAndStuffedHM],
    ["serving-step dump refuses E smear / live hung-44 then served", testServingStepDumpRefusesESmear],
    ["serving-step dump refuses overlay / missing own licenseE servingE", testServingStepDumpRefusesOverlayAndMissingOwnFields],
    ["serving-step X_n dump: same object, H/M from runTau2Turn", testHybridStateServingStepDumpAfterLicensedWrite],
    ["landed serving-step dump: licenseE ≠ servingE; live H/M kept", testLandedServingStepDumpESplit],
    ["live hang-obs-isku refuses old hung / replay / after= / pHit0813", testLiveHangObsIskuRefusesOldHungReplayAfterAndPHit],
    ["live hang-obs-isku THIS episode hung → I_sku omit after", testLiveHangObsIskuThisEpisodeHungThenIskuOmitAfter],
    ["live hang-obs-isku no hang keeps hole open", testLiveHangObsIskuNoHangKeepsHoleOpen],
    ["live hang-obs-isku pending key + landed JSON", testLiveHangObsIskuPendingKeyAndLandedJson],
    ["#12 reject cell stays controllerReplay; this cell is not a relabel", testRejectCell12StaysControllerReplay],
    ["r6 later timeout does not overwrite 1c3528c no-hang packet", testR6LaterTimeoutDoesNotOverwriteNoHangPacket],
    ["live hang-obs-isku TASK_ID 39 writes a new file", testLiveHangObsIskuTaskIdWritesNewFile],
    ["landed 39 no-hang does not overwrite 44 packets", testLanded39NoHangDoesNotOverwrite44Packets],
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
