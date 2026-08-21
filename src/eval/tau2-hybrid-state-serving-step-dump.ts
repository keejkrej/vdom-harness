/**
 * Serving-step X_n dump after a licensed I_sku write (critic hole (1) after #16).
 *
 * Reuses the #15/#16 controller (hung-44 license + fixture after). Writes
 * X_44.S=0813 / X_39.S=0731 on EXISTING HybridState objects, then runs ONE
 * real serving turn (runTau2Turn) that mutates that same X: messages → X.H,
 * traces → X.M. Not ping. Not get_state. Not a HybridState assembled from
 * servingByTask. Not stuffed hung44LicenseObs / #15 pong / sourceEval / a
 * fake user line.
 *
 * Not a score. Not invented p_hit(0813). Not a τ² result. Trainer I_weight
 * stays off. Live airline improveLoop still omits after=.
 */
import { writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { RuntimeDOM } from "../reconciler.js";
import { findNode } from "../ir.js";
import { tau2Graph } from "./tau2-graph.js";
import { type ControlledBatch } from "./tau2-control.js";
import {
  ISKU_MOUNT_CELL_FILE,
  ISKU_REJECT_CELL_FILE,
  SOURCE_EVAL,
  runIskuMountCellController,
} from "./tau2-isku-mount-cell.js";
import {
  CATALOG_JUMP_MODEL,
  SERVING_MODEL,
  sameCTopology,
} from "./tau2-weight.js";
import {
  compactC,
  cGraphHash,
  nModelOf,
  nodeListOf,
  sOnState,
  type HybridState,
} from "./tau2-hybrid-state.js";
import { runFresh39AfterMount } from "./tau2-hybrid-state-s-dump.js";
import {
  runTau2Turn,
  type Tau2TurnResult,
} from "./tau2-turn.js";
import {
  resolveChatConfig,
  resolveProvider,
  type Completion,
  type CompleteOpts,
  type Message,
  type Provider,
} from "../providers.js";
import { type CatalogPointer } from "./tau2-types.js";

export const HYBRID_STATE_SERVING_STEP_DUMP_FILE = "hybrid-state-serving-step-dump.json";

export const HYBRID_STATE_SERVING_STEP_DUMP_READING =
  "Serving-step X_n dump after licensed write: same HybridState the turn mutated; H/M from that runTau2Turn, not assembled; not a score";

export const SERVING_STEP_DUMP_IS_NOT =
  "ping / get_state S0 / controller-only empty H" as const;

export const LIVE_TURN_REJECT_NO_KEY =
  "no OPENROUTER_API_KEY; live serving id not faked; mock turn wrote H/M onto existing X";

const STUFFED_NEEDLES = [
  "Reply with the single word pong.",
  "reconstructed hung=true/timeout",
  "improve-live-0731-iweight-44-hung.json",
  "improve-live-0731-self-3944-postgate.json",
  "airline-live-self-3944-postgate-r0.json",
] as const;

/**
 * Mock provider for a real runTau2Turn when no live key.
 * Output is produced by this invocation (turn id + sku). Not copied obs.
 */
export class ServingStepMockProvider implements Provider {
  name = "serving-step-mock";
  model?: string;
  lastTurnId = "";

  constructor(model?: string) {
    this.model = model;
  }

  async complete(msgs: Message[], opts?: CompleteOpts): Promise<string> {
    return (await this.completeTurn(msgs, opts)).content;
  }

  async completeTurn(_msgs: Message[], opts?: CompleteOpts): Promise<Completion> {
    const sku = this.model ?? opts?.model ?? "unknown";
    this.lastTurnId = `serving-step-turn:${sku}:${Date.now()}`;
    return { content: this.lastTurnId, servedModel: this.model };
  }
}

export function defaultServingStepProvider(sku: string): {
  provider: Provider;
  mock: boolean;
} {
  if (resolveChatConfig()) {
    return { provider: resolveProvider(sku), mock: false };
  }
  return { provider: new ServingStepMockProvider(sku), mock: true };
}

export type ServingStepDumpView = {
  H: HybridState["H"];
  M: HybridState["M"];
  E: { taskId?: string; hung?: boolean; arm?: string; termination?: string };
  C: ReturnType<typeof compactC>;
  S: CatalogPointer;
  S_on_state: true;
  hLength: number;
  mLength: number;
};

export type ServingStepStamp = {
  op: "runTau2Turn";
  notPing: true;
  notGetState: true;
  hFromTurn: true;
  mFromTurn: true;
  sameObjectAsTurnX: true;
  model: string;
  hLength: number;
  mLength: number;
  hRoles: string[];
  firstTrace?: { nodeKey: string; role: string; ts: number };
  assistantChars: number;
  assistantPreview: string;
};

export type HybridStateServingStepDump = {
  benchmark: "tau2-bench";
  kind: "hybrid-state-serving-step-dump";
  not_a_sota_result: true;
  closedLoop: false;
  maxRounds: 0;
  domain: "airline";
  taskIds: ["39", "44"];
  arm44: "I_sku";
  license: "hung";
  slice: "I_sku";
  applyScope: { waitKept: string[]; looped: string[]; weighted: string[] };
  servingPaused: false;
  trained: false;
  notFineTuning: true;
  notInventedPHit0813: true;
  pHit0813: null;
  protocolCell: true;
  fixtureAfter: true;
  incompleteFixture: true;
  notTau2Lift: true;
  omitAfter: false;
  liveAirlineImproveLoopOmitsAfter: true;
  liveServe: false;
  liveServingId: string | null;
  servingIdNotFaked: true;
  liveTurnRejected: boolean;
  liveTurnRejectReason: string | null;
  mockProviderTurn: boolean;
  vsRejectCell: typeof ISKU_REJECT_CELL_FILE;
  vsMountCell: typeof ISKU_MOUNT_CELL_FILE;
  jumped: boolean;
  jumpedIs: "S write on X_n, then one serving-step turn on that same X";
  mounted: boolean;
  sourceOfTruth: "X_n the turn mutated";
  dumpIsNot: typeof SERVING_STEP_DUMP_IS_NOT;
  notAssembledFromServingByTask: true;
  sameObjectAsTurnX: true;
  servingByTaskIs: "derived cache from X.S, not the lookup";
  servingStep: ServingStepStamp;
  X_44: ServingStepDumpView;
  X_39: ServingStepDumpView;
  C: {
    nModel: string;
    nModelUnchanged: boolean;
    topologyUnchanged: boolean;
    graphHashBefore: string;
    graphHashAfter: string;
    nodeListBefore: string[];
    nodeListAfter: string[];
    sameNodeList: boolean;
  };
  fresh39: {
    S: CatalogPointer;
    inherited0813: false;
  };
  sourceEval: readonly string[];
  hung44LicenseObs: "reconstructed hung=true/timeout fixture citing sourceEval; not a new 0731 timeout; not copied into H";
  trainerIWeight: "unimplemented; off the claim";
  reading: typeof HYBRID_STATE_SERVING_STEP_DUMP_READING;
};

export function stuffedServingStepReason(X: HybridState): string | null {
  const blob = JSON.stringify({ H: X.H, M: X.M });
  for (const needle of STUFFED_NEEDLES) {
    if (blob.includes(needle)) return `stuffed H/M contains ${needle}`;
  }
  return null;
}

export function assertServingStepHM(X: HybridState, turn?: Tau2TurnResult): void {
  if (!sOnState(X)) {
    throw new Error("dump refused: HybridState has no own S; will not assemble from servingByTask");
  }
  if (X.H.length === 0 || X.M.length === 0) {
    throw new Error("dump refused: empty H/M; controller-only X is not a serving-step");
  }
  const stuffed = stuffedServingStepReason(X);
  if (stuffed) {
    throw new Error(`dump refused: ${stuffed}`);
  }
  if (turn) {
    const assistant = [...X.H].reverse().find((m) => m.role === "assistant");
    if (!assistant || assistant.content !== turn.content) {
      throw new Error("dump refused: X.H assistant is not this turn's completion");
    }
    if (!X.M.some((t) => turn.traces.some((tr) => tr.ts === t.ts && tr.output === t.output))) {
      throw new Error("dump refused: X.M is not this turn's traces");
    }
  }
}

function viewOfServingStep(X: HybridState, requireHM: boolean): ServingStepDumpView {
  if (!sOnState(X)) {
    throw new Error("dump refused: HybridState has no own S; will not assemble from servingByTask");
  }
  if (requireHM) assertServingStepHM(X);
  return {
    H: X.H,
    M: X.M,
    E: {
      taskId: X.E.taskId,
      hung: X.E.hung,
      arm: X.E.arm,
      termination: X.E.termination,
    },
    C: compactC(X.C),
    S: X.S,
    S_on_state: true,
    hLength: X.H.length,
    mLength: X.M.length,
  };
}

export function servingStepStamp(X: HybridState, turn: Tau2TurnResult): ServingStepStamp {
  const assistant = [...X.H].reverse().find((m) => m.role === "assistant");
  const preview = (assistant?.content ?? "").slice(0, 80);
  const first = X.M[0];
  return {
    op: "runTau2Turn",
    notPing: true,
    notGetState: true,
    hFromTurn: true,
    mFromTurn: true,
    sameObjectAsTurnX: true,
    model: X.S.sku,
    hLength: X.H.length,
    mLength: X.M.length,
    hRoles: X.H.map((m) => m.role),
    firstTrace: first
      ? { nodeKey: first.nodeKey, role: first.role, ts: first.ts }
      : undefined,
    assistantChars: (assistant?.content ?? "").length,
    assistantPreview: preview,
  };
}

/** One serving turn on the existing X. Mutates that object. No fake user line. */
export async function runServingStepOnExistingX(
  X: HybridState,
  opts?: {
    provider?: Provider;
    policy?: string;
    messages?: Message[];
  },
): Promise<{ X: HybridState; turn: Tau2TurnResult; sameRef: true }> {
  const incoming = opts?.messages ?? [];
  const turn = await runTau2Turn({
    policy: opts?.policy ?? "",
    tools: [],
    messages: incoming,
    graph: X.C,
    model: X.S.sku,
    provider: opts?.provider,
    X,
  });
  return { X, turn, sameRef: true };
}

export async function buildHybridStateServingStepDump(opts?: {
  ctrl?: ControlledBatch;
  provider?: Provider;
  messages?: Message[];
}): Promise<{
  ctrl: ControlledBatch;
  dump: HybridStateServingStepDump;
  X_44: HybridState;
  X_39: HybridState;
  turn: Tau2TurnResult;
  sameRef: true;
}> {
  const start = tau2Graph("one-shot", SERVING_MODEL);
  const hashBefore = cGraphHash(start);
  const { ctrl } = opts?.ctrl
    ? { ctrl: opts.ctrl }
    : runIskuMountCellController();
  const X_44 = ctrl.X["44"];
  const X_39 = ctrl.X["39"];
  if (!X_44 || !X_39) {
    throw new Error("controller did not install HybridState X_44 / X_39");
  }
  if (!sOnState(X_44) || !sOnState(X_39)) {
    throw new Error("X_n has no own S field; dump is not servingByTask");
  }

  const resolved = opts?.provider
    ? { provider: opts.provider, mock: opts.provider instanceof ServingStepMockProvider || opts.provider.name === "serving-step-mock" || opts.provider.name.startsWith("deterministic") }
    : defaultServingStepProvider(X_44.S.sku);
  const liveKey = Boolean(resolveChatConfig());
  const mockTurn = resolved.mock || !liveKey;

  const beforeRef = X_44;
  const { X: turned, turn } = await runServingStepOnExistingX(X_44, {
    provider: resolved.provider,
    messages: opts?.messages ?? [],
  });
  if (turned !== beforeRef || turned !== ctrl.X["44"] || turned !== ctrl.episodes.find((e) => e.taskId === "44")?.X) {
    throw new Error("serving-step X is not the same object the controller held");
  }
  assertServingStepHM(turned, turn);

  const cAfter = ctrl.graphSku ?? ctrl.graphC0 ?? start;
  const hashAfter = cGraphHash(cAfter);
  const nModel = nModelOf(cAfter);
  const nodesBefore = nodeListOf(start);
  const nodesAfter = nodeListOf(cAfter);
  const mounted = ctrl.gate?.action === "mount";
  const jumped = Boolean(mounted && X_44.S.sku === CATALOG_JUMP_MODEL && X_44.S.servingPaused === false);
  const fresh = runFresh39AfterMount(ctrl.graphC0 ?? start);

  const dump: HybridStateServingStepDump = {
    benchmark: "tau2-bench",
    kind: "hybrid-state-serving-step-dump",
    not_a_sota_result: true,
    closedLoop: false,
    maxRounds: 0,
    domain: "airline",
    taskIds: ["39", "44"],
    arm44: "I_sku",
    license: "hung",
    slice: "I_sku",
    applyScope: {
      waitKept: [...ctrl.applyScope.waitKept],
      looped: [...ctrl.applyScope.looped],
      weighted: [...ctrl.applyScope.weighted],
    },
    servingPaused: false,
    trained: false,
    notFineTuning: true,
    notInventedPHit0813: true,
    pHit0813: null,
    protocolCell: true,
    fixtureAfter: true,
    incompleteFixture: true,
    notTau2Lift: true,
    omitAfter: false,
    liveAirlineImproveLoopOmitsAfter: true,
    liveServe: false,
    liveServingId: !mockTurn && liveKey ? (turn.servedModel ?? null) : null,
    servingIdNotFaked: true,
    liveTurnRejected: mockTurn,
    liveTurnRejectReason: mockTurn ? LIVE_TURN_REJECT_NO_KEY : null,
    mockProviderTurn: mockTurn,
    vsRejectCell: ISKU_REJECT_CELL_FILE,
    vsMountCell: ISKU_MOUNT_CELL_FILE,
    jumped,
    jumpedIs: "S write on X_n, then one serving-step turn on that same X",
    mounted: Boolean(mounted),
    sourceOfTruth: "X_n the turn mutated",
    dumpIsNot: SERVING_STEP_DUMP_IS_NOT,
    notAssembledFromServingByTask: true,
    sameObjectAsTurnX: true,
    servingByTaskIs: "derived cache from X.S, not the lookup",
    servingStep: servingStepStamp(X_44, turn),
    X_44: viewOfServingStep(X_44, true),
    X_39: viewOfServingStep(X_39, false),
    C: {
      nModel,
      nModelUnchanged: nModel === SERVING_MODEL,
      topologyUnchanged: sameCTopology(cAfter, start),
      graphHashBefore: hashBefore,
      graphHashAfter: hashAfter,
      nodeListBefore: nodesBefore,
      nodeListAfter: nodesAfter,
      sameNodeList: nodesBefore.join(",") === nodesAfter.join(","),
    },
    fresh39: {
      S: fresh.S,
      inherited0813: false,
    },
    sourceEval: SOURCE_EVAL,
    hung44LicenseObs:
      "reconstructed hung=true/timeout fixture citing sourceEval; not a new 0731 timeout; not copied into H",
    trainerIWeight: "unimplemented; off the claim",
    reading: HYBRID_STATE_SERVING_STEP_DUMP_READING,
  };
  return { ctrl, dump, X_44, X_39, turn, sameRef: true };
}

export function hybridStateServingStepDumpPath(repoRoot?: string): string {
  const root =
    repoRoot ??
    join(dirname(fileURLToPath(import.meta.url)), "..", "..");
  return join(root, "eval", "tau2", HYBRID_STATE_SERVING_STEP_DUMP_FILE);
}

export function writeHybridStateServingStepDump(
  dump: HybridStateServingStepDump,
  path?: string,
): string {
  const out = path ?? hybridStateServingStepDumpPath();
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, `${JSON.stringify(dump, null, 2)}\n`);
  return out;
}

export async function runHybridStateServingStepDump(): Promise<{
  dump: HybridStateServingStepDump;
  path: string;
}> {
  const { dump } = await buildHybridStateServingStepDump();
  const path = writeHybridStateServingStepDump(dump);
  return { dump, path };
}

function parseArgs(argv: string[]): { write: boolean } {
  return { write: !argv.includes("--no-write") };
}

async function main(): Promise<void> {
  const { write } = parseArgs(process.argv.slice(2));
  const start = tau2Graph("one-shot", SERVING_MODEL);
  const dom = new RuntimeDOM();
  dom.reconcile(start);
  const { ctrl, dump } = await buildHybridStateServingStepDump();
  if (write) {
    const path = writeHybridStateServingStepDump(dump);
    process.stdout.write(`${JSON.stringify({ wrote: path, dump }, null, 2)}\n`);
  } else {
    process.stdout.write(`${JSON.stringify({ dump }, null, 2)}\n`);
  }
  const solve = findNode(ctrl.graphC0 ?? start, "solve");
  if (solve?.model && solve.model !== SERVING_MODEL) {
    process.stderr.write("warning: C n.model moved off 0731\n");
  }
}

const selfPath = fileURLToPath(import.meta.url);
const invoked = Boolean(
  process.argv[1] &&
    (selfPath === process.argv[1] ||
      selfPath.endsWith(process.argv[1]) ||
      process.argv[1].endsWith("tau2-hybrid-state-serving-step-dump.ts")),
);
if (invoked) {
  main().catch((err) => {
    process.stderr.write(`${err instanceof Error ? err.message : String(err)}\n`);
    process.exit(1);
  });
}
