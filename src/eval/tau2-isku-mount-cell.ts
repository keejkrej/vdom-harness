/**
 * Honest live I_sku ACCEPT protocol cell.
 *
 * Not a τ² result. Not a Pro-vs-Flash score. Does not invent p_hit(0813).
 * Does not sell p_hit(0813)−p_hit(0731). No new airline table.
 *
 * I_sku is a gated catalog rebind cell, not the contribution.
 * Jump iff the later serving model id from a live serve is 0813.
 * servingPaused stays false. 0813 existing is not a gate.
 * Gate needs a measured after-eval — here a fixture after, labeled as such.
 *
 * #12 reject cell omitted after (sidecar always rejects). This path calls
 * I_sku WITH after, then does ONE live OpenRouter completion on 0813.
 */
import { writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { findNode } from "../ir.js";
import { RuntimeDOM } from "../reconciler.js";
import {
  resolveChatConfig,
  resolveProvider,
  type Provider,
} from "../providers.js";
import { observeTau2 } from "./tau2-obs.js";
import { tau2Graph } from "./tau2-graph.js";
import { type Tau2Obs } from "./tau2-types.js";
import {
  controlBatch,
  controllerServingLog,
  servingModelForTask,
  servingProviderForTask,
  type ControlledBatch,
} from "./tau2-control.js";
import {
  CATALOG_JUMP_MODEL,
  SERVING_MODEL,
  catalogPointer,
} from "./tau2-weight.js";

export const ISKU_MOUNT_CELL_FILE = "improve-live-0731-isku-44-mount.json";
export const ISKU_REJECT_CELL_FILE = "improve-live-0731-isku-44-reject.json";

export const ISKU_MOUNT_CELL_NOTE =
  "Protocol cell, not a score. Hung-44 license replay (same sources as #12 reject) " +
  "through controlBatch / Obs; I_sku called WITH fixture after (after=1, before=0); " +
  "then one live OpenRouter completion on 0813. Jump iff that serving model id is 0813. " +
  "fixtureAfter / incompleteFixture / not a τ² lift. Not measured 0813 on airline. " +
  "Not invented p_hit(0813). Not p_hit(0813)−p_hit(0731). Not a Pro-vs-Flash score. " +
  "Catalog rebind, not fine-tuning. servingPaused stays false.";

export const SOURCE_EVAL = [
  "improve-live-0731-iweight-44-hung.json",
  "improve-live-0731-self-3944-postgate.json",
  "airline-live-self-3944-postgate-r0.json",
] as const;

const LIVE_PING = "Reply with the single word pong.";

export type LiveServeRecord = {
  ok: boolean;
  servingModelId: string | null;
  clientModel: string | null;
  reason?: string;
  contentChars?: number;
};

export type IskuMountCellReport = {
  benchmark: "tau2-bench";
  kind: "catalog-rebind";
  not_a_sota_result: true;
  closedLoop: false;
  maxRounds: 0;
  domain: "airline";
  taskIds: ["44"];
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
  vsRejectCell: typeof ISKU_REJECT_CELL_FILE;
  jumped: boolean;
  mounted: boolean;
  rejected: boolean;
  gate: {
    action: "mount" | "reject";
    before: number;
    after: number | null;
    kind: "fixtureAfter";
    incompleteFixture: true;
    notTau2Lift: true;
    reason: string;
  };
  servingModelAfter: string | null;
  proposedModel: typeof CATALOG_JUMP_MODEL;
  controllerServing: string;
  live: true;
  controllerReplay: true;
  liveServe: LiveServeRecord;
  nTurns: 9;
  nHungTrials: 1;
  nTurnsAre: string;
  sourceEval: readonly string[];
  reading: string;
};

export type IskuMountCellOpts = {
  before?: number;
  after?: number;
  provider?: Provider;
  dom?: RuntimeDOM;
  /** Attempt one live 0813 completion. Default false (unit / no-key). */
  liveServe?: boolean;
  /** Allow a injected mock provider to stand in for the live call. Tests only. */
  mockLive?: Provider;
};

/** Same hung-44 license Obs as #12 reject / post-gate 39/44 replay. */
export function hung44LicenseObs(): Tau2Obs {
  return observeTau2({
    traces: [],
    taskId: "44",
    reward: null,
    hung: true,
    termination: "timeout",
    actions: [],
  });
}

/** Post-gate 39 completed policy miss (I_loop). Proves S does not spray. */
export function completedMiss39Obs(): Tau2Obs {
  return observeTau2({
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
}

/** True when a recorded serving model id is 0813. Existence of the SKU is not a gate. */
export function servingIdIs0813(id: string | null | undefined): boolean {
  if (!id) return false;
  return id === CATALOG_JUMP_MODEL || id.includes("0813");
}

/**
 * Controller path: hung-44 license through controlBatch / Obs, I_sku WITH after.
 * Fixture after=1 > before=0. No live serve. Sets S for 44 to 0813; 39 stays 0731.
 */
export function runIskuMountCellController(opts?: IskuMountCellOpts): {
  ctrl: ControlledBatch;
  obs: Tau2Obs[];
  before: number;
  after: number;
} {
  const before = opts?.before ?? 0;
  const after = opts?.after ?? 1;
  const start = tau2Graph("one-shot", SERVING_MODEL);
  const dom = opts?.dom ?? new RuntimeDOM();
  if (!opts?.dom) dom.reconcile(start);
  const obs = [completedMiss39Obs(), hung44LicenseObs()];
  const ctrl = controlBatch(obs, {
    loopExhausted: false,
    graph: start,
    before,
    after,
    provider: opts?.provider,
    dom,
    serving: catalogPointer(SERVING_MODEL),
  });
  return { ctrl, obs, before, after };
}

/**
 * One OpenRouter completion AFTER mount with serving SKU = 0813.
 * Records the actual model id from the client / response. Does not fake it.
 */
export async function liveServe0813(opts?: {
  servingSku?: string;
  provider?: Provider;
  allowMock?: boolean;
}): Promise<LiveServeRecord> {
  const sku = opts?.servingSku ?? CATALOG_JUMP_MODEL;
  const cfg = resolveChatConfig();
  if (!cfg && !opts?.allowMock) {
    return {
      ok: false,
      servingModelId: null,
      clientModel: null,
      reason: "no OPENROUTER_API_KEY; live 0813 serve not run; serving id not faked",
    };
  }
  const provider = opts?.provider ?? resolveProvider(sku);
  try {
    const msgs = [{ role: "user" as const, content: LIVE_PING }];
    const turn = provider.completeTurn
      ? await provider.completeTurn(msgs, { model: sku, temperature: 0 })
      : {
          content: await provider.complete(msgs, { model: sku, temperature: 0 }),
          servedModel: provider.model ?? sku,
        };
    const recorded = turn.servedModel ?? provider.model ?? null;
    return {
      ok: true,
      servingModelId: recorded,
      clientModel: provider.model ?? sku,
      contentChars: (turn.content ?? "").length,
    };
  } catch (err) {
    return {
      ok: false,
      servingModelId: null,
      clientModel: provider.model ?? sku,
      reason: err instanceof Error ? err.message : String(err),
    };
  }
}

function gateReason(before: number, after: number): string {
  return (
    `fixtureAfter after=${after} > before=${before}; incompleteFixture; ` +
    "not a τ² lift; not measured 0813 on airline; 0813 existing is not a gate"
  );
}

function readingFor(report: Pick<IskuMountCellReport, "jumped" | "rejected" | "servingModelAfter" | "liveServe">): string {
  const live = report.liveServe;
  if (report.jumped && report.servingModelAfter && servingIdIs0813(report.servingModelAfter)) {
    return (
      "Protocol cell, not a score. Replay of saved live hung-44 traces through the " +
      "I_sku controller with fixture after (not omit-after #12 reject); S for 44 " +
      "rebound to 0813; one live OpenRouter completion recorded serving model id 0813. " +
      "Not invented p_hit(0813). Not a τ² lift. Not a Pro-vs-Flash score."
    );
  }
  const why = live.reason ?? "live 0813 serve did not record serving model id 0813";
  return (
    "Protocol cell, not a score. Replay of saved live hung-44 traces through the " +
    "I_sku controller with fixture after (not omit-after #12 reject); S for 44 " +
    `may be 0813. Live 0813 serve did not jump: ${why}. ` +
    "Not invented p_hit(0813). Not a τ² lift."
  );
}

export function buildIskuMountCellReport(opts: {
  ctrl: ControlledBatch;
  before: number;
  after: number;
  live: LiveServeRecord;
}): IskuMountCellReport {
  const { ctrl, before, after, live } = opts;
  const mounted = ctrl.gate?.action === "mount";
  const recorded = live.ok ? live.servingModelId : null;
  const jumped = Boolean(mounted && live.ok && servingIdIs0813(recorded));
  const rejected = !jumped;
  const report: IskuMountCellReport = {
    benchmark: "tau2-bench",
    kind: "catalog-rebind",
    not_a_sota_result: true,
    closedLoop: false,
    maxRounds: 0,
    domain: "airline",
    taskIds: ["44"],
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
    vsRejectCell: ISKU_REJECT_CELL_FILE,
    jumped,
    mounted,
    rejected,
    gate: {
      action: ctrl.gate?.action ?? "reject",
      before,
      after,
      kind: "fixtureAfter",
      incompleteFixture: true,
      notTau2Lift: true,
      reason: gateReason(before, after),
    },
    servingModelAfter: recorded,
    proposedModel: CATALOG_JUMP_MODEL,
    controllerServing: controllerServingLog(ctrl).text,
    live: true,
    controllerReplay: true,
    liveServe: live,
    nTurns: 9,
    nHungTrials: 1,
    nTurnsAre:
      "nine turns of one hung-44 trial, all stamped hung/timeout; not nine independent hangs",
    sourceEval: SOURCE_EVAL,
    reading: "",
  };
  report.reading = readingFor(report);
  return report;
}

export async function runIskuMountCell(opts?: IskuMountCellOpts): Promise<{
  ctrl: ControlledBatch;
  report: IskuMountCellReport;
}> {
  const { ctrl, before, after } = runIskuMountCellController(opts);
  const sku44 = servingModelForTask(ctrl, "44") ?? SERVING_MODEL;
  let live: LiveServeRecord;
  if (opts?.liveServe) {
    const fallback = opts.mockLive ?? resolveProvider(sku44);
    const provider = opts.dom
      ? servingProviderForTask(ctrl, "44", fallback, opts.dom)
      : fallback;
    live = await liveServe0813({
      servingSku: sku44,
      provider: opts.mockLive ?? (resolveChatConfig() ? provider : undefined),
      allowMock: Boolean(opts.mockLive),
    });
  } else {
    live = {
      ok: false,
      servingModelId: null,
      clientModel: null,
      reason: "liveServe=false; controller-only fixture after; serving id not faked",
    };
  }
  const report = buildIskuMountCellReport({ ctrl, before, after, live });
  return { ctrl, report };
}

export function iskuMountCellEvalPath(repoRoot?: string): string {
  const root =
    repoRoot ??
    join(dirname(fileURLToPath(import.meta.url)), "..", "..");
  return join(root, "eval", "tau2", ISKU_MOUNT_CELL_FILE);
}

export function writeIskuMountCell(report: IskuMountCellReport, path?: string): string {
  const out = path ?? iskuMountCellEvalPath();
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, `${JSON.stringify(report, null, 2)}\n`);
  return out;
}

function parseArgs(argv: string[]): { liveServe: boolean; write: boolean } {
  const liveServe = !argv.includes("--no-live");
  const write = !argv.includes("--no-write");
  return { liveServe, write };
}

async function main(): Promise<void> {
  const { liveServe, write } = parseArgs(process.argv.slice(2));
  const start = tau2Graph("one-shot", SERVING_MODEL);
  const dom = new RuntimeDOM();
  dom.reconcile(start);
  const { ctrl, report } = await runIskuMountCell({ liveServe, dom });
  if (write) {
    const path = writeIskuMountCell(report);
    process.stdout.write(`${JSON.stringify({ wrote: path, report }, null, 2)}\n`);
  } else {
    process.stdout.write(`${JSON.stringify({ report }, null, 2)}\n`);
  }
  const solve = findNode(ctrl.graphC0 ?? start, "solve");
  if (solve?.model && solve.model !== SERVING_MODEL) {
    process.stderr.write("warning: C n.model moved off 0731\n");
  }
  if (report.rejected) process.exitCode = 2;
}

const selfPath = fileURLToPath(import.meta.url);
const invoked = Boolean(
  process.argv[1] &&
    (selfPath === process.argv[1] ||
      selfPath.endsWith(process.argv[1]) ||
      process.argv[1].endsWith("tau2-isku-mount-cell.ts")),
);
if (invoked) {
  main().catch((err) => {
    process.stderr.write(`${err instanceof Error ? err.message : String(err)}\n`);
    process.exit(1);
  });
}
