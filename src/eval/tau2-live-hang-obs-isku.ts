/**
 * Live closed-loop Obs cell: a FRESH 0731 airline episode, then hung-first I_sku.
 *
 * Not a score. Not a dump. Not #12's controllerReplay of saved hung-44.
 * Not #15's mount (fixture after + 0813 serve). Not another X_n dump.
 *
 * If the episode hangs: Obs on THOSE traces chooses I_sku; improveLoop-style
 * I_sku omits after=; gate rejects; serving stays 0731; jumped=false.
 * If it does not hang: write that honestly (freshHang=false, holeOpen=true).
 * Do not stuff a reconstructed hang. Do not invent p_hit(0813).
 *
 * No OPENROUTER_API_KEY → pending JSON. Do not copy old hung files.
 */
import { writeFileSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { observeTau2 } from "./tau2-obs.js";
import { controlBatch } from "./tau2-control.js";
import { applyISku, CATALOG_JUMP_MODEL, SERVING_MODEL } from "./tau2-weight.js";
import { tau2Graph } from "./tau2-graph.js";
import { type Tau2Obs } from "./tau2-types.js";

export const LIVE_HANG_OBS_ISKU_FILE = "improve-live-0731-hang-obs-isku.json";
export const LIVE_HANG_OBS_ISKU_R6_FILE = "improve-live-0731-hang-obs-isku-r6.json";
export const LIVE_HANG_OBS_ISKU_39_FILE = "improve-live-0731-hang-obs-isku-39.json";
export const LIVE_HANG_OBS_ISKU_TASK_DEFAULT = "44";

export const FORBIDDEN_HANG_SOURCES = [
  "improve-live-0731-iweight-44-hung.json",
  "improve-live-0731-self-3944-postgate.json",
  "airline-live-self-3944-postgate-r0.json",
] as const;

export const LIVE_HANG_OBS_ISKU_READING =
  "live Obs of this episode; not a controller replay of saved hung-44; " +
  "not a score; not a dump; not live hung-44 then served as a mount.";

export const GATE_OMIT_AFTER_REASON =
  "no measured after-eval; 0813 existing is not a gate";

export type LiveHangObsArm = "I_sku" | "I_loop" | "wait" | null;

export type LiveHangObsIskuGate = {
  action: "reject" | "mount" | null;
  after: null;
  before?: number;
  reason: string;
};

export type LiveHangObsIskuReport = {
  benchmark: "tau2-bench";
  kind: "live-closed-loop-obs";
  not_a_sota_result: true;
  closedLoop: boolean;
  domain: "airline";
  taskIds: string[];
  numTrials: 1;
  model: typeof SERVING_MODEL;
  live: true;
  controllerReplay: false;
  freshHang: boolean;
  hung: boolean;
  holeOpen: boolean;
  pendingKey: boolean;
  /** Slice / I_sku arm when present. Null / omitted when I_sku is not licensed. */
  arm?: LiveHangObsArm;
  obs: {
    arm: LiveHangObsArm;
    hung: boolean;
    taskId: string | null;
    termination: string | null;
    nSuccessProxy: number | null;
  };
  applyScope: { waitKept: string[]; looped: string[]; weighted: string[] };
  omitAfter: true;
  iSkuRequestOmitsAfter: true;
  iSkuRequest: { op: "i_sku"; before: number } | null;
  jumped: boolean;
  servingPaused: false;
  servingModelAfter: typeof SERVING_MODEL | null;
  proposedModel: typeof CATALOG_JUMP_MODEL;
  pHit0813: null;
  notInventedPHit0813: true;
  trained: false;
  trainerOff: true;
  liveAirlineImproveLoopOmitsAfter: true;
  gate: LiveHangObsIskuGate;
  sourceEval: string[];
  sourceEvalIs: "this run";
  vsRejectCell: "improve-live-0731-isku-44-reject.json";
  reading: string;
};

export type LiveHangEpisodeInput = {
  taskId: string;
  hung: boolean;
  termination?: string | null;
  reward?: number | null;
  nSuccessProxy?: number | null;
  arm?: Exclude<LiveHangObsArm, null>;
  lastActions?: string[];
  evalFile?: string;
};

function blobOf(value: unknown): string {
  return JSON.stringify(value);
}

function hasForbiddenHangSource(value: unknown): string | null {
  const blob = blobOf(value);
  for (const name of FORBIDDEN_HANG_SOURCES) {
    if (blob.includes(name)) return name;
  }
  return null;
}

export function assertLiveHangObsIskuCell(report: unknown): asserts report is LiveHangObsIskuReport {
  if (!report || typeof report !== "object") {
    throw new Error("live hang-obs-isku cell refused: report is not an object");
  }
  const r = report as Record<string, unknown>;
  const forbidden = hasForbiddenHangSource(r);
  if (forbidden) {
    throw new Error(
      `live hang-obs-isku cell refused: sourceEval-of-old-hung (${forbidden}); ` +
        "this cell is live Obs of THIS episode, not saved hung-44 / postgate",
    );
  }
  if (r.controllerReplay === true) {
    throw new Error(
      "live hang-obs-isku cell refused: controllerReplay=true; #12 is the replay; this cell is not",
    );
  }
  if (r.pHit0813 !== null && r.pHit0813 !== undefined) {
    throw new Error("live hang-obs-isku cell refused: pHit0813 set; do not invent p_hit(0813)");
  }
  if ("after" in r && r.after !== undefined && r.after !== null) {
    throw new Error("live hang-obs-isku cell refused: after= present; this cell omits after=");
  }
  const gate = r.gate as Record<string, unknown> | undefined;
  if (gate && "after" in gate && gate.after !== undefined && gate.after !== null) {
    throw new Error("live hang-obs-isku cell refused: gate.after present; omit after=");
  }
  if (r.omitAfter !== true) {
    throw new Error("live hang-obs-isku cell refused: omitAfter must be true");
  }
  const req = r.iSkuRequest as Record<string, unknown> | null | undefined;
  if (req && "after" in req && req.after !== undefined) {
    throw new Error("live hang-obs-isku cell refused: I_sku request has after=");
  }
  if (r.iSkuRequestOmitsAfter !== true) {
    throw new Error("live hang-obs-isku cell refused: iSkuRequestOmitsAfter must be true");
  }
  const reading = String(r.reading ?? "");
  for (const needle of [
    "live Obs of this episode",
    "not a controller replay of saved hung-44",
    "not a score",
    "not a dump",
    "not live hung-44 then served as a mount",
  ]) {
    if (!reading.includes(needle)) {
      throw new Error(`live hang-obs-isku cell refused: reading missing “${needle}”`);
    }
  }
  if (reading.includes("not a new timeout")) {
    throw new Error(
      "live hang-obs-isku cell refused: reading says “not a new timeout” (#12 replay smear)",
    );
  }
  if (r.servingPaused !== false) {
    throw new Error("live hang-obs-isku cell refused: servingPaused must be false");
  }
  if (r.trained === true || r.trainerOff === false) {
    throw new Error("live hang-obs-isku cell refused: trainer must stay off");
  }
}

function readingFor(opts: { pendingKey: boolean; freshHang: boolean; hung: boolean }): string {
  const base = LIVE_HANG_OBS_ISKU_READING;
  if (opts.pendingKey) {
    return (
      "live Obs of this episode pending OPENROUTER_API_KEY; " +
      "not a controller replay of saved hung-44; not a score; not a dump; " +
      "not live hung-44 then served as a mount. Live JSON is pending a key. " +
      "No hang was faked."
    );
  }
  if (!opts.freshHang || !opts.hung) {
    return (
      "live Obs of this episode; episode did not hang (freshHang=false); " +
      "hole remains open; not a controller replay of saved hung-44; " +
      "not a score; not a dump; not live hung-44 then served as a mount."
    );
  }
  return (
    "live Obs of this episode; hung-first Obs chose I_sku; omit after; " +
    "gate rejected; serving stayed 0731; not a controller replay of saved hung-44; " +
    "not a score; not a dump; not live hung-44 then served as a mount."
  );
}

export function pendingLiveHangObsIskuReport(taskId = LIVE_HANG_OBS_ISKU_TASK_DEFAULT): LiveHangObsIskuReport {
  const report: LiveHangObsIskuReport = {
    benchmark: "tau2-bench",
    kind: "live-closed-loop-obs",
    not_a_sota_result: true,
    closedLoop: false,
    domain: "airline",
    taskIds: [taskId],
    numTrials: 1,
    model: SERVING_MODEL,
    live: true,
    controllerReplay: false,
    freshHang: false,
    hung: false,
    holeOpen: true,
    pendingKey: true,
    arm: null,
    obs: {
      arm: null,
      hung: false,
      taskId: null,
      termination: null,
      nSuccessProxy: null,
    },
    applyScope: { waitKept: [], looped: [], weighted: [] },
    omitAfter: true,
    iSkuRequestOmitsAfter: true,
    iSkuRequest: null,
    jumped: false,
    servingPaused: false,
    servingModelAfter: SERVING_MODEL,
    proposedModel: CATALOG_JUMP_MODEL,
    pHit0813: null,
    notInventedPHit0813: true,
    trained: false,
    trainerOff: true,
    liveAirlineImproveLoopOmitsAfter: true,
    gate: {
      action: null,
      after: null,
      reason: "live episode pending OPENROUTER_API_KEY; I_sku not fired; after omitted",
    },
    sourceEval: [liveHangObsIskuFilename(taskId)],
    sourceEvalIs: "this run",
    vsRejectCell: "improve-live-0731-isku-44-reject.json",
    reading: readingFor({ pendingKey: true, freshHang: false, hung: false }),
  };
  assertLiveHangObsIskuCell(report);
  return report;
}

/**
 * Build the cell from THIS episode. Refuses old hung files, after=, pHit0813,
 * and controllerReplay. Does not reconstruct a hang when the episode completed.
 */
export function buildLiveHangObsIskuReport(opts: {
  episode: LiveHangEpisodeInput;
  pendingKey?: boolean;
  /** Measured I_sku sidecar/gate after omit-after. Tests may pass the reject. */
  gate?: LiveHangObsIskuGate;
  applyScope?: { waitKept: string[]; looped: string[]; weighted: string[] };
  sourceEval?: string[];
  controllerReplay?: boolean;
  pHit0813?: null | number;
  after?: number;
}): LiveHangObsIskuReport {
  if (opts.pendingKey) return pendingLiveHangObsIskuReport(opts.episode.taskId ?? LIVE_HANG_OBS_ISKU_TASK_DEFAULT);
  if (opts.controllerReplay) {
    throw new Error(
      "live hang-obs-isku cell refused: controllerReplay=true; #12 is the replay; this cell is not",
    );
  }
  if (opts.pHit0813 !== undefined && opts.pHit0813 !== null) {
    throw new Error("live hang-obs-isku cell refused: pHit0813 set; do not invent p_hit(0813)");
  }
  if (opts.after !== undefined) {
    throw new Error("live hang-obs-isku cell refused: after= present; this cell omits after=");
  }
  const sourceEval =
    opts.sourceEval ?? [opts.episode.evalFile ?? liveHangObsIskuFilename(opts.episode.taskId)];
  const forbidden = hasForbiddenHangSource(sourceEval);
  if (forbidden) {
    throw new Error(
      `live hang-obs-isku cell refused: sourceEval-of-old-hung (${forbidden}); ` +
        "this cell is live Obs of THIS episode, not saved hung-44 / postgate",
    );
  }

  const hung = Boolean(opts.episode.hung);
  const freshHang = hung;
  const holeOpen = !freshHang;
  const arm: LiveHangObsArm =
    opts.episode.arm ??
    (hung ? "I_sku" : opts.episode.nSuccessProxy === 1 ? "wait" : "I_loop");
  const applyScope =
    opts.applyScope ??
    (hung
      ? { waitKept: [], looped: [], weighted: [opts.episode.taskId] }
      : arm === "wait"
        ? { waitKept: [opts.episode.taskId], looped: [], weighted: [] }
        : { waitKept: [], looped: [opts.episode.taskId], weighted: [] });

  const iSkuFired = hung && arm === "I_sku";
  const gate: LiveHangObsIskuGate =
    opts.gate ??
    (iSkuFired
      ? {
          action: "reject",
          after: null,
          before: 0,
          reason: GATE_OMIT_AFTER_REASON,
        }
      : {
          action: null,
          after: null,
          reason:
            "episode completed; I_sku not licensed; after omitted; hole remains open",
        });

  if (iSkuFired && (arm !== "I_sku" || !applyScope.weighted.includes(opts.episode.taskId))) {
    throw new Error("live hang-obs-isku cell refused: hung episode must choose I_sku and weight that task");
  }

  const report: LiveHangObsIskuReport = {
    benchmark: "tau2-bench",
    kind: "live-closed-loop-obs",
    not_a_sota_result: true,
    closedLoop: iSkuFired,
    domain: "airline",
    taskIds: [opts.episode.taskId],
    numTrials: 1,
    model: SERVING_MODEL,
    live: true,
    controllerReplay: false,
    freshHang,
    hung,
    holeOpen,
    pendingKey: false,
    arm: iSkuFired ? "I_sku" : null,
    obs: {
      arm,
      hung,
      taskId: opts.episode.taskId,
      termination: opts.episode.termination ?? (hung ? "timeout" : null),
      nSuccessProxy: hung ? 0 : (opts.episode.nSuccessProxy ?? null),
    },
    applyScope,
    omitAfter: true,
    iSkuRequestOmitsAfter: true,
    iSkuRequest: iSkuFired ? { op: "i_sku", before: 0 } : null,
    jumped: false,
    servingPaused: false,
    servingModelAfter: SERVING_MODEL,
    proposedModel: CATALOG_JUMP_MODEL,
    pHit0813: null,
    notInventedPHit0813: true,
    trained: false,
    trainerOff: true,
    liveAirlineImproveLoopOmitsAfter: true,
    gate,
    sourceEval,
    sourceEvalIs: "this run",
    vsRejectCell: "improve-live-0731-isku-44-reject.json",
    reading: readingFor({ pendingKey: false, freshHang, hung }),
  };
  assertLiveHangObsIskuCell(report);
  return report;
}

/** Hung-first controller on THIS episode's Obs, I_sku with after omitted. */
export function runLiveHangObsIskuController(obs: Tau2Obs): {
  obs: Tau2Obs;
  applyScope: { waitKept: string[]; looped: string[]; weighted: string[] };
  gate: LiveHangObsIskuGate;
  iSkuFired: boolean;
} {
  const start = tau2Graph("one-shot", SERVING_MODEL);
  const ctrl = controlBatch([obs], { graph: start, before: 0 });
  const iSkuFired = ctrl.applied.includes("I_sku");
  let gate: LiveHangObsIskuGate;
  if (iSkuFired) {
    const decided = applyISku({ graph: start, before: 0 });
    gate = {
      action: decided.action,
      after: null,
      before: decided.before,
      reason: decided.reason,
    };
    if (decided.after !== null) {
      throw new Error("live hang-obs-isku cell refused: after= present; this cell omits after=");
    }
    if (decided.action !== "reject" || decided.jumped) {
      throw new Error("live hang-obs-isku cell refused: omit-after must reject; serving stays 0731");
    }
  } else {
    gate = {
      action: null,
      after: null,
      reason: "episode completed; I_sku not licensed; after omitted; hole remains open",
    };
  }
  return {
    obs,
    applyScope: {
      waitKept: [...ctrl.applyScope.waitKept],
      looped: [...ctrl.applyScope.looped],
      weighted: [...ctrl.applyScope.weighted],
    },
    gate,
    iSkuFired,
  };
}

/** Obs of THIS hung episode. Not hung44LicenseObs / sourceEval replay. */
export function thisEpisodeHungObs(taskId = LIVE_HANG_OBS_ISKU_TASK_DEFAULT): Tau2Obs {
  return observeTau2({
    traces: [],
    taskId,
    reward: null,
    hung: true,
    termination: "timeout",
    actions: [],
  });
}

export function liveHangObsIskuFilename(taskId = LIVE_HANG_OBS_ISKU_TASK_DEFAULT): string {
  const tid = String(taskId);
  return tid === LIVE_HANG_OBS_ISKU_TASK_DEFAULT
    ? LIVE_HANG_OBS_ISKU_FILE
    : `improve-live-0731-hang-obs-isku-${tid}.json`;
}

export function liveHangObsIskuEvalPath(repoRoot?: string, taskId = LIVE_HANG_OBS_ISKU_TASK_DEFAULT): string {
  const root =
    repoRoot ??
    join(dirname(fileURLToPath(import.meta.url)), "..", "..");
  return join(root, "eval", "tau2", liveHangObsIskuFilename(taskId));
}

export function liveHangObsIskuR6EvalPath(repoRoot?: string): string {
  const root =
    repoRoot ??
    join(dirname(fileURLToPath(import.meta.url)), "..", "..");
  return join(root, "eval", "tau2", LIVE_HANG_OBS_ISKU_R6_FILE);
}

export function liveHangObsIsku39EvalPath(repoRoot?: string): string {
  return liveHangObsIskuEvalPath(repoRoot, "39");
}

export function writeLiveHangObsIsku(report: LiveHangObsIskuReport, path?: string): string {
  assertLiveHangObsIskuCell(report);
  const taskId = report.taskIds[0] ?? LIVE_HANG_OBS_ISKU_TASK_DEFAULT;
  const out = path ?? liveHangObsIskuEvalPath(undefined, taskId);
  const base = out.split(/[\\/]/).pop() ?? "";
  if (
    (base === LIVE_HANG_OBS_ISKU_FILE || base === LIVE_HANG_OBS_ISKU_R6_FILE) &&
    taskId !== LIVE_HANG_OBS_ISKU_TASK_DEFAULT
  ) {
    throw new Error(
      "live hang-obs-isku refused to overwrite the 44 / r6 packets; " +
        `TASK_ID=${taskId} writes ${liveHangObsIskuFilename(taskId)}`,
    );
  }
  if (base === LIVE_HANG_OBS_ISKU_R6_FILE) {
    throw new Error(
      "live hang-obs-isku refused to overwrite r6; " +
        "task 44 still writes the historical improve-live-0731-hang-obs-isku.json",
    );
  }
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, `${JSON.stringify(report, null, 2)}\n`);
  return out;
}

export function readLiveHangObsIsku(path?: string): LiveHangObsIskuReport {
  const p = path ?? liveHangObsIskuEvalPath();
  const report = JSON.parse(readFileSync(p, "utf8"));
  assertLiveHangObsIskuCell(report);
  return report;
}

export function parseLiveHangObsIskuArgs(argv: string[]): {
  pendingKey: boolean;
  write: boolean;
  taskId: string;
  out?: string;
} {
  const pendingKey = argv.includes("--pending-key") || !process.env.OPENROUTER_API_KEY;
  const write = !argv.includes("--no-write");
  let taskId = LIVE_HANG_OBS_ISKU_TASK_DEFAULT;
  let out: string | undefined;
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]!;
    if (arg === "--task-id" || arg === "--live-hang-obs-isku") {
      const next = argv[i + 1];
      if (next && !next.startsWith("-")) {
        taskId = next;
        i += 1;
      }
      continue;
    }
    if (arg === "--out" || arg === "--live-hang-obs-isku-out") {
      const next = argv[i + 1];
      if (next && !next.startsWith("-")) {
        out = next;
        i += 1;
      }
      continue;
    }
    if (arg === "--pending-key" || arg === "--no-write") continue;
    if (!arg.startsWith("-")) {
      taskId = arg;
    }
  }
  return { pendingKey, write, taskId, out };
}

function main(): void {
  const { pendingKey, write, taskId, out } = parseLiveHangObsIskuArgs(process.argv.slice(2));
  const report = pendingKey
    ? pendingLiveHangObsIskuReport(taskId)
    : (() => {
        throw new Error(
          "live hang-obs-isku TS CLI does not run the airline episode; " +
            "use PYTHONPATH=python python3 -m tau2_vdom.improve --live-hang-obs-isku [TASK_ID]",
        );
      })();
  if (write) {
    const path = writeLiveHangObsIsku(report, out);
    process.stdout.write(`${JSON.stringify({ wrote: path, report }, null, 2)}\n`);
    return;
  }
  process.stdout.write(`${JSON.stringify({ report }, null, 2)}\n`);
}

const selfPath = fileURLToPath(import.meta.url);
const invoked = Boolean(
  process.argv[1] &&
    (selfPath === process.argv[1] ||
      selfPath.endsWith(process.argv[1]) ||
      process.argv[1].endsWith("tau2-live-hang-obs-isku.ts")),
);
if (invoked) {
  try {
    main();
  } catch (err) {
    process.stderr.write(`${err instanceof Error ? err.message : String(err)}\n`);
    process.exit(1);
  }
}
