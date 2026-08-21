/**
 * X_n.S dump after a licensed I_sku write.
 *
 * Reuses the #15 mount-cell controller (hung-44 license + fixture after).
 * Does NOT re-run a live 0813 ping. jumped is the S write, not an OpenRouter call.
 * Not a score. Not invented p_hit(0813). Not a τ² result.
 *
 * Dump serializes X_n.S from the HybridState object. Not ping / get_state S0.
 * Not assembled from servingByTask.
 */
import { writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { RuntimeDOM } from "../reconciler.js";
import { findNode } from "../ir.js";
import { tau2Graph } from "./tau2-graph.js";
import {
  controlBatch,
  type ControlledBatch,
} from "./tau2-control.js";
import {
  completedMiss39Obs,
  ISKU_MOUNT_CELL_FILE,
  ISKU_REJECT_CELL_FILE,
  SOURCE_EVAL,
  runIskuMountCellController,
} from "./tau2-isku-mount-cell.js";
import {
  CATALOG_JUMP_MODEL,
  SERVING_MODEL,
  catalogPointer,
  sameCTopology,
} from "./tau2-weight.js";
import {
  compactC,
  cGraphHash,
  hybridState,
  nModelOf,
  nodeListOf,
  sOnState,
  type HybridState,
} from "./tau2-hybrid-state.js";
import { type CatalogPointer, type Tau2Obs } from "./tau2-types.js";

export const HYBRID_STATE_S_DUMP_FILE = "hybrid-state-s-dump.json";

export const HYBRID_STATE_S_DUMP_READING =
  "X_n.S dump after licensed write; not a score; not a new 0813 serve";

export type HybridStateDumpView = {
  H: HybridState["H"];
  M: HybridState["M"];
  E: { taskId?: string; hung?: boolean; arm?: string; termination?: string };
  C: ReturnType<typeof compactC>;
  S: CatalogPointer;
  S_on_state: true;
};

export type HybridStateSDump = {
  benchmark: "tau2-bench";
  kind: "hybrid-state-s-dump";
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
  liveServe: false;
  notANew0813Serve: true;
  vsRejectCell: typeof ISKU_REJECT_CELL_FILE;
  vsMountCell: typeof ISKU_MOUNT_CELL_FILE;
  jumped: boolean;
  jumpedIs: "S write on X_n, not a new OpenRouter ping";
  mounted: boolean;
  sourceOfTruth: "X_n.S";
  dumpIsNot: "ping / get_state S0";
  notAssembledFromServingByTask: true;
  servingByTaskIs: "derived cache from X.S, not the lookup";
  X_44: HybridStateDumpView;
  X_39: HybridStateDumpView;
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
  hung44LicenseObs: "reconstructed hung=true/timeout fixture citing sourceEval; not a new 0731 timeout";
  trainerIWeight: "unimplemented; off the claim";
  reading: typeof HYBRID_STATE_S_DUMP_READING;
};

function viewOf(X: HybridState): HybridStateDumpView {
  if (!sOnState(X)) {
    throw new Error("dump refused: HybridState has no own S; will not assemble from servingByTask");
  }
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
  };
}

export function runFresh39AfterMount(graphC0: ControlledBatch["graphC0"]): HybridState {
  const start = graphC0 ?? tau2Graph("one-shot", SERVING_MODEL);
  const fresh = controlBatch([completedMiss39Obs()], { graph: start });
  const X = fresh.X["39"] ?? fresh.episodes[0]?.X;
  if (!X) {
    return hybridState({
      E: completedMiss39Obs(),
      C: start,
      S: catalogPointer(SERVING_MODEL),
    });
  }
  return X;
}

export function buildHybridStateSDump(opts?: {
  ctrl?: ControlledBatch;
  obs?: Tau2Obs[];
  graphBeforeHash?: string;
}): { ctrl: ControlledBatch; dump: HybridStateSDump; X_44: HybridState; X_39: HybridState } {
  const start = tau2Graph("one-shot", SERVING_MODEL);
  const hashBefore = opts?.graphBeforeHash ?? cGraphHash(start);
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
  const cAfter = ctrl.graphSku ?? ctrl.graphC0 ?? start;
  const hashAfter = cGraphHash(cAfter);
  const nModel = nModelOf(cAfter);
  const nodesBefore = nodeListOf(start);
  const nodesAfter = nodeListOf(cAfter);
  const mounted = ctrl.gate?.action === "mount";
  // jumped is the S write, not a new OpenRouter ping.
  const jumped = Boolean(mounted && X_44.S.sku === CATALOG_JUMP_MODEL && X_44.S.servingPaused === false);
  const fresh = runFresh39AfterMount(ctrl.graphC0 ?? start);
  const dump: HybridStateSDump = {
    benchmark: "tau2-bench",
    kind: "hybrid-state-s-dump",
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
    liveServe: false,
    notANew0813Serve: true,
    vsRejectCell: ISKU_REJECT_CELL_FILE,
    vsMountCell: ISKU_MOUNT_CELL_FILE,
    jumped,
    jumpedIs: "S write on X_n, not a new OpenRouter ping",
    mounted: Boolean(mounted),
    sourceOfTruth: "X_n.S",
    dumpIsNot: "ping / get_state S0",
    notAssembledFromServingByTask: true,
    servingByTaskIs: "derived cache from X.S, not the lookup",
    X_44: viewOf(X_44),
    X_39: viewOf(X_39),
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
      "reconstructed hung=true/timeout fixture citing sourceEval; not a new 0731 timeout",
    trainerIWeight: "unimplemented; off the claim",
    reading: HYBRID_STATE_S_DUMP_READING,
  };
  return { ctrl, dump, X_44, X_39 };
}

export function hybridStateSDumpPath(repoRoot?: string): string {
  const root =
    repoRoot ??
    join(dirname(fileURLToPath(import.meta.url)), "..", "..");
  return join(root, "eval", "tau2", HYBRID_STATE_S_DUMP_FILE);
}

export function writeHybridStateSDump(dump: HybridStateSDump, path?: string): string {
  const out = path ?? hybridStateSDumpPath();
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, `${JSON.stringify(dump, null, 2)}\n`);
  return out;
}

export function runHybridStateSDump(): { dump: HybridStateSDump; path: string } {
  const { dump } = buildHybridStateSDump();
  const path = writeHybridStateSDump(dump);
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
  const { ctrl, dump } = buildHybridStateSDump();
  if (write) {
    const path = writeHybridStateSDump(dump);
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
      process.argv[1].endsWith("tau2-hybrid-state-s-dump.ts")),
);
if (invoked) {
  main().catch((err) => {
    process.stderr.write(`${err instanceof Error ? err.message : String(err)}\n`);
    process.exit(1);
  });
}
