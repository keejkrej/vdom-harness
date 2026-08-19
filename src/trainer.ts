import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { type Trace, type AgentNode, node } from "./ir.js";
import {
  registerProvider,
  reverseEachWord,
  type Provider,
  type Message,
  type CompleteOpts,
} from "./providers.js";

/**
 * Weight / adapter artifact produced by an out-of-process (or injectable) trainer.
 * Real HF Jobs / LoRA can fill `uri` and leave runtime binding to registerProvider.
 */
export type AdapterArtifact = {
  id: string;
  baseModel: string;
  technique: string;
  /** Opaque handle: file path, hf:// URI, job id, etc. */
  uri?: string;
  /** Model id to bind via AgentNode.model after a successful gate. */
  resultModelId: string;
  meta?: Record<string, unknown>;
};

export type TrainOpts = {
  baseModel: string;
  technique?: string;
  /** Override the model id registered for chat binding after train. */
  resultModelId?: string;
};

export type Trainer = {
  train(traces: Trace[], opts: TrainOpts): Promise<AdapterArtifact>;
};

/**
 * Two-clock I_weight:
 * - Fast clock: serving continues on old θ. `servingPaused` is always false.
 * - Slow clock: an async trainer consumes traces of incomplete episodes
 *   (transfer / hung / crash / reward 0 with early transfer).
 * - Gate: mount only if a held-out eval does not regress; else keep old θ.
 *
 * 0731 via OpenRouter cannot take an adapter. A successful mount here is a
 * **surrogate θ** (prompt-prefix / logit-bias stand-in) or a FakeTrainer
 * protocol stub — never a claimed 0731 LoRA.
 */
export type TrainJobStatus = "running" | "done" | "failed";
export type TrainerKind = "fake" | "surrogate";
export type IncompleteReason = "transfer" | "hung" | "crash" | "reward0-early-transfer";

export type TrainExample = Trace & {
  taskId?: string;
  trial?: number;
  reward?: number | null;
  hung?: boolean;
  termination?: string | null;
  reason?: IncompleteReason | string;
};

export type TrainJobGate = {
  arm: "I_weight";
  action: "mount" | "reject";
  before: number;
  after: number;
  reason: string;
};

export type TrainJob = {
  id: string;
  status: TrainJobStatus;
  tracesUsed: TrainExample[];
  artifact?: AdapterArtifact;
  artifactPointer?: string;
  gate?: TrainJobGate;
  servingPaused: false;
  trainerKind: TrainerKind;
  baseModel: string;
  /** File-backed prompt-prefix / logit-bias stand-in — not API model weights. */
  surrogate: boolean;
  not0731Weights: true;
  error?: string;
  createdAt: number;
  updatedAt: number;
};

export const SURROGATE_NOTE =
  "Surrogate θ (prompt-prefix / logit-bias stand-in). Not 0731 weights. " +
  "OpenRouter deepseek/deepseek-v4-flash-0731 cannot take an adapter.";

export const FROZEN_API_MODEL = "deepseek/deepseek-v4-flash-0731";

const jobs = new Map<string, TrainJob>();
let jobSeq = 0;

export function isFrozenApiModel(model: string): boolean {
  return /deepseek-v4-flash-0731|deepseek\/deepseek-v4/i.test(model);
}

export function asTrainExample(raw: unknown): TrainExample {
  const t = (raw ?? {}) as Partial<TrainExample>;
  return {
    nodeKey: t.nodeKey ?? "solve",
    role: t.role ?? "solve",
    input: t.input ?? (typeof raw === "string" ? raw : JSON.stringify(raw ?? {})),
    output: t.output ?? "",
    ts: typeof t.ts === "number" ? t.ts : Date.now(),
    taskId: t.taskId,
    trial: t.trial,
    reward: t.reward,
    hung: t.hung,
    termination: t.termination,
    reason: t.reason,
  };
}

export function getTrainJob(id: string): TrainJob | undefined {
  return jobs.get(id);
}

export function listTrainJobs(): TrainJob[] {
  return [...jobs.values()];
}

export function activeTrainJob(): TrainJob | undefined {
  const all = listTrainJobs();
  return all[all.length - 1];
}

export function clearTrainJobs(): void {
  jobs.clear();
  jobSeq = 0;
}

export function persistTrainJob(job: TrainJob, root = process.cwd()): string {
  const dir = join(root, "eval", "tau2");
  mkdirSync(dir, { recursive: true });
  const path = join(dir, "latest-iweight-job.json");
  writeFileSync(path, `${JSON.stringify(job, null, 2)}\n`);
  return path;
}

export function recordTrainJobGate(id: string, gate: TrainJobGate, persist = true): TrainJob | undefined {
  const job = jobs.get(id);
  if (!job) return undefined;
  job.gate = gate;
  job.updatedAt = Date.now();
  if (persist) persistTrainJob(job);
  return job;
}

/**
 * Slow-clock spawn. Returns immediately with status=running.
 * Serving is never paused. Callers poll getTrainJob / i_weight_status.
 */
export function spawnTrainJob(opts: {
  trainer: Trainer;
  traces: unknown[];
  trainOpts: TrainOpts;
  trainerKind?: TrainerKind;
  persist?: boolean;
}): TrainJob {
  jobSeq += 1;
  const tracesUsed = opts.traces.map(asTrainExample);
  const trainerKind = opts.trainerKind ?? trainerKindOf(opts.trainer);
  const persist = opts.persist !== false;
  const job: TrainJob = {
    id: `train-job-${jobSeq}`,
    status: "running",
    tracesUsed,
    servingPaused: false,
    trainerKind,
    baseModel: opts.trainOpts.baseModel,
    surrogate: trainerKind === "surrogate",
    not0731Weights: true,
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
  jobs.set(job.id, job);
  if (persist) persistTrainJob(job);

  void opts.trainer
    .train(tracesUsed, opts.trainOpts)
    .then((art) => {
      const live = jobs.get(job.id);
      if (!live) return;
      live.status = "done";
      live.artifact = art;
      live.artifactPointer = art.uri ?? art.id;
      live.updatedAt = Date.now();
      if (persist) persistTrainJob(live);
    })
    .catch((err: unknown) => {
      const live = jobs.get(job.id);
      if (!live) return;
      live.status = "failed";
      live.error = err instanceof Error ? err.message : String(err);
      live.updatedAt = Date.now();
      if (persist) persistTrainJob(live);
    });

  return job;
}

export async function waitTrainJob(id: string, timeoutMs = 5000): Promise<TrainJob> {
  const start = Date.now();
  for (;;) {
    const job = jobs.get(id);
    if (!job) throw new Error(`waitTrainJob: missing ${id}`);
    if (job.status !== "running") return job;
    if (Date.now() - start > timeoutMs) {
      throw new Error(`waitTrainJob: timeout ${id}`);
    }
    await new Promise((r) => setTimeout(r, 5));
  }
}

export function trainerKindOf(trainer: Trainer): TrainerKind {
  return trainer.constructor?.name === "SurrogateTrainer" ? "surrogate" : "fake";
}

/** Local held-out for the slow clock. Surrogate cannot complete incomplete episodes. */
export function localHeldOutScore(job: TrainJob): number {
  if (job.status !== "done" || !job.artifact) return 0;
  if (job.trainerKind === "fake" && job.artifact.meta?.stub && !job.artifact.meta?.failing) {
    return 1;
  }
  return 0;
}

const artifacts = new Map<string, AdapterArtifact>();

export function registerArtifact(artifact: AdapterArtifact): void {
  artifacts.set(artifact.id, artifact);
}

export function getArtifact(id: string): AdapterArtifact | undefined {
  return artifacts.get(id);
}

export function clearArtifactRegistry(): void {
  artifacts.clear();
}

/** Provider that always applies the word-reverse lesson (simulates a trained adapter). */
export class AdaptedProvider implements Provider {
  name: string;
  model: string;

  constructor(model: string) {
    this.model = model;
    this.name = `adapted:${model}`;
  }

  async complete(msgs: Message[], opts?: CompleteOpts): Promise<string> {
    const role = (opts?.role ?? "").toLowerCase();
    const text = msgs.map((m) => m.content).join("\n");
    const inputLine = text.match(/Input:\s*(.+)/i);
    const input = inputLine?.[1]?.trim().split("\n")[0]?.trim() ?? "";

    if (
      role === "solve" ||
      role === "actor" ||
      role === "one-shot" ||
      role === "oneshot" ||
      role === "generator" ||
      role === "actor-retry" ||
      role === ""
    ) {
      return reverseEachWord(input);
    }
    return reverseEachWord(input);
  }
}

/**
 * Test / demo trainer: no GPU, no HF Jobs.
 * Emits an AdapterArtifact and registers an AdaptedProvider under resultModelId.
 */
export class FakeTrainer implements Trainer {
  private seq = 0;

  async train(_traces: Trace[], opts: TrainOpts): Promise<AdapterArtifact> {
    this.seq += 1;
    const technique = opts.technique ?? "fake-lora";
    const id = `adapter-${opts.baseModel}-${technique}-${this.seq}`;
    const resultModelId = opts.resultModelId ?? `adapted:${id}`;
    const artifact: AdapterArtifact = {
      id,
      baseModel: opts.baseModel,
      technique,
      uri: `fake://${id}`,
      resultModelId,
      meta: { stub: true },
    };
    registerArtifact(artifact);
    registerProvider(resultModelId, new AdaptedProvider(resultModelId));
    return artifact;
  }
}

/**
 * Trainer that registers a still-failing provider — used to exercise eval reject / rollback.
 */
export class FailingTrainer implements Trainer {
  private seq = 0;

  async train(_traces: Trace[], opts: TrainOpts): Promise<AdapterArtifact> {
    this.seq += 1;
    const technique = opts.technique ?? "fail-lora";
    const id = `adapter-fail-${opts.baseModel}-${technique}-${this.seq}`;
    const resultModelId = opts.resultModelId ?? `adapted-fail:${id}`;
    const artifact: AdapterArtifact = {
      id,
      baseModel: opts.baseModel,
      technique,
      uri: `fake://fail/${id}`,
      resultModelId,
      meta: { stub: true, failing: true },
    };
    registerArtifact(artifact);
    // Keep the naive whole-string reverse behavior via DeterministicProvider semantics:
    // register a provider that returns the wrong answer for solve.
    registerProvider(resultModelId, {
      name: `failing:${resultModelId}`,
      model: resultModelId,
      async complete(msgs: Message[]): Promise<string> {
        const text = msgs.map((m) => m.content).join("\n");
        const inputLine = text.match(/Input:\s*(.+)/i);
        const input = inputLine?.[1]?.trim().split("\n")[0]?.trim() ?? "";
        return [...input].reverse().join("");
      },
    });
    return artifact;
  }
}

/**
 * File-backed surrogate θ. Prepends a prompt prefix and records a logit-bias
 * stand-in. This is **not** a 0731 finetune and does not raise API-model p_hit.
 */
export class SurrogatePrefixProvider implements Provider {
  name: string;
  model: string;
  prefix: string;

  constructor(model: string, prefix: string) {
    this.model = model;
    this.name = `surrogate:${model}`;
    this.prefix = prefix;
  }

  async complete(msgs: Message[], _opts?: CompleteOpts): Promise<string> {
    const text = [this.prefix, ...msgs.map((m) => m.content)].join("\n");
    const inputLine = text.match(/Input:\s*(.+)/i);
    const input = inputLine?.[1]?.trim().split("\n")[0]?.trim() ?? "";
    return [...input].reverse().join("");
  }
}

export class SurrogateTrainer implements Trainer {
  private seq = 0;

  async train(traces: Trace[], opts: TrainOpts): Promise<AdapterArtifact> {
    this.seq += 1;
    const safeBase = opts.baseModel.replace(/[^a-zA-Z0-9._-]+/g, "-");
    const id = `surrogate-${safeBase}-${this.seq}`;
    const resultModelId = opts.resultModelId ?? `surrogate:${id}`;
    const prefix = deriveSurrogatePrefix(traces);
    const dir = join(process.cwd(), "eval", "tau2");
    mkdirSync(dir, { recursive: true });
    const filePath = join(dir, `surrogate-theta-${id}.json`);
    const payload = {
      kind: "surrogate-theta",
      not0731Weights: true,
      baseModel: opts.baseModel,
      frozenApiModel: FROZEN_API_MODEL,
      apiFrozen: isFrozenApiModel(opts.baseModel),
      promptPrefix: prefix,
      logitBiasStandIn: { transfer_to_human_agents: -1 },
      tracesUsed: traces.length,
      note: SURROGATE_NOTE,
    };
    writeFileSync(filePath, `${JSON.stringify(payload, null, 2)}\n`);
    const artifact: AdapterArtifact = {
      id,
      baseModel: opts.baseModel,
      technique: opts.technique ?? "surrogate-prefix",
      uri: `file://${filePath}`,
      resultModelId,
      meta: {
        surrogate: true,
        not0731Weights: true,
        stub: false,
        note: SURROGATE_NOTE,
        tracesUsed: traces.length,
      },
    };
    registerArtifact(artifact);
    registerProvider(resultModelId, new SurrogatePrefixProvider(resultModelId, prefix));
    return artifact;
  }
}

export function deriveSurrogatePrefix(traces: Trace[]): string {
  const reasons = traces
    .map((t) => ("reason" in t ? String((t as TrainExample).reason ?? "") : ""))
    .filter(Boolean);
  const unique = [...new Set(reasons)];
  return [
    SURROGATE_NOTE,
    unique.length ? `incomplete reasons: ${unique.join(", ")}` : "incomplete episode traces",
    "Do not transfer or hang; finish the episode on the current tools.",
  ].join("\n");
}

/** Build a proposed adapter node referencing a trained artifact. */
export function proposeAdapter(opts: {
  key: string;
  artifact: AdapterArtifact;
  objective?: string;
}): AgentNode {
  return node({
    key: opts.key,
    kind: "adapter",
    role: "adapter",
    objective: opts.objective ?? `Adapter ${opts.artifact.technique} on ${opts.artifact.baseModel}`,
    status: "proposed",
    adapterRef: opts.artifact.id,
    modelRef: opts.artifact.baseModel,
    artifactRef: opts.artifact.uri ?? opts.artifact.id,
    technique: opts.artifact.technique,
  });
}

/**
 * Documented extension point for real weight updates.
 * Wire Hugging Face Jobs / TRL / peft LoRA here — keep training out-of-process.
 *
 * Example sketch (not executed by this repo):
 *   hf jobs run … trl sft --model $BASE --dataset $TRACES --output-dir $OUT
 *   then registerProvider(resultModelId, openAiCompatiblePointingAtServingEndpoint)
 */
export type HfJobsTrainerConfig = {
  baseModel: string;
  technique?: "lora" | "qlora" | "sft";
  /** Command or job template the host runs out-of-process. */
  jobCommand?: string;
};

export function describeHfJobsExtension(cfg: HfJobsTrainerConfig): string {
  return [
    "HF Jobs / TRL extension point (not implemented in-process):",
    `  baseModel=${cfg.baseModel}`,
    `  technique=${cfg.technique ?? "lora"}`,
    "  1. Export traces to a dataset",
    "  2. Launch out-of-process train (hf jobs / skypilot / local GPU)",
    "  3. Register AdapterArtifact { id, uri, resultModelId }",
    "  4. registerProvider(resultModelId, servingClient)",
    "  5. Gate via lifecycle.gateAdapter before updating AgentNode.model",
    cfg.jobCommand ? `  jobCommand=${cfg.jobCommand}` : "  jobCommand=<your train launcher>",
  ].join("\n");
}
