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
