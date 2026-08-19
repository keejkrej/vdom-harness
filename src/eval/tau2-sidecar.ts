/**
 * Stdio JSONL sidecar. Python HalfDuplexAgent sends one turn per line.
 * Logs go to stderr so stdout stays machine-readable.
 *
 * Serving does not pause on I_loop / I_weight: set_technique and i_loop
 * mutate in-process state; i_weight_spawn returns immediately.
 */
import { createProvider, DeterministicProvider, type Message, type ToolSpec } from "../providers.js";
import { runTau2Turn } from "./tau2-turn.js";
import { type AgentGraph } from "../ir.js";
import { FakeTrainer } from "../trainer.js";
import { type Tau2Obs, type Tau2Technique, type Tau2TurnResponse } from "./tau2-types.js";
import { applyILoop, gateWeightMount, type GraphDiffOp } from "./tau2-improve.js";
import { tau2Graph } from "./tau2-graph.js";
import { createInterface } from "node:readline";

type Incoming = {
  op?: string;
  id?: string;
  policy?: string;
  tools?: ToolSpec[];
  messages?: Message[];
  technique?: Tau2Technique;
  model?: string;
  graph?: AgentGraph;
  before?: number;
  after?: number;
  obs?: Tau2Obs | Tau2Obs[];
};

type SidecarReply = Tau2TurnResponse & {
  technique?: Tau2Technique;
  graph?: AgentGraph;
  graphDiff?: GraphDiffOp[];
  servingPaused?: boolean;
  spawned?: boolean;
  done?: boolean;
  gate?: ReturnType<typeof gateWeightMount>;
};

let currentTechnique: Tau2Technique = "one-shot";
let currentGraph: AgentGraph = tau2Graph("one-shot");
let weightJob: { spawned: boolean; done: boolean; artifactId?: string } = {
  spawned: false,
  done: false,
};

function write(obj: SidecarReply): void {
  process.stdout.write(`${JSON.stringify(obj)}\n`);
}

function setTechnique(next: Tau2Technique, graph?: AgentGraph): void {
  currentTechnique = next;
  currentGraph = graph ?? tau2Graph(next);
}

async function handle(line: string): Promise<void> {
  let req: Incoming;
  try {
    req = JSON.parse(line) as Incoming;
  } catch (err) {
    write({
      op: "error",
      id: "parse",
      error: err instanceof Error ? err.message : "invalid json",
    });
    return;
  }

  const id = req.id ?? req.op ?? "?";

  if (req.op === "ping") {
    write({
      op: "ok",
      id,
      content: "pong",
      technique: currentTechnique,
      servingPaused: false,
    });
    return;
  }

  if (req.op === "get_state") {
    write({
      op: "ok",
      id,
      technique: currentTechnique,
      graph: currentGraph,
      servingPaused: false,
    });
    return;
  }

  if (req.op === "set_technique") {
    const next = (req.technique ?? "one-shot") as Tau2Technique;
    setTechnique(next, req.graph);
    write({
      op: "ok",
      id,
      technique: currentTechnique,
      graph: currentGraph,
      servingPaused: false,
    });
    return;
  }

  if (req.op === "i_loop") {
    const applied = applyILoop(req.graph ?? currentGraph, req.obs);
    currentTechnique = applied.techniqueAfter;
    currentGraph = applied.graphAfter;
    write({
      op: "ok",
      id,
      technique: currentTechnique,
      graph: currentGraph,
      graphDiff: applied.graphDiff,
      servingPaused: false,
      content: applied.applied ? "applied" : "exhausted",
    });
    return;
  }

  if (req.op === "i_weight_spawn") {
    weightJob = { spawned: true, done: false };
    // Async trainer: do not block this request. Serving keeps answering.
    const trainer = new FakeTrainer();
    void trainer.train([], { baseModel: "base", technique: "fake-lora" }).then((art) => {
      weightJob = { spawned: true, done: true, artifactId: art.id };
    });
    write({
      op: "ok",
      id,
      spawned: true,
      done: false,
      servingPaused: false,
    });
    return;
  }

  if (req.op === "i_weight_status") {
    write({
      op: "ok",
      id,
      spawned: weightJob.spawned,
      done: weightJob.done,
      servingPaused: false,
      content: weightJob.artifactId,
    });
    return;
  }

  if (req.op === "i_weight_gate") {
    const gate = gateWeightMount(Number(req.before ?? 0), Number(req.after ?? 0));
    write({
      op: "ok",
      id,
      gate,
      servingPaused: false,
    });
    return;
  }

  if (req.op !== "turn") {
    write({ op: "error", id, error: `unknown op ${req.op ?? ""}` });
    return;
  }

  try {
    const technique = req.technique ?? currentTechnique;
    const result = await runTau2Turn({
      policy: req.policy ?? "",
      tools: req.tools ?? [],
      messages: req.messages ?? [],
      technique,
      graph: req.graph ?? currentGraph,
      model: req.model,
      provider: req.model === "deterministic" ? new DeterministicProvider() : createProvider(),
    });
    write({
      op: "ok",
      id,
      content: result.content,
      tool_calls: result.toolCalls,
      traces: result.traces,
      technique,
      servingPaused: false,
    });
  } catch (err) {
    write({
      op: "error",
      id,
      error: err instanceof Error ? err.message : String(err),
    });
  }
}

async function main(): Promise<void> {
  process.stderr.write("vdom tau2 sidecar ready\n");
  const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of rl) {
    const trimmed = line.trim();
    if (trimmed.length === 0) continue;
    await handle(trimmed);
  }
}

main().catch((err) => {
  process.stderr.write(String(err) + "\n");
  process.exit(1);
});
