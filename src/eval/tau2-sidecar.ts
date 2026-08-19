/**
 * Stdio JSONL sidecar. Python HalfDuplexAgent sends one turn per line.
 * Logs go to stderr so stdout stays machine-readable.
 */
import { createProvider, type Message, type ToolSpec } from "../providers.js";
import { runTau2Turn } from "./tau2-turn.js";
import { type Tau2Technique, type Tau2TurnResponse } from "./tau2-types.js";
import { createInterface } from "node:readline";

type Incoming = {
  op?: string;
  id?: string;
  policy?: string;
  tools?: ToolSpec[];
  messages?: Message[];
  technique?: Tau2Technique;
  model?: string;
};

function write(obj: Tau2TurnResponse): void {
  process.stdout.write(`${JSON.stringify(obj)}\n`);
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

  if (req.op === "ping") {
    write({ op: "ok", id: req.id ?? "ping", content: "pong" });
    return;
  }

  if (req.op !== "turn") {
    write({ op: "error", id: req.id ?? "?", error: `unknown op ${req.op ?? ""}` });
    return;
  }

  const id = req.id ?? "turn";
  try {
    const result = await runTau2Turn({
      policy: req.policy ?? "",
      tools: req.tools ?? [],
      messages: req.messages ?? [],
      technique: req.technique,
      model: req.model,
      provider: createProvider(),
    });
    write({
      op: "ok",
      id,
      content: result.content,
      tool_calls: result.toolCalls,
      traces: result.traces,
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
