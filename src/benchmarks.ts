import { type AgentGraph, flatten } from "./ir.js";
import { type Provider } from "./providers.js";
import { runGraph, type RunResult } from "./runtime.js";

export type Task = {
  id: string;
  input: string;
  expected: string;
  grade(output: string): number;
};

function normalize(s: string): string {
  return s.trim().replace(/\s+/g, " ");
}

function exactGrade(expected: string): (output: string) => number {
  const want = normalize(expected);
  return (output: string) => (normalize(output) === want ? 1 : 0);
}

export const WORD_REVERSE: Task = {
  id: "word-reverse",
  input: "dom virtual",
  expected: "mod lautriv",
  grade: exactGrade("mod lautriv"),
};

export const WORD_REVERSE_HELLO: Task = {
  id: "word-reverse-hello",
  input: "hello world",
  expected: "olleh dlrow",
  grade: exactGrade("olleh dlrow"),
};

export type BenchmarkResult = RunResult & {
  score: number;
  nodes: number;
};

export async function runBenchmark(
  g: AgentGraph,
  task: Task,
  provider: Provider,
): Promise<BenchmarkResult> {
  const result = await runGraph(g, task.input, provider);
  return {
    ...result,
    score: task.grade(result.final),
    nodes: flatten(g).length,
  };
}
