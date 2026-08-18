import { createProvider } from "./providers.js";
import {
  compilePaper,
  SELF_REFINE_ABSTRACT,
  REFLEXION_ABSTRACT,
  oneShotGraph,
} from "./papers.js";
import { WORD_REVERSE, runBenchmark } from "./benchmarks.js";
import { applySelfRefineMutation } from "./scientist.js";
import { reconcile, formatOps } from "./reconciler.js";

function padLabel(label: string, width: number): string {
  return label + " ".repeat(Math.max(1, width - label.length));
}

async function main(): Promise<void> {
  const provider = createProvider();

  console.log("vdom — virtual DOM for agents");
  console.log("");

  const selfRefine = compilePaper(SELF_REFINE_ABSTRACT);
  const sr = await runBenchmark(selfRefine, WORD_REVERSE, provider);
  console.log(`${padLabel("self-refine", 19)}score=${sr.score.toFixed(2)}`);

  const reflexion = compilePaper(REFLEXION_ABSTRACT);
  const rx = await runBenchmark(reflexion, WORD_REVERSE, provider);
  console.log(`${padLabel("reflexion", 19)}score=${rx.score.toFixed(2)}`);

  console.log("");
  console.log("Runtime evolution:");

  const v1 = oneShotGraph();
  const v1b = await runBenchmark(v1, WORD_REVERSE, provider);
  console.log(
    `${padLabel("v1 one-shot", 24)}score=${v1b.score.toFixed(2)} nodes=${v1b.nodes}`,
  );

  const v2 = applySelfRefineMutation(v1);
  const v2b = await runBenchmark(v2, WORD_REVERSE, provider);
  console.log(
    `${padLabel("v2 one-shot+critic", 24)}score=${v2b.score.toFixed(2)} nodes=${v2b.nodes}`,
  );

  console.log("");
  const playback = v2b.traces.length > 0 ? v2b.traces : sr.traces;
  for (const t of playback) {
    if (t.role === "solve") {
      console.log(`[solve]   ${WORD_REVERSE.input}`);
    } else if (t.role === "critic") {
      const start = t.output.split(".")[0] ?? t.output;
      console.log(`[critic]  ${start}...`);
    } else if (t.role === "refine") {
      console.log(`[refine]  ${t.output}`);
    }
  }

  console.log("");
  const rec = reconcile(v1, v2);
  console.log("reconcile v1 → v2");
  console.log(formatOps(rec.ops));

  console.log("");
  if (process.env.OPENAI_API_KEY) {
    console.log(
      "OPENAI_API_KEY is set — researchLoop can compile arbitrary paper text with a live model.",
    );
  } else {
    console.log("No OPENAI_API_KEY — deterministic provider is active.");
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
