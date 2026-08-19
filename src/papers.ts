/**
 * Source → graph compilers.
 *
 * Papers are the current compiled examples (Self-Refine, Reflexion).
 * `compilePaper` / `compileSource` accept any untrusted text — blogs, repos,
 * X posts, other agents, conversation, traces. Unknown sources fall through
 * to a one-shot solve node. The compiler is a convenience, not the product.
 */
import { type AgentGraph, node, graph } from "./ir.js";

export const SELF_REFINE_ABSTRACT = `
Self-Refine (Madaan et al.) improves generation by iterative self-feedback.
A generator produces an output, a feedback model critiques it, and a refinement
model revises the output using that critique. Repeat until quality saturates.
`.trim();

export const REFLEXION_ABSTRACT = `
Reflexion (Shinn et al.) reinforces language agents through linguistic feedback.
An actor attempts a task, a reflection model verbalizes a lesson, and episodic
memory stores it so the next trial can improve without weight updates.
`.trim();

export function oneShotGraph(objective = "Solve the task in a single pass"): AgentGraph {
  return graph({
    id: "oneshot",
    version: 1,
    meta: { technique: "one-shot" },
    root: node({
      key: "solve",
      role: "solve",
      objective,
      technique: "one-shot",
    }),
  });
}

/**
 * Self-Refine topology (Madaan et al.):
 *   solve (generator)
 *     └── critic (feedback)
 *           └── refine (refinement)
 */
export function selfRefineGraph(): AgentGraph {
  return graph({
    id: "self-refine",
    version: 1,
    meta: { paper: "Self-Refine", technique: "self-refine" },
    root: node({
      key: "solve",
      role: "solve",
      objective: "Generate an initial solution",
      paper: "Self-Refine",
      technique: "self-refine",
      children: [
        node({
          key: "critic",
          role: "critic",
          objective: "Critique the generator output",
          paper: "Self-Refine",
          technique: "self-refine",
          children: [
            node({
              key: "refine",
              role: "refine",
              objective: "Revise the answer using the critique",
              paper: "Self-Refine",
              technique: "self-refine",
            }),
          ],
        }),
      ],
    }),
  });
}

/**
 * Reflexion topology (Shinn et al.):
 *   actor
 *     ├── reflect
 *     └── memory (episodic)
 *
 * The runtime loop re-runs the actor after writing memory.
 */
export function reflexionGraph(): AgentGraph {
  return graph({
    id: "reflexion",
    version: 1,
    meta: { paper: "Reflexion", technique: "reflexion" },
    root: node({
      key: "actor",
      role: "actor",
      objective: "Attempt the task",
      capabilities: ["retry"],
      paper: "Reflexion",
      technique: "reflexion",
      children: [
        node({
          key: "reflect",
          role: "reflect",
          objective: "Verbalize a lesson from the failed attempt",
          paper: "Reflexion",
          technique: "reflexion",
        }),
        node({
          key: "memory",
          kind: "memory",
          role: "memory",
          objective: "Episodic store of reflections",
          persistence: "session",
          paper: "Reflexion",
          technique: "reflexion",
        }),
      ],
    }),
  });
}

/**
 * Compile untrusted source text into an AgentGraph.
 * Known Self-Refine / Reflexion citations route to fixtures; everything
 * else is a one-shot. Prefer `compileSource` in new call sites.
 */
export function compilePaper(text: string): AgentGraph {
  const t = text.toLowerCase();
  if (/self-refine|self refine|madaan/.test(t)) return selfRefineGraph();
  if (/reflexion|shinn/.test(t)) return reflexionGraph();
  return oneShotGraph();
}

/** Alias: papers are one source among blogs, repos, X, agents, traces. */
export const compileSource = compilePaper;
