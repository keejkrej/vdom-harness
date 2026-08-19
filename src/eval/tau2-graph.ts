import { type AgentGraph, node, graph } from "../ir.js";
import { type Tau2Technique } from "./tau2-types.js";

const OBJECTIVE =
  "Help the user according to the domain policy. Use tools when needed. Each turn is either a tool call or a message to the user, never both.";

/**
 * AgentGraph the sidecar walks for one τ² turn.
 * Same IR as the word-reverse fixtures — policy/tools arrive as turn context.
 */
export function tau2Graph(technique: Tau2Technique = "one-shot", model?: string): AgentGraph {
  if (technique === "self-refine") {
    return graph({
      id: "tau2-self-refine",
      version: 1,
      meta: { paper: "Self-Refine", technique: "self-refine", benchmark: "tau2-bench" },
      root: node({
        key: "solve",
        role: "solve",
        objective: "Draft an action given the conversation",
        paper: "Self-Refine",
        technique: "self-refine",
        model,
        children: [
          node({
            key: "critic",
            role: "critic",
            objective: "Critique whether the next step follows policy and uses the right tool",
            paper: "Self-Refine",
            technique: "self-refine",
            model,
            children: [
              node({
                key: "refine",
                role: "refine",
                objective: OBJECTIVE,
                paper: "Self-Refine",
                technique: "self-refine",
                model,
              }),
            ],
          }),
        ],
      }),
    });
  }

  if (technique === "reflexion") {
    return graph({
      id: "tau2-reflexion",
      version: 1,
      meta: { paper: "Reflexion", technique: "reflexion", benchmark: "tau2-bench" },
      root: node({
        key: "actor",
        role: "actor",
        objective: OBJECTIVE,
        capabilities: ["retry"],
        paper: "Reflexion",
        technique: "reflexion",
        model,
        children: [
          node({
            key: "reflect",
            role: "reflect",
            objective: "Verbalize a lesson from a failed tool call",
            paper: "Reflexion",
            technique: "reflexion",
            model,
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

  return graph({
    id: "tau2-oneshot",
    version: 1,
    meta: { technique: "one-shot", benchmark: "tau2-bench" },
    root: node({
      key: "solve",
      role: "solve",
      objective: OBJECTIVE,
      technique: "one-shot",
      model,
    }),
  });
}
