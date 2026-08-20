import {
  type AgentGraph,
  type AgentNode,
  type Trace,
  kindOf,
  modelId,
} from "./ir.js";
import { type Provider, type Message, resolveProvider } from "./providers.js";
import { RuntimeDOM } from "./reconciler.js";

export type RunResult = {
  traces: Trace[];
  outputs: Record<string, string>;
  final: string;
};

function isMemory(n: AgentNode): boolean {
  return kindOf(n) === "memory" || n.role === "memory";
}

function isReflect(n: AgentNode): boolean {
  return n.role === "reflect" || n.role === "reflection";
}

function isGrouping(n: AgentNode): boolean {
  const k = kindOf(n);
  return (
    k === "supervisor" ||
    k === "goal" ||
    k === "barrier" ||
    k === "channel" ||
    k === "policy" ||
    k === "tool" ||
    k === "artifact" ||
    k === "adapter" ||
    // Unmounted / proposed capabilities are structural only.
    (k === "capability" && n.status !== "mounted")
  );
}

function isMountedCapability(n: AgentNode): boolean {
  return kindOf(n) === "capability" && n.status === "mounted";
}

function isReflexionHost(n: AgentNode): boolean {
  const kids = n.children ?? [];
  const hasReflect = kids.some(isReflect);
  const hasMemory = kids.some(isMemory);
  const retryCap = n.capabilities?.includes("retry") ?? false;
  return (n.role === "actor" || retryCap) && hasReflect && hasMemory;
}

function buildSystem(n: AgentNode): string {
  const lines = [`Role: ${n.role}`, `Objective: ${n.objective}`];
  if (n.prompt) lines.push(`Prompt: ${n.prompt}`);
  if (n.technique) lines.push(`Technique: ${n.technique}`);
  return lines.join("\n");
}

function buildUser(
  task: string,
  parentOutputs: string[],
  memories: string[],
  traces: Trace[],
): string {
  const parts = [`Input: ${task}`];
  if (parentOutputs.length > 0) {
    parts.push(`Parent outputs:\n${parentOutputs.join("\n")}`);
  }
  if (memories.length > 0) {
    parts.push(`Memory:\n${memories.join("\n")}`);
  }
  if (traces.length > 0) {
    parts.push(`Traces:\n${traces.map((t) => `[${t.role}] ${t.output}`).join("\n")}`);
  }
  return parts.join("\n\n");
}

/**
 * Prefer S (per-episode CatalogPointer) over a bound PhysicalNode.provider.
 * Bound is a cache; spraying one client onto every key leaks 0813 onto I_loop.
 * n.model is a derived projection of C, not the paper S coordinate.
 * Official later-serving helpers must pass S. Omitting servingSku is not a jump.
 */
export function providerForNode(
  n: AgentNode,
  fallback: Provider,
  dom?: RuntimeDOM,
  servingSku?: string,
): Provider {
  if (servingSku) return resolveProvider(servingSku);
  const bound = dom?.current.get(n.key)?.provider;
  if (bound) return bound;
  if (modelId(n.model) != null) return resolveProvider(n.model);
  return fallback;
}

export async function runGraph(
  g: AgentGraph,
  task: string,
  provider: Provider,
  dom?: RuntimeDOM,
  servingSku?: string,
): Promise<RunResult> {
  if (dom) dom.reconcile(g);

  const outputs: Record<string, string> = {};
  const traces: Trace[] = [];
  const memoryStore: Record<string, string> = {};

  const allMemories = (): string[] => Object.values(memoryStore);

  const runAgent = async (n: AgentNode, parentOutputs: string[]): Promise<string> => {
    const msgs: Message[] = [
      { role: "system", content: buildSystem(n) },
      { role: "user", content: buildUser(task, parentOutputs, allMemories(), traces) },
    ];
    const client = providerForNode(n, provider, dom, servingSku);
    const output = await client.complete(msgs, {
      role: n.role,
      model: servingSku ?? client.model ?? modelId(n.model),
    });
    outputs[n.key] = output;
    const trace: Trace = {
      nodeKey: n.key,
      role: n.role,
      input: task,
      output,
      ts: Date.now(),
    };
    traces.push(trace);
    const phys = dom?.current.get(n.key);
    if (phys) phys.traces.push(trace);
    return output;
  };

  const runCapability = async (n: AgentNode, parentOutputs: string[]): Promise<string> => {
    const phys = dom?.current.get(n.key);
    const fn = phys?.capability;
    if (!fn) {
      throw new Error(
        `mounted capability ${n.key} has no loaded module (moduleId=${n.moduleId ?? "?"})`,
      );
    }
    const input = parentOutputs[parentOutputs.length - 1] ?? task;
    const output = await fn(input);
    outputs[n.key] = output;
    const trace: Trace = {
      nodeKey: n.key,
      role: n.role,
      input,
      output,
      ts: Date.now(),
    };
    traces.push(trace);
    if (phys) phys.traces.push(trace);
    return output;
  };

  const walk = async (n: AgentNode, parentOutputs: string[]): Promise<void> => {
    if (isReflexionHost(n)) {
      await runReflexionNode(n, parentOutputs);
      return;
    }

    if (isMemory(n)) {
      const content = parentOutputs[parentOutputs.length - 1] ?? n.prompt ?? "";
      memoryStore[n.key] = content;
      outputs[n.key] = content;
      for (const child of n.children ?? []) {
        await walk(child, [...parentOutputs, content]);
      }
      return;
    }

    if (isMountedCapability(n)) {
      const out = await runCapability(n, parentOutputs);
      for (const child of n.children ?? []) {
        await walk(child, [...parentOutputs, out]);
      }
      return;
    }

    if (isGrouping(n)) {
      for (const child of n.children ?? []) await walk(child, parentOutputs);
      return;
    }

    const out = await runAgent(n, parentOutputs);
    for (const child of n.children ?? []) {
      await walk(child, [...parentOutputs, out]);
    }
  };

  const runReflexionNode = async (n: AgentNode, parentOutputs: string[]): Promise<void> => {
    const kids = n.children ?? [];
    const reflectNode = kids.find(isReflect);
    const memoryNode = kids.find(isMemory);

    // First attempt — typically fails the word-reverse benchmark.
    await runAgent(n, parentOutputs);

    if (reflectNode) {
      await runAgent(reflectNode, [outputs[n.key] ?? ""]);
    }
    if (memoryNode) {
      const lesson = reflectNode ? (outputs[reflectNode.key] ?? "") : "";
      memoryStore[memoryNode.key] = lesson;
      outputs[memoryNode.key] = lesson;
      for (const child of memoryNode.children ?? []) {
        await walk(child, [lesson]);
      }
    }

    // Retry with episodic memory in context.
    await runAgent(n, parentOutputs);

    for (const child of kids) {
      if (isReflect(child) || isMemory(child)) continue;
      await walk(child, [...parentOutputs, outputs[n.key] ?? ""]);
    }
  };

  await walk(g.root, []);

  const final = traces.length > 0 ? traces[traces.length - 1]!.output : "";
  return { traces, outputs, final };
}

/** Explicit Reflexion helper: actor → reflect → memory → actor. */
export async function runReflexion(
  g: AgentGraph,
  task: string,
  provider: Provider,
  dom?: RuntimeDOM,
): Promise<RunResult> {
  return runGraph(g, task, provider, dom);
}
