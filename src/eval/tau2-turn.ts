import { flatten, type AgentGraph, type Trace } from "../ir.js";
import {
  type Completion,
  type Message,
  type Provider,
  type ToolSpec,
  createProvider,
} from "../providers.js";
import { tau2Graph } from "./tau2-graph.js";
import { executeGraphSelfTool, gymToolCalls, withGraphSelfTools, type GraphToolEdit } from "./tau2-graph-tools.js";
import { serializeKernelC, stripGoldIds } from "./tau2-kernel.js";
import { type Tau2Technique } from "./tau2-types.js";
import { AIRLINE_POLICY_CHECKLIST } from "./tau2-policy.js";

const MAX_SELF_TOOL_ITERS = 4;

const AGENT_INSTRUCTION = `You are a customer service agent that helps the user according to the policy below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.
Always follow the policy. Prefer tools over guessing.
You have get_agent_graph and set_agent_graph. You may read and rewrite your own graph when the current C is failing the user or the policy. You should get_agent_graph before set_agent_graph. Domain tools go to the environment; get/set are local and never forwarded to the gym. Do not invent reservation IDs.`;

export type Tau2TurnOpts = {
  policy: string;
  tools: ToolSpec[];
  messages: Message[];
  technique?: Tau2Technique;
  graph?: AgentGraph;
  provider?: Provider;
  model?: string;
};

export type Tau2TurnResult = Completion & {
  traces: Trace[];
  system: string;
  graph: AgentGraph;
  graphEdits: GraphToolEdit[];
  servingPaused: false;
};

function completeTurn(
  provider: Provider,
  msgs: Message[],
  opts: { role: string; tools?: ToolSpec[]; model?: string },
): Promise<Completion> {
  if (provider.completeTurn) {
    return provider.completeTurn(msgs, opts);
  }
  return provider.complete(msgs, opts).then((content) => ({ content }));
}

function lastContent(msgs: Message[]): string {
  const last = msgs[msgs.length - 1];
  return last?.content ?? "";
}

function toolFailed(msgs: Message[]): boolean {
  const last = [...msgs].reverse().find((m) => m.role === "tool");
  if (!last) return false;
  const c = last.content.toLowerCase();
  return last.content.startsWith("Error") || c.includes("error") || c.includes("traceback");
}

function policyChecklistText(graph: AgentGraph): string {
  const nodes = flatten(graph).map((f) => f.node);
  const n =
    nodes.find((x) => x.key === "policy-checklist") ??
    nodes.find((x) => x.kind === "policy") ??
    nodes.find((x) => x.role === "policy" || x.role === "policy-checklist");
  const text = stripGoldIds((n?.prompt ?? n?.objective ?? "").trim());
  return text || AIRLINE_POLICY_CHECKLIST;
}

function systemFor(policy: string, graph: AgentGraph, extra?: string): Message {
  const parts = [
    AGENT_INSTRUCTION,
    "",
    serializeKernelC(graph),
    "",
    "<policy>",
    policy,
    "</policy>",
  ];
  if (extra) parts.push("", extra);
  return { role: "system", content: parts.join("\n") };
}

function pushTrace(
  traces: Trace[],
  nodeKey: string,
  role: string,
  input: string,
  output: string,
): void {
  traces.push({ nodeKey, role, input, output, ts: Date.now() });
}

function formatCompletion(c: Completion): string {
  if (c.toolCalls && c.toolCalls.length > 0) {
    return c.toolCalls.map((t) => `${t.name}(${JSON.stringify(t.arguments)})`).join("; ");
  }
  return c.content;
}

function liveExtra(graph: AgentGraph, fallback?: string): string | undefined {
  const nodes = flatten(graph).map((f) => f.node);
  const policy = nodes.find((n) => n.kind === "policy" || n.key === "policy-checklist");
  if (policy?.prompt) return stripGoldIds(policy.prompt);
  return fallback;
}

async function actingTurn(opts: {
  policy: string;
  graph: AgentGraph;
  extra?: string;
  convo: Message[];
  provider: Provider;
  role: string;
  tools: ToolSpec[];
  model?: string;
  traces: Trace[];
  input: string;
}): Promise<{ acted: Completion; graph: AgentGraph; system: string; edits: GraphToolEdit[] }> {
  let graph = opts.graph;
  let extra = liveExtra(graph, opts.extra);
  const convo: Message[] = [...opts.convo];
  const tools = withGraphSelfTools(opts.tools);
  const edits: GraphToolEdit[] = [];
  let system = systemFor(opts.policy, graph, extra).content;

  for (let i = 0; i < MAX_SELF_TOOL_ITERS; i++) {
    extra = liveExtra(graph, extra);
    const sys = systemFor(opts.policy, graph, extra);
    system = sys.content;
    const acted = await completeTurn(opts.provider, [sys, ...convo], {
      role: opts.role,
      tools,
      model: opts.model,
    });
    const selfCalls = (acted.toolCalls ?? []).filter((c) =>
      c.name === "get_agent_graph" || c.name === "set_agent_graph",
    );
    if (selfCalls.length === 0) {
      const gym = gymToolCalls(acted.toolCalls);
      pushTrace(opts.traces, opts.role, opts.role, opts.input, formatCompletion({ ...acted, toolCalls: gym }));
      return { acted: { ...acted, toolCalls: gym }, graph, system, edits };
    }

    convo.push({
      role: "assistant",
      content: acted.content,
      tool_calls: selfCalls,
    });
    for (const tc of selfCalls) {
      const result = executeGraphSelfTool(tc, graph);
      graph = result.graph;
      edits.push(result.edit);
      pushTrace(opts.traces, "kernel", "self", tc.name, result.edit.reason);
      convo.push({
        role: "tool",
        name: tc.name,
        tool_call_id: tc.id,
        content: result.content,
      });
    }
  }

  const bound: Completion = { content: "Updated kernel C." };
  pushTrace(opts.traces, opts.role, opts.role, opts.input, bound.content);
  return { acted: bound, graph, system, edits };
}

function finish(
  acted: Completion,
  traces: Trace[],
  system: string,
  graph: AgentGraph,
  edits: GraphToolEdit[],
): Tau2TurnResult {
  return {
    ...acted,
    traces,
    system,
    graph,
    graphEdits: edits,
    servingPaused: false,
  };
}

/**
 * One τ² half-duplex turn through the vdom AgentGraph.
 * Orchestrator (Python) owns the gym/user/tools; get/set_agent_graph stay here.
 */
export async function runTau2Turn(opts: Tau2TurnOpts): Promise<Tau2TurnResult> {
  const technique = opts.technique ?? "one-shot";
  const provider = opts.provider ?? createProvider();
  let graph = opts.graph ?? tau2Graph(technique, opts.model);
  const traces: Trace[] = [];
  const input = lastContent(opts.messages);
  const convo = opts.messages;

  if (technique === "self-refine") {
    const thinkMsgs: Message[] = [
      systemFor(
        opts.policy,
        graph,
        "Before acting, critique the situation: what does policy require, and which tool (if any) is next?",
      ),
      ...convo,
      { role: "user", content: "Write a short critique of the next action. Do not call tools." },
    ];
    const critique = await completeTurn(provider, thinkMsgs, {
      role: "critic",
      model: opts.model,
    });
    pushTrace(traces, "critic", "critic", input, critique.content);

    const out = await actingTurn({
      policy: opts.policy,
      graph,
      extra: `Critique from the critic node:\n${critique.content}`,
      convo,
      provider,
      role: "refine",
      tools: opts.tools,
      model: opts.model,
      traces,
      input,
    });
    return finish(out.acted, traces, out.system, out.graph, out.edits);
  }

  if (technique === "policy-checklist") {
    const out = await actingTurn({
      policy: opts.policy,
      graph,
      extra: policyChecklistText(graph),
      convo,
      provider,
      role: "policy-checklist",
      tools: opts.tools,
      model: opts.model,
      traces,
      input,
    });
    return finish(out.acted, traces, out.system, out.graph, out.edits);
  }

  if (technique === "validator") {
    const thinkMsgs: Message[] = [
      systemFor(
        opts.policy,
        graph,
        "Critique the last action. If policy forbids it, name the correct tool (including transfer).",
      ),
      ...convo,
      { role: "user", content: "Write a short critique. Do not call tools." },
    ];
    const critique = await completeTurn(provider, thinkMsgs, {
      role: "critic",
      model: opts.model,
    });
    pushTrace(traces, "critic", "critic", input, critique.content);

    const out = await actingTurn({
      policy: opts.policy,
      graph,
      extra: `Validator critique:\n${critique.content}`,
      convo,
      provider,
      role: "validator",
      tools: opts.tools,
      model: opts.model,
      traces,
      input,
    });
    return finish(out.acted, traces, out.system, out.graph, out.edits);
  }

  if (technique === "reflexion" && toolFailed(convo)) {
    const reflectMsgs: Message[] = [
      systemFor(opts.policy, graph, "A tool call just failed. Verbalize a lesson, then the actor will retry."),
      ...convo,
      { role: "user", content: "What went wrong and what should change on the retry?" },
    ];
    const lesson = await completeTurn(provider, reflectMsgs, {
      role: "reflect",
      model: opts.model,
    });
    pushTrace(traces, "reflect", "reflect", input, lesson.content);
    pushTrace(traces, "memory", "memory", lesson.content, lesson.content);

    const out = await actingTurn({
      policy: opts.policy,
      graph,
      extra: `Episodic memory:\n${lesson.content}`,
      convo,
      provider,
      role: "actor",
      tools: opts.tools,
      model: opts.model,
      traces,
      input,
    });
    return finish(out.acted, traces, out.system, out.graph, out.edits);
  }

  const out = await actingTurn({
    policy: opts.policy,
    graph,
    convo,
    provider,
    role: graph.root.role,
    tools: opts.tools,
    model: opts.model,
    traces,
    input,
  });
  return finish(out.acted, traces, out.system, out.graph, out.edits);
}

export function filterGymToolCalls(calls?: ToolCallOut[]): ToolCallOut[] | undefined {
  return gymToolCalls(calls);
}
