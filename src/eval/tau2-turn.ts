import { type AgentGraph, type Trace } from "../ir.js";
import {
  type Completion,
  type Message,
  type Provider,
  type ToolSpec,
  createProvider,
} from "../providers.js";
import { tau2Graph } from "./tau2-graph.js";
import { type Tau2Technique } from "./tau2-types.js";
import { AIRLINE_POLICY_CHECKLIST } from "./tau2-policy.js";

const AGENT_INSTRUCTION = `You are a customer service agent that helps the user according to the policy below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.
Always follow the policy. Prefer tools over guessing.`;

export type Tau2TurnOpts = {
  policy: string;
  tools: ToolSpec[];
  messages: Message[];
  technique?: Tau2Technique;
  graph?: AgentGraph;
  provider?: Provider;
  model?: string;
};

export type Tau2TurnResult = Completion & { traces: Trace[] };

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

function systemFor(policy: string, extra?: string): Message {
  const parts = [AGENT_INSTRUCTION, "", "<policy>", policy, "</policy>"];
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

/**
 * One τ² half-duplex turn through the vdom AgentGraph.
 * Orchestrator (Python) owns the gym/user/tools; we only decide the next AssistantMessage.
 */
export async function runTau2Turn(opts: Tau2TurnOpts): Promise<Tau2TurnResult> {
  const technique = opts.technique ?? "one-shot";
  const provider = opts.provider ?? createProvider();
  const graph = opts.graph ?? tau2Graph(technique, opts.model);
  const traces: Trace[] = [];
  const input = lastContent(opts.messages);
  const convo = opts.messages;

  if (technique === "self-refine") {
    const thinkMsgs: Message[] = [
      systemFor(
        opts.policy,
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

    const actMsgs: Message[] = [
      systemFor(opts.policy, `Critique from the critic node:\n${critique.content}`),
      ...convo,
    ];
    const acted = await completeTurn(provider, actMsgs, {
      role: "refine",
      tools: opts.tools,
      model: opts.model,
    });
    pushTrace(traces, "refine", "refine", input, formatCompletion(acted));
    return { ...acted, traces };
  }

  if (technique === "policy-checklist") {
    const extra =
      graph.root.children?.find((c) => c.key === "policy-checklist")?.prompt ??
      AIRLINE_POLICY_CHECKLIST;
    const actMsgs: Message[] = [systemFor(opts.policy, extra), ...convo];
    const acted = await completeTurn(provider, actMsgs, {
      role: "policy-checklist",
      tools: opts.tools,
      model: opts.model,
    });
    pushTrace(traces, "policy-checklist", "critic", input, formatCompletion(acted));
    return { ...acted, traces };
  }

  if (technique === "validator") {
    const thinkMsgs: Message[] = [
      systemFor(
        opts.policy,
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

    const actMsgs: Message[] = [
      systemFor(opts.policy, `Validator critique:\n${critique.content}`),
      ...convo,
    ];
    const acted = await completeTurn(provider, actMsgs, {
      role: "validator",
      tools: opts.tools,
      model: opts.model,
    });
    pushTrace(traces, "validator", "validator", input, formatCompletion(acted));
    return { ...acted, traces };
  }

  if (technique === "reflexion" && toolFailed(convo)) {
    const reflectMsgs: Message[] = [
      systemFor(opts.policy, "A tool call just failed. Verbalize a lesson, then the actor will retry."),
      ...convo,
      { role: "user", content: "What went wrong and what should change on the retry?" },
    ];
    const lesson = await completeTurn(provider, reflectMsgs, {
      role: "reflect",
      model: opts.model,
    });
    pushTrace(traces, "reflect", "reflect", input, lesson.content);
    pushTrace(traces, "memory", "memory", lesson.content, lesson.content);

    const retryMsgs: Message[] = [
      systemFor(opts.policy, `Episodic memory:\n${lesson.content}`),
      ...convo,
    ];
    const retried = await completeTurn(provider, retryMsgs, {
      role: "actor",
      tools: opts.tools,
      model: opts.model,
    });
    pushTrace(traces, "actor", "actor", input, formatCompletion(retried));
    return { ...retried, traces };
  }

  const root = graph.root;
  const actMsgs: Message[] = [systemFor(opts.policy), ...convo];
  const acted = await completeTurn(provider, actMsgs, {
    role: root.role,
    tools: opts.tools,
    model: opts.model,
  });
  pushTrace(traces, root.key, root.role, input, formatCompletion(acted));
  return { ...acted, traces };
}
