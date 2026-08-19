import { type AgentNode, modelId } from "./ir.js";

/** Official OpenRouter model for live τ² / τ³ evals. Not the 0424 preview. */
export const DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-v4-flash-0731";
export const DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1";

export type MessageRole = "system" | "user" | "assistant" | "tool";

export type ToolSpec = {
  name: string;
  description?: string;
  parameters?: Record<string, unknown>;
};

export type ToolCallOut = {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
};

export type Message = {
  role: MessageRole;
  content: string;
  name?: string;
  tool_call_id?: string;
  tool_calls?: ToolCallOut[];
};

export type Completion = {
  content: string;
  toolCalls?: ToolCallOut[];
};

export type CompleteOpts = {
  role?: string;
  temperature?: number;
  /** Override the provider's default model for this call (OpenAI-compatible). */
  model?: string;
  /** OpenAI-style tool schemas. Used by completeTurn; ignored by complete(). */
  tools?: ToolSpec[];
};

export type Provider = {
  name: string;
  /** Bound model id when this provider was resolved for a specific AgentNode.model. */
  model?: string;
  complete(msgs: Message[], opts?: CompleteOpts): Promise<string>;
  /** One conversational turn: text XOR tool calls (τ² protocol). */
  completeTurn?(msgs: Message[], opts?: CompleteOpts): Promise<Completion>;
};

export type ChatConfig = {
  apiKey: string;
  baseUrl: string;
  model: string;
  viaOpenRouter: boolean;
};

/**
 * Resolve OpenAI-compatible chat settings.
 * OPENROUTER_API_KEY alone → OpenRouter + deepseek-v4-flash-0731.
 * OPENAI_API_KEY alone → existing OpenAI defaults (gpt-4o-mini).
 * No key → null (DeterministicProvider).
 */
export function resolveChatConfig(): ChatConfig | null {
  const openrouter = process.env.OPENROUTER_API_KEY?.trim();
  const openai = process.env.OPENAI_API_KEY?.trim();
  const apiKey = openai || openrouter;
  if (!apiKey) return null;

  const envBase = process.env.OPENAI_BASE_URL?.trim();
  const viaOpenRouter =
    Boolean(openrouter) &&
    (!openai || (envBase ?? "").includes("openrouter.ai"));
  const baseUrl =
    envBase ||
    (viaOpenRouter ? DEFAULT_OPENROUTER_BASE_URL : "https://api.openai.com/v1");
  const defaultModel = baseUrl.includes("openrouter.ai")
    ? DEFAULT_OPENROUTER_MODEL
    : "gpt-4o-mini";
  return {
    apiKey,
    baseUrl,
    model: process.env.OPENAI_MODEL?.trim() || defaultModel,
    viaOpenRouter: baseUrl.includes("openrouter.ai"),
  };
}

/** Test / custom providers keyed by model id. */
const registry = new Map<string, Provider>();

/** Lazily built providers for model ids (OpenAI or tagged deterministic). */
const byModel = new Map<string, Provider>();

let defaultProvider: Provider | undefined;

/** Register a provider for a model id (tests inject fakes here). */
export function registerProvider(id: string, provider: Provider): void {
  registry.set(id, provider);
}

/** Drop registry + caches. Used by tests for isolation. */
export function clearProviderRegistry(): void {
  registry.clear();
  byModel.clear();
  defaultProvider = undefined;
}

function createProviderForId(id: string): Provider {
  const cfg = resolveChatConfig();
  if (cfg) {
    return new OpenAICompatibleProvider(cfg.apiKey, cfg.baseUrl, id, cfg.viaOpenRouter);
  }
  return new DeterministicProvider(id);
}

/**
 * Resolve the chat client for an AgentNode.model value.
 * Missing model → process-global createProvider() singleton (same fallback as today).
 * Registered ids win; otherwise cache one provider per model id.
 */
export function resolveProvider(model?: AgentNode["model"]): Provider {
  const id = modelId(model);
  if (id) {
    const registered = registry.get(id);
    if (registered) return registered;
    let cached = byModel.get(id);
    if (!cached) {
      cached = createProviderForId(id);
      byModel.set(id, cached);
    }
    return cached;
  }
  if (!defaultProvider) defaultProvider = createProvider();
  return defaultProvider;
}

const CRITIQUE =
  "The transformation is incorrect. Reverse each word independently, not the whole string.";

const LESSON = "reverse each word independently; do not reverse the entire string.";

export function reverseEntire(s: string): string {
  return [...s].reverse().join("");
}

export function reverseEachWord(s: string): string {
  return s
    .split(/\s+/)
    .filter((w) => w.length > 0)
    .map((w) => [...w].reverse().join(""))
    .join(" ");
}

/** Pull the puzzle phrase from a chat transcript. */
export function extractInput(msgs: Message[]): string {
  const text = msgs.map((m) => m.content).join("\n");

  const inputLine = text.match(/Input:\s*(.+)/i);
  if (inputLine?.[1]) {
    return inputLine[1].trim().split("\n")[0]!.trim();
  }

  const quoted = [...text.matchAll(/"([^"]+)"/g)];
  if (quoted.length > 0) {
    return quoted[quoted.length - 1]![1]!;
  }

  const lastUser = [...msgs].reverse().find((m) => m.role === "user");
  if (lastUser) {
    const line = lastUser.content
      .split("\n")
      .map((l) => l.trim())
      .find((l) => l.length > 0 && !l.endsWith(":"));
    if (line) return line.replace(/^[-*]\s*/, "");
  }
  return "";
}

function inferRole(msgs: Message[], opts?: { role?: string }): string {
  if (opts?.role) return opts.role.toLowerCase();
  for (const m of msgs) {
    const match = m.content.match(/^Role:\s*(\S+)/im);
    if (match?.[1]) return match[1].toLowerCase();
  }
  return "";
}

function hasWordReverseLesson(msgs: Message[]): boolean {
  const text = msgs.map((m) => m.content).join("\n").toLowerCase();
  return (
    text.includes("reverse each word") ||
    (text.includes("independently") && text.includes("reverse"))
  );
}

function selfRefineGraphJson(): string {
  return JSON.stringify(
    {
      id: "evolved-self-refine",
      version: 2,
      meta: { technique: "self-refine", source: "scientist" },
      root: {
        key: "solve",
        kind: "agent",
        role: "solve",
        objective: "Generate an initial solution",
        paper: "Self-Refine",
        technique: "self-refine",
        children: [
          {
            key: "critic",
            kind: "agent",
            role: "critic",
            objective: "Critique the generator output",
            children: [
              {
                key: "refine",
                kind: "agent",
                role: "refine",
                objective: "Produce a corrected solution from the critique",
              },
            ],
          },
        ],
      },
    },
    null,
    2,
  );
}

/**
 * Role-aware, credential-free provider. The demo and tests are reproducible
 * because each role implements a fixed transformation of the extracted input.
 */
export class DeterministicProvider implements Provider {
  name: string;
  model?: string;

  constructor(model?: string) {
    this.model = model;
    this.name = model ? `deterministic:${model}` : "deterministic";
  }

  async complete(msgs: Message[], opts?: CompleteOpts): Promise<string> {
    const role = inferRole(msgs, opts);
    const input = extractInput(msgs);

    if (role === "critic" || role === "feedback") {
      return CRITIQUE;
    }

    if (role === "refine" || role === "refiner") {
      return reverseEachWord(input);
    }

    if (role === "reflect" || role === "reflection") {
      return LESSON;
    }

    if (role === "scientist") {
      return selfRefineGraphJson();
    }

    if (
      role === "solve" ||
      role === "actor" ||
      role === "one-shot" ||
      role === "oneshot" ||
      role === "actor-retry" ||
      role === "generator"
    ) {
      if (hasWordReverseLesson(msgs)) return reverseEachWord(input);
      return reverseEntire(input);
    }

    const lastUser = [...msgs].reverse().find((m) => m.role === "user");
    return lastUser?.content ?? input;
  }

  /**
   * Credential-free τ² mock turns. Word-reverse complete() is unchanged.
   * Scripted tool calls let `--domain mock` smoke without an API key.
   */
  async completeTurn(msgs: Message[], opts?: CompleteOpts): Promise<Completion> {
    const tools = opts?.tools ?? [];
    const scripted = scriptedTau2MockTurn(msgs, tools, { role: opts?.role });
    if (scripted) return scripted;
    const content = await this.complete(msgs, opts);
    return { content };
  }
}

function transcript(msgs: Message[]): string {
  return msgs
    .map((m) => {
      const tools = (m.tool_calls ?? [])
        .map((t) => `${t.name} ${JSON.stringify(t.arguments)}`)
        .join(" ");
      return `${m.role}: ${m.content} ${tools}`;
    })
    .join("\n");
}

function lastToolMessage(msgs: Message[]): Message | undefined {
  return [...msgs].reverse().find((m) => m.role === "tool");
}

function lastAssistantToolName(msgs: Message[]): string | undefined {
  const last = [...msgs].reverse().find((m) => (m.tool_calls?.length ?? 0) > 0);
  return last?.tool_calls?.[0]?.name;
}

function toolFailed(msg: Message): boolean {
  const c = msg.content.toLowerCase();
  return msg.content.startsWith("Error") || c.includes("error") || c.includes("traceback");
}

const NAIVE_ROLES = new Set(["solve", "actor", "one-shot", "oneshot", "generator"]);

export type ScriptedTurnOpts = {
  role?: string;
};

function isUpdateIntent(text: string): boolean {
  return (
    text.includes("completed") ||
    text.includes("mark task") ||
    text.includes("mark an existing") ||
    text.includes("update_task_status") ||
    (text.includes("update") && text.includes("status"))
  );
}

/**
 * Deterministic mock-domain policy for create/update/transfer tasks.
 *
 * Naive one-shot (role solve/actor) fails official `update_task_1`: it stays
 * in the create_task attractor. Self-Refine refine role calls the right tool.
 * create_task_1 is unchanged (one-shot already hits).
 */
export function scriptedTau2MockTurn(
  msgs: Message[],
  tools: ToolSpec[],
  opts?: ScriptedTurnOpts,
): Completion | undefined {
  if (tools.length === 0) return undefined;
  const names = new Set(tools.map((t) => t.name));
  const text = transcript(msgs).toLowerCase();
  const lastTool = lastToolMessage(msgs);
  const role = (opts?.role ?? "").toLowerCase();
  const naive = NAIVE_ROLES.has(role);

  if (lastTool) {
    if (toolFailed(lastTool)) {
      return { content: "The previous tool call failed. I will try a different approach." };
    }
    const lastToolName =
      lastTool.name === "create_task" ||
      lastTool.name === "update_task_status" ||
      lastTool.name === "transfer_to_human_agents"
        ? lastTool.name
        : lastAssistantToolName(msgs);
    if (lastToolName === "update_task_status") {
      return { content: "The task status was updated successfully." };
    }
    if (lastToolName === "create_task" || lastTool.content.toLowerCase().includes("task")) {
      return { content: "The task was created successfully." };
    }
    return { content: "Done. The requested change is complete." };
  }

  if (
    names.has("create_task") &&
    (text.includes("important meeting") ||
      text.includes("create a new task") ||
      text.includes("create a task") ||
      text.includes("create_task"))
  ) {
    return {
      content: "",
      toolCalls: [
        {
          id: "call_create_task_1",
          name: "create_task",
          arguments: { user_id: "user_1", title: "Important Meeting" },
        },
      ],
    };
  }

  if (names.has("update_task_status") && isUpdateIntent(text) && naive) {
    // Metastable create_task loop — the I_loop diagnostic on mock update_task_1.
    if (names.has("create_task")) {
      return {
        content: "",
        toolCalls: [
          {
            id: "call_create_task_naive",
            name: "create_task",
            arguments: { user_id: "user_1", title: "Important Meeting" },
          },
        ],
      };
    }
    return { content: "I created a new task instead of updating the existing one." };
  }

  if (names.has("update_task_status") && isUpdateIntent(text)) {
    const taskId = text.includes("task_2") ? "task_2" : "task_1";
    return {
      content: "",
      toolCalls: [
        {
          id: "call_update_1",
          name: "update_task_status",
          arguments: { task_id: taskId, status: "completed" },
        },
      ],
    };
  }

  if (
    names.has("transfer_to_human_agents") &&
    (text.includes("delete") || text.includes("impossible") || text.includes("cannot"))
  ) {
    return {
      content: "",
      toolCalls: [
        {
          id: "call_transfer_1",
          name: "transfer_to_human_agents",
          arguments: {
            summary:
              "User needs to delete all their current tasks. This is not possible to do with the tools available.",
          },
        },
      ],
    };
  }

  return undefined;
}

type ChatCompletionResponse = {
  choices?: Array<{
    message?: {
      content?: string | null;
      tool_calls?: Array<{
        id?: string;
        function?: { name?: string; arguments?: string };
      }>;
    };
  }>;
};

function toOpenAIMessages(msgs: Message[]): Array<Record<string, unknown>> {
  return msgs.map((m) => {
    if (m.role === "tool") {
      return {
        role: "tool",
        tool_call_id: m.tool_call_id ?? m.name ?? "tool",
        content: m.content,
      };
    }
    if (m.tool_calls && m.tool_calls.length > 0) {
      return {
        role: m.role,
        content: m.content.length > 0 ? m.content : null,
        tool_calls: m.tool_calls.map((tc) => ({
          id: tc.id,
          type: "function",
          function: { name: tc.name, arguments: JSON.stringify(tc.arguments) },
        })),
      };
    }
    return { role: m.role, content: m.content };
  });
}

function toolsToOpenAI(tools: ToolSpec[]): Array<Record<string, unknown>> {
  return tools.map((t) => ({
    type: "function",
    function: {
      name: t.name,
      description: t.description ?? t.name,
      parameters: t.parameters ?? { type: "object", properties: {} },
    },
  }));
}

function parseToolCalls(
  raw: ChatCompletionResponse["choices"],
): ToolCallOut[] | undefined {
  const calls = raw?.[0]?.message?.tool_calls;
  if (!calls || calls.length === 0) return undefined;
  return calls.map((c, i) => {
    let args: Record<string, unknown> = {};
    const src = c.function?.arguments ?? "{}";
    try {
      const parsed = JSON.parse(src) as unknown;
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        args = parsed as Record<string, unknown>;
      }
    } catch {
      args = { _raw: src };
    }
    return {
      id: c.id && c.id.length > 0 ? c.id : `call_${i}`,
      name: c.function?.name ?? "unknown",
      arguments: args,
    };
  });
}

export class OpenAICompatibleProvider implements Provider {
  name = "openai";
  model: string;

  constructor(
    private readonly apiKey: string,
    private readonly baseUrl: string,
    model: string,
    private readonly viaOpenRouter = false,
  ) {
    this.model = model;
    if (viaOpenRouter) this.name = "openrouter";
  }

  private headers(): Record<string, string> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      Authorization: `Bearer ${this.apiKey}`,
    };
    if (this.viaOpenRouter) {
      headers["HTTP-Referer"] = "https://github.com/keejkrej/vdom-harness";
      headers["X-Title"] = "vdom-harness";
    }
    return headers;
  }

  async complete(msgs: Message[], opts?: CompleteOpts): Promise<string> {
    const turn = await this.completeTurn(msgs, opts);
    return turn.content;
  }

  async completeTurn(msgs: Message[], opts?: CompleteOpts): Promise<Completion> {
    const url = `${this.baseUrl.replace(/\/$/, "")}/chat/completions`;
    const body: Record<string, unknown> = {
      model: opts?.model ?? this.model,
      messages: toOpenAIMessages(msgs),
      temperature: opts?.temperature ?? 0,
    };
    if (opts?.tools && opts.tools.length > 0) {
      body.tools = toolsToOpenAI(opts.tools);
    }
    const res = await fetch(url, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const errBody = await res.text();
      throw new Error(`OpenAI-compatible provider ${res.status}: ${errBody}`);
    }
    const data = (await res.json()) as ChatCompletionResponse;
    const toolCalls = parseToolCalls(data.choices);
    const content = data.choices?.[0]?.message?.content ?? "";
    if (toolCalls && toolCalls.length > 0) {
      return { content: "", toolCalls };
    }
    return { content };
  }
}

export function createProvider(): Provider {
  const cfg = resolveChatConfig();
  if (cfg) {
    return new OpenAICompatibleProvider(
      cfg.apiKey,
      cfg.baseUrl,
      cfg.model,
      cfg.viaOpenRouter,
    );
  }
  return new DeterministicProvider();
}
