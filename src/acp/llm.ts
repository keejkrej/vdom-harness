import type { AgentConfig } from "./config.js";

/** OpenAI chat message, stored verbatim in session history. */
export type ChatPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } };

export type ChatToolCall = {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
};

export type ChatMessage =
  | { role: "system"; content: string }
  | { role: "user"; content: string | ChatPart[] }
  | { role: "assistant"; content: string | null; tool_calls?: ChatToolCall[]; reasoning_content?: string }
  | { role: "tool"; tool_call_id: string; content: string };

export type ToolSchema = {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
};

export type StreamHandlers = {
  onText?(delta: string): void | Promise<void>;
  onReasoning?(delta: string): void | Promise<void>;
};

export type Usage = { promptTokens: number; completionTokens: number; totalTokens: number };

export type TurnResult = {
  content: string;
  reasoning: string;
  toolCalls: ChatToolCall[];
  finishReason: string | null;
  usage?: Usage;
  model?: string;
};

export class LlmError extends Error {
  constructor(
    message: string,
    readonly status?: number,
    readonly retryable = false,
  ) {
    super(message);
  }
}

const RETRY_STATUSES = new Set([408, 409, 425, 429, 500, 502, 503, 504, 529]);
const MAX_ATTEMPTS = 5;
/** Abort a stream that has produced no bytes for this long. */
const IDLE_TIMEOUT_MS = 5 * 60_000;

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(signal.reason);
    const t = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(t);
        reject(signal.reason);
      },
      { once: true },
    );
  });
}

function headers(cfg: AgentConfig): Record<string, string> {
  const h: Record<string, string> = { "Content-Type": "application/json", ...cfg.headers };
  if (cfg.apiKey) h.Authorization = `Bearer ${cfg.apiKey}`;
  return h;
}

/** GET /models. Returns [] on any failure — discovery is best effort. */
export async function listModels(cfg: AgentConfig, timeoutMs = 8000): Promise<string[]> {
  try {
    const res = await fetch(`${cfg.baseUrl}/models`, {
      headers: headers(cfg),
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!res.ok) return [];
    const data = (await res.json()) as { data?: { id?: unknown }[]; models?: { name?: unknown }[] };
    const ids = (data.data ?? []).map((m) => m.id).concat((data.models ?? []).map((m) => m.name));
    return [...new Set(ids.filter((id): id is string => typeof id === "string" && id.length > 0))].sort();
  } catch {
    return [];
  }
}

type StreamChunk = {
  model?: string;
  choices?: {
    delta?: {
      content?: string | null;
      reasoning?: string | null;
      reasoning_content?: string | null;
      tool_calls?: {
        index?: number;
        id?: string;
        function?: { name?: string; arguments?: string };
      }[];
    };
    finish_reason?: string | null;
  }[];
  usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number } | null;
  error?: { message?: string } | string;
};

/**
 * One streamed chat completion. Retries transient failures only before any
 * delta reached the caller, so a retry never duplicates visible output.
 */
export type ThoughtLevel = "off" | "low" | "medium" | "high";
export const THOUGHT_LEVELS: ThoughtLevel[] = ["off", "low", "medium", "high"];

/** Models that rejected a reasoning parameter; we stop sending it to them. */
const noReasoning = new Set<string>();

function reasoningParams(cfg: AgentConfig, model: string, level: ThoughtLevel | undefined): Record<string, unknown> {
  if (!level || noReasoning.has(model)) return {};
  if (cfg.baseUrl.includes("openrouter.ai")) return { reasoning: { effort: level === "off" ? "none" : level } };
  return { reasoning_effort: level === "off" ? "none" : level };
}

export async function streamChat(
  cfg: AgentConfig,
  req: { model: string; messages: ChatMessage[]; tools: ToolSchema[]; signal: AbortSignal; thought?: ThoughtLevel },
  handlers: StreamHandlers = {},
): Promise<TurnResult> {
  const body: Record<string, unknown> = {
    model: req.model,
    messages: req.messages,
    stream: true,
    stream_options: { include_usage: true },
    ...reasoningParams(cfg, req.model, req.thought),
  };
  if (cfg.temperature !== undefined) body.temperature = cfg.temperature;
  if (req.tools.length > 0) {
    body.tools = req.tools.map((t) => ({ type: "function", function: t }));
    body.tool_choice = "auto";
  }

  let lastErr: unknown;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    let emitted = false;
    try {
      return await streamOnce(cfg, body, req.signal, handlers, () => {
        emitted = true;
      });
    } catch (err) {
      if (req.signal.aborted) throw err;
      lastErr = err;
      // Provider rejected the reasoning knob: drop it for this model and retry immediately.
      if (err instanceof LlmError && err.status === 400 && /reason|effort|think/i.test(err.message) && ("reasoning_effort" in body || "reasoning" in body)) {
        noReasoning.add(req.model);
        delete body.reasoning_effort;
        delete body.reasoning;
        attempt--;
        continue;
      }
      const retryable = err instanceof LlmError ? err.retryable : true; // network errors
      if (!retryable || emitted || attempt === MAX_ATTEMPTS) throw err;
      const delay = Math.min(30_000, 1000 * 2 ** (attempt - 1)) + Math.floor(Math.random() * 250);
      await sleep(delay, req.signal);
    }
  }
  throw lastErr;
}

async function streamOnce(
  cfg: AgentConfig,
  body: Record<string, unknown>,
  signal: AbortSignal,
  handlers: StreamHandlers,
  markEmitted: () => void,
): Promise<TurnResult> {
  const idle = new AbortController();
  let idleTimer = setTimeout(() => idle.abort(new LlmError("model stream idle timeout", undefined, true)), IDLE_TIMEOUT_MS);
  const bump = () => {
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => idle.abort(new LlmError("model stream idle timeout", undefined, true)), IDLE_TIMEOUT_MS);
  };

  try {
    const res = await fetch(`${cfg.baseUrl}/chat/completions`, {
      method: "POST",
      headers: headers(cfg),
      body: JSON.stringify(body),
      signal: AbortSignal.any([signal, idle.signal]),
    });
    if (!res.ok || !res.body) {
      const text = await res.text().catch(() => "");
      throw new LlmError(
        `${cfg.providerName} ${res.status} ${res.statusText}: ${text.slice(0, 2000)}`,
        res.status,
        RETRY_STATUSES.has(res.status),
      );
    }

    const out: TurnResult = { content: "", reasoning: "", toolCalls: [], finishReason: null };
    const calls = new Map<number, ChatToolCall>();
    const decoder = new TextDecoder();
    let buf = "";

    const handleData = async (data: string) => {
      if (data === "[DONE]") return;
      let chunk: StreamChunk;
      try {
        chunk = JSON.parse(data) as StreamChunk;
      } catch {
        return;
      }
      if (chunk.error) {
        const msg = typeof chunk.error === "string" ? chunk.error : (chunk.error.message ?? "unknown error");
        throw new LlmError(`${cfg.providerName} stream error: ${msg}`, undefined, false);
      }
      if (chunk.model) out.model = chunk.model;
      if (chunk.usage) {
        out.usage = {
          promptTokens: chunk.usage.prompt_tokens ?? 0,
          completionTokens: chunk.usage.completion_tokens ?? 0,
          totalTokens: chunk.usage.total_tokens ?? (chunk.usage.prompt_tokens ?? 0) + (chunk.usage.completion_tokens ?? 0),
        };
      }
      const choice = chunk.choices?.[0];
      if (!choice) return;
      if (choice.finish_reason) out.finishReason = choice.finish_reason;
      const d = choice.delta;
      if (!d) return;
      const reasoning = d.reasoning_content ?? d.reasoning;
      if (reasoning) {
        markEmitted();
        out.reasoning += reasoning;
        await handlers.onReasoning?.(reasoning);
      }
      if (d.content) {
        markEmitted();
        out.content += d.content;
        await handlers.onText?.(d.content);
      }
      for (const tc of d.tool_calls ?? []) {
        const idx = tc.index ?? calls.size;
        let call = calls.get(idx);
        if (!call) {
          call = { id: tc.id ?? "", type: "function", function: { name: "", arguments: "" } };
          calls.set(idx, call);
        }
        if (tc.id) call.id = tc.id;
        if (tc.function?.name) call.function.name += tc.function.name;
        if (tc.function?.arguments) call.function.arguments += tc.function.arguments;
      }
    };

    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      bump();
      buf += decoder.decode(value, { stream: true });
      let nl: number;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).replace(/\r$/, "");
        buf = buf.slice(nl + 1);
        if (line.startsWith("data:")) await handleData(line.slice(5).trim());
      }
    }
    if (buf.startsWith("data:")) await handleData(buf.slice(5).trim());

    out.toolCalls = [...calls.entries()]
      .sort(([a], [b]) => a - b)
      .map(([i, c]) => ({ ...c, id: c.id || `call_${Date.now().toString(36)}_${i}` }))
      .filter((c) => c.function.name.length > 0);
    return out;
  } catch (err) {
    if (idle.signal.aborted && !signal.aborted) throw idle.signal.reason;
    throw err;
  } finally {
    clearTimeout(idleTimer);
  }
}
