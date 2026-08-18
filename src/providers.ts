import { type AgentNode, modelId } from "./ir.js";

export type Message = {
  role: "system" | "user" | "assistant";
  content: string;
};

export type CompleteOpts = {
  role?: string;
  temperature?: number;
  /** Override the provider's default model for this call (OpenAI-compatible). */
  model?: string;
};

export type Provider = {
  name: string;
  /** Bound model id when this provider was resolved for a specific AgentNode.model. */
  model?: string;
  complete(msgs: Message[], opts?: CompleteOpts): Promise<string>;
};

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
  const key = process.env.OPENAI_API_KEY;
  if (key) {
    return new OpenAICompatibleProvider(
      key,
      process.env.OPENAI_BASE_URL ?? "https://api.openai.com/v1",
      id,
    );
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
}

type ChatCompletionResponse = {
  choices?: Array<{ message?: { content?: string | null } }>;
};

export class OpenAICompatibleProvider implements Provider {
  name = "openai";
  model: string;

  constructor(
    private readonly apiKey: string,
    private readonly baseUrl: string,
    model: string,
  ) {
    this.model = model;
  }

  async complete(msgs: Message[], opts?: CompleteOpts): Promise<string> {
    const url = `${this.baseUrl.replace(/\/$/, "")}/chat/completions`;
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: opts?.model ?? this.model,
        messages: msgs,
        temperature: opts?.temperature ?? 0,
      }),
    });
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`OpenAI-compatible provider ${res.status}: ${body}`);
    }
    const data = (await res.json()) as ChatCompletionResponse;
    return data.choices?.[0]?.message?.content ?? "";
  }
}

export function createProvider(): Provider {
  const key = process.env.OPENAI_API_KEY;
  if (key) {
    return new OpenAICompatibleProvider(
      key,
      process.env.OPENAI_BASE_URL ?? "https://api.openai.com/v1",
      process.env.OPENAI_MODEL ?? "gpt-4o-mini",
    );
  }
  return new DeterministicProvider();
}
