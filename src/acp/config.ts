import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { parseLadder } from "./routing.js";

/**
 * Coding-agent settings. Precedence: CLI flags > env > ~/.vdom/config.json > defaults.
 * One OpenAI-compatible endpoint; the model is picked per session.
 */
export type AgentConfig = {
  baseUrl: string;
  apiKey?: string;
  /** Default model id for new sessions. */
  model: string;
  /** Pinned model list. Empty → discovered from `${baseUrl}/models`. */
  models: string[];
  /** Where sessions are persisted for session/load and session/resume. */
  dataDir: string;
  /** Max model calls per prompt turn before stopReason=max_turn_requests. */
  maxSteps: number;
  /** Approximate context budget in tokens; older tool output is elided past this. */
  contextTokens: number;
  /** Shell used by the `bash` tool. */
  shell: string;
  /** Default timeout for the `bash` tool. */
  shellTimeoutMs: number;
  /** Process-wide: never ask permission (cursor `--force`, grok `--always-approve`). */
  fullAccess: boolean;
  /** Extra request headers (e.g. OpenRouter attribution). */
  headers: Record<string, string>;
  temperature?: number;
  providerName: string;
  /** Small model that reads each user message for frustration (undefined = off). */
  sentimentModel?: string;
  /** Model for the background diagnostician (default: the session model). */
  diagnosisModel?: string;
  /** Runtime guards on by default for every session (e.g. "verify_claims"). */
  guards: string[];
  /** Cheap-first model ladder, strongest last. Empty = routing off. */
  routeLadder: string[];
};

export const OLLAMA_CLOUD_BASE_URL = "https://ollama.com/v1";
export const OLLAMA_CLOUD_DEFAULT_MODEL = "gpt-oss:120b";

type FileConfig = Partial<Omit<AgentConfig, "headers">> & {
  /** Read the API key from this env var instead of storing it in the file. */
  apiKeyEnv?: string;
  headers?: Record<string, string>;
};

export function vdomHome(): string {
  return process.env.VDOM_HOME?.trim() || join(homedir(), ".vdom");
}

function readFileConfig(path: string): FileConfig {
  if (!existsSync(path)) return {};
  try {
    const parsed = JSON.parse(readFileSync(path, "utf8")) as unknown;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return parsed as FileConfig;
  } catch (err) {
    process.stderr.write(`vdom: ignoring unreadable config ${path}: ${String(err)}\n`);
  }
  return {};
}

function env(name: string): string | undefined {
  const v = process.env[name]?.trim();
  return v && v.length > 0 ? v : undefined;
}

function defaultShell(): string {
  if (env("VDOM_SHELL")) return env("VDOM_SHELL")!;
  if (process.platform !== "win32") return env("SHELL") ?? "/bin/bash";
  // Git Bash, never System32\bash.exe (that is WSL and sees a different filesystem).
  for (const p of [
    join(process.env.ProgramFiles ?? "C:\\Program Files", "Git", "bin", "bash.exe"),
    join(process.env.LOCALAPPDATA ?? "", "Programs", "Git", "bin", "bash.exe"),
  ]) {
    if (existsSync(p)) return p;
  }
  return "powershell.exe";
}

/** Infer endpoint + key from whichever provider key is present. */
function endpointFromEnv(): { baseUrl?: string; apiKey?: string; model?: string; providerName?: string } {
  if (env("VDOM_BASE_URL") || env("VDOM_API_KEY")) {
    return { baseUrl: env("VDOM_BASE_URL"), apiKey: env("VDOM_API_KEY"), providerName: "custom" };
  }
  if (env("OLLAMA_API_KEY")) {
    return {
      baseUrl: OLLAMA_CLOUD_BASE_URL,
      apiKey: env("OLLAMA_API_KEY"),
      model: OLLAMA_CLOUD_DEFAULT_MODEL,
      providerName: "ollama-cloud",
    };
  }
  if (env("OPENROUTER_API_KEY")) {
    return {
      baseUrl: "https://openrouter.ai/api/v1",
      apiKey: env("OPENROUTER_API_KEY"),
      model: "deepseek/deepseek-v4-flash-0731",
      providerName: "openrouter",
    };
  }
  if (env("OPENAI_API_KEY")) {
    return {
      baseUrl: env("OPENAI_BASE_URL") ?? "https://api.openai.com/v1",
      apiKey: env("OPENAI_API_KEY"),
      providerName: "openai",
    };
  }
  return {};
}

export type ConfigOverrides = Partial<AgentConfig> & { configPath?: string };

export function loadConfig(overrides: ConfigOverrides = {}): AgentConfig {
  const home = vdomHome();
  const file = readFileConfig(overrides.configPath ?? env("VDOM_CONFIG") ?? join(home, "config.json"));
  // Precedence: CLI > VDOM_* env > ~/.vdom/config.json > generic provider keys (OLLAMA_API_KEY, …) > local Ollama.
  const fromEnv = endpointFromEnv();
  const vdomEnv = fromEnv.providerName === "custom";
  const fileKey = (file.apiKeyEnv ? env(file.apiKeyEnv) : undefined) ?? file.apiKey;
  const fileWins = !vdomEnv && Boolean(file.baseUrl || fileKey);

  const baseUrl = (
    overrides.baseUrl ??
    (fileWins ? file.baseUrl : undefined) ??
    fromEnv.baseUrl ??
    file.baseUrl ??
    // Local Ollama needs no key.
    "http://127.0.0.1:11434/v1"
  ).replace(/\/+$/, "");
  const apiKey = overrides.apiKey ?? (vdomEnv ? fromEnv.apiKey : undefined) ?? fileKey ?? (fromEnv.baseUrl === baseUrl ? fromEnv.apiKey : undefined);
  const model =
    overrides.model ?? env("VDOM_MODEL") ?? file.model ?? (fileWins ? undefined : fromEnv.model) ?? (baseUrl.includes("ollama.com") ? OLLAMA_CLOUD_DEFAULT_MODEL : "gpt-oss:20b");
  const models = overrides.models ?? (env("VDOM_MODELS")?.split(",").map((s) => s.trim()).filter(Boolean)) ?? file.models ?? [];
  const headers: Record<string, string> = { ...(file.headers ?? {}) };
  if (baseUrl.includes("openrouter.ai")) {
    headers["HTTP-Referer"] ??= "https://github.com/keejkrej/vdom-harness";
    headers["X-Title"] ??= "vdom";
  }

  return {
    baseUrl,
    apiKey,
    model,
    models,
    dataDir: overrides.dataDir ?? file.dataDir ?? join(home, "sessions"),
    maxSteps: overrides.maxSteps ?? numEnv("VDOM_MAX_STEPS") ?? file.maxSteps ?? 200,
    contextTokens: overrides.contextTokens ?? numEnv("VDOM_CONTEXT_TOKENS") ?? file.contextTokens ?? 120_000,
    shell: overrides.shell ?? file.shell ?? defaultShell(),
    shellTimeoutMs: overrides.shellTimeoutMs ?? file.shellTimeoutMs ?? 120_000,
    fullAccess: overrides.fullAccess ?? file.fullAccess ?? false,
    headers,
    temperature: overrides.temperature ?? file.temperature,
    providerName:
      overrides.providerName ??
      (fileWins ? undefined : fromEnv.providerName) ??
      file.providerName ??
      (baseUrl.includes("ollama.com") ? "ollama-cloud" : baseUrl.includes("11434") ? "ollama" : "custom"),
    // Real-time sentiment interpreter: a small fast model; "off" disables it.
    sentimentModel: (() => {
      const v = overrides.sentimentModel ?? env("VDOM_SENTIMENT_MODEL") ?? file.sentimentModel;
      if (v === "off") return undefined;
      return v ?? (baseUrl.includes("ollama.com") ? "gpt-oss:20b" : model);
    })(),
    diagnosisModel: overrides.diagnosisModel ?? env("VDOM_DIAGNOSIS_MODEL") ?? file.diagnosisModel,
    guards: overrides.guards ?? file.guards ?? ["verify_claims"],
    routeLadder: (() => {
      const v =
        overrides.routeLadder ??
        (env("VDOM_ROUTING") === "off" ? [] : undefined) ??
        (env("VDOM_ROUTE_LADDER") ? parseLadder(env("VDOM_ROUTE_LADDER")!) : undefined) ??
        (env("VDOM_ROUTING") === "off" ? [] : file.routeLadder ?? []);
      return v;
    })(),
  };
}

function numEnv(name: string): number | undefined {
  const v = env(name);
  if (!v) return undefined;
  const n = Number(v);
  return Number.isFinite(n) && n > 0 ? n : undefined;
}
