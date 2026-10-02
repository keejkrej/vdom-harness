import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport, getDefaultEnvironment } from "@modelcontextprotocol/sdk/client/stdio.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import type { McpServer } from "@agentclientprotocol/sdk";
import type { ToolSchema } from "./llm.js";
import { trace } from "./trace.js";

/** Normalized server spec from ACP session/new or an mcp.json file. */
export type McpSpec =
  | { name: string; transport: "stdio"; command: string; args: string[]; env: Record<string, string>; cwd?: string }
  | { name: string; transport: "http"; url: string; headers: Record<string, string> };

export type McpTool = {
  /** mcp__<server>__<tool> */
  name: string;
  server: string;
  tool: string;
  schema: ToolSchema;
  readOnly: boolean;
};

const NAME_RE = /[^A-Za-z0-9_-]/g;
const CALL_TIMEOUT_MS = 120_000;
const CONNECT_TIMEOUT_MS = 30_000;
/** OpenAI caps function names at 64 chars. */
const MAX_TOOL_NAME = 64;

export function fromAcp(servers: McpServer[] | undefined): McpSpec[] {
  const out: McpSpec[] = [];
  for (const s of servers ?? []) {
    if ("type" in s && s.type === "http") {
      out.push({ name: s.name, transport: "http", url: s.url, headers: Object.fromEntries(s.headers.map((h) => [h.name, h.value])) });
    } else if (!("type" in s) || s.type === undefined) {
      const st = s as { name: string; command: string; args: string[]; env: { name: string; value: string }[] };
      out.push({ name: st.name, transport: "stdio", command: st.command, args: st.args ?? [], env: Object.fromEntries((st.env ?? []).map((e) => [e.name, e.value])) });
    }
    // sse / acp transports are not supported.
  }
  return out;
}

function expand(v: string): string {
  return v.replace(/\$\{([A-Za-z_][A-Za-z0-9_]*)\}/g, (_, k: string) => process.env[k] ?? "");
}

/** `{ "mcpServers": { name: { command, args, env, cwd } | { url, headers } } }` — Claude/Pi/Cursor shape. */
export function fromFile(path: string): McpSpec[] {
  if (!existsSync(path)) return [];
  try {
    const raw = JSON.parse(readFileSync(path, "utf8")) as { mcpServers?: Record<string, Record<string, unknown>> };
    const out: McpSpec[] = [];
    for (const [name, cfg] of Object.entries(raw.mcpServers ?? {})) {
      if (cfg.enabled === false || cfg.disabled === true) continue;
      if (typeof cfg.url === "string") {
        if (cfg.type === "sse") continue;
        const headers = Object.fromEntries(Object.entries((cfg.headers as Record<string, string>) ?? {}).map(([k, v]) => [k, expand(String(v))]));
        out.push({ name, transport: "http", url: expand(cfg.url), headers });
      } else if (typeof cfg.command === "string") {
        const env = Object.fromEntries(Object.entries((cfg.env as Record<string, string>) ?? {}).map(([k, v]) => [k, expand(String(v))]));
        out.push({
          name,
          transport: "stdio",
          command: cfg.command,
          args: Array.isArray(cfg.args) ? cfg.args.map(String) : [],
          env,
          ...(typeof cfg.cwd === "string" ? { cwd: cfg.cwd } : {}),
        });
      }
    }
    return out;
  } catch (err) {
    process.stderr.write(`vdom: ignoring invalid MCP config ${path}: ${String(err)}\n`);
    return [];
  }
}

/** User config, then project config (.vdom/mcp.json, .mcp.json); ACP-provided servers override by name. */
export function collectSpecs(home: string, cwd: string, acp: McpServer[] | undefined): McpSpec[] {
  const byName = new Map<string, McpSpec>();
  for (const s of [...fromFile(join(home, "mcp.json")), ...fromFile(join(cwd, ".vdom", "mcp.json")), ...fromFile(join(cwd, ".mcp.json")), ...fromAcp(acp)]) {
    byName.set(s.name, s);
  }
  return [...byName.values()];
}

type Conn = { spec: McpSpec; client?: Client; tools: McpTool[]; error?: string; instructions?: string };

/** One MCP client set per session. Connections start in the background. */
export class McpHub {
  private conns: Conn[] = [];
  private ready: Promise<void> = Promise.resolve();

  start(specs: McpSpec[], cwd: string): void {
    this.conns = specs.map((spec) => ({ spec, tools: [] }));
    this.ready = Promise.all(this.conns.map((c) => this.connect(c, cwd))).then(() => {});
  }

  /** Wait (bounded) for connections before the first model call. */
  async whenReady(timeoutMs = 15_000): Promise<void> {
    await Promise.race([this.ready, new Promise((r) => setTimeout(r, timeoutMs))]);
  }

  private async connect(c: Conn, cwd: string): Promise<void> {
    const client = new Client({ name: "vdom", version: "0.3.0" });
    try {
      const transport =
        c.spec.transport === "stdio"
          ? new StdioClientTransport({
              command: c.spec.command,
              args: c.spec.args,
              env: { ...getDefaultEnvironment(), ...c.spec.env },
              cwd: c.spec.cwd ?? cwd,
              stderr: "ignore",
            })
          : new StreamableHTTPClientTransport(new URL(c.spec.url), { requestInit: { headers: c.spec.headers } });
      await withTimeout(client.connect(transport), CONNECT_TIMEOUT_MS, `connect ${c.spec.name}`);
      c.client = client;
      c.instructions = client.getInstructions()?.slice(0, 8000);
      const server = c.spec.name.replace(NAME_RE, "_").slice(0, 24);
      let cursor: string | undefined;
      do {
        const page = await withTimeout(client.listTools(cursor ? { cursor } : {}), CONNECT_TIMEOUT_MS, `list tools ${c.spec.name}`);
        for (const t of page.tools) {
          const name = `mcp__${server}__${t.name.replace(NAME_RE, "_")}`.slice(0, MAX_TOOL_NAME);
          c.tools.push({
            name,
            server: c.spec.name,
            tool: t.name,
            readOnly: t.annotations?.readOnlyHint === true,
            schema: {
              name,
              description: `[MCP ${c.spec.name}] ${t.description ?? t.name}`.slice(0, 1024),
              parameters: (t.inputSchema as Record<string, unknown>) ?? { type: "object", properties: {} },
            },
          });
        }
        cursor = page.nextCursor;
      } while (cursor);
    } catch (err) {
      c.error = err instanceof Error ? err.message : String(err);
      process.stderr.write(`vdom: MCP server ${c.spec.name} unavailable: ${c.error}\n`);
      trace("mcp_error", { server: c.spec.name, spec: c.spec, error: c.error });
      await client.close().catch(() => {});
    }
  }

  tools(): McpTool[] {
    return this.conns.flatMap((c) => c.tools);
  }

  find(name: string): McpTool | undefined {
    return this.tools().find((t) => t.name === name);
  }

  /** Prompt section: connected servers, failures, and their instructions. */
  describe(): string {
    if (this.conns.length === 0) return "";
    return this.conns
      .map((c) => {
        const head = `- ${c.spec.name}: ${c.error ? `unavailable (${c.error})` : `${c.tools.length} tools`}`;
        return c.instructions ? `${head}\n  Instructions: ${c.instructions.replace(/\n/g, "\n  ")}` : head;
      })
      .join("\n");
  }

  async call(t: McpTool, args: Record<string, unknown>, signal: AbortSignal): Promise<{ text: string; isError: boolean; images: { mimeType: string; data: string }[] }> {
    const c = this.conns.find((x) => x.spec.name === t.server);
    if (!c?.client) throw new Error(`MCP server ${t.server} is not connected`);
    const res = await c.client.callTool({ name: t.tool, arguments: args }, undefined, { signal, timeout: CALL_TIMEOUT_MS });
    const parts: string[] = [];
    const images: { mimeType: string; data: string }[] = [];
    for (const item of (res.content as { type: string; text?: string; data?: string; mimeType?: string; resource?: { uri?: string; text?: string } }[]) ?? []) {
      if (item.type === "text" && item.text !== undefined) parts.push(item.text);
      else if (item.type === "image" && item.data && item.mimeType) images.push({ mimeType: item.mimeType, data: item.data });
      else if (item.type === "resource" && item.resource) parts.push(item.resource.text ?? `[resource ${item.resource.uri ?? ""}]`);
      else parts.push(`[${item.type} content]`);
    }
    if (parts.length === 0 && res.structuredContent) parts.push(JSON.stringify(res.structuredContent, null, 2));
    return { text: parts.join("\n") || "(no output)", isError: res.isError === true, images };
  }

  async close(): Promise<void> {
    await Promise.all(this.conns.map((c) => c.client?.close().catch(() => {})));
    this.conns = [];
  }
}

function withTimeout<T>(p: Promise<T>, ms: number, what: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error(`${what} timed out after ${ms}ms`)), ms);
    p.then(
      (v) => {
        clearTimeout(t);
        resolve(v);
      },
      (e) => {
        clearTimeout(t);
        reject(e);
      },
    );
  });
}
