import { cloneGraph, flatten, type AgentGraph, type AgentNode } from "../ir.js";
import { reconcile, formatOps } from "../reconciler.js";

/**
 * The coding agent is an AgentGraph like every other vdom society. Its system
 * prompt is compiled from the graph; the agent may rewrite the graph mid-session
 * via set_agent_graph and the reconciler reports the diff. Host code is never
 * patched — only prompt topology changes.
 */
export function codingGraph(): AgentGraph {
  return {
    id: "vdom-coder",
    version: 1,
    root: {
      key: "coder",
      kind: "agent",
      role: "coder",
      objective:
        "Complete the user's software engineering task in their repository: understand the code, make correct minimal changes, verify them, and report plainly.",
      persistence: "session",
      children: [
        {
          key: "explore",
          kind: "policy",
          role: "explore",
          objective: "Ground every change in the actual code.",
          prompt:
            "Before editing, locate the relevant code with grep/find and read it. Read a file before you edit it. Match the surrounding style, naming, and comment density. Check for an AGENTS.md / CLAUDE.md / CONTRIBUTING.md and follow it.",
        },
        {
          key: "edit",
          kind: "policy",
          role: "edit",
          objective: "Make focused, reviewable changes.",
          prompt:
            "Use edit for targeted changes (several disjoint edits in one call) and write only for new files or full rewrites. Do not reformat unrelated code. Do not add files the task does not need. Never print or commit secrets.",
        },
        {
          key: "verify",
          kind: "policy",
          role: "verify",
          objective: "Prove the change works.",
          prompt:
            "Run the project's own type-check, lint, and tests for what you touched (look at package.json scripts, Makefile, pyproject, etc.). If something fails, fix it or say exactly what failed. Never claim success you did not observe.",
        },
        {
          key: "git",
          kind: "policy",
          role: "git",
          objective: "Treat version control as the user's.",
          prompt:
            "Do not commit, push, rebase, reset, or open pull requests unless the user asks. When asked: branch off the default branch first, write a concise commit message describing why, and use `gh` for GitHub. Never force-push to the default branch. Never skip hooks.",
        },
        {
          key: "report",
          kind: "policy",
          role: "report",
          objective: "Communicate like a senior engineer.",
          prompt:
            "Be concise. Use todo_write for multi-step tasks. When done, summarize what changed (with file paths), how it was verified, and anything left undone. Ask only when a decision is genuinely the user's; otherwise pick the sensible default and say so.",
        },
      ],
    },
  };
}

function renderNode(n: AgentNode, depth: number): string {
  const head = `${"#".repeat(Math.min(6, depth + 2))} ${n.role} (${n.key})`;
  const body = [n.objective, n.prompt].filter(Boolean).join("\n");
  const kids = (n.children ?? []).map((c) => renderNode(c, depth + 1));
  return [head, body, ...kids].join("\n\n");
}

export type PromptEnv = {
  cwd: string;
  roots: string[];
  platform: string;
  shell: string;
  model: string;
  mode: string;
  date: string;
  gitBranch?: string;
};

export type PromptResources = {
  /** SYSTEM.md: replaces the preamble and the tool/rule sections. */
  system?: string;
  /** APPEND_SYSTEM.md */
  append?: string;
  contextFiles: { path: string; content: string }[];
  skills: string;
  tools: { name: string; snippet: string; guidelines?: string[] }[];
  mcp: string;
  /** Where the running harness's own source lives (for self-debugging). */
  self?: string;
};

const PREAMBLE =
  "You are vdom, an expert coding agent working in the user's repository through tools. An editor drives you over the Agent Client Protocol: the user sees your messages, tool calls, diffs, and plan. You help by reading files, running commands, editing code, and writing new files — and you finish the task end to end.";

const PLAN_MODE = `You are in plan mode (read-only). Imperative requests to implement changes mean: plan the implementation, do not execute it.
Explore first with non-mutating reads, searches, and checks. Do not edit or write files, change configuration, run formatters or code generation that rewrite tracked files, commit, or install packages.
Finish with a decision-complete plan: the files to change and how, the order of steps, how to verify, and open questions that only the user can answer.`;

function section(name: string, body: string): string {
  return body.trim() ? `<${name}>\n${body.trim()}\n</${name}>` : "";
}

const posix = (p: string) => p.replace(/\\/g, "/");

export function compileSystemPrompt(g: AgentGraph, env: PromptEnv, res: PromptResources): string {
  const root = g.root;
  const parts: string[] = [];
  if (res.system) {
    parts.push(res.system.trim());
  } else {
    parts.push(PREAMBLE);
    parts.push(
      section(
        "tools",
        `${res.tools.map((t) => `- ${t.name}: ${t.snippet}`).join("\n")}\nOther tools may be available (e.g. from MCP servers); their descriptions say what they do.`,
      ),
    );
    const rules = [
      ...new Set([
        ...res.tools.flatMap((t) => t.guidelines ?? []),
        "Call independent read-only tools in parallel in a single response.",
        "If the user declines a tool call, do not retry it; adjust or ask.",
        "Be concise in your responses",
        "Show file paths clearly when working with files",
      ]),
    ];
    parts.push(section("rules", rules.map((r) => `- ${r}`).join("\n")));
  }
  const graphRules = (root.children ?? []).map((c) => renderNode(c, 0)).join("\n\n");
  parts.push(section("working_rules", `AgentGraph ${g.id} v${g.version}. Objective: ${root.objective}${root.prompt ? `\n${root.prompt}` : ""}\n\n${graphRules}`));
  if (env.mode === "plan") parts.push(section("plan_mode", PLAN_MODE));
  if (res.append) parts.push(section("addendum", res.append));
  if (res.contextFiles.length) {
    const files = res.contextFiles.map((f) => `<project_instructions path="${posix(f.path)}">\n${f.content.trim()}\n</project_instructions>`).join("\n\n");
    parts.push(section("project_context", `Project-specific instructions and guidelines (more specific files come later and take precedence):\n\n${files}`));
  }
  if (res.skills) parts.push(section("skills", res.skills));
  if (res.mcp) parts.push(section("mcp_servers", res.mcp));
  if (res.self) parts.push(section("self", res.self));
  const envLines = [
    `Working directory: ${posix(env.cwd)}`,
    ...(env.roots.length > 1 ? [`Additional directories: ${env.roots.slice(1).map(posix).join(", ")}`] : []),
    `Platform: ${env.platform}`,
    `Shell (bash tool): ${env.shell}`,
    `Model: ${env.model}`,
    `Mode: ${env.mode}`,
    `Date: ${env.date}`,
    ...(env.gitBranch ? [`Git branch: ${env.gitBranch}`] : []),
  ];
  parts.push(section("environment", envLines.join("\n")));
  return parts.filter(Boolean).join("\n\n");
}

export function describeGraph(g: AgentGraph): string {
  return JSON.stringify(g, null, 2);
}

/** Validate an untrusted graph from the model and reconcile it against the live one. */
export function applyGraph(prev: AgentGraph, raw: unknown): { ok: true; graph: AgentGraph; diff: string } | { ok: false; reason: string } {
  let value = raw;
  if (typeof value === "string") {
    try {
      value = JSON.parse(value);
    } catch {
      return { ok: false, reason: "graph must be a JSON object" };
    }
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) return { ok: false, reason: "graph must be an object" };
  const obj = value as Partial<AgentGraph>;
  if (!obj.root || typeof obj.root !== "object") return { ok: false, reason: "graph.root is required" };
  const next: AgentGraph = {
    id: typeof obj.id === "string" ? obj.id : prev.id,
    version: prev.version + 1,
    root: obj.root as AgentNode,
    meta: obj.meta,
  };
  if (next.root.key !== "coder") return { ok: false, reason: "root key must stay `coder`" };
  const seen = new Set<string>();
  for (const { node } of flatten(next)) {
    if (typeof node.key !== "string" || !node.key) return { ok: false, reason: "every node needs a string key" };
    if (seen.has(node.key)) return { ok: false, reason: `duplicate key ${node.key}` };
    seen.add(node.key);
    if (typeof node.role !== "string") node.role = node.key;
    if (typeof node.objective !== "string") node.objective = "";
    // Executable kinds go through the capability/adapter gates, never through chat.
    if (node.kind === "capability" || node.kind === "adapter" || node.kind === "artifact") {
      return { ok: false, reason: `node ${node.key}: kind ${node.kind} cannot be mounted from set_agent_graph` };
    }
  }
  if (JSON.stringify(next.root).length > 40_000) return { ok: false, reason: "graph too large (40k chars max)" };
  const { ops } = reconcile(prev, next);
  const changed = ops.filter((o) => o.op !== "retain");
  return { ok: true, graph: cloneGraph(next), diff: changed.length ? formatOps(changed) : "  (no changes)" };
}
