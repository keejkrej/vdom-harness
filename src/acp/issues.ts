import { randomBytes } from "node:crypto";
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { vdomHome } from "./config.js";

/** Harness issues filed by the diagnostician; input to `vdom fix`. */
export type Issue = {
  id: string;
  createdAt: string;
  status: "open" | "fixing" | "fixed" | "wontfix" | "duplicate";
  sessionId: string;
  sessionDir: string;
  turn: number;
  /** What triggered diagnosis. */
  signal: { source: "keyword" | "interpreter" | "cancel" | "reject" | "fault" | "manual"; text?: string; detail?: string };
  title: string;
  /** harness = a vdom code/prompt/tool change would prevent it. */
  blame: "harness" | "model" | "env" | "unclear";
  category: string;
  whatHappened: string;
  rootCause: string;
  proposedFix: string;
  /** Harness files the fix likely touches. */
  files: string[];
  /** How to reproduce (prompt, fixture, or test idea). */
  repro: string;
  evidence: string[];
  severity: "high" | "medium" | "low";
};

export const issuesDir = (): string => join(vdomHome(), "issues");

export function fileIssue(i: Omit<Issue, "id" | "createdAt" | "status">): Issue {
  const d = new Date();
  const id = `I-${d.toISOString().slice(0, 10).replace(/-/g, "")}-${randomBytes(2).toString("hex")}`;
  const issue: Issue = { id, createdAt: d.toISOString(), status: "open", ...i };
  mkdirSync(issuesDir(), { recursive: true });
  writeFileSync(join(issuesDir(), `${id}.json`), JSON.stringify(issue, null, 2));
  return issue;
}

export function loadIssue(id: string): Issue | undefined {
  const p = join(issuesDir(), `${id}.json`);
  return existsSync(p) ? (JSON.parse(readFileSync(p, "utf8")) as Issue) : undefined;
}

export function saveIssue(i: Issue): void {
  writeFileSync(join(issuesDir(), `${i.id}.json`), JSON.stringify(i, null, 2));
}

export function listIssues(): Issue[] {
  if (!existsSync(issuesDir())) return [];
  return readdirSync(issuesDir())
    .filter((f) => f.endsWith(".json"))
    .map((f) => {
      try {
        return JSON.parse(readFileSync(join(issuesDir(), f), "utf8")) as Issue;
      } catch {
        return undefined;
      }
    })
    .filter((x): x is Issue => Boolean(x))
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

export function renderIssue(i: Issue): string {
  return [
    `# ${i.id} — ${i.title}`,
    "",
    `status ${i.status} · ${i.severity} · blame ${i.blame} · ${i.category} · session ${i.sessionId} turn ${i.turn}`,
    `signal: ${i.signal.source}${i.signal.text ? ` "${i.signal.text}"` : ""}${i.signal.detail ? ` — ${i.signal.detail}` : ""}`,
    "",
    "## What happened",
    i.whatHappened,
    "",
    "## Root cause",
    i.rootCause,
    "",
    "## Proposed harness fix",
    i.proposedFix,
    i.files.length ? `\nFiles: ${i.files.join(", ")}` : "",
    "",
    "## Reproduction",
    i.repro,
    "",
    `Evidence: ${i.evidence.join(", ")} (in ${i.sessionDir}/events.jsonl)`,
  ].join("\n");
}
