import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { basename, dirname, join, parse, resolve } from "node:path";

/**
 * Project/user resources, Pi/DSH-compatible:
 * - context files: ~/.vdom/AGENTS.md, then filesystem root → cwd
 * - SYSTEM.md (replaces the preamble) / APPEND_SYSTEM.md
 * - skills (SKILL.md) and prompt templates (/name args)
 */

export type ContextFile = { path: string; content: string };
export type Skill = { name: string; description: string; path: string; baseDir: string; modelInvocable: boolean };
export type Template = { name: string; description: string; argumentHint?: string; body: string; path: string };

const CONTEXT_NAMES = ["AGENTS.override.md", "AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD"];
const LOCAL_NAMES = ["AGENTS.local.md", "CLAUDE.local.md"];
const MAX_CONTEXT_BYTES = 64 * 1024;
const MAX_FILE_BYTES = 1024 * 1024;

function readText(p: string): string | undefined {
  try {
    const st = statSync(p);
    if (!st.isFile() || st.size > MAX_FILE_BYTES) return undefined;
    return readFileSync(p, "utf8").replace(/^﻿/, "");
  } catch {
    return undefined;
  }
}

/** cwd, its parent, …, filesystem root. */
function ancestors(cwd: string): string[] {
  const out: string[] = [];
  let d = resolve(cwd);
  for (;;) {
    out.push(d);
    const up = dirname(d);
    if (up === d) return out;
    d = up;
  }
}

function repoRoot(cwd: string): string | undefined {
  return ancestors(cwd).find((d) => existsSync(join(d, ".git")));
}

export function loadContextFiles(home: string, cwd: string): ContextFile[] {
  const files: ContextFile[] = [];
  const seen = new Set<string>();
  const seenContent = new Set<string>();
  const add = (p: string) => {
    const key = process.platform === "win32" ? p.toLowerCase() : p;
    if (seen.has(key)) return;
    const content = readText(p);
    if (content === undefined || !content.trim()) return;
    seen.add(key);
    if (seenContent.has(content.trim())) return;
    seenContent.add(content.trim());
    files.push({ path: p, content });
  };
  for (const n of ["AGENTS.md", "CLAUDE.md"]) {
    const p = join(home, n);
    if (existsSync(p)) {
      add(p);
      break;
    }
  }
  for (const dir of ancestors(cwd).reverse()) {
    const main = CONTEXT_NAMES.map((n) => join(dir, n)).find((p) => existsSync(p));
    if (main) add(main);
    for (const n of LOCAL_NAMES) if (existsSync(join(dir, n))) add(join(dir, n));
  }
  // Budget: drop broadest files first, then truncate the most specific.
  let total = files.reduce((n, f) => n + Buffer.byteLength(f.content), 0);
  while (total > MAX_CONTEXT_BYTES && files.length > 1) total -= Buffer.byteLength(files.shift()!.content);
  if (total > MAX_CONTEXT_BYTES && files[0]) files[0].content = `${files[0].content.slice(0, MAX_CONTEXT_BYTES)}\n[truncated]`;
  return files;
}

/** Project-level file wins over the user-level one. */
export function loadPromptFile(home: string, cwd: string, name: "SYSTEM.md" | "APPEND_SYSTEM.md"): string | undefined {
  return readText(join(cwd, ".vdom", name)) ?? readText(join(home, name));
}

export function parseFrontmatter(src: string): { meta: Record<string, string>; body: string } {
  const m = src.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
  if (!m) return { meta: {}, body: src };
  const meta: Record<string, string> = {};
  let key: string | undefined;
  for (const line of m[1]!.split(/\r?\n/)) {
    const kv = line.match(/^([A-Za-z0-9_-]+):\s*(.*)$/);
    if (kv) {
      key = kv[1]!;
      meta[key] = kv[2]!.trim().replace(/^["']|["']$/g, "");
    } else if (key && /^\s+\S/.test(line)) {
      meta[key] = `${meta[key] ? `${meta[key]} ` : ""}${line.trim()}`;
    }
  }
  return { meta, body: src.slice(m[0].length) };
}

function listDir(d: string): string[] {
  try {
    return readdirSync(d);
  } catch {
    return [];
  }
}

function isDir(p: string): boolean {
  try {
    return statSync(p).isDirectory();
  } catch {
    return false;
  }
}

function skillRoots(home: string, cwd: string): string[] {
  const user = homedir();
  const roots = [join(cwd, ".vdom", "skills")];
  const top = repoRoot(cwd);
  for (const d of ancestors(cwd)) {
    roots.push(join(d, ".agents", "skills"), join(d, ".claude", "skills"));
    if (d === top) break;
  }
  roots.push(join(home, "skills"), join(user, ".agents", "skills"), join(user, ".claude", "skills"));
  return [...new Set(roots)];
}

function readSkill(file: string, fallbackName: string): Skill | undefined {
  const src = readText(file);
  if (!src) return undefined;
  const { meta } = parseFrontmatter(src);
  if (!meta.description) return undefined;
  return {
    name: (meta.name || fallbackName).trim(),
    description: meta.description.slice(0, 1024),
    path: file,
    baseDir: dirname(file),
    modelInvocable: meta["disable-model-invocation"] !== "true",
  };
}

export function loadSkills(home: string, cwd: string): Skill[] {
  const byName = new Map<string, Skill>();
  const visit = (dir: string, depth: number, isRoot: boolean) => {
    if (depth > 4) return;
    const skillMd = join(dir, "SKILL.md");
    if (existsSync(skillMd)) {
      const s = readSkill(skillMd, basename(dir));
      if (s && !byName.has(s.name)) byName.set(s.name, s);
      return;
    }
    for (const e of listDir(dir)) {
      if (e.startsWith(".") || e === "node_modules") continue;
      const p = join(dir, e);
      if (isDir(p)) visit(p, depth + 1, false);
      else if (isRoot && e.endsWith(".md")) {
        const s = readSkill(p, parse(e).name);
        if (s && !byName.has(s.name)) byName.set(s.name, s);
      }
    }
  };
  for (const r of skillRoots(home, cwd)) if (isDir(r)) visit(r, 0, true);
  return [...byName.values()].sort((a, b) => a.name.localeCompare(b.name));
}

export function loadTemplates(home: string, cwd: string): Template[] {
  const byName = new Map<string, Template>();
  const dirs = [join(cwd, ".vdom", "prompts"), join(cwd, ".claude", "commands"), join(home, "prompts"), join(homedir(), ".claude", "commands")];
  for (const d of dirs) {
    for (const e of listDir(d)) {
      if (!e.endsWith(".md")) continue;
      const name = parse(e).name;
      if (byName.has(name)) continue;
      const src = readText(join(d, e));
      if (src === undefined) continue;
      const { meta, body } = parseFrontmatter(src);
      const firstLine = body.split("\n").find((l) => l.trim())?.trim() ?? name;
      byName.set(name, {
        name,
        description: meta.description || (firstLine.length > 60 ? `${firstLine.slice(0, 60)}...` : firstLine),
        ...(meta["argument-hint"] ? { argumentHint: meta["argument-hint"] } : {}),
        body,
        path: join(d, e),
      });
    }
  }
  return [...byName.values()].sort((a, b) => a.name.localeCompare(b.name));
}

/** Shell-style split honoring single and double quotes. */
export function splitArgs(s: string): string[] {
  const out: string[] = [];
  const re = /"((?:[^"\\]|\\.)*)"|'([^']*)'|(\S+)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(s))) out.push(m[1] !== undefined ? m[1].replace(/\\(.)/g, "$1") : (m[2] ?? m[3]!));
  return out;
}

/** $1 $2 … $@ $ARGUMENTS ${N:-def} ${@:-def} ${@:N} ${@:N:L} — one pass, not recursive. */
export function substituteArgs(body: string, argText: string): string {
  const args = splitArgs(argText);
  const all = args.join(" ");
  return body.replace(/\$\{(@|ARGUMENTS|\d+)(?::-([^}]*)|:(\d+)(?::(\d+))?)?\}|\$(ARGUMENTS|@|\d+)/g, (_m, braced: string, def: string | undefined, from: string | undefined, len: string | undefined, bare: string | undefined) => {
    const key = braced ?? bare!;
    if (key === "@" || key === "ARGUMENTS") {
      if (from !== undefined) {
        const start = Math.max(0, Number(from) - 1);
        return args.slice(start, len !== undefined ? start + Number(len) : undefined).join(" ");
      }
      return all || (def ?? "");
    }
    const v = args[Number(key) - 1];
    return v ?? def ?? "";
  });
}

function escapeXml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

export function skillsPromptSection(skills: Skill[]): string {
  const visible = skills.filter((s) => s.modelInvocable);
  if (visible.length === 0) return "";
  return [
    "The following skills provide specialized instructions for specific tasks.",
    "Use the read tool to load a skill's file when the task matches its description.",
    "When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.",
    "",
    "<available_skills>",
    ...visible.map((s) => `  <skill>\n    <name>${escapeXml(s.name)}</name>\n    <description>${escapeXml(s.description)}</description>\n    <location>${escapeXml(s.path.replace(/\\/g, "/"))}</location>\n  </skill>`),
    "</available_skills>",
  ].join("\n");
}

/** `/skill:name args` → inlined skill body. */
export function expandSkill(skill: Skill, args: string): string {
  const src = readText(skill.path) ?? "";
  const { body } = parseFrontmatter(src);
  const text = `<skill name="${escapeXml(skill.name)}" location="${escapeXml(skill.path.replace(/\\/g, "/"))}">\nReferences are relative to ${skill.baseDir.replace(/\\/g, "/")}.\n\n${body.trim()}\n</skill>`;
  return args.trim() ? `${text}\n\n${args.trim()}` : text;
}
