import { spawnSync, type SpawnSyncReturns } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync, appendFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { vdomHome } from "./config.js";

/**
 * Prod / staging environments for the harness's own code.
 *
 * - prod:    git worktree of the source repo on branch `vdom/prod`, built.
 *            Clients launch it through the stable shim ~/.vdom/bin/vdom.
 * - staging: one worktree per fix on `vdom/staging/<name>`, cut from prod.
 * - gate:    build + tests (+ an optional reproduction command) in staging.
 * - promote: fast-forward prod to the gated staging commit and rebuild.
 * - rollback: move prod back to the previous promoted commit.
 */

export type GateRecord = { commit: string; passed: boolean; at: string; steps: { cmd: string; ok: boolean; ms: number; tail: string }[] };

export type Staging = { name: string; branch: string; path: string; base: string; createdAt: string; ticket?: string; gate?: GateRecord };

export type EnvState = {
  version: 1;
  /** Source repository the worktrees belong to. */
  repo: string;
  prod: { path: string; branch: string; commit: string; history: { commit: string; at: string; from?: string; note?: string }[] };
  staging: Record<string, Staging>;
  /** Commands that must pass in staging before promotion. */
  gate: string[];
};

export const envRoot = (): string => join(vdomHome(), "env");
const statePath = (): string => join(envRoot(), "state.json");
const logPath = (): string => join(envRoot(), "log.ndjson");

export function loadEnv(): EnvState | undefined {
  try {
    return JSON.parse(readFileSync(statePath(), "utf8")) as EnvState;
  } catch {
    return undefined;
  }
}

function saveEnv(s: EnvState): void {
  mkdirSync(envRoot(), { recursive: true });
  writeFileSync(statePath(), JSON.stringify(s, null, 2));
}

function log(event: string, data: Record<string, unknown>): void {
  mkdirSync(envRoot(), { recursive: true });
  appendFileSync(logPath(), `${JSON.stringify({ t: new Date().toISOString(), event, ...data })}\n`);
}

export class EnvError extends Error {}

function run(cmd: string, args: string[], cwd: string, timeoutMs = 600_000): SpawnSyncReturns<string> {
  return spawnSync(cmd, args, { cwd, encoding: "utf8", timeout: timeoutMs, windowsHide: true, shell: process.platform === "win32" && (cmd === "npm" || cmd === "npx"), maxBuffer: 64 * 1024 * 1024 });
}

function git(cwd: string, ...args: string[]): string {
  const r = run("git", args, cwd, 120_000);
  if (r.status !== 0) throw new EnvError(`git ${args.join(" ")} failed in ${cwd}: ${(r.stderr || r.stdout).trim()}`);
  return r.stdout.trim();
}

function shell(cmd: string, cwd: string, timeoutMs = 1_200_000): { ok: boolean; out: string; ms: number } {
  const t0 = Date.now();
  const r = spawnSync(cmd, { cwd, encoding: "utf8", shell: true, timeout: timeoutMs, windowsHide: true, maxBuffer: 64 * 1024 * 1024, env: { ...process.env, VDOM_TRACE: "" } });
  return { ok: r.status === 0, out: `${r.stdout ?? ""}${r.stderr ?? ""}`, ms: Date.now() - t0 };
}

function lockHash(dir: string): string {
  try {
    return readFileSync(join(dir, "package-lock.json"), "utf8");
  } catch {
    return "";
  }
}

/** Dependencies: share prod's node_modules when the lockfile matches, else npm ci. */
function installDeps(dir: string, shareFrom?: string): void {
  if (existsSync(join(dir, "node_modules"))) return;
  if (shareFrom && existsSync(join(shareFrom, "node_modules")) && lockHash(dir) === lockHash(shareFrom)) {
    symlinkSync(join(shareFrom, "node_modules"), join(dir, "node_modules"), process.platform === "win32" ? "junction" : "dir");
    return;
  }
  const r = shell("npm ci --no-audit --no-fund", dir);
  if (!r.ok) throw new EnvError(`npm ci failed in ${dir}:\n${r.out.slice(-3000)}`);
}

function build(dir: string): void {
  const r = shell("npm run build", dir);
  if (!r.ok) throw new EnvError(`build failed in ${dir}:\n${r.out.slice(-3000)}`);
}

/** Stable launchers so clients (T3, editors, scripts) always start prod. */
function writeShims(prodPath: string): string {
  const bin = join(vdomHome(), "bin");
  mkdirSync(bin, { recursive: true });
  const cli = join(prodPath, "dist", "acp", "cli.js");
  // On Windows a plain `@node "...cli.js" %*` shim loses multi-line prompts:
  // cmd.exe cuts an argument at the first newline before `%*` even expands.
  // `%cmdcmdline%` still holds the raw, untruncated command line, so the .cmd
  // stashes it in an env var and the node launcher re-splits it (newlines and
  // < > included) before re-spawning the real CLI.
  writeFileSync(join(bin, "vdom.cmd"), `@echo off\r\nsetlocal enabledelayedexpansion\r\nset "VDOM_RAW_CMDLINE=!cmdcmdline!"\r\nnode "${join(bin, "vdom-launch.cjs")}" %*\r\n`);
  writeFileSync(join(bin, "vdom-launch.cjs"), launcherSource(cli));
  writeFileSync(join(bin, "vdom"), `#!/bin/sh\nexec node "${cli.replace(/\\/g, "/")}" "$@"\n`, { mode: 0o755 });
  return bin;
}

/** The node launcher behind vdom.cmd: re-spawn the CLI with the untruncated command line. */
function launcherSource(cli: string): string {
  const lines = [
    "// Generated by vdom env init: cmd.exe cuts batch arguments at the first",
    "// newline and reads bare < or > as redirects, so vdom.cmd stashes the raw",
    "// command line (%cmdcmdline%) in VDOM_RAW_CMDLINE and this launcher re-splits",
    "// it instead of trusting the already-truncated argv.",
    "const { spawnSync } = require('node:child_process');",
    `const cli = ${JSON.stringify(cli)};`,
    "const raw = process.env.VDOM_RAW_CMDLINE || '';",
    "// Windows-style split: quotes toggle quoting, doubled quotes are literal.",
    "function splitArgs(s) {",
    "  const out = []; let cur = ''; let quoted = false;",
    "  for (let i = 0; i < s.length; i++) {",
    "    const c = s[i];",
    "    if (quoted) {",
    "      if (c === '\"' && s[i + 1] === '\"') { cur += '\"'; i++; }",
    "      else if (c === '\"') quoted = false;",
    "      else cur += c;",
    "    } else if (c === '\"') quoted = true;",
    "    else if (c === ' ' || c === '\t') { if (cur) { out.push(cur); cur = ''; } }",
    "    else cur += c;",
    "  }",
    "  if (cur) out.push(cur);",
    "  return out;",
    "}",
    "// The raw line starts with the shell and the batch path; drop everything up",
    "// to and including the vdom.cmd entry, which keeps the quoted multi-line",
    "// prompt intact.",
    "const parts = splitArgs(raw);",
    "// The raw line may wrap the batch and its arguments in outer quotes:",
    "// `cmd /c \"\"<vdom.cmd>\" <args>\"` (PowerShell) or `cmd /d /s /c \"<vdom.cmd> <args>\"`.",
    "// Unwrap that outer quoting before splitting so the batch path and the",
    "// quoted multi-line prompt survive as single arguments.",
    "let line = raw;",
    "const dbl = /\\s[\\/]c\\s+\"\"([\\s\\S]*)\"\\s*$/.exec(line);",
    "const sgl = /\\s[\\/]c\\s+\"([\\s\\S]*)\"\\s*$/.exec(line);",
    "if (dbl) line = dbl[1];",
    "else if (sgl) line = sgl[1];",
    "const parts = splitArgs(line);",
    "const batchIdx = parts.findIndex((p) => /vdom[.]cmd$/i.test(p));",
    "const recovered = batchIdx >= 0 ? parts.slice(batchIdx + 1) : process.argv.slice(2);",,
    "const r = spawnSync(process.execPath, [cli, ...recovered], { stdio: 'inherit', windowsHide: true });",
    "if (r.error) { console.error(String(r.error)); process.exit(1); }",
    "process.exit(r.status ?? 0);",
    "",
  ];
  return lines.join("\n");
}

export function initEnv(opts: { repo: string; ref?: string; gate?: string[] }): EnvState {
  if (loadEnv()) throw new EnvError(`already initialized (${statePath()}); use \`vdom env status\``);
  const repo = resolve(opts.repo);
  const commit = git(repo, "rev-parse", opts.ref ?? "HEAD");
  const prodPath = join(envRoot(), "prod");
  git(repo, "worktree", "add", "-B", "vdom/prod", prodPath, commit);
  installDeps(prodPath);
  build(prodPath);
  const state: EnvState = {
    version: 1,
    repo,
    prod: { path: prodPath, branch: "vdom/prod", commit, history: [{ commit, at: new Date().toISOString(), note: "init" }] },
    staging: {},
    gate: opts.gate ?? ["npm run build", "npm test"],
  };
  saveEnv(state);
  writeShims(prodPath);
  log("init", { repo, commit });
  return state;
}

function need(): EnvState {
  const s = loadEnv();
  if (!s) throw new EnvError("no environments yet; run `vdom env init --repo <path to vdom-harness>`");
  return s;
}

export function createStaging(name: string, ticket?: string): Staging {
  const s = need();
  if (!/^[a-z0-9][a-z0-9-]{0,60}$/.test(name)) throw new EnvError("staging name must be lowercase letters, digits, dashes");
  if (s.staging[name]) throw new EnvError(`staging ${name} already exists at ${s.staging[name]!.path}`);
  const path = join(envRoot(), "staging", name);
  const branch = `vdom/staging/${name}`;
  git(s.repo, "worktree", "add", "-B", branch, path, s.prod.commit);
  installDeps(path, s.prod.path);
  const st: Staging = { name, branch, path, base: s.prod.commit, createdAt: new Date().toISOString(), ...(ticket ? { ticket } : {}) };
  s.staging[name] = st;
  saveEnv(s);
  log("stage", { name, base: s.prod.commit, ticket });
  return st;
}

/** Commit whatever the fix left uncommitted so the gate and promotion refer to one commit. */
function commitAll(path: string, message: string): string {
  if (git(path, "status", "--porcelain")) {
    git(path, "add", "-A");
    git(path, "-c", "user.name=vdom", "-c", "user.email=vdom@localhost", "commit", "-q", "-m", message);
  }
  return git(path, "rev-parse", "HEAD");
}

export function gateStaging(name: string, extra: string[] = []): GateRecord {
  const s = need();
  const st = s.staging[name];
  if (!st) throw new EnvError(`unknown staging ${name}`);
  const commit = commitAll(st.path, `vdom fix: ${st.ticket ?? name}`);
  const steps: GateRecord["steps"] = [];
  for (const cmd of [...s.gate, ...extra]) {
    const r = shell(cmd, st.path);
    steps.push({ cmd, ok: r.ok, ms: r.ms, tail: r.out.slice(-4000) });
    if (!r.ok) break;
  }
  const gate: GateRecord = { commit, passed: steps.length === s.gate.length + extra.length && steps.every((x) => x.ok), at: new Date().toISOString(), steps };
  st.gate = gate;
  saveEnv(s);
  log("gate", { name, commit, passed: gate.passed, steps: steps.map((x) => ({ cmd: x.cmd, ok: x.ok, ms: x.ms })) });
  return gate;
}

export function promote(name: string, opts: { force?: boolean } = {}): { from: string; to: string; diffstat: string } {
  const s = need();
  const st = s.staging[name];
  if (!st) throw new EnvError(`unknown staging ${name}`);
  const head = git(st.path, "rev-parse", "HEAD");
  if (!opts.force) {
    if (!st.gate?.passed) throw new EnvError(`staging ${name} has not passed the gate; run \`vdom env gate ${name}\``);
    if (st.gate.commit !== head || git(st.path, "status", "--porcelain")) throw new EnvError(`staging ${name} changed after the gate ran; gate it again`);
  }
  const from = s.prod.commit;
  const prodLockBefore = lockHash(s.prod.path);
  git(s.prod.path, "merge", "--ff-only", st.branch);
  const to = git(s.prod.path, "rev-parse", "HEAD");
  if (lockHash(s.prod.path) !== prodLockBefore) {
    rmSync(join(s.prod.path, "node_modules"), { recursive: true, force: true });
    installDeps(s.prod.path);
  }
  build(s.prod.path);
  const diffstat = git(s.prod.path, "diff", "--stat", from, to);
  s.prod.commit = to;
  s.prod.history.push({ commit: to, at: new Date().toISOString(), from: name, ...(st.ticket ? { note: st.ticket } : {}) });
  saveEnv(s);
  log("promote", { name, from, to });
  return { from, to, diffstat };
}

export function rollback(): { from: string; to: string } {
  const s = need();
  if (s.prod.history.length < 2) throw new EnvError("nothing to roll back to");
  const current = s.prod.history.pop()!;
  const prev = s.prod.history[s.prod.history.length - 1]!;
  git(s.prod.path, "reset", "-q", "--hard", prev.commit);
  build(s.prod.path);
  s.prod.commit = prev.commit;
  saveEnv(s);
  log("rollback", { from: current.commit, to: prev.commit });
  return { from: current.commit, to: prev.commit };
}

export function dropStaging(name: string): void {
  const s = need();
  const st = s.staging[name];
  if (!st) throw new EnvError(`unknown staging ${name}`);
  git(s.repo, "worktree", "remove", "--force", st.path);
  delete s.staging[name];
  saveEnv(s);
  log("drop", { name });
}

export function describeEnv(): string {
  const s = loadEnv();
  if (!s) return "No environments. Run `vdom env init --repo <path>`.";
  const lines = [
    `repo     ${s.repo}`,
    `prod     ${s.prod.commit.slice(0, 10)}  ${s.prod.path}`,
    `shim     ${join(vdomHome(), "bin", process.platform === "win32" ? "vdom.cmd" : "vdom")}`,
    `gate     ${s.gate.join(" && ")}`,
    `history  ${s.prod.history.map((h) => `${h.commit.slice(0, 7)}${h.from ? `(${h.from})` : ""}`).join(" → ")}`,
  ];
  for (const st of Object.values(s.staging)) {
    const g = st.gate ? `${st.gate.passed ? "PASSED" : "FAILED"} @ ${st.gate.commit.slice(0, 7)}` : "not gated";
    lines.push(`staging  ${st.name}  base ${st.base.slice(0, 7)}  ${g}  ${st.path}${st.ticket ? `  [${st.ticket}]` : ""}`);
  }
  return lines.join("\n");
}
