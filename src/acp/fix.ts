import { spawnSync } from "node:child_process";
import { cpSync, existsSync, mkdirSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import type { AgentConfig } from "./config.js";
import { createStaging, envRoot, EnvError, gateStaging, loadEnv, promote, type GateRecord } from "./envs.js";
import { loadIssue, renderIssue, saveIssue, type Issue } from "./issues.js";

/**
 * Long-term half of self-improvement: turn a filed issue into a gated harness
 * fix. Stage from prod → drive the agent on the staging worktree → the gate
 * (not the agent's report) decides: the new regression test must FAIL on the
 * prod base and the full suite must PASS on staging → optionally promote.
 */

export type FixResult = { issue: Issue; staging: string; gate?: GateRecord; red?: { ok: boolean; detail: string }; promoted?: { from: string; to: string } };

const TEST_FILE_RE = /(^|\/)(test|tests|__tests__)\/|[._-](test|spec|selftest)\.[cm]?[jt]sx?$|_test\.(py|go)$|test_.*\.py$/;

export function fixBrief(issue: Issue, stagingPath: string): string {
  return [
    "You are fixing a bug in your OWN harness (vdom). The working directory is a staging checkout of the harness; change code here only.",
    "The bug report below was written by vdom's diagnostician from a real session where a user was frustrated.",
    "",
    renderIssue(issue),
    "",
    "Required procedure:",
    "1. Reproduce: add a regression test to the existing test files (e.g. src/acp/unit-selftest.ts or src/acp/acp-selftest.ts) that FAILS on the current code because of this bug. Run it and observe the failure.",
    "2. Fix the harness so the test passes, keeping the change minimal and in the style of the surrounding code.",
    "3. Run the full suite (`npm run build` and `npm test`) and make it pass.",
    "4. Report: root cause, the fix (files), the regression test, and the exact test output you observed.",
    "An automated gate will independently re-run your regression test against the unfixed code (it must fail there) and the full suite against your fix (it must pass). Unverified claims will be caught.",
    `Staging path: ${stagingPath}`,
  ].join("\n");
}

function git(cwd: string, ...args: string[]): string {
  const r = spawnSync("git", args, { cwd, encoding: "utf8", windowsHide: true, maxBuffer: 32 * 1024 * 1024 });
  if (r.status !== 0) throw new EnvError(`git ${args.join(" ")}: ${(r.stderr || r.stdout).trim()}`);
  return r.stdout.trim();
}

/**
 * Red check: copy the staging commit's changed test files onto a clean
 * checkout of the base commit and run the test command there. It must fail —
 * otherwise the "regression test" does not reproduce the bug.
 */
export function redCheck(stagingPath: string, base: string, testCmd: string): { ok: boolean; detail: string } {
  const changed = git(stagingPath, "diff", "--name-only", `${base}..HEAD`).split("\n").filter(Boolean);
  const tests = changed.filter((f) => TEST_FILE_RE.test(f));
  if (tests.length === 0) return { ok: false, detail: "no regression test: the fix changed no test files" };
  const src = changed.filter((f) => !TEST_FILE_RE.test(f));
  if (src.length === 0) return { ok: false, detail: "only test files changed; nothing was fixed" };
  const tmp = join(envRoot(), "red", `${Date.now()}`);
  mkdirSync(dirname(tmp), { recursive: true });
  git(stagingPath, "worktree", "add", "--detach", tmp, base);
  try {
    for (const f of tests) {
      if (existsSync(join(stagingPath, f))) {
        mkdirSync(dirname(join(tmp, f)), { recursive: true });
        cpSync(join(stagingPath, f), join(tmp, f));
      }
    }
    if (existsSync(join(stagingPath, "node_modules")) && !existsSync(join(tmp, "node_modules"))) {
      spawnSync(process.platform === "win32" ? "cmd" : "ln", process.platform === "win32" ? ["/c", "mklink", "/J", join(tmp, "node_modules"), join(stagingPath, "node_modules")] : ["-s", join(stagingPath, "node_modules"), join(tmp, "node_modules")], { windowsHide: true });
    }
    const r = spawnSync(testCmd, { cwd: tmp, shell: true, encoding: "utf8", timeout: 1_200_000, windowsHide: true, maxBuffer: 64 * 1024 * 1024, env: { ...process.env, VDOM_TRACE: "" } });
    const out = `${r.stdout ?? ""}${r.stderr ?? ""}`.slice(-2500);
    return r.status === 0
      ? { ok: false, detail: `regression test passes on the unfixed base (${tests.join(", ")}): it does not reproduce the bug\n${out}` }
      : { ok: true, detail: `tests fail on the unfixed base as expected (${tests.join(", ")})\n${out}` };
  } finally {
    try {
      git(stagingPath, "worktree", "remove", "--force", tmp);
    } catch {
      rmSync(tmp, { recursive: true, force: true });
    }
  }
}

export type DriveFix = (cwd: string, prompt: string) => Promise<string[]>;

export async function fixIssue(opts: { issueId: string; cfg: AgentConfig; drive: DriveFix; promote: boolean; testCmd?: string; log?: (s: string) => void }): Promise<FixResult> {
  const log = opts.log ?? ((s: string) => process.stdout.write(`${s}\n`));
  const issue = loadIssue(opts.issueId);
  if (!issue) throw new EnvError(`no issue ${opts.issueId}`);
  const env = loadEnv();
  if (!env) throw new EnvError("no prod environment; run `vdom env init --repo <harness repo>` first");
  const name = issue.id.toLowerCase();
  const st = env.staging[name] ?? createStaging(name, issue.id);
  issue.status = "fixing";
  saveIssue(issue);
  log(`staging ${name} at ${st.path} (base ${st.base.slice(0, 10)})`);

  const stops = await opts.drive(st.path, fixBrief(issue, st.path));
  log(`agent finished: ${stops.join(", ")}`);

  const gate = gateStaging(name);
  log(`gate ${gate.passed ? "PASSED" : "FAILED"} at ${gate.commit.slice(0, 10)}: ${gate.steps.map((s) => `${s.ok ? "✓" : "✗"} ${s.cmd}`).join("  ")}`);
  const red = gate.passed ? redCheck(st.path, st.base, opts.testCmd ?? "npm test") : undefined;
  if (red) log(`red check ${red.ok ? "PASSED" : "FAILED"}: ${red.detail.split("\n")[0]}`);

  const result: FixResult = { issue, staging: name, gate, ...(red ? { red } : {}) };
  if (gate.passed && red?.ok) {
    issue.status = "fixed";
    if (opts.promote) {
      result.promoted = promote(name);
      log(`promoted prod ${result.promoted.from.slice(0, 10)} → ${result.promoted.to.slice(0, 10)}`);
    } else log(`ready: \`vdom env promote ${name}\``);
  } else {
    issue.status = "open";
  }
  saveIssue(issue);
  return result;
}
