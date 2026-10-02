/**
 * prod/staging environments + the fix gate, on a toy repo with a scripted
 * "fixer" (no model). Proves the gate decides, not the agent's report.
 */
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const home = mkdtempSync(join(tmpdir(), "vdom-envhome-"));
process.env.VDOM_HOME = home;
const { loadConfig } = await import("./config.js");
const { initEnv, loadEnv, rollback, describeEnv } = await import("./envs.js");
const { fixIssue } = await import("./fix.js");
const { fileIssue, loadIssue } = await import("./issues.js");

const git = (cwd: string, ...args: string[]) => {
  const r = spawnSync("git", args, { cwd, encoding: "utf8" });
  assert.equal(r.status, 0, `git ${args.join(" ")}: ${r.stderr}`);
  return r.stdout.trim();
};

// Toy harness: lib.js has a bug; `npm test` runs every test/*.js.
const repo = mkdtempSync(join(tmpdir(), "vdom-toyrepo-"));
mkdirSync(join(repo, "test"));
writeFileSync(join(repo, "package.json"), JSON.stringify({ name: "toy", version: "1.0.0", type: "module", scripts: { build: "node -e \"0\"", test: "node test/run.js" } }));
writeFileSync(join(repo, "package-lock.json"), JSON.stringify({ name: "toy", version: "1.0.0", lockfileVersion: 3, requires: true, packages: { "": { name: "toy", version: "1.0.0" } } }));
writeFileSync(join(repo, "lib.js"), "export const clamp = (x, lo, hi) => Math.max(lo, x);\n");
writeFileSync(
  join(repo, "test", "run.js"),
  'import { readdirSync } from "node:fs";\nfor (const f of readdirSync(new URL(".", import.meta.url)).filter((f) => f.endsWith(".test.js"))) await import(`./${f}`);\nconsole.log("all tests ok");\n',
);
writeFileSync(join(repo, "test", "basic.test.js"), 'import { clamp } from "../lib.js";\nif (clamp(-5, 0, 10) !== 0) throw new Error("low");\n');
git(repo, "init", "-q");
git(repo, "add", "-A");
git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base");

initEnv({ repo });
const env0 = loadEnv()!;
assert.ok(existsSync(join(home, "bin", process.platform === "win32" ? "vdom.cmd" : "vdom")));
console.log("ok env init (prod worktree + shim)");

const cfg = loadConfig();
const newIssue = (title: string) =>
  fileIssue({ sessionId: "s", sessionDir: "d", turn: 1, signal: { source: "manual" }, title, blame: "harness", category: "bug", whatHappened: "", rootCause: "", proposedFix: "", files: ["lib.js"], repro: "", evidence: [], severity: "medium" });
const quiet = () => {};

// 1. Proper red → green fix is promoted.
{
  const issue = newIssue("clamp ignores the upper bound");
  const r = await fixIssue({
    issueId: issue.id,
    cfg,
    promote: true,
    log: quiet,
    drive: async (cwd) => {
      writeFileSync(join(cwd, "test", "upper.test.js"), 'import { clamp } from "../lib.js";\nif (clamp(50, 0, 10) !== 10) throw new Error("upper bound ignored");\n');
      writeFileSync(join(cwd, "lib.js"), "export const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));\n");
      return ["end_turn"];
    },
  });
  assert.equal(r.gate?.passed, true);
  assert.equal(r.red?.ok, true, r.red?.detail);
  assert.ok(r.promoted);
  assert.match(readFileSync(join(env0.prod.path, "lib.js"), "utf8"), /Math\.min\(hi/);
  assert.equal(loadIssue(issue.id)?.status, "fixed");
  console.log("ok red→green fix gated and promoted to prod");
}

// 2. A "fix" with no regression test is not promotable, whatever the agent says.
{
  const issue = newIssue("cosmetic");
  const r = await fixIssue({
    issueId: issue.id,
    cfg,
    promote: true,
    log: quiet,
    drive: async (cwd) => {
      writeFileSync(join(cwd, "lib.js"), "// clamp a number\nexport const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));\n");
      return ["end_turn"];
    },
  });
  assert.equal(r.gate?.passed, true);
  assert.equal(r.red?.ok, false);
  assert.match(r.red!.detail, /no regression test/);
  assert.equal(r.promoted, undefined);
  assert.equal(loadIssue(issue.id)?.status, "open");
  console.log("ok fix without a regression test is rejected");
}

// 3. A test that already passes on the base does not reproduce anything.
{
  const issue = newIssue("tautology");
  const r = await fixIssue({
    issueId: issue.id,
    cfg,
    promote: true,
    log: quiet,
    drive: async (cwd) => {
      writeFileSync(join(cwd, "test", "taut.test.js"), "if (1 !== 1) throw new Error();\n");
      writeFileSync(join(cwd, "lib.js"), "export const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x)); // same\n");
      return ["end_turn"];
    },
  });
  assert.equal(r.red?.ok, false);
  assert.match(r.red!.detail, /does not reproduce/);
  assert.equal(r.promoted, undefined);
  console.log("ok non-reproducing test is rejected");
}

// 4. A fix that breaks the suite fails the gate.
{
  const issue = newIssue("breaks things");
  const r = await fixIssue({
    issueId: issue.id,
    cfg,
    promote: true,
    log: quiet,
    drive: async (cwd) => {
      writeFileSync(join(cwd, "lib.js"), "export const clamp = () => 0;\n");
      writeFileSync(join(cwd, "test", "zero.test.js"), 'import { clamp } from "../lib.js";\nif (clamp(5, 0, 10) !== 0) throw new Error();\n');
      return ["end_turn"];
    },
  });
  assert.equal(r.gate?.passed, false);
  assert.equal(r.promoted, undefined);
  console.log("ok suite-breaking fix fails the gate");
}

// 5. Rollback restores the previous prod commit.
{
  const before = loadEnv()!.prod.commit;
  const rb = rollback();
  assert.equal(rb.from, before);
  assert.equal(loadEnv()!.prod.commit, env0.prod.commit);
  assert.match(readFileSync(join(env0.prod.path, "lib.js"), "utf8"), /Math\.max\(lo, x\);/);
  assert.match(describeEnv(), /history/);
  console.log("ok rollback");
}

console.log("env selftest passed");
