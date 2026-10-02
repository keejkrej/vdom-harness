/**
 * Unit tests for the `vdom issues` CLI: `new` (via fileIssue, signal "manual"),
 * `list`, `show`, and `close`. Exercises the real command through a child process
 * so parseArgs → issuesCommand → issues.js are covered together.
 */
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const here = fileURLToPath(new URL(".", import.meta.url));
const cli = join(here, "cli.ts");
const home = mkdtempSync(join(tmpdir(), "vdom-issues-home-"));

const run = (...args: string[]) =>
  spawnSync(process.execPath, ["--import", "tsx", cli, "issues", ...args], {
    env: { ...process.env, VDOM_HOME: home },
    encoding: "utf8",
  });

// new requires a title and a "what happened" description.
{
  const r = run("new");
  assert.equal(r.status, 2);
  assert.match(r.stderr, /usage: vdom issues new/);
  console.log("ok issues new usage");
}

// new files an issue through fileIssue (source "manual") and prints the id.
{
  const r = run(
    "new",
    "--title",
    "Manual issue from selftest",
    "--blame",
    "harness",
    "--severity",
    "high",
    "--files",
    "src/acp/cli.ts, src/acp/issues.ts",
    "--repro",
    "run vdom issues new",
    "--fix",
    "none needed",
    "the model claimed success without running tests",
  );
  assert.equal(r.status, 0, r.stderr);
  const id = r.stdout.trim();
  assert.match(id, /^I-\d{8}-[0-9a-f]{4}$/);
  console.log("ok issues new prints id");

  // Listed by `vdom issues`.
  const list = run("list");
  assert.equal(list.status, 0);
  assert.match(list.stdout, new RegExp(id));
  assert.match(list.stdout, /Manual issue from selftest/);

  // show renders the full record with the manual signal.
  const show = run("show", id);
  assert.equal(show.status, 0);
  assert.match(show.stdout, /the model claimed success without running tests/);
  assert.match(show.stdout, /signal: manual/);
  assert.match(show.stdout, /blame harness/);

  // close flips status (default fixed) and persists.
  const close = run("close", id);
  assert.equal(close.status, 0);
  assert.match(close.stdout, new RegExp(`${id} fixed`));
  const after = run("show", id, "--json");
  assert.equal(after.status, 0);
  assert.match(after.stdout, /"status": "fixed"/);

  // close with an explicit status.
  const r2 = run(
    "new",
    "--title",
    "Second manual issue",
    "another thing went wrong",
  );
  const id2 = r2.stdout.trim();
  const close2 = run("close", id2, "--status", "wontfix");
  assert.equal(close2.status, 0);
  const after2 = run("show", id2, "--json");
  assert.match(after2.stdout, /"status": "wontfix"/);
  console.log("ok issues list/show/close");
}

// unknown close target errors.
{
  const r = run("close", "I-00000000-0000");
  assert.equal(r.status, 1);
  assert.match(r.stderr, /no issue/);
  console.log("ok issues close unknown id");
}

// flag validation.
{
  const badBlame = spawnSync(process.execPath, ["--import", "tsx", cli, "issues", "new", "--title", "t", "--blame", "nope", "x"], {
    env: { ...process.env, VDOM_HOME: home },
    encoding: "utf8",
  });
  assert.notEqual(badBlame.status, 0);
  assert.match(badBlame.stderr, /--blame must be harness\|model\|env\|unclear/);
  console.log("ok issues flag validation");
}

console.log("issues selftest passed");
