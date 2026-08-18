import { flatten, cloneGraph, node, graph, type AgentGraph } from "./ir.js";
import { reconcile, propsChanged, RuntimeDOM } from "./reconciler.js";
import { DeterministicProvider, reverseEntire, reverseEachWord } from "./providers.js";
import {
  compilePaper,
  SELF_REFINE_ABSTRACT,
  REFLEXION_ABSTRACT,
  oneShotGraph,
  selfRefineGraph,
  reflexionGraph,
} from "./papers.js";
import { WORD_REVERSE, WORD_REVERSE_HELLO, runBenchmark } from "./benchmarks.js";
import { applySelfRefineMutation, evolveOnce } from "./scientist.js";

let passed = 0;
let failed = 0;

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) {
    failed += 1;
    throw new Error(`FAIL: ${msg}`);
  }
  passed += 1;
}

function assertEq<T>(actual: T, expected: T, msg: string): void {
  if (actual !== expected) {
    failed += 1;
    throw new Error(`FAIL: ${msg}\n  expected: ${JSON.stringify(expected)}\n  actual:   ${JSON.stringify(actual)}`);
  }
  passed += 1;
}

async function testFlattenClone(): Promise<void> {
  const g = selfRefineGraph();
  const flat = flatten(g);
  assertEq(flat.length, 3, "self-refine flattens to 3 nodes");
  assertEq(flat[0]!.node.key, "solve", "preorder starts at solve");
  assertEq(flat[1]!.node.key, "critic", "critic is second");
  assertEq(flat[1]!.parentKey, "solve", "critic parent is solve");
  assertEq(flat[2]!.node.key, "refine", "refine is third");
  assertEq(flat[2]!.parentKey, "critic", "refine parent is critic");

  const copy = cloneGraph(g);
  assert(copy !== g, "cloneGraph returns a new object");
  assert(copy.root !== g.root, "cloneGraph deep-clones root");
  copy.root.role = "mutated";
  assertEq(g.root.role, "solve", "mutating clone does not change original");
}

async function testReconcile(): Promise<void> {
  const v1 = oneShotGraph();
  const v2 = applySelfRefineMutation(v1);
  const rec = reconcile(v1, v2);
  const kinds = rec.ops.map((o) => `${o.op}:${o.node.key}`);
  assertEq(kinds.join(","), "retain:solve,mount:critic,mount:refine", "v1→v2 retain solve, mount critic+refine");

  const empty = reconcile(undefined, v1);
  assertEq(empty.ops[0]?.op, "mount", "first reconcile mounts root");

  const same = reconcile(v1, cloneGraph(v1));
  assert(same.ops.every((o) => o.op === "retain"), "identical graph is all retain");

  const updated: AgentGraph = graph({
    id: "u",
    version: 2,
    root: node({ key: "solve", role: "solve", objective: "changed objective" }),
  });
  const upd = reconcile(v1, updated);
  assertEq(upd.ops[0]?.op, "update", "objective change is update");
  assert(propsChanged(v1.root, updated.root), "propsChanged detects objective");

  const withChild: AgentGraph = graph({
    id: "c",
    version: 1,
    root: node({
      key: "solve",
      role: "solve",
      objective: v1.root.objective,
      children: [node({ key: "specialist", role: "specialist", objective: "extra" })],
    }),
  });
  const gone = reconcile(withChild, v1);
  const unmount = gone.ops.find((o) => o.op === "unmount");
  assertEq(unmount?.node.key, "specialist", "removed child is unmounted");
  assertEq(gone.ops.find((o) => o.node.key === "solve")?.op, "retain", "parent retained when only child removed");

  const dom = new RuntimeDOM();
  const r1 = dom.reconcile(v1);
  assertEq(r1.ops[0]?.op, "mount", "DOM first pass mounts");
  assertEq(dom.current.get("solve")?.status, "mounted", "physical status mounted");
  const r2 = dom.reconcile(v2);
  assertEq(dom.current.get("solve")?.status, "retained", "physical status retained");
  assertEq(dom.current.get("critic")?.status, "mounted", "critic mounted on DOM");
  assertEq(r2.ops.filter((o) => o.op === "unmount").length, 0, "no unmounts v1→v2");
}

async function testGrade(): Promise<void> {
  assertEq(WORD_REVERSE.grade("mod lautriv"), 1, "exact match scores 1");
  assertEq(WORD_REVERSE.grade("  mod   lautriv  "), 1, "whitespace-normalized match scores 1");
  assertEq(WORD_REVERSE.grade("lautriv mod"), 0, "whole-string reverse scores 0");
  assertEq(WORD_REVERSE.grade("dom virtual"), 0, "identity scores 0");
  assertEq(WORD_REVERSE_HELLO.grade("olleh dlrow"), 1, "second task grades");
  assertEq(reverseEntire("dom virtual"), "lautriv mod", "naive reverse");
  assertEq(reverseEachWord("dom virtual"), "mod lautriv", "per-word reverse");
}

async function testPapers(): Promise<void> {
  const sr = compilePaper(SELF_REFINE_ABSTRACT);
  assertEq(sr.meta?.technique, "self-refine", "Madaan abstract → self-refine");
  assertEq(sr.root.technique, "self-refine", "root technique self-refine");
  assertEq(flatten(sr).map((f) => f.node.key).join(","), "solve,critic,refine", "self-refine keys");

  const rx = compilePaper(REFLEXION_ABSTRACT);
  assertEq(rx.meta?.technique, "reflexion", "Shinn abstract → reflexion");
  const rxKeys = flatten(rx).map((f) => f.node.key);
  assert(rxKeys.includes("actor"), "reflexion has actor");
  assert(rxKeys.includes("reflect"), "reflexion has reflect");
  assert(rxKeys.includes("memory"), "reflexion has memory");

  const one = compilePaper("a generic method with no known citation");
  assertEq(one.meta?.technique, "one-shot", "unknown paper → one-shot");
  assertEq(flatten(one).length, 1, "one-shot is a single node");

  assertEq(compilePaper("self refine without hyphen").meta?.technique, "self-refine", "self refine phrase");
}

async function testScores(): Promise<void> {
  const provider = new DeterministicProvider();

  const sr = await runBenchmark(selfRefineGraph(), WORD_REVERSE, provider);
  assertEq(sr.score, 1, "self-refine scores 1");
  assertEq(sr.final, "mod lautriv", "self-refine final is per-word reverse");
  assertEq(sr.nodes, 3, "self-refine node count");

  const rx = await runBenchmark(reflexionGraph(), WORD_REVERSE, provider);
  assertEq(rx.score, 1, "reflexion scores 1");
  assertEq(rx.final, "mod lautriv", "reflexion final is per-word reverse");

  const one = await runBenchmark(oneShotGraph(), WORD_REVERSE, provider);
  assertEq(one.score, 0, "one-shot scores 0");
  assertEq(one.final, "lautriv mod", "one-shot is whole-string reverse");
  assertEq(one.nodes, 1, "one-shot node count");
}

async function testEvolution(): Promise<void> {
  const provider = new DeterministicProvider();
  const v1 = oneShotGraph();
  const a = await runBenchmark(v1, WORD_REVERSE, provider);
  assertEq(a.score, 0, "evolution start score 0");
  assertEq(a.nodes, 1, "evolution start nodes 1");

  const v2 = applySelfRefineMutation(v1);
  const b = await runBenchmark(v2, WORD_REVERSE, provider);
  assertEq(b.score, 1, "evolution end score 1");
  assertEq(b.nodes, 3, "evolution end nodes 3");

  const evolved = await evolveOnce(v1, a.traces, a.score, provider);
  const c = await runBenchmark(evolved, WORD_REVERSE, provider);
  assertEq(c.score, 1, "scientist-emitted graph scores 1");
  assert(flatten(evolved).some((f) => f.node.key === "critic"), "scientist adds critic");
  assert(flatten(evolved).some((f) => f.node.key === "refine"), "scientist adds refine");
}

async function main(): Promise<void> {
  const tests: Array<[string, () => Promise<void>]> = [
    ["flatten / clone", testFlattenClone],
    ["reconcile mount/update/unmount/retain", testReconcile],
    ["word-reverse grade", testGrade],
    ["paper compiler", testPapers],
    ["self-refine + reflexion scores", testScores],
    ["evolution 0 → 1", testEvolution],
  ];

  for (const [name, fn] of tests) {
    try {
      await fn();
      console.log(`ok  ${name}`);
    } catch (err) {
      console.error(`not ok  ${name}`);
      console.error(err instanceof Error ? err.message : err);
      process.exitCode = 1;
    }
  }

  console.log("");
  console.log(`${passed} passed, ${failed} failed`);
  if (failed > 0) process.exitCode = 1;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
