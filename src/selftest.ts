import { flatten, cloneGraph, node, graph, modelId, findNode, type AgentGraph } from "./ir.js";
import { reconcile, propsChanged, RuntimeDOM } from "./reconciler.js";
import {
  DeterministicProvider,
  reverseEntire,
  reverseEachWord,
  registerProvider,
  clearProviderRegistry,
  resolveProvider,
  resolveChatConfig,
  DEFAULT_OPENROUTER_MODEL,
  type Provider,
  type Message,
  type CompleteOpts,
} from "./providers.js";
import {
  compilePaper,
  compileSource,
  SELF_REFINE_ABSTRACT,
  REFLEXION_ABSTRACT,
  oneShotGraph,
  selfRefineGraph,
  reflexionGraph,
} from "./papers.js";
import { WORD_REVERSE, WORD_REVERSE_HELLO, runBenchmark } from "./benchmarks.js";
import { applySelfRefineMutation, evolveOnce } from "./scientist.js";
import { runGraph, providerForNode } from "./runtime.js";
import {
  registerCapability,
  clearCapabilityRegistry,
  proposeCapability,
  sandboxValidate,
} from "./capability.js";
import {
  FakeTrainer,
  FailingTrainer,
  SurrogateTrainer,
  clearArtifactRegistry,
  getArtifact,
  describeHfJobsExtension,
  spawnTrainJob,
  waitTrainJob,
  recordTrainJobGate,
  clearTrainJobs,
  localHeldOutScore,
  isFrozenApiModel,
  FROZEN_API_MODEL,
} from "./trainer.js";
import {
  gateCapability,
  gateAdapter,
  unmountAdapterOnFailure,
  mountedImprovementKeys,
} from "./lifecycle.js";
import { improveLoop, pickMode, tracesLookIncomplete } from "./improve.js";

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
  assertEq(one.meta?.technique, "one-shot", "unknown source → one-shot");
  assertEq(flatten(one).length, 1, "one-shot is a single node");

  assertEq(compilePaper("self refine without hyphen").meta?.technique, "self-refine", "self refine phrase");
  assertEq(compileSource, compilePaper, "compileSource aliases compilePaper");
  assertEq(compileSource("a blog post about nothing in particular").meta?.technique, "one-shot", "blog-shaped text → one-shot");
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

/** Fake provider that records which model id handled each complete() call. */
class RecordingProvider implements Provider {
  name: string;
  model: string;
  calls: CompleteOpts[] = [];

  constructor(model: string) {
    this.model = model;
    this.name = `mock:${model}`;
  }

  async complete(_msgs: Message[], opts?: CompleteOpts): Promise<string> {
    this.calls.push({ ...opts });
    // Role-aware enough for a one-shot solve walk.
    const role = (opts?.role ?? "").toLowerCase();
    if (role === "solve" || role === "actor" || role === "one-shot" || role === "oneshot") {
      return reverseEntire("dom virtual");
    }
    return `ok:${this.model}`;
  }
}

async function testModelBinding(): Promise<void> {
  clearProviderRegistry();

  assertEq(modelId("mock-a"), "mock-a", "string model id");
  assertEq(modelId({ intelligence: "gpt-x" }), "gpt-x", "object intelligence id");
  assertEq(modelId(undefined), undefined, "missing model is undefined");
  assertEq(modelId({ cost: "low" }), undefined, "object without intelligence");

  const mockA = new RecordingProvider("mock-a");
  const mockB = new RecordingProvider("mock-b");
  registerProvider("mock-a", mockA);
  registerProvider("mock-b", mockB);

  const withModel = (model: string, version: number): AgentGraph =>
    graph({
      id: "model-bind",
      version,
      root: node({
        key: "solve",
        role: "solve",
        objective: "Solve the task in a single pass",
        technique: "one-shot",
        model,
      }),
    });

  const v1 = withModel("mock-a", 1);
  const v2 = withModel("mock-b", 2);
  const v2again = withModel("mock-b", 3);

  // Diff-only: model change → update; same model → retain.
  const diff = reconcile(v1, v2);
  assertEq(diff.ops.map((o) => `${o.op}:${o.node.key}`).join(","), "update:solve", "model change yields update:solve");
  const same = reconcile(v2, v2again);
  assertEq(same.ops[0]?.op, "retain", "identical model retains");

  const dom = new RuntimeDOM();
  const r1 = dom.reconcile(v1);
  assertEq(r1.ops[0]?.op, "mount", "DOM mounts model-a graph");
  const phys1 = dom.current.get("solve");
  assert(phys1?.provider === mockA, "mount binds mock-a provider");
  assertEq(phys1?.provider?.model, "mock-a", "bound provider model is mock-a");

  const r2 = dom.reconcile(v2);
  assertEq(r2.ops[0]?.op, "update", "DOM update on model swap");
  const phys2 = dom.current.get("solve");
  assert(phys2?.provider === mockB, "update rebinds to mock-b");
  assert(phys2?.provider !== mockA, "old provider discarded on model change");
  assertEq(dom.current.get("solve")?.status, "updated", "physical status updated");

  const r3 = dom.reconcile(v2again);
  assertEq(r3.ops[0]?.op, "retain", "DOM retain when model unchanged");
  const phys3 = dom.current.get("solve");
  assert(phys3?.provider === mockB, "retain keeps the same client instance");
  assertEq(dom.current.get("solve")?.status, "retained", "physical status retained");

  // Execution actually calls the bound provider (not the fallback).
  const fallback = new DeterministicProvider("unused-fallback");
  mockB.calls = [];
  const run = await runGraph(v2again, WORD_REVERSE.input, fallback, dom);
  assert(mockB.calls.length >= 1, "bound mock-b received complete()");
  assertEq(mockB.calls[0]?.role, "solve", "bound provider saw solve role");
  assertEq(run.final, reverseEntire("dom virtual"), "runGraph used bound provider output");

  // Without DOM, n.model still selects the registered provider.
  mockA.calls = [];
  await runGraph(v1, WORD_REVERSE.input, fallback);
  assert(mockA.calls.length >= 1, "runGraph resolves n.model without DOM");

  // Missing model → same fallback as today.
  const noModel = oneShotGraph();
  const fb = new DeterministicProvider();
  assert(providerForNode(noModel.root, fb) === fb, "missing model uses fallback provider");
  assertEq(resolveProvider(undefined).name, "deterministic", "resolveProvider() default is deterministic without key");
  assertEq(resolveChatConfig(), null, "no API key → resolveChatConfig is null");
  assertEq(DEFAULT_OPENROUTER_MODEL, "deepseek/deepseek-v4-flash-0731", "default OpenRouter model is 0731");

  // Object-form model binds via intelligence.
  const objGraph = graph({
    id: "obj-model",
    version: 1,
    root: node({
      key: "solve",
      role: "solve",
      objective: "x",
      model: { intelligence: "mock-a", cost: "low" },
    }),
  });
  const dom2 = new RuntimeDOM();
  dom2.reconcile(objGraph);
  assert(dom2.current.get("solve")?.provider === mockA, "object model binds via intelligence");

  clearProviderRegistry();
}

function resetImprovementFixtures(): void {
  clearCapabilityRegistry();
  clearArtifactRegistry();
  clearProviderRegistry();
  registerCapability({
    id: "reverse-each-word",
    run: (input) => reverseEachWord(input),
    source: "fn:reverseEachWord",
  });
  registerCapability({
    id: "reverse-entire",
    run: (input) => reverseEntire(input),
    source: "fn:reverseEntire",
  });
}

async function testCapabilityMount(): Promise<void> {
  resetImprovementFixtures();
  const provider = new DeterministicProvider();
  const base = oneShotGraph();
  const before = await runBenchmark(base, WORD_REVERSE, provider);
  assertEq(before.score, 0, "capability path starts at 0");

  const proposal = proposeCapability({
    key: "harness-cap",
    source: "module:reverse-each-word",
  });
  const dom = new RuntimeDOM();
  const gate = await gateCapability({
    base,
    proposal,
    task: WORD_REVERSE,
    provider,
    dom,
  });
  assertEq(gate.action, "mount", "capability gate mounts on pass");
  assertEq(gate.score, 1, "capability eval scores 1");
  assertEq(
    mountedImprovementKeys(gate.graph).capabilities.join(","),
    "harness-cap",
    "capability listed as mounted",
  );

  const phys = dom.current.get("harness-cap");
  assert(phys?.capability != null, "PhysicalNode loads capability fn");
  assertEq(phys?.descriptor.status, "mounted", "descriptor status mounted");

  const after = await runBenchmark(gate.graph, WORD_REVERSE, provider, dom);
  assertEq(after.score, 1, "mounted capability scores 1 in use");
  assertEq(after.final, "mod lautriv", "capability output is per-word reverse");
  resetImprovementFixtures();
}

async function testCapabilityReject(): Promise<void> {
  resetImprovementFixtures();
  const provider = new DeterministicProvider();
  const base = oneShotGraph();

  // Untrusted raw source — sandbox must reject (scientist safety).
  const raw = proposeCapability({
    key: "evil",
    source: "eval('process.exit(1)')",
  });
  const sand = sandboxValidate(raw.source!);
  assertEq(sand.ok, false, "raw source fails sandbox");

  const gateRaw = await gateCapability({
    base,
    proposal: raw,
    task: WORD_REVERSE,
    provider,
  });
  assertEq(gateRaw.action, "reject", "untrusted capability rejected");
  assertEq(gateRaw.graph.version, base.version, "live graph unchanged on sandbox reject");

  // Trusted module that still fails the task eval.
  const bad = proposeCapability({
    key: "bad-cap",
    source: "module:reverse-entire",
  });
  const gateBad = await gateCapability({
    base,
    proposal: bad,
    task: WORD_REVERSE,
    provider,
  });
  assertEq(gateBad.action, "reject", "failing capability not mounted");
  assertEq(gateBad.score, 0, "failing capability scores 0");
  assertEq(
    mountedImprovementKeys(gateBad.graph).capabilities.length,
    0,
    "no capability mounted after fail",
  );
  assertEq(findNode(gateBad.graph, "bad-cap"), undefined, "rejected cap absent from live graph");
  resetImprovementFixtures();
}

async function testAdapterMount(): Promise<void> {
  resetImprovementFixtures();
  const baseModel = "base-naive";
  registerProvider(baseModel, new DeterministicProvider(baseModel));

  const base = graph({
    id: "adapter-start",
    version: 1,
    root: node({
      key: "solve",
      role: "solve",
      objective: "Solve the task in a single pass",
      technique: "one-shot",
      model: baseModel,
    }),
  });

  const provider = new DeterministicProvider();
  const before = await runBenchmark(base, WORD_REVERSE, provider);
  assertEq(before.score, 0, "adapter path starts at 0");

  const trainer = new FakeTrainer();
  const artifact = await trainer.train(before.traces, {
    baseModel,
    technique: "fake-lora",
  });
  assert(getArtifact(artifact.id) != null, "artifact registered");

  const dom = new RuntimeDOM();
  dom.reconcile(base);
  assertEq(dom.current.get("solve")?.provider?.model, baseModel, "pre-train binds base model");

  const gate = await gateAdapter({
    base,
    artifact,
    targetKey: "solve",
    task: WORD_REVERSE,
    provider,
    dom,
  });
  assertEq(gate.action, "mount", "adapter gate mounts on pass");
  assertEq(gate.score, 1, "adapter eval scores 1");

  const solve = findNode(gate.graph, "solve");
  assertEq(solve?.model, artifact.resultModelId, "model pointer updated after mount");
  assertEq(
    mountedImprovementKeys(gate.graph).adapters.join(","),
    "adapter",
    "adapter listed as mounted",
  );

  const physAdapter = dom.current.get("adapter");
  assert(physAdapter?.adapter != null, "PhysicalNode holds adapter artifact");
  assertEq(physAdapter?.adapter?.id, artifact.id, "adapter artifact id matches");

  const physSolve = dom.current.get("solve");
  assertEq(physSolve?.provider?.model, artifact.resultModelId, "solve rebound to adapted model");

  const after = await runBenchmark(gate.graph, WORD_REVERSE, provider, dom);
  assertEq(after.score, 1, "adapted model scores 1");
  resetImprovementFixtures();
}

async function testAdapterRollback(): Promise<void> {
  resetImprovementFixtures();
  const baseModel = "base-naive";
  registerProvider(baseModel, new DeterministicProvider(baseModel));

  const base = graph({
    id: "adapter-fail",
    version: 1,
    root: node({
      key: "solve",
      role: "solve",
      objective: "Solve",
      technique: "one-shot",
      model: baseModel,
    }),
  });

  const provider = new DeterministicProvider();
  const failing = new FailingTrainer();
  const artifact = await failing.train([], { baseModel, technique: "fail-lora" });

  const rejectGate = await gateAdapter({
    base,
    artifact,
    targetKey: "solve",
    task: WORD_REVERSE,
    provider,
  });
  assertEq(rejectGate.action, "reject", "failing adapter rejected at gate");
  assertEq(findNode(rejectGate.graph, "solve")?.model, baseModel, "model pointer unchanged on reject");
  assertEq(findNode(rejectGate.graph, "adapter"), undefined, "adapter not in live graph");

  // Mount a good adapter, then force rollback.
  const good = await new FakeTrainer().train([], { baseModel, technique: "fake-lora" });
  const dom = new RuntimeDOM();
  const mountGate = await gateAdapter({
    base,
    artifact: good,
    targetKey: "solve",
    task: WORD_REVERSE,
    provider,
    dom,
  });
  assertEq(mountGate.action, "mount", "good adapter mounts");
  assertEq(findNode(mountGate.graph, "solve")?.model, good.resultModelId, "pointer after mount");

  const rollback = await unmountAdapterOnFailure({
    current: mountGate.graph,
    adapterKey: "adapter",
    targetKey: "solve",
    previousModel: baseModel,
    task: WORD_REVERSE,
    provider,
    dom,
  });
  assertEq(rollback.action, "rollback", "rollback action");
  assertEq(findNode(rollback.graph, "solve")?.model, baseModel, "model restored on rollback");
  assertEq(findNode(rollback.graph, "adapter"), undefined, "adapter unmounted from graph");
  assertEq(dom.current.get("adapter"), undefined, "adapter removed from DOM");
  assertEq(dom.current.get("solve")?.provider?.model, baseModel, "DOM rebound to base model");
  resetImprovementFixtures();
}

async function testImproveLoopCapability(): Promise<void> {
  resetImprovementFixtures();
  const history = await improveLoop({
    task: WORD_REVERSE,
    provider: new DeterministicProvider(),
    maxIters: 3,
    mode: "capability",
    capabilitySource: "module:reverse-each-word",
    start: oneShotGraph(),
  });
  const last = history[history.length - 1]!;
  assert(last.benchmark.score === 1 || last.gate?.action === "mount", "improveLoop capability reaches 1");
  const mounted = history.find((h) => h.gate?.action === "mount");
  assert(mounted != null, "improveLoop recorded a capability mount");
  resetImprovementFixtures();
}

async function testTwoClockTrainJob(): Promise<void> {
  resetImprovementFixtures();
  clearTrainJobs();
  const traces = [
    {
      nodeKey: "solve",
      role: "solve",
      input: "finish the episode",
      output: "transfer_to_human_agents",
      ts: 1,
      reason: "reward0-early-transfer",
      taskId: "incomplete_fixture_1",
      reward: 0,
    },
  ];
  const job = spawnTrainJob({
    trainer: new FakeTrainer(),
    traces,
    trainOpts: { baseModel: "base", technique: "fake-lora" },
    persist: false,
  });
  assertEq(job.servingPaused, false, "spawn never pauses serve");
  assert(
    job.status === "running" || job.status === "done",
    "slow clock is running or already done",
  );
  assertEq(job.not0731Weights, true, "job never claims 0731 weights");
  assertEq(job.tracesUsed.length, 1, "incomplete traces recorded");

  const done = await waitTrainJob(job.id);
  assertEq(done.status, "done", "FakeTrainer job completes");
  assert(done.artifactPointer != null, "artifact pointer set");
  assertEq(localHeldOutScore(done), 1, "FakeTrainer held-out is the protocol unit test");

  const gated = recordTrainJobGate(
    done.id,
    { arm: "I_weight", action: "reject", before: 0, after: 0, reason: "no raise" },
    false,
  );
  assertEq(gated?.gate?.action, "reject", "honest reject recorded");
  assertEq(gated?.servingPaused, false, "gate never pauses serve");
  assert(
    !String(done.artifact?.resultModelId ?? "").includes("deepseek-v4-pro-0813"),
    "FakeTrainer is not a catalog jump",
  );

  const sjob = spawnTrainJob({
    trainer: new SurrogateTrainer(),
    traces,
    trainOpts: { baseModel: FROZEN_API_MODEL, technique: "surrogate-prefix" },
    persist: false,
  });
  const sdone = await waitTrainJob(sjob.id);
  assertEq(sdone.surrogate, true, "SurrogateTrainer labeled surrogate");
  assertEq(sdone.not0731Weights, true, "0731 base model still not 0731 weights");
  assertEq(sdone.servingPaused, false, "surrogate spawn does not pause serve");
  assert(String(sdone.artifact?.uri ?? "").startsWith("file://"), "file-backed artifact");
  assert(String(sdone.artifact?.meta?.note ?? "").includes("Not 0731"), "artifact says not 0731");
  assertEq(localHeldOutScore(sdone), 0, "surrogate cannot raise p_hit — honest reject");
  assert(isFrozenApiModel(FROZEN_API_MODEL), "0731 is the frozen API model");
  clearTrainJobs();
  resetImprovementFixtures();
}

async function testPickModeConsultsIncompleteTraces(): Promise<void> {
  assertEq(pickMode("topology", 0, []), "topology", "explicit topology is not rewritten");
  assertEq(pickMode("auto", 0, []), "capability", "auto iter 0 is still capability");
  assertEq(pickMode("auto", 1, []), "adapter", "auto iter 1 is still adapter");
  assertEq(pickMode("auto", 2, []), "topology", "auto iter 2+ is still topology");
  assertEq(tracesLookIncomplete([]), false, "empty traces are not incomplete");
  const incomplete = [
    {
      nodeKey: "solve",
      role: "solve",
      input: "finish the episode",
      output: "transfer_to_human_agents",
      ts: 1,
      hung: true,
      reason: "hung",
    },
  ];
  assertEq(tracesLookIncomplete(incomplete), true, "hung / transfer traces look incomplete");
  assertEq(pickMode("auto", 0, incomplete), "adapter", "auto + incomplete traces → adapter / I_sku");
  assertEq(pickMode("capability", 0, incomplete), "capability", "explicit capability ignores traces");
}

async function testHfExtensionDoc(): Promise<void> {
  const doc = describeHfJobsExtension({ baseModel: "gpt-ish", technique: "lora" });
  assert(doc.includes("HF Jobs"), "HF Jobs extension doc present");
  assert(doc.includes("gateAdapter"), "docs mention gateAdapter");
}

async function main(): Promise<void> {
  const tests: Array<[string, () => Promise<void>]> = [
    ["flatten / clone", testFlattenClone],
    ["reconcile mount/update/unmount/retain", testReconcile],
    ["word-reverse grade", testGrade],
    ["source compiler", testPapers],
    ["self-refine + reflexion scores", testScores],
    ["evolution 0 → 1", testEvolution],
    ["model bind / swap / retain", testModelBinding],
    ["capability propose→pass→mount→use", testCapabilityMount],
    ["capability fail→no mount", testCapabilityReject],
    ["adapter train→mount→model pointer", testAdapterMount],
    ["adapter fail eval→unmount/rollback", testAdapterRollback],
    ["improveLoop capability path", testImproveLoopCapability],
    ["two-clock I_weight TrainJob", testTwoClockTrainJob],
    ["pickMode auto consults incomplete traces", testPickModeConsultsIncompleteTraces],
    ["HF Jobs extension docs", testHfExtensionDoc],
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
