import { type AgentGraph, type Trace } from "../ir.js";
import { type Completion, type Message, type ToolSpec } from "../providers.js";

export const TAU2_PAPER_REPO = "https://github.com/keejkrej/agent-stochastic-dynamics";
export const TAU2_BENCH_REPO = "https://github.com/sierra-research/tau2-bench";

export type Tau2Technique =
  | "one-shot"
  | "self-refine"
  | "reflexion"
  | "validator"
  | "policy-checklist";

export type Tau2MissedAction = {
  name: string;
  arguments?: Record<string, unknown>;
};

/** Persisted tau2 RewardInfo. On airline, ACTION / nl_assertions are diagnostics only. */
export type Tau2RewardInfo = {
  reward?: number | null;
  action_checks?: Array<{
    action?: { name?: string; arguments?: Record<string, unknown> };
    name?: string;
    arguments?: Record<string, unknown>;
    action_match?: boolean;
    action_reward?: number;
    tool_type?: string;
  }>;
  communicate_checks?: Array<{ info?: string; met?: boolean; justification?: string }>;
  nl_assertions?: Array<{ nl_assertion?: string; met?: boolean; justification?: string }>;
  db_check?: { db_match?: boolean; db_reward?: number } | null;
  missedActions?: Tau2MissedAction[];
  reward_basis?: string[];
  reward_breakdown?: Record<string, number> | null;
};

export type Tau2TurnRequest = {
  op: "turn";
  id: string;
  policy: string;
  tools: ToolSpec[];
  messages: Message[];
  technique?: Tau2Technique;
  graph?: AgentGraph;
  model?: string;
};

export type Tau2TurnResponse = {
  op: "ok" | "error";
  id: string;
  content?: string;
  tool_calls?: Completion["toolCalls"];
  traces?: Trace[];
  error?: string;
};

export type Tau2ActionLog = {
  kind: "text" | "tool";
  text: string;
  toolName?: string;
  toolArgs?: Record<string, unknown>;
  ok?: boolean;
  repeat?: boolean;
};

/** Features the paper repo's observe() can consume. */
export type Tau2Obs = {
  nSteps: number;
  nSuccessProxy: number;
  lastActions: string[];
  channels: string[];
  critique: string;
  toolFailures: number;
  repeatActions: number;
  /** Paper arm Obs should emit: I_loop | I_sku | wait. I_weight is a trainer-stub alias; I_catalog is a prior name. */
  arm?: "I_loop" | "I_sku" | "I_catalog" | "I_weight" | "wait";
  /** Official task id. Required for per-task C on mixed wait-hit / miss batches. */
  taskId?: string;
  missedActions?: Tau2MissedAction[];
  refusedCancel?: boolean;
  inventedPolicy?: boolean;
  hung?: boolean;
  /** user_stop | transfer | timeout | crash | … when the runner recorded one. */
  termination?: string;
  /** Typed I_loop graph when the miss is a refused / never-called cancel or update. */
  techniqueRecommendation?: Tau2Technique;
};

/**
 * Paper S — serving catalog pointer, held beside C.
 * Specified state is X=(H,M,E,C,S). S is not AgentNode.model (that field is C,
 * or a derived projection for PhysicalNode.provider).
 * I_loop never writes S. I_sku mount writes only S, and only onto the
 * weighted episodes of that batch.
 * Per-episode controller coordinate, not a process-global servingSku.
 * Source of truth is HybridState.S on the X_n object itself.
 */
export type CatalogPointer = {
  sku: string;
  servingPaused: false;
};

export type ServingSku = CatalogPointer;

/**
 * Specified runtime state X=(H,M,E,C,S).
 * S lives ON this object. Not ControlledEpisode.serving alone, not a
 * process Map (`servingByTask`) as the lookup, and not a post-hoc
 * assembly that reads that Map then stuffs S into a new object.
 */
export type HybridHistory = Message[];
export type HybridMemory = Trace[];
export type HybridEnv = Tau2Obs;
export type HybridController = AgentGraph;

export type HybridState = {
  H: HybridHistory;
  M: HybridMemory;
  E: HybridEnv;
  C: HybridController;
  S: CatalogPointer;
};

/**
 * Mixed-batch split:
 * wait+hit keep C0; completed I_loop miss get C1; incomplete / I_sku keep C0
 * topology and read S (not n.model) for the serving SKU.
 */
export type ApplyScope = {
  waitKept: string[];
  looped: string[];
  /** Hung / timeout / transfer / no-write. C topology stays C0; serving reads S. */
  weighted: string[];
};

export type Tau2SimulationLog = {
  taskId: string;
  trial: number;
  reward: number | null;
  pHit: 0 | 1 | null;
  termination: string | null;
  actions: Tau2ActionLog[];
  traces: Trace[];
  obs: Tau2Obs;
  messages?: unknown[];
  rewardInfo?: Tau2RewardInfo | null;
  hung?: boolean;
};

export type Tau2EvalFile = {
  benchmark: "tau2-bench";
  domain: string;
  agent: "vdom";
  model: string;
  provider: string;
  technique: Tau2Technique;
  paperRepo: typeof TAU2_PAPER_REPO;
  tau2Repo: typeof TAU2_BENCH_REPO;
  metricNote: string;
  live: boolean;
  smoke: boolean;
  passHatK: Record<string, number> | null;
  avgReward: number | null;
  simulations: Tau2SimulationLog[];
};
