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
  /** Paper arm Obs should emit: I_loop | I_weight | wait */
  arm?: "I_loop" | "I_weight" | "wait";
  missedActions?: Tau2MissedAction[];
  refusedCancel?: boolean;
  inventedPolicy?: boolean;
  hung?: boolean;
  /** Typed I_loop graph when the miss is a refused / never-called cancel or update. */
  techniqueRecommendation?: Tau2Technique;
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
