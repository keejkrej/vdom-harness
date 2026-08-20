import { type Trace } from "../ir.js";
import { type Completion } from "../providers.js";
import { type Tau2ActionLog, type Tau2Obs, type Tau2RewardInfo } from "./tau2-types.js";
import { recommendIntervention } from "./tau2-improve.js";
import {
  detectInventedPolicy,
  detectRefusedCancel,
  missedActionsFromRewardInfo,
  missedPolicyWrites,
  policyCritique,
  shouldRecommendPolicy,
} from "./tau2-policy.js";

export function actionFromCompletion(turn: Completion): Tau2ActionLog {
  const tc = turn.toolCalls?.[0];
  if (tc) {
    return {
      kind: "tool",
      text: tc.name,
      toolName: tc.name,
      toolArgs: tc.arguments,
      ok: true,
    };
  }
  return { kind: "text", text: turn.content };
}

export function markRepeats(actions: Tau2ActionLog[]): Tau2ActionLog[] {
  const seen = new Map<string, number>();
  return actions.map((a) => {
    const key = a.kind === "tool" ? `tool:${a.toolName}:${JSON.stringify(a.toolArgs ?? {})}` : `text:${a.text}`;
    const n = (seen.get(key) ?? 0) + 1;
    seen.set(key, n);
    return { ...a, repeat: n > 1 };
  });
}

export function observeTau2(opts: {
  traces: Trace[];
  actions: Tau2ActionLog[];
  reward: number | null;
  toolFailures?: number;
  rewardInfo?: Tau2RewardInfo | null;
  hung?: boolean;
  termination?: string;
  messages?: Array<{ role?: string; content?: string }>;
  taskId?: string;
}): Tau2Obs {
  const actions = markRepeats(opts.actions);
  const lastActions = actions.map((a) =>
    a.kind === "tool" ? a.toolName ?? a.text : `text:${a.text.slice(0, 80)}`,
  );
  const repeatActions = actions.filter((a) => a.repeat).length;
  const toolFailures = opts.toolFailures ?? actions.filter((a) => a.ok === false).length;
  const hung = Boolean(opts.hung);
  const pHit = hung ? 0 : opts.reward != null && opts.reward >= 1 - 1e-6 ? 1 : 0;
  const missedActions = missedActionsFromRewardInfo(opts.rewardInfo);
  const refusedCancel = detectRefusedCancel(actions, opts.messages);
  const inventedPolicy = detectInventedPolicy(actions, opts.messages);
  const missedPolicy = missedPolicyWrites(missedActions);
  const critique = policyCritique({
    pHit,
    hung,
    refusedCancel,
    inventedPolicy,
    missedPolicy,
    toolFailures,
    repeatActions,
  });
  const obs: Tau2Obs = {
    nSteps: tracesOrActions(opts.traces, actions),
    nSuccessProxy: pHit,
    lastActions,
    channels: actions.some((a) => a.kind === "tool") ? ["env"] : ["samp"],
    critique,
    toolFailures,
    repeatActions,
    missedActions,
    refusedCancel,
    inventedPolicy,
    hung,
  };
  if (opts.termination) obs.termination = opts.termination;
  if (
    pHit !== 1 &&
    shouldRecommendPolicy({
      refusedCancel,
      inventedPolicy,
      missedActions,
    })
  ) {
    obs.techniqueRecommendation = "policy-checklist";
  }
  obs.arm = recommendIntervention(obs);
  if (opts.taskId) obs.taskId = opts.taskId;
  return obs;
}

function tracesOrActions(traces: Trace[], actions: Tau2ActionLog[]): number {
  return traces.length > 0 ? traces.length : actions.length;
}
