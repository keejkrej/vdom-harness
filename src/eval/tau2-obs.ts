import { type Trace } from "../ir.js";
import { type Completion } from "../providers.js";
import { type Tau2ActionLog, type Tau2Obs } from "./tau2-types.js";
import { recommendIntervention } from "./tau2-improve.js";

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
}): Tau2Obs {
  const actions = markRepeats(opts.actions);
  const lastActions = actions.map((a) =>
    a.kind === "tool" ? a.toolName ?? a.text : `text:${a.text.slice(0, 80)}`,
  );
  const repeatActions = actions.filter((a) => a.repeat).length;
  const toolFailures = opts.toolFailures ?? actions.filter((a) => a.ok === false).length;
  const pHit = opts.reward != null && opts.reward >= 1 - 1e-6 ? 1 : 0;
  const critique =
    pHit === 1
      ? "path measure hits S; wait"
      : toolFailures > 0
        ? "tool failures in trajectory; inspect env channel"
        : repeatActions > 0
          ? "repeat actions; loop mutation or wait"
          : "episode unfinished or miss; inspect cascade / tools";
  const obs: Tau2Obs = {
    nSteps: tracesOrActions(opts.traces, actions),
    nSuccessProxy: pHit,
    lastActions,
    channels: actions.some((a) => a.kind === "tool") ? ["env"] : ["samp"],
    critique,
    toolFailures,
    repeatActions,
  };
  obs.arm = recommendIntervention(obs);
  return obs;
}

function tracesOrActions(traces: Trace[], actions: Tau2ActionLog[]): number {
  return traces.length > 0 ? traces.length : actions.length;
}
