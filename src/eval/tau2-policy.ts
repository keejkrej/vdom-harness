/**
 * Official τ² airline cancel/modify gates (from tau2-bench airline/policy.md).
 * Used as the system text of the policy-checklist I_loop node.
 * Not generic "think harder".
 */
export const POLICY_WRITE_TOOLS = new Set([
  "cancel_reservation",
  "update_reservation_flights",
  "update_reservation_baggages",
  "update_reservation_passengers",
  "book_reservation",
]);

/**
 * Official cancel/modify gates from data/tau2/domains/airline/policy.md.
 * Encodes rules only — never gold reservation IDs. Airline reward is DB × COMMUNICATE;
 * ACTION / nl_assertions are diagnostics, not reward_basis.
 */
export const AIRLINE_POLICY_CHECKLIST = `Official τ² airline policy checklist (source: tau2-bench data/tau2/domains/airline/policy.md). The current time in that wiki is 2024-05-15 15:00:00 EST. Encode these rules. Never hard-code reservation IDs. Do not invent extra restrictions. Do not say you have no mechanism when a listed tool exists.

Airline reward is the DB end-state (times communicate_info, which is often empty). An extra cancel of an ineligible reservation, or a missed eligible write, zeros the score. ACTION traces and nl_assertions are diagnostics only.

Cancel flight (official gates):
- Obtain user id (user must provide it) and reservation id (help locate it with tools if needed).
- Obtain the reason for cancellation (change of plan, airline cancelled flight, or other).
- If any portion of the flight has already been flown, you cannot cancel it. Do not call cancel_reservation on flown or partially flown trips.
- Otherwise a reservation may be cancelled if ANY of these is true:
  1. The booking was made within the last 24 hours.
  2. The flight is cancelled by the airline.
  3. It is a business cabin reservation.
  4. It is economy (not basic economy) and the user has travel insurance.
- Economy + travel insurance is eligible. Do not invent a "personal reason", "change of plan", "give up the seat", or "not health/weather" block that refuses an insured economy cancel when the user asked to cancel and did not state they are healthy.
- If the user states they are healthy, or that this is not a health or weather claim, insurance does not apply. Refuse that cancel. Do not call cancel_reservation on that reservation.
- Basic economy without insurance, outside the 24-hour window, and not airline-cancelled, is ineligible. Do not cancel it.
- The API does not check these rules. You must check them before calling cancel_reservation. An ineligible cancel still writes the DB and zeros the reward.
- Refunds go to the original payment methods in 5–7 business days.

If the user asks to cancel and the reservation is eligible under those gates, CALL cancel_reservation. If it is ineligible, say the real reason from this list (already flown; basic economy without insurance; user is healthy so insurance does not apply). Never invent bars such as "no-show is not possible" or "I have no mechanism".

When the user says "cancel all upcoming" or "cancel flights longer than X hours":
- Enumerate every reservation with get_user_details / get_reservation_details.
- Check each reservation against the official gates (and the user's duration filter, if any).
- Cancel every eligible reservation. Do not stop after the first two.
- Leave every ineligible reservation untouched.

Modify flight / cabin (official):
- Basic economy itineraries cannot change origin, destination, or trip type.
- Cabin can be changed on all reservations, including basic economy, unless any segment has already been flown.
- Cabin class must stay the same across every flight in a reservation.
- If a cancel is ineligible, refuse it, then continue: complete every eligible cabin upgrade the user asked for. Do not stop after the first two update_reservation_flights.

Do not provide procedures that are not in the policy or tools. One tool call per turn.`;

export type MissedAction = {
  name: string;
  arguments?: Record<string, unknown>;
};

export type RewardInfoLike = {
  reward?: number | null;
  action_checks?: Array<{
    action?: { name?: string; arguments?: Record<string, unknown> };
    name?: string;
    arguments?: Record<string, unknown>;
    action_match?: boolean;
  }>;
  communicate_checks?: Array<{ info?: string; met?: boolean; justification?: string }>;
  nl_assertions?: Array<{ nl_assertion?: string; met?: boolean; justification?: string }>;
  missedActions?: MissedAction[];
};

export type PolicyObsHint = {
  refusedCancel?: boolean;
  inventedPolicy?: boolean;
  missedActions?: MissedAction[];
  hung?: boolean;
  techniqueRecommendation?: string;
};

const REFUSE_CANCEL =
  /unable to cancel|cannot cancel|can't cancel|can not cancel|no way for me to|no mechanism|not possible to cancel|i(?:'m| am) unable to|unfortunately.{0,60}cancel|i(?:'m| am) afraid.{0,60}cancel|i(?:'m| am) sorry.{0,60}unable/i;

const INVENTED_POLICY =
  /no-?show|no mechanism|i have no (?:way|mechanism)|there is no way for me to|not possible to (?:make|process) (?:a )?no-?show|i have no (?:tool|api) to cancel|personal reason|change of plan is not (?:covered|eligible)|not a (?:valid|covered) (?:personal )?reason/i;

export function missedActionsFromRewardInfo(info?: RewardInfoLike | null): MissedAction[] {
  if (!info) return [];
  if (Array.isArray(info.missedActions) && info.missedActions.length > 0) {
    return info.missedActions
      .filter((a) => typeof a?.name === "string" && a.name.length > 0)
      .map((a) => ({ name: a.name, arguments: a.arguments ?? {} }));
  }
  const missed: MissedAction[] = [];
  for (const check of info.action_checks ?? []) {
    if (check.action_match !== false) continue;
    const name = check.action?.name ?? check.name;
    if (!name) continue;
    missed.push({
      name,
      arguments: check.action?.arguments ?? check.arguments ?? {},
    });
  }
  return missed;
}

export function isPolicyWriteTool(name: string | undefined): boolean {
  return !!name && POLICY_WRITE_TOOLS.has(name);
}

export function missedPolicyWrites(missed: MissedAction[]): MissedAction[] {
  return missed.filter((a) => isPolicyWriteTool(a.name));
}

function textsFrom(actions: Array<{ text?: string; kind?: string }>, messages?: Array<{ role?: string; content?: string }>): string {
  const actionText = actions
    .filter((a) => a.kind === "text" || typeof a.text === "string")
    .map((a) => a.text ?? "")
    .join("\n");
  const msgText = (messages ?? [])
    .filter((m) => m.role === "assistant")
    .map((m) => m.content ?? "")
    .join("\n");
  return `${actionText}\n${msgText}`;
}

export function detectRefusedCancel(
  actions: Array<{ text?: string; kind?: string; toolName?: string }>,
  messages?: Array<{ role?: string; content?: string }>,
): boolean {
  const blob = textsFrom(actions, messages);
  return REFUSE_CANCEL.test(blob);
}

export function detectInventedPolicy(
  actions: Array<{ text?: string; kind?: string }>,
  messages?: Array<{ role?: string; content?: string }>,
): boolean {
  const blob = textsFrom(actions, messages);
  return INVENTED_POLICY.test(blob);
}

export function shouldRecommendPolicy(hint: PolicyObsHint): boolean {
  if (hint.techniqueRecommendation === "policy-checklist") return true;
  if (hint.refusedCancel || hint.inventedPolicy) return true;
  return missedPolicyWrites(hint.missedActions ?? []).length > 0;
}

export function policyCritique(opts: {
  pHit: number;
  hung?: boolean;
  refusedCancel?: boolean;
  inventedPolicy?: boolean;
  missedPolicy?: MissedAction[];
  toolFailures?: number;
  repeatActions?: number;
}): string {
  if (opts.pHit === 1) return "path measure hits S; wait";
  if (opts.hung) return "trial hung or skipped; keep task in the set (null reward), retry once";
  const missed = opts.missedPolicy ?? [];
  if (opts.refusedCancel || opts.inventedPolicy || missed.length > 0) {
    const names = missed.map((a) => a.name).join(", ") || "cancel/update";
    return `user asked cancel/update and agent refused or never called the tool (${names}); I_loop policy-checklist`;
  }
  if ((opts.toolFailures ?? 0) > 0) return "tool failures in trajectory; inspect env channel";
  if ((opts.repeatActions ?? 0) > 0) return "repeat actions; loop mutation or wait";
  return "episode unfinished or miss; inspect cascade / tools";
}
