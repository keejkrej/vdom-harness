/**
 * In-session "that was bad" detection from the user's next message.
 * Cheap, deterministic, and deliberately biased toward recall: a false
 * positive costs one background diagnosis; a miss loses the best signal we get.
 */

export type FeedbackSignal = {
  signal: "frustration" | "correction" | "repeat" | "cancel" | "reject";
  /** 0..1 */
  strength: number;
  matched: string;
};

const FRUSTRATION = [
  /\bw\s?t\s?f+\b/i,
  /\bwth\b/i,
  /\bf+u+\b/i,
  /\bfuck(ing|ed)?\b/i,
  /\bffs\b/i,
  /\bomfg\b/i,
  /\bshit\b/i,
  /\bstupid\b/i,
  /\bidiot(ic)?\b/i,
  /\bdumb\b/i,
  /\buseless\b/i,
  /\bnonsense\b/i,
  /\bare you (kidding|serious)\b/i,
  /\bwhat the (hell|heck)\b/i,
  /\?{3,}|!{3,}/,
  /\bseriously\b/i,
];

const CORRECTION = [
  /^\s*no[,.! ]/i,
  /^\s*nope\b/i,
  /\bthat'?s (not|wrong)\b/i,
  /\bnot what i (asked|said|meant|wanted)\b/i,
  /\bi (said|told you|asked( you)?)\b/i,
  /\bwhy (did|would) you\b/i,
  /\byou (didn'?t|did not|ignored|forgot|broke|deleted|removed)\b/i,
  /\bstop\b/i,
  /\bundo\b|\brevert\b/i,
  /\bagain\?/i,
  /\bwrong\b/i,
  /\bdoesn'?t work\b|\bstill (broken|failing|fails)\b/i,
];

function shouting(text: string): boolean {
  const letters = text.replace(/[^A-Za-z]/g, "");
  return letters.length >= 8 && letters === letters.toUpperCase();
}

export function detectFeedback(text: string, previousUserText?: string): FeedbackSignal | undefined {
  const t = text.trim();
  if (!t) return undefined;
  for (const re of FRUSTRATION) {
    const m = t.match(re);
    if (m) return { signal: "frustration", strength: 0.9, matched: m[0] };
  }
  if (shouting(t)) return { signal: "frustration", strength: 0.7, matched: "ALL CAPS" };
  // Corrections only count near the start or in short messages; long specs often contain "no"/"wrong".
  const head = t.slice(0, 160);
  for (const re of CORRECTION) {
    const m = head.match(re);
    if (m && (t.length < 400 || m.index! < 40)) return { signal: "correction", strength: 0.6, matched: m[0] };
  }
  if (previousUserText && normalize(previousUserText) === normalize(t) && t.length > 8) {
    return { signal: "repeat", strength: 0.6, matched: "repeated the previous request" };
  }
  return undefined;
}

function normalize(s: string): string {
  return s.toLowerCase().replace(/\s+/g, " ").trim();
}
