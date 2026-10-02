/**
 * Multi-edit engine. Every oldText is matched against the ORIGINAL content
 * (not after earlier edits). Exact match first; if any edit needs it, all
 * edits are matched in a whitespace/quote/dash-normalized space and only the
 * touched lines are rewritten. BOM and the file's line ending are preserved.
 */

export type Edit = { oldText: string; newText: string };

export class EditError extends Error {}

const BOM = "﻿";

export function detectEol(s: string): "\r\n" | "\n" {
  const crlf = s.indexOf("\r\n");
  const lf = s.indexOf("\n");
  return crlf >= 0 && crlf <= lf ? "\r\n" : "\n";
}

/** Normalization used for fuzzy matching. Must not change line count. */
export function fuzzyNormalize(s: string): string {
  return s
    .normalize("NFKC")
    .replace(/[‘-‛]/g, "'")
    .replace(/[“-‟]/g, '"')
    .replace(/[‐-―−]/g, "-")
    .replace(/[  -   　]/g, " ")
    .split("\n")
    .map((l) => l.replace(/[ \t]+$/, ""))
    .join("\n");
}

function occurrences(hay: string, needle: string): number[] {
  const out: number[] = [];
  for (let i = hay.indexOf(needle); i >= 0; i = hay.indexOf(needle, i + Math.max(1, needle.length))) out.push(i);
  return out;
}

/** Accept the shapes models actually send: edits as JSON string, one object, or legacy top-level fields. */
export function coerceEdits(args: Record<string, unknown>): Edit[] {
  let raw: unknown = args.edits;
  if (typeof raw === "string") {
    try {
      raw = JSON.parse(raw);
    } catch {
      throw new EditError("edits must be an array of {oldText, newText}");
    }
  }
  const list: unknown[] = Array.isArray(raw) ? raw : raw && typeof raw === "object" ? [raw] : [];
  const legacyOld = args.oldText ?? args.old_string;
  const legacyNew = args.newText ?? args.new_string;
  if (typeof legacyOld === "string" && typeof legacyNew === "string") list.push({ oldText: legacyOld, newText: legacyNew });
  const edits = list.map((e, i) => {
    const o = (e ?? {}) as Record<string, unknown>;
    const oldText = o.oldText ?? o.old_string ?? o.old;
    const newText = o.newText ?? o.new_string ?? o.new;
    if (typeof oldText !== "string" || typeof newText !== "string") {
      throw new EditError(`edits[${i}] needs string oldText and newText`);
    }
    return { oldText, newText };
  });
  if (edits.length === 0) throw new EditError("edits must contain at least one {oldText, newText}");
  return edits;
}

type Span = { start: number; end: number; newText: string; index: number };

function locate(content: string, edits: Edit[], path: string, label: (i: number) => string): Span[] {
  const spans: Span[] = [];
  for (let i = 0; i < edits.length; i++) {
    const e = edits[i]!;
    if (e.oldText.length === 0) throw new EditError(`${edits.length > 1 ? `edits[${i}].oldText` : "oldText"} must not be empty in ${path}.`);
    const hits = occurrences(content, e.oldText);
    if (hits.length === 0) {
      throw new EditError(
        edits.length > 1
          ? `Could not find edits[${i}] in ${path}. The oldText must match exactly including all whitespace and newlines.`
          : `Could not find the exact text in ${path}. The old text must match exactly including all whitespace and newlines.`,
      );
    }
    if (hits.length > 1) {
      throw new EditError(
        edits.length > 1
          ? `Found ${hits.length} occurrences of edits[${i}] in ${path}. Each oldText must be unique. Please provide more context to make it unique.`
          : `Found ${hits.length} occurrences of the text in ${path}. The text must be unique. Please provide more context to make it unique.`,
      );
    }
    spans.push({ start: hits[0]!, end: hits[0]! + e.oldText.length, newText: e.newText, index: i });
  }
  spans.sort((a, b) => a.start - b.start);
  for (let k = 1; k < spans.length; k++) {
    if (spans[k]!.start < spans[k - 1]!.end) {
      throw new EditError(`${label(spans[k - 1]!.index)} and ${label(spans[k]!.index)} overlap in ${path}. Merge them into one edit or target disjoint regions.`);
    }
  }
  return spans;
}

function splice(content: string, spans: Span[]): string {
  let out = "";
  let pos = 0;
  for (const s of spans) {
    out += content.slice(pos, s.start) + s.newText;
    pos = s.end;
  }
  return out + content.slice(pos);
}

/** Line index (0-based) of a char offset. */
function lineOf(s: string, offset: number): number {
  let n = 0;
  for (let i = s.indexOf("\n"); i >= 0 && i < offset; i = s.indexOf("\n", i + 1)) n++;
  return n;
}

export type EditResult = { before: string; after: string; firstChangedLine: number; replaced: number; fuzzy: boolean };

export function applyEdits(original: string, editsIn: Edit[], path: string): EditResult {
  const hasBom = original.startsWith(BOM);
  const body = hasBom ? original.slice(1) : original;
  const eol = detectEol(body);
  const lf = body.replace(/\r\n/g, "\n");
  const edits = editsIn.map((e) => ({ oldText: e.oldText.replace(/\r\n/g, "\n"), newText: e.newText.replace(/\r\n/g, "\n") }));
  const label = (i: number) => `edits[${i}]`;

  let result: string;
  let fuzzy = false;
  let firstLine: number;
  const exactOk = edits.every((e) => e.oldText.length > 0 && occurrences(lf, e.oldText).length === 1);
  if (exactOk) {
    const spans = locate(lf, edits, path, label);
    result = splice(lf, spans);
    firstLine = lineOf(lf, spans[0]!.start);
  } else {
    // Fuzzy space keeps line structure, so line numbers map 1:1 to the original.
    const normContent = fuzzyNormalize(lf);
    const normEdits = edits.map((e) => ({ oldText: fuzzyNormalize(e.oldText), newText: e.newText }));
    const spans = locate(normContent, normEdits, path, label);
    fuzzy = true;
    const origLines = lf.split("\n");
    const normLines = normContent.split("\n");
    // Rewrite only the lines each edit touches; untouched lines keep their bytes.
    // Group spans whose line ranges touch; each group's lines are rebuilt once.
    // l1 is the line holding the first char AFTER the span, so a trailing "\n" in oldText stays inside the block.
    const groups: { l0: number; l1: number; spans: Span[] }[] = [];
    for (const s of spans) {
      const l0 = lineOf(normContent, s.start);
      const l1 = lineOf(normContent, s.end);
      const g = groups[groups.length - 1];
      if (g && l0 <= g.l1) {
        g.l1 = Math.max(g.l1, l1);
        g.spans.push(s);
      } else groups.push({ l0, l1, spans: [s] });
    }
    const pieces: string[] = [];
    let lineCursor = 0;
    for (const g of groups) {
      const lineStart = normLines.slice(0, g.l0).reduce((n, l) => n + l.length + 1, 0);
      const block = normLines.slice(g.l0, g.l1 + 1).join("\n");
      const local = g.spans.map((s) => ({ ...s, start: s.start - lineStart, end: s.end - lineStart }));
      pieces.push(...origLines.slice(lineCursor, g.l0), splice(block, local));
      lineCursor = g.l1 + 1;
    }
    pieces.push(...origLines.slice(lineCursor));
    result = pieces.join("\n");
    firstLine = lineOf(normContent, spans[0]!.start);
  }
  if (result === lf) throw new EditError(`No changes made to ${path}. The replacement produced identical content.`);
  const after = (hasBom ? BOM : "") + (eol === "\r\n" ? result.replace(/\n/g, "\r\n") : result);
  return { before: original, after, firstChangedLine: firstLine + 1, replaced: edits.length, fuzzy };
}

/** Per-file serialization of mutations; different files run in parallel. */
const queues = new Map<string, Promise<unknown>>();
export async function withFileLock<T>(key: string, fn: () => Promise<T>): Promise<T> {
  const k = process.platform === "win32" ? key.toLowerCase() : key;
  const prev = queues.get(k) ?? Promise.resolve();
  const run = prev.then(fn, fn);
  const tail = run.catch(() => {});
  queues.set(k, tail);
  try {
    return await run;
  } finally {
    if (queues.get(k) === tail) queues.delete(k);
  }
}
