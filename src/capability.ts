import { type AgentNode, node } from "./ir.js";

/**
 * Approved capability modules. Raw scientist JSON / source is never executed
 * unless it resolves to a registered module through the sandbox gate.
 */
export type CapabilityFn = (input: string) => string | Promise<string>;

export type CapabilityModule = {
  id: string;
  run: CapabilityFn;
  /** Optional exact source string that maps to this module. */
  source?: string;
};

const registry = new Map<string, CapabilityModule>();

export function registerCapability(mod: CapabilityModule): void {
  registry.set(mod.id, mod);
}

export function getCapability(id: string): CapabilityModule | undefined {
  return registry.get(id);
}

export function clearCapabilityRegistry(): void {
  registry.clear();
}

export function listCapabilities(): string[] {
  return [...registry.keys()];
}

/**
 * Sandbox validation: source alone is untrusted.
 * Accepts:
 *   - `module:<registered-id>`
 *   - exact `source` match against a registered module
 * Rejects everything else (including free-form code from tryParseGraph).
 */
export function sandboxValidate(
  source: string,
): { ok: true; moduleId: string } | { ok: false; reason: string } {
  const trimmed = source.trim();
  if (!trimmed) return { ok: false, reason: "empty capability source" };

  const moduleRef = trimmed.match(/^module:(.+)$/);
  if (moduleRef?.[1]) {
    const id = moduleRef[1].trim();
    if (!registry.has(id)) {
      return { ok: false, reason: `unknown module ref: ${id}` };
    }
    return { ok: true, moduleId: id };
  }

  for (const mod of registry.values()) {
    if (mod.source != null && mod.source === trimmed) {
      return { ok: true, moduleId: mod.id };
    }
  }

  return {
    ok: false,
    reason: "untrusted capability source; register a module or use module:<id>",
  };
}

/** Build a proposed capability node (not mounted, not executable). */
export function proposeCapability(opts: {
  key: string;
  source: string;
  objective?: string;
  role?: string;
}): AgentNode {
  return node({
    key: opts.key,
    kind: "capability",
    role: opts.role ?? "capability",
    objective: opts.objective ?? "Proposed harness capability",
    source: opts.source,
    status: "proposed",
  });
}

/**
 * Promote a proposed capability through sandbox validation.
 * Does not mount — caller must eval-gate before setting status to mounted.
 */
export function validateCapability(n: AgentNode): AgentNode {
  if ((n.kind ?? "agent") !== "capability") {
    throw new Error(`validateCapability: expected kind capability, got ${n.kind}`);
  }
  const source = n.source ?? (n.moduleId ? `module:${n.moduleId}` : "");
  const result = sandboxValidate(source);
  if (!result.ok) {
    return { ...n, status: "rejected", moduleId: undefined };
  }
  return { ...n, status: "validated", moduleId: result.moduleId };
}

/** Mark a validated capability as mounted (caller already passed eval). */
export function mountCapability(n: AgentNode): AgentNode {
  if (n.status !== "validated" && n.status !== "mounted") {
    throw new Error(`mountCapability: expected validated/mounted, got ${n.status}`);
  }
  if (!n.moduleId) throw new Error("mountCapability: missing moduleId");
  return { ...n, status: "mounted" };
}

export function rejectCapability(n: AgentNode, _reason?: string): AgentNode {
  return { ...n, status: "rejected" };
}

export function unmountCapability(n: AgentNode): AgentNode {
  return { ...n, status: "unmounted", moduleId: n.moduleId };
}
