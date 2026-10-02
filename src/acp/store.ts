import { existsSync } from "node:fs";
import { mkdir, readdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { AgentGraph } from "../ir.js";
import type { CompactionState } from "./compaction.js";
import type { ChatMessage, ThoughtLevel } from "./llm.js";

export type Mode = "agent" | "ask" | "plan";

/**
 * Resume snapshot (derived cache). The canonical history is events.jsonl in
 * the same directory: ~/.vdom/sessions/--<cwd>--/<id>/{events.jsonl,snapshot.json,blobs/}.
 */
export type SessionRecord = {
  version: 1;
  id: string;
  cwd: string;
  roots: string[];
  title?: string;
  model: string;
  mode: Mode;
  thought: ThoughtLevel;
  graph: AgentGraph;
  /** Conversation without the system prompt (recompiled every step). */
  messages: ChatMessage[];
  /** Routing position + decay counters, so a resumed session keeps its rung. */
  route?: { position: number; hold: number; prevClean: boolean };
  /** Rolling summary + file tracking from earlier compactions. */
  compaction: CompactionState;
  forkedFrom?: string;
  createdAt: string;
  updatedAt: string;
};

const ID_RE = /^[A-Za-z0-9_-]{1,128}$/;

/** Pi-style directory key for a cwd: --C--Users-me-repo-- */
export function cwdKey(cwd: string): string {
  return `--${cwd.replace(/^[\\/]+/, "").replace(/[\\/:]+/g, "-").replace(/[^A-Za-z0-9._-]/g, "_").slice(0, 120)}--`;
}

export class SessionStore {
  /** Session directory by id (stable even if the client later reports another cwd). */
  private dirs = new Map<string, string>();

  constructor(readonly root: string) {}

  dirFor(rec: Pick<SessionRecord, "id" | "cwd">): string {
    if (!ID_RE.test(rec.id)) throw new Error(`invalid session id ${rec.id}`);
    let d = this.dirs.get(rec.id);
    if (!d) {
      d = join(this.root, cwdKey(rec.cwd), rec.id);
      this.dirs.set(rec.id, d);
    }
    return d;
  }

  async save(rec: SessionRecord): Promise<void> {
    const dir = this.dirFor(rec);
    await mkdir(dir, { recursive: true });
    rec.updatedAt = new Date().toISOString();
    const target = join(dir, "snapshot.json");
    const tmp = `${target}.${process.pid}.tmp`;
    await writeFile(tmp, JSON.stringify(rec), "utf8");
    await rename(tmp, target);
  }

  /** Find a session directory by id across cwd folders (and legacy flat files). */
  async locate(id: string): Promise<string | undefined> {
    if (!ID_RE.test(id)) return undefined;
    const known = this.dirs.get(id);
    if (known) return known;
    if (!existsSync(this.root)) return undefined;
    for (const d of await readdir(this.root)) {
      const p = join(this.root, d, id);
      if (existsSync(join(p, "snapshot.json")) || existsSync(join(p, "events.jsonl"))) {
        this.dirs.set(id, p);
        return p;
      }
    }
    return undefined;
  }

  async load(id: string): Promise<SessionRecord | undefined> {
    const dir = await this.locate(id);
    let p = dir ? join(dir, "snapshot.json") : join(this.root, `${id}.json`);
    if (!existsSync(p)) return undefined;
    const rec = JSON.parse(await readFile(p, "utf8")) as SessionRecord;
    if (rec.version !== 1) return undefined;
    if (!dir) {
      // Legacy flat file: migrate into the directory layout on next save.
      this.dirFor(rec);
      p = "";
    }
    return rec;
  }

  async delete(id: string): Promise<void> {
    const dir = await this.locate(id);
    if (dir) await rm(dir, { recursive: true, force: true });
    await rm(join(this.root, `${id}.json`), { force: true });
    this.dirs.delete(id);
  }

  /** Drop never-prompted sessions (editor probes) older than maxAgeMs. */
  async pruneEmpty(maxAgeMs = 24 * 3600_000): Promise<void> {
    const cutoff = new Date(Date.now() - maxAgeMs).toISOString();
    for (const rec of await this.list()) {
      if (rec.messages.length === 0 && rec.updatedAt < cutoff) await this.delete(rec.id);
    }
  }

  async list(): Promise<SessionRecord[]> {
    if (!existsSync(this.root)) return [];
    const out: SessionRecord[] = [];
    const tryRead = async (p: string) => {
      try {
        const rec = JSON.parse(await readFile(p, "utf8")) as SessionRecord;
        if (rec.version === 1) out.push(rec);
      } catch {
        /* partial or foreign file */
      }
    };
    for (const entry of await readdir(this.root, { withFileTypes: true })) {
      if (entry.isFile() && entry.name.endsWith(".json")) await tryRead(join(this.root, entry.name));
      else if (entry.isDirectory()) {
        for (const id of await readdir(join(this.root, entry.name)).catch(() => [] as string[])) {
          const snap = join(this.root, entry.name, id, "snapshot.json");
          if (existsSync(snap)) {
            this.dirs.set(id, join(this.root, entry.name, id));
            await tryRead(snap);
          }
        }
      }
    }
    return out.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
  }
}
