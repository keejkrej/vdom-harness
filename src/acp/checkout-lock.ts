import { createHash } from "node:crypto";
import { mkdir, open, readFile, rm } from "node:fs/promises";
import { join, resolve } from "node:path";
import { vdomHome } from "./config.js";

/**
 * Cross-process lock for one checkout (I-20261002-cf19).
 * Two `vdom acp` processes editing one directory interleave reads and edits.
 * Sessions inside one process share the lock: the suite and a long-lived
 * agent open many sessions in one cwd, and in-process edits are already
 * serialized. A crashed holder is stolen when its pid is gone.
 */
export type CheckoutHolder = { pid: number; cwd: string; startedAt: string };

export type TakeResult = { ok: true; stole?: CheckoutHolder } | { ok: false; holder: CheckoutHolder };

export function pidAlive(pid: number): boolean {
  if (!pid || pid === process.pid) return true;
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    return (err as NodeJS.ErrnoException).code === "EPERM";
  }
}

export function checkoutKey(cwd: string): string {
  const full = resolve(cwd);
  const key = process.platform === "win32" ? full.toLowerCase() : full;
  return createHash("sha1").update(key).digest("hex").slice(0, 16);
}

function lockFile(cwd: string, dir: string): string {
  return join(dir, `${checkoutKey(cwd)}.json`);
}

async function readHolder(path: string): Promise<CheckoutHolder | undefined> {
  try {
    const parsed = JSON.parse(await readFile(path, "utf8")) as CheckoutHolder;
    if (!parsed || typeof parsed.pid !== "number") return undefined;
    return parsed;
  } catch {
    return undefined;
  }
}

export async function takeCheckoutLock(opts: {
  cwd: string;
  dir?: string;
  alive?: (pid: number) => boolean;
}): Promise<TakeResult> {
  const dir = opts.dir ?? join(vdomHome(), "locks");
  const alive = opts.alive ?? pidAlive;
  const path = lockFile(opts.cwd, dir);
  await mkdir(dir, { recursive: true });
  const mine: CheckoutHolder = { pid: process.pid, cwd: resolve(opts.cwd), startedAt: new Date().toISOString() };

  for (let attempt = 0; attempt < 3; attempt++) {
    let fh;
    try {
      fh = await open(path, "wx");
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== "EEXIST") throw err;
      const holder = await readHolder(path);
      if (holder?.pid === process.pid) return { ok: true };
      if (holder && alive(holder.pid)) return { ok: false, holder };
      await rm(path, { force: true });
      continue;
    }
    try {
      await fh.writeFile(JSON.stringify(mine));
    } finally {
      await fh.close();
    }
    return { ok: true };
  }
  const holder = await readHolder(path);
  if (holder?.pid === process.pid) return { ok: true };
  if (holder && alive(holder.pid)) return { ok: false, holder };
  throw new Error(`could not acquire checkout lock for ${opts.cwd}`);
}

export async function releaseCheckoutLock(opts: { cwd: string; dir?: string }): Promise<void> {
  const dir = opts.dir ?? join(vdomHome(), "locks");
  const path = lockFile(opts.cwd, dir);
  const holder = await readHolder(path);
  if (holder?.pid === process.pid) await rm(path, { force: true });
}
