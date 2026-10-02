import type { ConfigOverrides } from "./config.js";
import type { ApprovePolicy } from "./client.js";

/** Options for `vdom run` / `vdom client` (driving an agent over ACP). */
export type DriveFlags = {
  prompts: string[];
  session?: string;
  continueLast: boolean;
  mode?: string;
  thought?: string;
  approve?: ApprovePolicy;
  trace?: string;
  showThinking: boolean;
  verbose: boolean;
  timeoutSec?: number;
  /** Agent command for `vdom client` (default: this vdom). */
  agent?: string;
};

export type Parsed = {
  command: string;
  positional: string[];
  overrides: ConfigOverrides;
  cwd?: string;
  json: boolean;
  drive: DriveFlags;
  /** env / fix / inspect flags */
  ops: { repo?: string; ref?: string; repro: string[]; since?: string; promote: boolean; force: boolean; faults: boolean };
};

export function parseArgs(argv: string[]): Parsed {
  const overrides: ConfigOverrides = {};
  const positional: string[] = [];
  const drive: DriveFlags = { prompts: [], continueLast: false, showThinking: false, verbose: false };
  let cwd: string | undefined;
  let json = false;
  const ops: Parsed["ops"] = { repro: [], promote: false, force: false, faults: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    const next = () => {
      const v = argv[++i];
      if (v === undefined) throw new Error(`${a} needs a value`);
      return v;
    };
    switch (a) {
      case "--model":
      case "-m":
        overrides.model = next();
        break;
      case "--base-url":
      case "--endpoint":
      case "-e":
        overrides.baseUrl = next();
        break;
      case "--config":
        overrides.configPath = next();
        break;
      case "--cwd":
        cwd = next();
        break;
      case "--force":
      case "--yolo":
      case "--always-approve":
        overrides.fullAccess = true;
        break;
      case "--format":
        json = next() === "json";
        break;
      case "--json":
        json = true;
        break;
      case "--permission-mode": {
        const v = next();
        if (v === "bypassPermissions" || v === "full-access") overrides.fullAccess = true;
        break;
      }
      // Driving flags.
      case "-p":
      case "--prompt":
        drive.prompts.push(next());
        break;
      case "--session":
      case "-s":
        drive.session = next();
        break;
      case "--continue":
      case "-c":
        drive.continueLast = true;
        break;
      case "--mode":
        drive.mode = next();
        break;
      case "--thought":
      case "--thinking-level":
        drive.thought = next();
        break;
      case "--approve": {
        const v = next();
        if (!["allow", "always", "reject", "prompt"].includes(v)) throw new Error("--approve must be allow|always|reject|prompt");
        drive.approve = v as ApprovePolicy;
        break;
      }
      case "--trace":
        drive.trace = next();
        break;
      case "--show-thinking":
        drive.showThinking = true;
        break;
      case "--verbose":
      case "-V":
        drive.verbose = true;
        break;
      case "--timeout":
        drive.timeoutSec = Number(next());
        break;
      case "--repo":
        ops.repo = next();
        break;
      case "--ref":
        ops.ref = next();
        break;
      case "--repro":
        ops.repro.push(next());
        break;
      case "--since":
        ops.since = next();
        break;
      case "--promote":
        ops.promote = true;
        break;
      case "--faults":
        ops.faults = true;
        break;
      case "--force-promote":
        ops.force = true;
        break;
      case "--agent":
        drive.agent = next();
        break;
      // Accepted for drop-in compatibility; default behaviour already matches.
      case "--auto-review":
      case "--stdio":
        break;
      case "-h":
      case "--help":
        positional.unshift("help");
        break;
      case "-v":
      case "--version":
        positional.unshift("version");
        break;
      default:
        if (a.startsWith("--")) process.stderr.write(`vdom: ignoring unknown flag ${a}\n`);
        else positional.push(a);
    }
  }
  let command = positional.shift() ?? "acp";
  // `grok agent stdio` shape.
  if (command === "agent" && positional[0] === "stdio") {
    positional.shift();
    command = "acp";
  }
  return { command, positional, overrides, cwd, json, drive, ops };
}

/** Split a command string honoring double/single quotes. */
export function splitCommand(s: string): string[] {
  const out: string[] = [];
  const re = /"((?:[^"\\]|\\.)*)"|'([^']*)'|(\S+)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(s))) out.push(m[1] ?? m[2] ?? m[3]!);
  return out;
}
