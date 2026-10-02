/**
 * Minimal ACP agent over stdio used by acp-selftest. At startup it spawns a
 * grandchild process that heartbeats a file every 500ms; its session/prompt
 * never completes (the turn stays open forever). Regression fixture for
 * I-20261002-7e86: when the driving client dies (stdin close / signal /
 * exit), it must kill this whole process tree; a bare child.kill() leaves the
 * grandchild running.
 *
 * Plain .mjs on purpose: it runs under bare `node` with the temp workspace as
 * cwd, so no TypeScript loader may be required.
 */
import { spawn } from "node:child_process";
import { appendFileSync, writeFileSync } from "node:fs";
import { randomUUID } from "node:crypto";
import { PROTOCOL_VERSION } from "@agentclientprotocol/sdk";

const marker = process.env.VDOM_TREE_MARKER;
const heartbeat = process.env.VDOM_TREE_HEARTBEAT;

const info = { agent: process.pid, grandchild: 0, prompted: false };
const save = () => writeFileSync(marker, `${JSON.stringify(info)}\n`);

// Grandchild: heartbeats the file every 500ms forever — liveness without
// trusting pids. On POSIX it stays in this agent's process group so a group
// kill reaches it; on Windows it is detached from the console but still the
// agent's child, which taskkill /T must reach through the parent link.
const grandchild = spawn(
  process.execPath,
  [
    "-e",
    `const fs = require("node:fs");
fs.writeFileSync(${JSON.stringify(heartbeat)}, "");
setInterval(() => { try { fs.appendFileSync(${JSON.stringify(heartbeat)}, "."); } catch {} }, 500);
setInterval(() => {}, 60000);`,
  ],
  { stdio: "ignore", detached: process.platform === "win32", windowsHide: true },
);
grandchild.unref();
info.grandchild = grandchild.pid;
save();

const send = (o) => process.stdout.write(`${JSON.stringify(o)}\n`);
process.stdin.resume();
process.stdin.on("data", (d) => {
  for (const line of d.toString("utf8").split("\n")) {
    if (!line.trim()) continue;
    let msg;
    try {
      msg = JSON.parse(line);
    } catch {
      continue;
    }
    const reply = (result) => send({ jsonrpc: "2.0", id: msg.id, result });
    switch (msg.method) {
      case "initialize":
        reply({
          protocolVersion: PROTOCOL_VERSION,
          agentCapabilities: {},
          authMethods: [],
          agentInfo: { name: "tree-agent", title: "tree agent", version: "0.0.0" },
        });
        break;
      case "authenticate":
        reply({});
        break;
      case "session/new":
        reply({ sessionId: `tree-${randomUUID().slice(0, 8)}`, modes: [], configOptions: [] });
        break;
      case "session/prompt":
        // Mid-turn forever: acknowledge by flag, never send the response.
        info.prompted = true;
        save();
        break;
      default:
        reply({});
    }
  }
});