/** Minimal stdio MCP server used by acp-selftest. */
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({ name: "echo", version: "1.0.0" }, { instructions: "Echo server for tests." });

server.registerTool(
  "echo",
  { description: "Echo text back", inputSchema: { text: z.string() }, annotations: { readOnlyHint: true } },
  async ({ text }) => ({ content: [{ type: "text", text: `echo: ${text}` }] }),
);

server.registerTool(
  "write_note",
  { description: "Pretend to write a note (mutating)", inputSchema: { note: z.string() } },
  async ({ note }) => ({ content: [{ type: "text", text: `saved ${note}` }] }),
);

await server.connect(new StdioServerTransport());
