#!/usr/bin/env node
// spec.ai example WebSocket client (Node.js)
//
// Usage:
//   node client.js [--url ws://localhost:3040/ws] [--token <api-key>]

const WebSocket = require("ws");

const args = process.argv.slice(2);
let url = "ws://localhost:3040/ws";
let token = null;

for (let i = 0; i < args.length; i++) {
  if (args[i] === "--url" && args[i + 1]) url = args[++i];
  if (args[i] === "--token" && args[i + 1]) token = args[++i];
}

const sessionId = `example-${Date.now()}`;
const query = "how to configure authentication";

console.log(`Connecting to ${url}`);
console.log(`Session: ${sessionId}`);
console.log(`Query: "${query}"\n`);

const wsOptions = {};
if (token) {
  wsOptions.headers = { Authorization: `Bearer ${token}` };
  console.log("Using Bearer token authentication\n");
}

const ws = new WebSocket(url, wsOptions);

ws.on("open", () => {
  console.log("Connected!\n");
  simulateTyping();
});

ws.on("message", (data) => {
  const msg = JSON.parse(data.toString());
  const ts = new Date().toISOString().slice(11, 23);

  switch (msg.type) {
    case "speculating":
      console.log(`[${ts}] SPECULATING: "${msg.query}"`);
      break;
    case "speculation_ready":
      console.log(
        `[${ts}] SPECULATION READY: ${msg.num_results} results in ${msg.latency_ms}ms`
      );
      break;
    case "results":
      console.log(
        `\n[${ts}] RESULTS (verdict: ${msg.cache_verdict}, latency: ${msg.latency_ms}ms)`
      );
      if (msg.documents && msg.documents.length > 0) {
        msg.documents.forEach((doc, i) => {
          const text = doc.text.length > 80 ? doc.text.slice(0, 80) + "..." : doc.text;
          console.log(`  ${i + 1}. [${doc.score.toFixed(3)}] ${text}`);
        });
      } else {
        console.log("  (no documents returned)");
      }
      ws.send(JSON.stringify({ type: "close", session_id: sessionId }));
      setTimeout(() => ws.close(), 500);
      break;
    case "error":
      console.error(`[${ts}] ERROR: ${msg.message}`);
      break;
  }
});

ws.on("close", () => {
  console.log("\nConnection closed.");
  process.exit(0);
});

ws.on("error", (err) => {
  console.error("WebSocket error:", err.message);
  process.exit(1);
});

async function simulateTyping() {
  for (let i = 1; i <= query.length; i++) {
    const partial = query.slice(0, i);

    ws.send(
      JSON.stringify({
        type: "keystroke",
        session_id: sessionId,
        text: partial,
      })
    );

    process.stdout.write(
      `\rTyping: "${partial}"` + " ".repeat(query.length - partial.length)
    );

    await sleep(50 + Math.random() * 100);
  }

  console.log("\n");
  await sleep(500);

  console.log(`Submitting: "${query}"\n`);
  ws.send(
    JSON.stringify({
      type: "submit",
      session_id: sessionId,
      query: query,
    })
  );
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
