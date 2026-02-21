#!/usr/bin/env python3
"""spec.ai example WebSocket client (Python).

Usage:
    python client.py [--url ws://localhost:3040/ws] [--token <api-key>]
"""

import argparse
import asyncio
import json
import random
import time

try:
    import websockets
except ImportError:
    print("Install websockets: pip install websockets")
    raise SystemExit(1)


async def main():
    parser = argparse.ArgumentParser(description="spec.ai WebSocket example client")
    parser.add_argument("--url", default="ws://localhost:3040/ws", help="WebSocket URL")
    parser.add_argument("--token", default=None, help="Bearer token for authentication")
    args = parser.parse_args()

    session_id = f"example-{int(time.time() * 1000)}"
    query = "how to configure authentication"

    print(f"Connecting to {args.url}")
    print(f"Session: {session_id}")
    print(f'Query: "{query}"\n')

    headers = {}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"
        print("Using Bearer token authentication\n")

    async with websockets.connect(args.url, additional_headers=headers) as ws:
        print("Connected!\n")

        listener_task = asyncio.create_task(listen(ws))

        await simulate_typing(ws, session_id, query)

        await asyncio.sleep(0.5)
        print(f'Submitting: "{query}"\n')
        await ws.send(json.dumps({
            "type": "submit",
            "session_id": session_id,
            "query": query,
        }))

        # Wait for results
        await asyncio.sleep(3)

        await ws.send(json.dumps({
            "type": "close",
            "session_id": session_id,
        }))

        listener_task.cancel()
        try:
            await listener_task
        except asyncio.CancelledError:
            pass

    print("\nConnection closed.")


async def listen(ws):
    """Listen for server messages and print them."""
    try:
        async for raw in ws:
            msg = json.loads(raw)
            ts = time.strftime("%H:%M:%S", time.localtime())

            if msg["type"] == "speculating":
                print(f'[{ts}] SPECULATING: "{msg["query"]}"')

            elif msg["type"] == "speculation_ready":
                print(
                    f"[{ts}] SPECULATION READY: "
                    f'{msg["num_results"]} results in {msg["latency_ms"]}ms'
                )

            elif msg["type"] == "results":
                verdict = msg["cache_verdict"]
                latency = msg["latency_ms"]
                print(f"\n[{ts}] RESULTS (verdict: {verdict}, latency: {latency}ms)")
                docs = msg.get("documents", [])
                if docs:
                    for i, doc in enumerate(docs, 1):
                        score = doc["score"]
                        text = doc["text"][:80]
                        if len(doc["text"]) > 80:
                            text += "..."
                        print(f"  {i}. [{score:.3f}] {text}")
                else:
                    print("  (no documents returned)")

            elif msg["type"] == "error":
                print(f"[{ts}] ERROR: {msg['message']}")

    except websockets.ConnectionClosed:
        pass


async def simulate_typing(ws, session_id: str, query: str):
    """Simulate typing character by character with realistic delays."""
    for i in range(1, len(query) + 1):
        partial = query[:i]

        await ws.send(json.dumps({
            "type": "keystroke",
            "session_id": session_id,
            "text": partial,
        }))

        padding = " " * (len(query) - len(partial))
        print(f'\rTyping: "{partial}"{padding}', end="", flush=True)

        await asyncio.sleep(0.05 + random.random() * 0.1)

    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
