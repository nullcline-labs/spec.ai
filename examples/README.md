# spec.ai Example Clients

Example WebSocket clients demonstrating the keystroke streaming, speculation, and submission flow.

## Prerequisites

A running spec.ai server:

```bash
# From project root
cargo run --release

# Or with Docker Compose
docker compose up -d
docker compose exec ollama ollama pull nomic-embed-text
```

## JavaScript Client (Node.js)

Requires Node.js 18+.

```bash
cd examples/js
npm install
node client.js

# With auth:
node client.js --token my-secret-key

# Custom server:
node client.js --url ws://example.com:3040/ws
```

## Python Client

Requires Python 3.9+.

```bash
cd examples/python
pip install websockets

python client.py

# With auth:
python client.py --token my-secret-key

# Custom server:
python client.py --url ws://example.com:3040/ws
```

## What the examples demonstrate

1. Connect to the WebSocket endpoint (`/ws`)
2. Simulate typing "how to configure authentication" character by character
3. Show speculation notifications as the server pre-fetches results
4. Submit the final query and display cached results with cache verdict
5. Close the session cleanly
