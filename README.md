# spec.ai

Speculative retrieval engine that reduces RAG latency by pre-executing embedding + vector search while users type.

## How It Works

```
User types "how to configur..."  →  spec.ai embeds + searches in background
User presses Enter               →  results served from cache in ~5ms instead of ~300ms
```

1. Client streams keystrokes via WebSocket
2. Server debounces (300ms pause) then speculatively embeds + searches
3. Results cached per session with cosine similarity gating
4. On submit: **Hit** (serve cached, ~5ms) / **Partial** (refresh) / **Miss** (fresh retrieval)

## Quick Start

```bash
# Build
cargo build --release

# Run (connects to local Ollama + vectors.db)
cargo run --release

# With OpenAI
SPECAI_EMBEDDING_API_KEY=sk-... cargo run --release -- \
  --embedding-url https://api.openai.com/v1/embeddings \
  --vectorsdb-url http://localhost:3030 \
  --collection my_docs
```

## WebSocket Protocol

```json
// Client sends keystrokes
{"type": "keystroke", "session_id": "abc", "text": "how to configur"}

// Client submits final query
{"type": "submit", "session_id": "abc", "query": "how to configure auth"}

// Server responds with results
{"type": "results", "session_id": "abc", "documents": [...], "cache_verdict": "Hit", "latency_ms": 5}
```

## REST API

| Method | Path     | Description                      |
|--------|----------|----------------------------------|
| GET    | /ws      | WebSocket endpoint               |
| GET    | /health  | Health check                     |
| GET    | /stats   | Cache hit rate, latencies        |
| POST   | /submit  | Synchronous submission           |
| GET    | /metrics | Prometheus metrics               |

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 3040 | Listen port |
| `--embedding-url` | `http://localhost:11434/v1/embeddings` | Embedding API |
| `--embedding-model` | `text-embedding-3-small` | Model name |
| `--vectorsdb-url` | `http://localhost:3030` | vectors.db URL |
| `--collection` | `default` | Collection to search |
| `--similarity-threshold` | 0.92 | Cache hit threshold |
| `--top-k` | 10 | Results per search |
| `--debounce-ms` | 300 | Keystroke debounce |
| `--cache-ttl` | 120 | Cache TTL (seconds) |

## Docker

```bash
docker build -t spec-ai .
docker run -p 3040:3040 spec-ai \
  --embedding-url http://host.docker.internal:11434/v1/embeddings \
  --vectorsdb-url http://host.docker.internal:3030
```

## License

AGPL-3.0
