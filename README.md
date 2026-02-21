# spec.ai

Speculative retrieval engine that reduces RAG latency by pre-executing embedding + vector search while users type.

## How It Works

```
User types "how to configur..."  →  spec.ai embeds + searches in background
User presses Enter               →  results served from cache in ~5ms instead of ~300ms
```

1. Client streams keystrokes via WebSocket
2. Server debounces then speculatively embeds + searches
3. Results cached per session with cosine similarity gating
4. On submit: **Hit** (serve cached, ~5ms) / **Partial** (refresh) / **Miss** (fresh retrieval)

## Architecture

```
crates/
├── core/           specai-core (no server deps)
│   ├── config.rs           constants & thresholds
│   ├── types.rs            Document, Embedding, CacheVerdict, EngineStats
│   ├── similarity.rs       cosine similarity + SimilarityGate
│   ├── circuit_breaker.rs  circuit breaker (Closed → Open → HalfOpen)
│   ├── cache/
│   │   ├── mod.rs          SpeculativeCache (DashMap + ring buffer)
│   │   └── eviction.rs     TTL background sweep
│   ├── embedder/
│   │   ├── mod.rs          Embedder trait
│   │   ├── http.rs         HttpEmbedder (OpenAI-compatible)
│   │   └── guarded.rs      GuardedEmbedder (with circuit breaker)
│   ├── retriever/
│   │   ├── mod.rs          Retriever trait
│   │   ├── vectorsdb.rs    VectorsDbRetriever (vectors.db REST API)
│   │   └── guarded.rs      GuardedRetriever (with circuit breaker)
│   └── engine.rs           orchestrator (speculate + submit)
└── server/         specai-server
    ├── main.rs             CLI (clap) + startup + graceful shutdown
    └── api/
        ├── mod.rs          router + middleware stack
        ├── auth.rs         optional API key authentication
        ├── validation.rs   input validation (session_id, query)
        ├── errors.rs       ApiError → HTTP status codes
        ├── handlers.rs     REST handlers + AppState
        ├── models.rs       request/response DTOs
        ├── metrics.rs      Prometheus metrics recording
        ├── docs.rs         OpenAPI spec (utoipa)
        └── ws.rs           WebSocket handler + debounce + rate limiting
```

## Quick Start

```bash
# Build
cargo build --release

# Run (connects to local Ollama + vectors.db)
cargo run --release

# With OpenAI embeddings
SPECAI_EMBEDDING_API_KEY=sk-... cargo run --release -- \
  --embedding-url https://api.openai.com/v1/embeddings \
  --embedding-model text-embedding-3-small \
  --vectorsdb-url http://localhost:3030 \
  --collection my_docs

# With authentication enabled
cargo run --release -- --api-key my-secret-key
```

## API Endpoints

| Method | Path       | Description                          | Auth |
|--------|------------|--------------------------------------|------|
| GET    | `/ws`      | WebSocket for keystroke streaming     | Yes  |
| GET    | `/health`  | Liveness probe (always responds)     | No   |
| GET    | `/ready`   | Readiness probe (checks dependencies)| Yes  |
| GET    | `/stats`   | Engine stats (hits, misses, latency) | Yes  |
| POST   | `/submit`  | REST submission (non-WS clients)     | Yes  |
| GET    | `/metrics` | Prometheus metrics                   | Yes  |
| GET    | `/docs`    | OpenAPI JSON spec                    | Yes  |

## WebSocket Protocol

```json
// Client → Server: keystroke (triggers speculative search after debounce)
{"type": "keystroke", "session_id": "abc", "text": "how to configur"}

// Client → Server: submit final query
{"type": "submit", "session_id": "abc", "query": "how to configure auth"}

// Client → Server: close session
{"type": "close", "session_id": "abc"}

// Server → Client: speculation started
{"type": "speculating", "session_id": "abc", "query": "how to configur"}

// Server → Client: speculation complete
{"type": "speculation_ready", "session_id": "abc", "query": "how to configur", "num_results": 10, "latency_ms": 145}

// Server → Client: final results
{"type": "results", "session_id": "abc", "documents": [...], "cache_verdict": "Hit", "latency_ms": 5}

// Server → Client: error
{"type": "error", "message": "Speculation failed: ..."}
```

### Input Validation

- **session_id**: required, max 128 chars, alphanumeric + `-` + `_` only
- **query/text**: required, max 2000 chars

Invalid input returns a `{"type": "error", ...}` message over WebSocket or HTTP 400 on REST.

### Rate Limiting

Each WebSocket connection is rate-limited to 20 keystrokes per second per session. Excess keystrokes receive an error message and are dropped.

## Authentication

Authentication is **optional** and disabled by default. Enable it with `--api-key`:

```bash
cargo run --release -- --api-key my-secret-key
```

When enabled, all endpoints except `/health` require the header:

```
Authorization: Bearer my-secret-key
```

For WebSocket connections, auth is checked on the HTTP upgrade request.

## Configuration

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--port` / `-p` | `3040` | Listen port |
| `--embedding-url` | `http://localhost:11434/v1/embeddings` | Embedding API endpoint |
| `--embedding-model` | `text-embedding-3-small` | Embedding model name |
| `--embedding-api-key` | — | Bearer token for embedding API |
| `--vectorsdb-url` | `http://localhost:3030` | vectors.db base URL |
| `--collection` | `default` | Collection to search |
| `--vectorsdb-api-key` | — | Bearer token for vectors.db |
| `--similarity-threshold` | `0.92` | Cosine threshold for cache Hit |
| `--partial-threshold` | `0.80` | Cosine threshold for Partial |
| `--top-k` | `10` | Results per retrieval |
| `--debounce-ms` | `300` | Keystroke debounce interval (ms) |
| `--cache-ttl` | `120` | Cache entry TTL (seconds) |
| `--rate-limit-rps` | `200` | Global rate limit (req/sec) |
| `--max-concurrent` | `512` | Max concurrent requests |
| `--request-timeout` | `30` | Request timeout (seconds) |
| `--max-body-size` | `1048576` | Max request body (bytes) |
| `--api-key` | — | API key for server authentication |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `SPECAI_EMBEDDING_API_KEY` | Bearer token for embedding API (fallback for `--embedding-api-key`) |
| `SPECAI_VECTORSDB_API_KEY` | Bearer token for vectors.db (fallback for `--vectorsdb-api-key`) |
| `RUST_LOG` | Log level (e.g. `specai_server=debug,specai_core=debug`) |

## Resilience

### Circuit Breaker

External service calls (embedding API and vectors.db) are protected by a circuit breaker:

- **Closed** (normal): requests flow through
- **Open** (after 5 consecutive failures): requests fail immediately for 30 seconds, avoiding cascading timeouts
- **Half-Open** (after recovery timeout): one probe request is allowed through to test if the service recovered

When the circuit is open, the engine returns a 503 error instead of waiting for a timeout.

### Cache Verdicts

| Verdict | Cosine Similarity | Action |
|---------|------------------|--------|
| **Hit** | >= 0.92 | Return cached results (~5ms) |
| **Partial** | >= 0.80 | Cached results exist but may be stale; fresh retrieval performed |
| **Miss** | < 0.80 | No useful cached results; full embed + search |

### Session Cleanup

- Sessions are automatically evicted after the TTL expires (default 120s)
- When a WebSocket client disconnects (gracefully or abruptly), all its sessions are cleaned up immediately

## Observability

### Health Checks

- **`GET /health`** — Liveness probe. Always returns `200 OK` with uptime and session counts. Use for container orchestration liveness checks.
- **`GET /ready`** — Readiness probe. Verifies connectivity to the embedding API and vectors.db. Returns `200` when both are reachable, `503` otherwise.

### Prometheus Metrics

Available at `GET /metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | counter | HTTP requests by method/path/status |
| `http_request_duration_seconds` | histogram | Request latency by method/path/status |
| `specai_predictions_total` | gauge | Total speculative predictions executed |
| `specai_submissions_total` | gauge | Total submissions processed |
| `specai_cache_hits` | gauge | Cache hit count |
| `specai_cache_partials` | gauge | Cache partial hit count |
| `specai_cache_misses` | gauge | Cache miss count |
| `specai_avg_speculation_latency_ms` | gauge | Average speculation latency |
| `specai_avg_submission_latency_ms` | gauge | Average submission latency |
| `specai_active_sessions` | gauge | Currently active sessions |
| `specai_cached_entries` | gauge | Total cached entries |
| `specai_cache_hit_rate` | gauge | Cache hit rate (hits / submissions) |

### OpenAPI

The full API specification is available at `GET /docs` as JSON. Compatible with Swagger UI, Redoc, and other OpenAPI tools.

## Docker

```bash
docker build -t spec-ai .
docker run -p 3040:3040 spec-ai \
  --embedding-url http://host.docker.internal:11434/v1/embeddings \
  --vectorsdb-url http://host.docker.internal:3030
```

The Docker image uses a multi-stage build (Rust 1.88 builder + Debian slim runtime), runs as a non-root user, and includes a health check against `/health`.

## Development

```bash
cargo build                      # debug build
cargo build --release            # optimized (fat LTO)
cargo test                       # run all 93 tests
cargo clippy -- -D warnings      # lint (zero warnings required)
cargo fmt --check                # format check
```

## License

AGPL-3.0
