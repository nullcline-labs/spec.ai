<p align="center">
  <img src="https://win98icons.alexmeub.com/icons/png/doctor_watson.png" width="128" height="128" alt="spec.ai logo">
</p>

<h1 align="center">spec.ai</h1>

<p align="center">
  Speculative retrieval engine that reduces RAG latency by pre-executing embedding + vector search while users type.
</p>

<p align="center">
  <a href="https://github.com/nullcline-labs/spec.ai/actions/workflows/ci.yml">
    <img src="https://github.com/nullcline-labs/spec.ai/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
</p>

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
│   ├── audit.rs            structured query audit logging (specai_audit target)
│   ├── cache/
│   │   ├── mod.rs          SpeculativeCache (DashMap + ring buffer)
│   │   └── eviction.rs     TTL background sweep
│   ├── embedder/
│   │   ├── mod.rs          Embedder trait
│   │   ├── http.rs         HttpEmbedder (OpenAI-compatible)
│   │   ├── guarded.rs      GuardedEmbedder (circuit breaker decorator)
│   │   └── cached.rs       CachedEmbedder (embedding deduplication cache)
│   ├── retriever/
│   │   ├── mod.rs          Retriever trait
│   │   ├── vectorsdb.rs    VectorsDbRetriever (multi-collection fan-out search)
│   │   └── guarded.rs      GuardedRetriever (circuit breaker decorator)
│   ├── reranker/
│   │   ├── mod.rs          Reranker trait + WeightedReranker
│   │   └── text_relevance.rs  Jaccard text similarity
│   └── engine.rs           orchestrator (speculate + submit + rerank + audit)
└── server/         specai-server
    ├── main.rs             CLI (clap) + TOML config + TLS + startup
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

### Embedder Composition Chain

```
HttpEmbedder → GuardedEmbedder (circuit breaker) → CachedEmbedder (dedup cache)
```

Cache hits skip both circuit breaker and HTTP calls entirely.

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

# With TOML config file
cargo run --release -- --config specai.toml

# With TLS
cargo run --release -- --tls-cert cert.pem --tls-key key.pem

# Multi-collection search
cargo run --release -- --collection "docs,faq,guides"

# With result re-ranking
cargo run --release -- --rerank --rerank-alpha 0.7
```

## Docker Compose

A full local development stack is provided via Docker Compose:

```bash
docker compose up -d
```

This starts three services:
- **Ollama** — local embedding model server
- **vectors.db** — vector database
- **spec.ai** — the retrieval engine (auto-builds from source)

All services include health checks and `depends_on` conditions. See `docker-compose.override.example.yml` for development overrides.

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

## Example Clients

Ready-to-use WebSocket clients for testing and integration:

```bash
# Node.js
cd examples/js && npm install && node client.js

# Python
pip install websockets
python examples/python/client.py

# With authentication
node examples/js/client.js --token my-secret-key
python examples/python/client.py --token my-secret-key
```

Both clients demonstrate keystroke streaming, speculation notifications, and result display. See [`examples/README.md`](examples/README.md) for details.

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

### Config File (TOML)

spec.ai supports a TOML config file to avoid long CLI commands. Priority: **CLI > env var > config file > default**.

```bash
# Explicit path (error if not found)
cargo run --release -- --config /path/to/specai.toml

# Auto-discovery: looks for specai.toml in current directory
cargo run --release
```

See [`specai.example.toml`](specai.example.toml) for all available options. Unknown fields are rejected to catch typos.

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--config` / `-c` | — | Path to TOML config file |
| `--port` / `-p` | `3040` | Listen port |
| `--embedding-url` | `http://localhost:11434/v1/embeddings` | Embedding API endpoint |
| `--embedding-model` | `text-embedding-3-small` | Embedding model name |
| `--embedding-api-key` | — | Bearer token for embedding API |
| `--vectorsdb-url` | `http://localhost:3030` | vectors.db base URL |
| `--collection` | `default` | Comma-separated collection names |
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
| `--rerank` | `false` | Enable result re-ranking |
| `--rerank-alpha` | `0.7` | Re-rank weight (1.0 = vector only, 0.0 = text only) |
| `--allowed-origins` | — | Comma-separated allowed WebSocket origins (CSRF) |
| `--tls-cert` | — | TLS certificate PEM (requires `--tls-key`) |
| `--tls-key` | — | TLS private key PEM (requires `--tls-cert`) |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `SPECAI_EMBEDDING_API_KEY` | Bearer token for embedding API (fallback for `--embedding-api-key`) |
| `SPECAI_VECTORSDB_API_KEY` | Bearer token for vectors.db (fallback for `--vectorsdb-api-key`) |
| `RUST_LOG` | Log level (e.g. `specai_server=debug,specai_core=debug`) |

## Features

### Embedding Cache

Deduplicates embedding API calls by caching `query_text → Embedding` in a DashMap with TTL. When the same text is requested again (e.g. identical keystrokes from multiple sessions), the cached embedding is returned without hitting the embedding API or circuit breaker.

- Max 1000 entries, 5-minute TTL (configurable in `config.rs`)
- Lazy eviction on insert when at capacity
- Hit/miss counters exposed via `EngineStats`

### Multi-Collection Search

Search across multiple vector collections in parallel:

```bash
cargo run --release -- --collection "docs,faq,guides"
```

Results from all collections are merged, sorted by score descending, and truncated to `top_k`. If any collection fails, the entire search returns an error.

### Result Re-ranking

Optional post-retrieval re-ranking that combines vector similarity with text relevance:

```bash
cargo run --release -- --rerank --rerank-alpha 0.7
```

`score = alpha * vector_score + (1 - alpha) * jaccard_similarity`

- `alpha = 1.0`: pure vector score (no re-ranking effect)
- `alpha = 0.0`: pure text relevance
- Default `alpha = 0.7`: balanced

The `Reranker` trait is extensible — swap in a cross-encoder or other model.

### Query Audit Log

All speculations and submissions are logged via a dedicated tracing target `specai_audit`:

```bash
RUST_LOG=specai_audit=info cargo run --release
```

Events: `speculation_started`, `speculation_complete`, `submission_received`, `submission_result` — each with structured fields (session_id, query, verdict, num_results, latency_ms).

### TLS / HTTPS

Enable HTTPS/WSS with certificate and key PEM files:

```bash
cargo run --release -- --tls-cert cert.pem --tls-key key.pem
```

Both `--tls-cert` and `--tls-key` are required together. When omitted, the server runs plain HTTP as before. Uses `rustls` (no OpenSSL dependency).

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
| **StaleFallback** | >= 0.80 | Partial hit but retrieval failed; serve stale cached results |
| **Miss** | < 0.80 | No useful cached results; full embed + search |

### Stale Fallback

When a Partial cache hit occurs but the vector database is unreachable, the engine falls back to serving stale cached results instead of returning an error. This ensures graceful degradation — users still get results (possibly slightly outdated) rather than a failure.

### Session Cleanup

- Sessions are automatically evicted after the TTL expires (default 120s)
- When a WebSocket client disconnects (gracefully or abruptly), all its sessions are cleaned up immediately
- Global cache capacity is limited to 50,000 entries to prevent OOM

## Security

### WebSocket Origin Validation

Restrict which origins can connect via WebSocket to prevent CSRF attacks:

```bash
cargo run --release -- --allowed-origins "https://app.example.com,https://staging.example.com"
```

When configured, connections without a matching `Origin` header receive a 403 Forbidden response. When not configured, all origins are allowed (backwards compatible).

### Per-IP Connection Limiting

Each IP address is limited to 50 concurrent WebSocket connections. Excess connections receive a 429 Too Many Requests response. The counter is decremented when connections close.

### Auth Brute Force Protection

When API key authentication is enabled, repeated failures from the same IP trigger a lockout:

- After **10 failed attempts** within a 5-minute window, the IP receives 429 Too Many Requests
- The lockout expires after 5 minutes of inactivity
- Successful authentication resets the failure counter

### Speculation Timeout

Each speculative search operation (embed + retrieve) has a 10-second timeout. If the operation doesn't complete in time, it's cancelled and an error message is sent to the client.

### Startup Embedding Validation

On startup, the server sends a test embedding request to validate connectivity and logs the embedding dimension. If the dimension is outside the expected range (64–4096), a warning is emitted. This is non-blocking — the server starts regardless.

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
| `specai_stale_fallbacks` | gauge | Stale fallback count (Partial + retriever failure) |
| `specai_auth_failures_total` | counter | Authentication failure count |
| `specai_speculation_timeouts_total` | counter | Speculation timeout count |

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
cargo test                       # run all 140 tests
cargo clippy -- -D warnings      # lint (zero warnings required)
cargo fmt --check                # format check
```

## License

AGPL-3.0
