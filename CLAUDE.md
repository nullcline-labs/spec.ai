# spec.ai — Speculative Retrieval Engine

## Overview

Rust-based speculative retrieval engine that reduces RAG latency by pre-executing embedding + vector search while users type. Uses WebSocket keystroke streaming with server-side debounce, a DashMap-based speculative cache, and cosine similarity gating (Hit/Partial/Miss).

## Build & Test

```bash
cargo build                      # debug build
cargo build --release            # optimized (fat LTO)
cargo test                       # run all 93 tests
cargo clippy -- -D warnings      # lint
cargo fmt --check                # format check
```

## Run

```bash
# Default: connects to local Ollama + vectors.db
cargo run --release

# With OpenAI embeddings + remote vectors.db
SPECAI_EMBEDDING_API_KEY=sk-... cargo run --release -- \
  --embedding-url https://api.openai.com/v1/embeddings \
  --embedding-model text-embedding-3-small \
  --vectorsdb-url http://vectors.example.com:3030 \
  --collection my_docs \
  --port 3040

# With authentication
cargo run --release -- --api-key my-secret-key
```

## Architecture

```
crates/
├── core/           specai-core (no server deps)
│   ├── config.rs           constants (thresholds, timeouts, limits)
│   ├── types.rs            Document, Embedding, CacheVerdict, EngineStats
│   ├── similarity.rs       cosine similarity + SimilarityGate
│   ├── circuit_breaker.rs  circuit breaker (Closed → Open → HalfOpen)
│   ├── cache/
│   │   ├── mod.rs          SpeculativeCache (DashMap + ring buffer per session)
│   │   └── eviction.rs     TTL background sweep
│   ├── embedder/
│   │   ├── mod.rs          Embedder trait + health_check
│   │   ├── http.rs         HttpEmbedder (OpenAI-compatible /v1/embeddings)
│   │   └── guarded.rs      GuardedEmbedder (circuit breaker wrapper)
│   ├── retriever/
│   │   ├── mod.rs          Retriever trait + health_check
│   │   ├── vectorsdb.rs    VectorsDbRetriever (calls vectors.db REST API)
│   │   └── guarded.rs      GuardedRetriever (circuit breaker wrapper)
│   └── engine.rs           Engine orchestrator (speculate + submit + check_readiness)
└── server/         specai-server
    ├── main.rs             CLI (clap) + startup + graceful shutdown
    └── api/
        ├── mod.rs          router + tower middleware stack + RouterConfig
        ├── auth.rs         optional API key authentication middleware
        ├── validation.rs   input validation (session_id, query)
        ├── errors.rs       ApiError enum → HTTP status codes
        ├── handlers.rs     AppState + REST handlers (/health, /ready, /stats, /submit)
        ├── models.rs       request/response DTOs (with utoipa::ToSchema)
        ├── metrics.rs      Prometheus metrics recording (all EngineStats fields)
        ├── docs.rs         OpenAPI spec generation (utoipa)
        └── ws.rs           WebSocket handler + debounce + rate limiting + session cleanup
```

## Key Design Decisions

- Core is async (must call external HTTP services)
- DashMap for cache (high-contention from many WS sessions)
- Ring buffer per session (max 5 entries, bounds memory)
- Trait objects (Arc<dyn Embedder/Retriever>) for swappable providers
- GuardedEmbedder/GuardedRetriever add circuit breaker via decorator pattern
- Debounce is server-side (consistent behavior regardless of client)
- Three-tier verdict: Hit (>=0.92) / Partial (>=0.80) / Miss (<0.80)
- Circuit breaker: 5 consecutive failures -> open for 30s -> half-open probe
- Optional API key auth (disabled by default, backwards compatible)
- Input validation on both REST and WebSocket handlers
- Per-session keystroke rate limiting (20/sec)
- Session cleanup on WebSocket disconnect (graceful or abrupt)

## API Endpoints

| Method | Path      | Description                          | Auth |
|--------|-----------|--------------------------------------|------|
| GET    | /ws       | WebSocket for keystroke streaming     | Yes  |
| GET    | /health   | Liveness probe (always responds)     | No   |
| GET    | /ready    | Readiness probe (checks dependencies)| Yes  |
| GET    | /stats    | Engine stats (hits, misses, latency) | Yes  |
| POST   | /submit   | REST submission (non-WS clients)     | Yes  |
| GET    | /metrics  | Prometheus metrics                   | Yes  |
| GET    | /docs     | OpenAPI JSON spec                    | Yes  |

## WebSocket Protocol

```json
// Client → Server
{"type": "keystroke", "session_id": "abc", "text": "how to configur"}
{"type": "submit", "session_id": "abc", "query": "how to configure auth"}
{"type": "close", "session_id": "abc"}

// Server → Client
{"type": "speculating", "session_id": "abc", "query": "how to configur"}
{"type": "speculation_ready", "session_id": "abc", "query": "how to configur", "num_results": 10, "latency_ms": 145}
{"type": "results", "session_id": "abc", "documents": [...], "cache_verdict": "Hit", "latency_ms": 5}
{"type": "error", "message": "Speculation failed: ..."}
```

### Input Validation

- session_id: non-empty, max 128 chars, alphanumeric + `-` + `_`
- query/text: non-empty, max 2000 chars

## Environment Variables

| Variable                   | Description                        |
|----------------------------|------------------------------------|
| SPECAI_EMBEDDING_API_KEY   | Bearer token for embedding API     |
| SPECAI_VECTORSDB_API_KEY   | Bearer token for vectors.db        |
| RUST_LOG                   | Log level (e.g. specai_server=debug) |

## Configuration Constants (config.rs)

| Constant                         | Default   | Description                      |
|----------------------------------|-----------|----------------------------------|
| DEFAULT_DEBOUNCE_MS              | 300       | Keystroke debounce interval      |
| DEFAULT_SIMILARITY_THRESHOLD     | 0.92      | Cosine threshold for cache hit   |
| DEFAULT_PARTIAL_THRESHOLD        | 0.80      | Threshold for partial hit        |
| DEFAULT_CACHE_TTL_SECS           | 120       | Cache entry TTL                  |
| MAX_ENTRIES_PER_SESSION          | 5         | Ring buffer size per session     |
| DEFAULT_TOP_K                    | 10        | Results per retrieval            |
| DEFAULT_PORT                     | 3040      | HTTP/WS port                     |
| RATE_LIMIT_RPS                   | 200       | Global rate limit (req/sec)      |
| MAX_CONCURRENT_REQUESTS          | 512       | Max concurrent requests          |
| REQUEST_TIMEOUT_SECS             | 30        | Request timeout                  |
| MAX_REQUEST_BODY_BYTES           | 1048576   | Max request body (1MB)           |
| MAX_SESSION_ID_LENGTH            | 128       | Max session ID length            |
| MAX_QUERY_LENGTH                 | 2000      | Max query length                 |
| MAX_KEYSTROKES_PER_SECOND        | 20        | Per-session WS rate limit        |
| CIRCUIT_BREAKER_FAILURE_THRESHOLD| 5         | Failures before circuit opens    |
| CIRCUIT_BREAKER_RECOVERY_SECS    | 30        | Seconds before half-open         |

## Code Conventions

- `cargo clippy -- -D warnings` must pass (zero warnings)
- `cargo fmt` enforced
- No panics in handlers (all errors via ApiError)
- Structured JSON logging via tracing
- All new features must include tests
- 93 tests total (33 core unit, 9 server unit, 51 server integration)
