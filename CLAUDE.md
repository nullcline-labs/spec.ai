# spec.ai — Speculative Retrieval Engine

## Overview

Rust-based speculative retrieval engine that reduces RAG latency by pre-executing embedding + vector search while users type. Uses WebSocket keystroke streaming with server-side debounce, a DashMap-based speculative cache, and cosine similarity gating (Hit/Partial/Miss).

## Build & Test

```bash
cargo build                      # debug build
cargo build --release            # optimized (fat LTO)
cargo test                       # run all 140 tests
cargo clippy -- -D warnings      # lint
cargo fmt --check                # format check
```

## Run

```bash
# Default: connects to local Ollama + vectors.db
cargo run --release

# With config file
cargo run --release -- --config specai.toml

# With OpenAI embeddings + remote vectors.db
SPECAI_EMBEDDING_API_KEY=sk-... cargo run --release -- \
  --embedding-url https://api.openai.com/v1/embeddings \
  --embedding-model text-embedding-3-small \
  --vectorsdb-url http://vectors.example.com:3030 \
  --collection my_docs \
  --port 3040

# With authentication
cargo run --release -- --api-key my-secret-key

# With TLS
cargo run --release -- --tls-cert cert.pem --tls-key key.pem

# Docker Compose (full stack)
docker compose up -d
```

## Architecture

```
crates/
├── core/           specai-core (no server deps)
│   ├── config.rs           29 constants (thresholds, timeouts, limits, security)
│   ├── types.rs            Document, Embedding, CacheVerdict (Hit/Partial/Miss/StaleFallback), EngineStats
│   ├── similarity.rs       cosine similarity + SimilarityGate
│   ├── circuit_breaker.rs  circuit breaker (Closed → Open → HalfOpen)
│   ├── audit.rs            structured query audit logging (specai_audit target)
│   ├── cache/
│   │   ├── mod.rs          SpeculativeCache (DashMap + ring buffer per session)
│   │   └── eviction.rs     TTL background sweep
│   ├── embedder/
│   │   ├── mod.rs          Embedder trait + health_check
│   │   ├── http.rs         HttpEmbedder (OpenAI-compatible /v1/embeddings)
│   │   ├── guarded.rs      GuardedEmbedder (circuit breaker wrapper)
│   │   └── cached.rs       CachedEmbedder (embedding dedup cache, DashMap + TTL)
│   ├── retriever/
│   │   ├── mod.rs          Retriever trait + health_check
│   │   ├── vectorsdb.rs    VectorsDbRetriever (multi-collection parallel fan-out)
│   │   └── guarded.rs      GuardedRetriever (circuit breaker wrapper)
│   ├── reranker/
│   │   ├── mod.rs          Reranker trait + WeightedReranker (alpha-blended scoring)
│   │   └── text_relevance.rs  Jaccard text similarity
│   └── engine.rs           Engine orchestrator (speculate + submit + rerank + audit)
└── server/         specai-server
    ├── main.rs             CLI (clap, 23 args) + TOML config + TLS + embedding validation + startup
    └── api/
        ├── mod.rs          router + tower middleware stack + RouterConfig
        ├── auth.rs         API key auth + brute force protection (IP lockout)
        ├── validation.rs   input validation (session_id, query)
        ├── errors.rs       ApiError enum → HTTP status codes
        ├── handlers.rs     AppState (8 fields) + REST handlers
        ├── models.rs       request/response DTOs (with utoipa::ToSchema)
        ├── metrics.rs      Prometheus metrics (15 metrics: gauges + counters)
        ├── docs.rs         OpenAPI spec generation (utoipa)
        └── ws.rs           WebSocket handler + debounce + rate limiting + Origin check + per-IP limit
```

### Embedder Composition Chain

```
HttpEmbedder → GuardedEmbedder (circuit breaker) → CachedEmbedder (dedup cache)
```

## Key Design Decisions

- Core is async (must call external HTTP services)
- DashMap for cache (high-contention from many WS sessions)
- Ring buffer per session (max 5 entries, bounds memory)
- Trait objects (Arc<dyn Embedder/Retriever/Reranker>) for swappable providers
- GuardedEmbedder/GuardedRetriever add circuit breaker via decorator pattern
- CachedEmbedder wraps GuardedEmbedder: cache hits skip circuit breaker + HTTP
- Multi-collection search via parallel fan-out + merge + sort by score
- Optional re-ranking: WeightedReranker (alpha * vector + (1-alpha) * jaccard)
- Debounce is server-side (consistent behavior regardless of client)
- Four-tier verdict: Hit (>=0.92) / Partial (>=0.80) / StaleFallback (Partial + retriever fail) / Miss (<0.80)
- Circuit breaker: 5 consecutive failures -> open for 30s -> half-open probe
- StaleFallback: on Partial verdict, if retriever fails, serve stale cached results instead of error
- Optional API key auth (disabled by default, backwards compatible)
- Auth brute force protection: IP lockout after 10 failures in 5 minutes
- Input validation on both REST and WebSocket handlers
- Per-session keystroke rate limiting (20/sec)
- Per-IP WebSocket connection limit (50 max, via ConnectInfo)
- WebSocket Origin validation (optional --allowed-origins, CSRF protection)
- Speculation timeout (10s, prevents task accumulation on slow backends)
- Session cleanup on WebSocket disconnect (graceful or abrupt)
- Global cache capacity limit (50k entries, prevents OOM)
- Startup embedding validation (non-blocking, warns on dimension anomalies)
- TOML config file support (priority: CLI > env > file > default)
- Optional TLS via rustls (--tls-cert + --tls-key)
- Structured audit logging via dedicated `specai_audit` tracing target
- Constructors return Result (no panics from reqwest client builder)

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
| RUST_LOG                   | Log level (e.g. specai_server=debug,specai_audit=info) |

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
| EMBEDDING_CACHE_MAX_ENTRIES      | 1000      | Max embedding cache entries      |
| EMBEDDING_CACHE_TTL_SECS         | 300       | Embedding cache TTL (5 min)      |
| DEFAULT_RERANK_ALPHA             | 0.7       | Re-rank weight (vector vs text)  |
| SPECULATION_TIMEOUT_SECS         | 10        | Speculation timeout (embed+search)|
| MAX_TOTAL_CACHE_ENTRIES          | 50000     | Global cache capacity limit      |
| MIN_EXPECTED_EMBEDDING_DIM       | 64        | Min embedding dimension (warn)   |
| MAX_EXPECTED_EMBEDDING_DIM       | 4096      | Max embedding dimension (warn)   |
| MAX_WS_CONNECTIONS_PER_IP        | 50        | Max WS connections per IP        |
| MAX_AUTH_FAILURES                | 10        | Auth failures before IP lockout  |
| AUTH_LOCKOUT_SECS                | 300       | Auth lockout duration (5 min)    |

## Code Conventions

- `cargo clippy -- -D warnings` must pass (zero warnings)
- `cargo fmt` enforced
- No panics in handlers (all errors via ApiError)
- Structured JSON logging via tracing
- All new features must include tests
- 140 tests total (53 core unit, 24 server unit, 12 main, 51 integration)
