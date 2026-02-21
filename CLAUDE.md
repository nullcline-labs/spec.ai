# spec.ai — Speculative Retrieval Engine

## Overview

Rust-based speculative retrieval engine that reduces RAG latency by pre-executing embedding + vector search while users type. Uses WebSocket keystroke streaming with server-side debounce, a DashMap-based speculative cache, and cosine similarity gating (Hit/Partial/Miss).

## Build & Test

```bash
cargo build                      # debug build
cargo build --release            # optimized (fat LTO)
cargo test                       # run all tests
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
```

## Architecture

```
crates/
├── core/           specai-core (no server deps)
│   ├── config.rs       constants (thresholds, timeouts, limits)
│   ├── types.rs        Document, Embedding, CacheVerdict, EngineStats
│   ├── similarity.rs   cosine similarity + SimilarityGate
│   ├── cache/
│   │   ├── mod.rs      SpeculativeCache (DashMap + ring buffer per session)
│   │   └── eviction.rs TTL background sweep
│   ├── embedder/
│   │   ├── mod.rs      Embedder trait
│   │   └── http.rs     HttpEmbedder (OpenAI-compatible /v1/embeddings)
│   ├── retriever/
│   │   ├── mod.rs      Retriever trait
│   │   └── vectorsdb.rs VectorsDbRetriever (calls vectors.db REST API)
│   └── engine.rs       Engine orchestrator (speculate + submit)
└── server/         specai-server
    ├── main.rs         CLI (clap) + startup + graceful shutdown
    └── api/
        ├── mod.rs      router + tower middleware stack
        ├── errors.rs   ApiError enum → HTTP status codes
        ├── handlers.rs AppState + REST handlers
        ├── models.rs   request/response DTOs
        ├── metrics.rs  Prometheus recording
        └── ws.rs       WebSocket handler + per-session debounce

## Key Design Decisions

- Core is async (must call external HTTP services)
- DashMap for cache (high-contention from many WS sessions)
- Ring buffer per session (max 5 entries, bounds memory)
- Trait objects (Arc<dyn Embedder/Retriever>) for swappable providers
- Debounce is server-side (consistent behavior regardless of client)
- Three-tier verdict: Hit (≥0.92) / Partial (≥0.80) / Miss (<0.80)

## API Endpoints

| Method | Path      | Description                          |
|--------|-----------|--------------------------------------|
| GET    | /ws       | WebSocket for keystroke streaming     |
| GET    | /health   | Health check (status, uptime)        |
| GET    | /stats    | Engine stats (hits, misses, latency) |
| POST   | /submit   | REST submission (non-WS clients)     |
| GET    | /metrics  | Prometheus metrics                   |

## WebSocket Protocol

```json
// Client → Server
{"type": "keystroke", "session_id": "abc", "text": "how to configur"}
{"type": "submit", "session_id": "abc", "query": "how to configure auth"}
{"type": "close", "session_id": "abc"}

// Server → Client
{"type": "speculating", "session_id": "abc", "query": "how to configur"}
{"type": "speculation_ready", "session_id": "abc", "num_results": 10, "latency_ms": 145}
{"type": "results", "session_id": "abc", "documents": [...], "cache_verdict": "Hit", "latency_ms": 5}
```

## Environment Variables

| Variable                   | Description                        |
|----------------------------|------------------------------------|
| SPECAI_EMBEDDING_API_KEY   | Bearer token for embedding API     |
| SPECAI_VECTORSDB_API_KEY   | Bearer token for vectors.db        |
| RUST_LOG                   | Log level (e.g. specai_server=debug) |

## Configuration Constants (config.rs)

| Constant                    | Default | Description                      |
|-----------------------------|---------|----------------------------------|
| DEFAULT_DEBOUNCE_MS         | 300     | Keystroke debounce interval      |
| DEFAULT_SIMILARITY_THRESHOLD| 0.92    | Cosine threshold for cache hit   |
| DEFAULT_PARTIAL_THRESHOLD   | 0.80    | Threshold for partial hit        |
| DEFAULT_CACHE_TTL_SECS      | 120     | Cache entry TTL                  |
| MAX_ENTRIES_PER_SESSION     | 5       | Ring buffer size per session     |
| DEFAULT_TOP_K               | 10      | Results per retrieval            |
| DEFAULT_PORT                | 3040    | HTTP/WS port                     |

## Code Conventions

- `cargo clippy -- -D warnings` must pass (zero warnings)
- `cargo fmt` enforced
- No panics in handlers (all errors via ApiError)
- Structured JSON logging via tracing
```
