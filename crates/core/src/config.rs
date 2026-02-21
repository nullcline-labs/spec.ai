/// Default WebSocket debounce interval in milliseconds.
pub const DEFAULT_DEBOUNCE_MS: u64 = 300;

/// Cosine similarity threshold for cache hits.
pub const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.92;

/// Cosine similarity threshold below which it's a full miss.
pub const DEFAULT_PARTIAL_THRESHOLD: f32 = 0.80;

/// Default TTL for cache entries in seconds.
pub const DEFAULT_CACHE_TTL_SECS: u64 = 120;

/// Maximum concurrent sessions.
pub const MAX_SESSIONS: usize = 10_000;

/// Max cached speculations per session (ring buffer).
pub const MAX_ENTRIES_PER_SESSION: usize = 5;

/// Default number of results to retrieve.
pub const DEFAULT_TOP_K: usize = 10;

/// Default HTTP port.
pub const DEFAULT_PORT: u16 = 3040;

/// Default embedding API endpoint.
pub const DEFAULT_EMBEDDING_URL: &str = "http://localhost:11434/v1/embeddings";

/// Default embedding model name.
pub const DEFAULT_EMBEDDING_MODEL: &str = "text-embedding-3-small";

/// Default vectors.db base URL.
pub const DEFAULT_VECTORSDB_URL: &str = "http://localhost:3030";

/// Default vectors.db collection name.
pub const DEFAULT_COLLECTION: &str = "default";

/// Timeout for external HTTP calls in seconds.
pub const EXTERNAL_REQUEST_TIMEOUT_SECS: u64 = 10;

/// Server request timeout in seconds.
pub const REQUEST_TIMEOUT_SECS: u64 = 30;

/// Global rate limit in requests per second.
pub const RATE_LIMIT_RPS: u64 = 200;

/// Maximum concurrent requests.
pub const MAX_CONCURRENT_REQUESTS: usize = 512;

/// Maximum request body size in bytes.
pub const MAX_REQUEST_BODY_BYTES: usize = 1024 * 1024;

/// Cache eviction sweep interval in seconds.
pub const CACHE_EVICTION_INTERVAL_SECS: u64 = 15;

/// Minimum query length to trigger speculation.
pub const MIN_QUERY_LENGTH: usize = 3;

/// Maximum session ID length in characters.
pub const MAX_SESSION_ID_LENGTH: usize = 128;

/// Maximum query length in characters.
pub const MAX_QUERY_LENGTH: usize = 2000;

/// Maximum keystrokes per second per session (WebSocket rate limit).
pub const MAX_KEYSTROKES_PER_SECOND: u64 = 20;

/// Circuit breaker: consecutive failures before opening.
pub const CIRCUIT_BREAKER_FAILURE_THRESHOLD: u64 = 5;

/// Circuit breaker: seconds to wait before half-open.
pub const CIRCUIT_BREAKER_RECOVERY_SECS: u64 = 30;

/// Maximum entries in the embedding deduplication cache.
pub const EMBEDDING_CACHE_MAX_ENTRIES: usize = 1000;

/// TTL for embedding cache entries in seconds.
pub const EMBEDDING_CACHE_TTL_SECS: u64 = 300;

/// Default re-ranking alpha weight (vector score weight).
pub const DEFAULT_RERANK_ALPHA: f32 = 0.7;

/// Timeout for a single speculation operation (embed + retrieve) in seconds.
pub const SPECULATION_TIMEOUT_SECS: u64 = 10;

/// Maximum total cache entries across all sessions.
pub const MAX_TOTAL_CACHE_ENTRIES: u64 = 50_000;

/// Minimum expected embedding dimension (below this, warn at startup).
pub const MIN_EXPECTED_EMBEDDING_DIM: usize = 64;

/// Maximum expected embedding dimension (above this, warn at startup).
pub const MAX_EXPECTED_EMBEDDING_DIM: usize = 4096;

/// Maximum concurrent WebSocket connections per IP address.
pub const MAX_WS_CONNECTIONS_PER_IP: usize = 50;

/// Maximum authentication failures per IP before lockout.
pub const MAX_AUTH_FAILURES: u32 = 10;

/// Duration of auth lockout in seconds after max failures.
pub const AUTH_LOCKOUT_SECS: u64 = 300;

/// Graceful shutdown drain timeout in seconds.
pub const GRACEFUL_SHUTDOWN_TIMEOUT_SECS: u64 = 30;

/// Default metrics update interval in seconds.
pub const DEFAULT_METRICS_INTERVAL_SECS: u64 = 15;
