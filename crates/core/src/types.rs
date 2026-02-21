use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// A unique session identifier.
pub type SessionId = String;

/// A dense embedding vector.
pub type Embedding = Vec<f32>;

/// A retrieved document from the vector database.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct Document {
    pub id: Uuid,
    pub text: String,
    pub score: f32,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// A complete speculative retrieval result stored in the cache.
#[derive(Debug, Clone)]
pub struct SpeculativeResult {
    pub query: String,
    pub embedding: Embedding,
    pub documents: Vec<Document>,
    pub created_at: std::time::Instant,
}

/// Outcome of a similarity gate check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheVerdict {
    /// Cached results are close enough to serve directly.
    Hit,
    /// Cached results are somewhat relevant but should be refreshed.
    Partial,
    /// No useful cached results.
    Miss,
}

impl std::fmt::Display for CacheVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CacheVerdict::Hit => write!(f, "Hit"),
            CacheVerdict::Partial => write!(f, "Partial"),
            CacheVerdict::Miss => write!(f, "Miss"),
        }
    }
}

/// Engine statistics.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct EngineStats {
    pub predictions_total: u64,
    pub submissions_total: u64,
    pub cache_hits: u64,
    pub cache_partials: u64,
    pub cache_misses: u64,
    pub avg_speculation_latency_ms: f64,
    pub avg_submission_latency_ms: f64,
    pub active_sessions: usize,
    pub cached_entries: usize,
}
