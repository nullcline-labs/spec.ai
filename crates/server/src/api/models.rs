use serde::{Deserialize, Serialize};
use specai_core::types::Document;

#[derive(Debug, Deserialize)]
pub struct SubmitRequest {
    pub session_id: String,
    pub query: String,
}

#[derive(Debug, Serialize)]
pub struct SubmitResponse {
    pub documents: Vec<Document>,
    pub cache_verdict: String,
    pub latency_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub active_sessions: usize,
    pub cached_entries: usize,
}

#[derive(Debug, Serialize)]
pub struct MessageResponse {
    pub message: String,
}
