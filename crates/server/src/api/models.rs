use serde::{Deserialize, Serialize};
use specai_core::types::Document;

#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct SubmitRequest {
    pub session_id: String,
    pub query: String,
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct SubmitResponse {
    pub documents: Vec<Document>,
    pub cache_verdict: String,
    pub latency_ms: u64,
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub active_sessions: usize,
    pub cached_entries: usize,
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct ReadyResponse {
    pub status: String,
    pub embedding_service: bool,
    pub vector_db: bool,
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct MessageResponse {
    pub message: String,
}
