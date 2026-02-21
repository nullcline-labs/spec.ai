use super::errors::ApiError;
use super::models::*;
use axum::extract::State;
use axum::Json;
use metrics_exporter_prometheus::PrometheusHandle;
use specai_core::engine::Engine;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Engine>,
    pub prometheus_handle: PrometheusHandle,
    pub start_time: Instant,
}

pub async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let stats = state.engine.stats();
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: state.start_time.elapsed().as_secs(),
        active_sessions: stats.active_sessions,
        cached_entries: stats.cached_entries,
    })
}

pub async fn stats(State(state): State<AppState>) -> Json<specai_core::types::EngineStats> {
    Json(state.engine.stats())
}

pub async fn submit(
    State(state): State<AppState>,
    Json(req): Json<SubmitRequest>,
) -> Result<Json<SubmitResponse>, ApiError> {
    if req.query.is_empty() {
        return Err(ApiError::BadRequest("Query must not be empty".into()));
    }
    if req.session_id.is_empty() {
        return Err(ApiError::BadRequest("session_id is required".into()));
    }

    let result = state.engine.submit(&req.session_id, &req.query).await?;

    Ok(Json(SubmitResponse {
        documents: result.documents,
        cache_verdict: result.verdict.to_string(),
        latency_ms: result.latency_ms,
    }))
}

pub async fn metrics_endpoint(State(state): State<AppState>) -> String {
    state.prometheus_handle.render()
}
