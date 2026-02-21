use super::errors::ApiError;
use super::models::*;
use super::validation;
use axum::extract::State;
use axum::Json;
use dashmap::DashMap;
use metrics_exporter_prometheus::PrometheusHandle;
use specai_core::engine::Engine;
use std::net::IpAddr;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Engine>,
    pub prometheus_handle: PrometheusHandle,
    pub start_time: Instant,
    pub debounce_ms: u64,
    pub api_key: Option<String>,
    pub allowed_origins: Option<Vec<String>>,
    pub ws_connections_per_ip: Arc<DashMap<IpAddr, AtomicUsize>>,
    pub auth_failures: Arc<DashMap<IpAddr, (u32, Instant)>>,
}

#[utoipa::path(get, path = "/health", responses((status = 200, body = HealthResponse)))]
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

#[utoipa::path(get, path = "/ready", responses((status = 200, body = ReadyResponse), (status = 503)))]
pub async fn ready(State(state): State<AppState>) -> Result<Json<ReadyResponse>, ApiError> {
    match state.engine.check_readiness().await {
        Ok(()) => Ok(Json(ReadyResponse {
            status: "ready".to_string(),
            embedding_service: true,
            vector_db: true,
        })),
        Err(e) => Err(ApiError::ServiceUnavailable(format!("Not ready: {}", e))),
    }
}

#[utoipa::path(get, path = "/stats", responses((status = 200, body = specai_core::types::EngineStats)))]
pub async fn stats(State(state): State<AppState>) -> Json<specai_core::types::EngineStats> {
    Json(state.engine.stats())
}

#[utoipa::path(post, path = "/submit", request_body = SubmitRequest, responses((status = 200, body = SubmitResponse), (status = 400)))]
pub async fn submit(
    State(state): State<AppState>,
    Json(req): Json<SubmitRequest>,
) -> Result<Json<SubmitResponse>, ApiError> {
    validation::validate_session_id(&req.session_id)
        .map_err(|e| ApiError::BadRequest(e.to_string()))?;
    validation::validate_query(&req.query).map_err(|e| ApiError::BadRequest(e.to_string()))?;

    let result = state.engine.submit(&req.session_id, &req.query).await?;

    Ok(Json(SubmitResponse {
        documents: result.documents,
        cache_verdict: result.verdict.to_string(),
        latency_ms: result.latency_ms,
    }))
}

#[utoipa::path(get, path = "/metrics", responses((status = 200)))]
pub async fn metrics_endpoint(State(state): State<AppState>) -> String {
    state.prometheus_handle.render()
}
