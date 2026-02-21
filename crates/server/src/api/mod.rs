pub mod auth;
pub mod docs;
pub mod errors;
pub mod handlers;
pub mod metrics;
pub mod models;
pub mod validation;
pub mod ws;

use axum::error_handling::HandleErrorLayer;
use axum::extract::DefaultBodyLimit;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{middleware, Router};
use handlers::AppState;
use specai_core::config;
use std::time::{Duration, Instant};
use tower::buffer::BufferLayer;
use tower::limit::{ConcurrencyLimitLayer, RateLimitLayer};
use tower::timeout::TimeoutLayer;
use tower::ServiceBuilder;
use tower_http::compression::CompressionLayer;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::Instrument;
use utoipa::OpenApi;

pub struct RouterConfig {
    pub rate_limit_rps: u64,
    pub max_concurrent: usize,
    pub request_timeout_secs: u64,
    pub max_body_size: usize,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            rate_limit_rps: config::RATE_LIMIT_RPS,
            max_concurrent: config::MAX_CONCURRENT_REQUESTS,
            request_timeout_secs: config::REQUEST_TIMEOUT_SECS,
            max_body_size: config::MAX_REQUEST_BODY_BYTES,
        }
    }
}

async fn request_id_middleware(
    req: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let request_id = uuid::Uuid::new_v4().to_string();
    let span = tracing::info_span!("request", request_id = %request_id);
    async move {
        let mut response = next.run(req).await;
        response.headers_mut().insert(
            axum::http::HeaderName::from_static("x-request-id"),
            axum::http::HeaderValue::from_str(&request_id).expect("UUID is valid ASCII"),
        );
        response
    }
    .instrument(span)
    .await
}

async fn metrics_middleware(
    req: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let method = req.method().to_string();
    let path = req.uri().path().to_string();
    let start = Instant::now();
    let response = next.run(req).await;
    metrics::record_request(&method, &path, response.status().as_u16(), start.elapsed());
    response
}

async fn docs_handler() -> impl axum::response::IntoResponse {
    axum::Json(docs::ApiDoc::openapi())
}

pub fn create_router(state: AppState, router_config: RouterConfig) -> Router {
    Router::new()
        .route("/ws", get(ws::ws_handler))
        .route("/health", get(handlers::health))
        .route("/ready", get(handlers::ready))
        .route("/stats", get(handlers::stats))
        .route("/submit", post(handlers::submit))
        .route("/metrics", get(handlers::metrics_endpoint))
        .route("/docs", get(docs_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth::auth_middleware,
        ))
        .layer(middleware::from_fn(metrics_middleware))
        .layer(middleware::from_fn(request_id_middleware))
        .layer(CompressionLayer::new())
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .layer(DefaultBodyLimit::max(router_config.max_body_size))
        .layer(
            ServiceBuilder::new()
                .layer(HandleErrorLayer::new(|err: tower::BoxError| async move {
                    if err.is::<tower::timeout::error::Elapsed>() {
                        StatusCode::REQUEST_TIMEOUT
                    } else {
                        StatusCode::TOO_MANY_REQUESTS
                    }
                }))
                .layer(BufferLayer::new(1024))
                .layer(ConcurrencyLimitLayer::new(router_config.max_concurrent))
                .layer(RateLimitLayer::new(
                    router_config.rate_limit_rps,
                    Duration::from_secs(1),
                ))
                .layer(TimeoutLayer::new(Duration::from_secs(
                    router_config.request_timeout_secs,
                ))),
        )
        .with_state(state)
}
