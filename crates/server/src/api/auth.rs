use axum::body::Body;
use axum::extract::connect_info::ConnectInfo;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde_json::json;
use specai_core::config;
use std::net::SocketAddr;
use std::time::Instant;

use super::handlers::AppState;

pub async fn auth_middleware(
    State(state): State<AppState>,
    req: Request<Body>,
    next: Next,
) -> Response {
    // If no API key configured, pass through (backwards compatible)
    let Some(ref expected_key) = state.api_key else {
        return next.run(req).await;
    };

    // Skip auth for /health (liveness probe)
    if req.uri().path() == "/health" {
        return next.run(req).await;
    }

    // Extract client IP for brute force protection
    let client_ip = req
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ci| ci.0.ip());

    // Phase 9: Check brute force lockout
    if let Some(ip) = client_ip {
        if let Some(entry) = state.auth_failures.get(&ip) {
            let (count, window_start) = entry.value();
            if *count >= config::MAX_AUTH_FAILURES
                && window_start.elapsed().as_secs() < config::AUTH_LOCKOUT_SECS
            {
                return (
                    StatusCode::TOO_MANY_REQUESTS,
                    axum::Json(
                        json!({"error": "Too many authentication failures, try again later"}),
                    ),
                )
                    .into_response();
            }
        }
    }

    // Check Authorization: Bearer <key>
    let auth_header = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok());

    match auth_header {
        Some(value) if value.starts_with("Bearer ") => {
            let token = &value[7..];
            if token == expected_key {
                // Reset failure counter on success
                if let Some(ip) = client_ip {
                    state.auth_failures.remove(&ip);
                }
                next.run(req).await
            } else {
                super::metrics::record_auth_failure();
                record_auth_failure_for_ip(&state, client_ip);
                (
                    StatusCode::UNAUTHORIZED,
                    axum::Json(json!({"error": "Invalid API key"})),
                )
                    .into_response()
            }
        }
        _ => {
            super::metrics::record_auth_failure();
            record_auth_failure_for_ip(&state, client_ip);
            (
                StatusCode::UNAUTHORIZED,
                axum::Json(json!({"error": "Missing or invalid Authorization header"})),
            )
                .into_response()
        }
    }
}

fn record_auth_failure_for_ip(state: &AppState, client_ip: Option<std::net::IpAddr>) {
    let Some(ip) = client_ip else { return };
    let mut entry = state.auth_failures.entry(ip).or_insert((0, Instant::now()));
    let (count, window_start) = entry.value_mut();

    // Reset window if expired
    if window_start.elapsed().as_secs() >= config::AUTH_LOCKOUT_SECS {
        *count = 0;
        *window_start = Instant::now();
    }
    *count += 1;
}
