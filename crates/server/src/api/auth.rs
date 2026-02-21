use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde_json::json;

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

    // Check Authorization: Bearer <key>
    let auth_header = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok());

    match auth_header {
        Some(value) if value.starts_with("Bearer ") => {
            let token = &value[7..];
            if token == expected_key {
                next.run(req).await
            } else {
                (
                    StatusCode::UNAUTHORIZED,
                    axum::Json(json!({"error": "Invalid API key"})),
                )
                    .into_response()
            }
        }
        _ => (
            StatusCode::UNAUTHORIZED,
            axum::Json(json!({"error": "Missing or invalid Authorization header"})),
        )
            .into_response(),
    }
}
