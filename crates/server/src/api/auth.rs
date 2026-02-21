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

/// Check if an IP is currently locked out.
/// Returns true if the IP has exceeded MAX_AUTH_FAILURES within AUTH_LOCKOUT_SECS.
pub fn is_ip_locked_out(
    auth_failures: &dashmap::DashMap<std::net::IpAddr, (u32, Instant)>,
    ip: std::net::IpAddr,
) -> bool {
    if let Some(entry) = auth_failures.get(&ip) {
        let (count, window_start) = entry.value();
        *count >= config::MAX_AUTH_FAILURES
            && window_start.elapsed().as_secs() < config::AUTH_LOCKOUT_SECS
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dashmap::DashMap;
    use std::net::{IpAddr, Ipv4Addr};
    use std::sync::Arc;

    fn test_ip() -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100))
    }

    fn make_test_state(api_key: Option<String>) -> AppState {
        use specai_core::cache::SpeculativeCache;
        use specai_core::engine::{Engine, EngineConfig};

        struct DummyEmbedder;
        #[async_trait::async_trait]
        impl specai_core::embedder::Embedder for DummyEmbedder {
            async fn embed(
                &self,
                _text: &str,
            ) -> Result<specai_core::types::Embedding, specai_core::embedder::EmbedError>
            {
                Ok(vec![1.0, 0.0, 0.0])
            }
        }
        struct DummyRetriever;
        #[async_trait::async_trait]
        impl specai_core::retriever::Retriever for DummyRetriever {
            async fn search(
                &self,
                _embedding: &specai_core::types::Embedding,
                _top_k: usize,
            ) -> Result<Vec<specai_core::types::Document>, specai_core::retriever::RetrieveError>
            {
                Ok(vec![])
            }
        }

        let engine = Arc::new(Engine::new(
            Arc::new(DummyEmbedder),
            Arc::new(DummyRetriever),
            Arc::new(SpeculativeCache::new(60, 5)),
            EngineConfig::default(),
            None,
        ));
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();

        AppState {
            engine,
            prometheus_handle: recorder.handle(),
            start_time: Instant::now(),
            debounce_ms: 50,
            api_key,
            allowed_origins: None,
            ws_connections_per_ip: Arc::new(DashMap::new()),
            auth_failures: Arc::new(DashMap::new()),
        }
    }

    #[test]
    fn test_record_failure_increments_count() {
        let state = make_test_state(Some("key".into()));
        let ip = Some(test_ip());

        record_auth_failure_for_ip(&state, ip);
        let entry = state.auth_failures.get(&test_ip()).unwrap();
        assert_eq!(entry.value().0, 1);

        drop(entry);
        record_auth_failure_for_ip(&state, ip);
        let entry = state.auth_failures.get(&test_ip()).unwrap();
        assert_eq!(entry.value().0, 2);
    }

    #[test]
    fn test_record_failure_ignores_none_ip() {
        let state = make_test_state(Some("key".into()));
        record_auth_failure_for_ip(&state, None);
        assert!(state.auth_failures.is_empty());
    }

    #[test]
    fn test_lockout_triggers_after_max_failures() {
        let state = make_test_state(Some("key".into()));
        let ip = test_ip();

        for _ in 0..config::MAX_AUTH_FAILURES {
            record_auth_failure_for_ip(&state, Some(ip));
        }

        assert!(is_ip_locked_out(&state.auth_failures, ip));
    }

    #[test]
    fn test_no_lockout_below_threshold() {
        let state = make_test_state(Some("key".into()));
        let ip = test_ip();

        for _ in 0..(config::MAX_AUTH_FAILURES - 1) {
            record_auth_failure_for_ip(&state, Some(ip));
        }

        assert!(!is_ip_locked_out(&state.auth_failures, ip));
    }

    #[test]
    fn test_lockout_expires_after_window() {
        let failures: DashMap<IpAddr, (u32, Instant)> = DashMap::new();
        let ip = test_ip();

        // Simulate failures from the past (beyond lockout window)
        let expired_time =
            Instant::now() - std::time::Duration::from_secs(config::AUTH_LOCKOUT_SECS + 1);
        failures.insert(ip, (config::MAX_AUTH_FAILURES, expired_time));

        assert!(!is_ip_locked_out(&failures, ip));
    }

    #[test]
    fn test_success_resets_counter() {
        let state = make_test_state(Some("key".into()));
        let ip = test_ip();

        // Accumulate failures
        for _ in 0..5 {
            record_auth_failure_for_ip(&state, Some(ip));
        }
        assert_eq!(state.auth_failures.get(&ip).unwrap().value().0, 5);

        // Simulate success: remove entry (same as auth_middleware does)
        state.auth_failures.remove(&ip);
        assert!(state.auth_failures.get(&ip).is_none());
        assert!(!is_ip_locked_out(&state.auth_failures, ip));
    }

    #[test]
    fn test_different_ips_tracked_independently() {
        let state = make_test_state(Some("key".into()));
        let ip1 = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1));
        let ip2 = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 2));

        for _ in 0..config::MAX_AUTH_FAILURES {
            record_auth_failure_for_ip(&state, Some(ip1));
        }

        assert!(is_ip_locked_out(&state.auth_failures, ip1));
        assert!(!is_ip_locked_out(&state.auth_failures, ip2));
    }
}
