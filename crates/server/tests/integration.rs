use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use metrics_exporter_prometheus::PrometheusHandle;
use specai_core::cache::SpeculativeCache;
use specai_core::embedder::EmbedError;
use specai_core::engine::{Engine, EngineConfig};
use specai_core::retriever::RetrieveError;
use specai_core::types::{Document, Embedding};
use specai_server::api::handlers::AppState;
use specai_server::api::{create_router, RouterConfig};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tower::ServiceExt;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Mock Embedder
// ---------------------------------------------------------------------------

struct MockEmbedder {
    response: Embedding,
}

impl MockEmbedder {
    fn new(response: Embedding) -> Self {
        Self { response }
    }
}

#[async_trait::async_trait]
impl specai_core::embedder::Embedder for MockEmbedder {
    async fn embed(&self, _text: &str) -> Result<Embedding, EmbedError> {
        Ok(self.response.clone())
    }
}

// ---------------------------------------------------------------------------
// Failing Mock Embedder (for readiness failure tests)
// ---------------------------------------------------------------------------

struct FailingEmbedder;

#[async_trait::async_trait]
impl specai_core::embedder::Embedder for FailingEmbedder {
    async fn embed(&self, _text: &str) -> Result<Embedding, EmbedError> {
        Err(EmbedError::InvalidResponse("mock failure".to_string()))
    }
}

// ---------------------------------------------------------------------------
// Mock Retriever
// ---------------------------------------------------------------------------

struct MockRetriever {
    documents: Vec<Document>,
}

impl MockRetriever {
    fn new(documents: Vec<Document>) -> Self {
        Self { documents }
    }
}

#[async_trait::async_trait]
impl specai_core::retriever::Retriever for MockRetriever {
    async fn search(
        &self,
        _embedding: &Embedding,
        _top_k: usize,
    ) -> Result<Vec<Document>, RetrieveError> {
        Ok(self.documents.clone())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_doc(text: &str) -> Document {
    Document {
        id: Uuid::new_v4(),
        text: text.to_string(),
        score: 0.95,
        metadata: HashMap::new(),
    }
}

fn make_prometheus_handle() -> PrometheusHandle {
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    recorder.handle()
}

fn test_router_config() -> RouterConfig {
    RouterConfig {
        rate_limit_rps: 1000,
        max_concurrent: 100,
        request_timeout_secs: 30,
        max_body_size: 1024 * 1024,
    }
}

fn make_test_app(api_key: Option<String>) -> axum::Router {
    let embedder = Arc::new(MockEmbedder::new(vec![1.0, 0.0, 0.0]));
    let retriever = Arc::new(MockRetriever::new(vec![
        make_doc("First test document about configuration"),
        make_doc("Second test document about authentication"),
    ]));
    let cache = Arc::new(SpeculativeCache::new(60, 5));
    let engine = Arc::new(Engine::new(
        embedder,
        retriever,
        cache,
        EngineConfig::default(),
    ));

    let state = AppState {
        engine,
        prometheus_handle: make_prometheus_handle(),
        start_time: Instant::now(),
        debounce_ms: 50,
        api_key,
    };
    create_router(state, test_router_config())
}

fn make_failing_app() -> axum::Router {
    let embedder = Arc::new(FailingEmbedder);
    let retriever = Arc::new(MockRetriever::new(vec![]));
    let cache = Arc::new(SpeculativeCache::new(60, 5));
    let engine = Arc::new(Engine::new(
        embedder,
        retriever,
        cache,
        EngineConfig::default(),
    ));

    let state = AppState {
        engine,
        prometheus_handle: make_prometheus_handle(),
        start_time: Instant::now(),
        debounce_ms: 50,
        api_key: None,
    };
    create_router(state, test_router_config())
}

async fn get_body(resp: axum::response::Response) -> String {
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    String::from_utf8(bytes.to_vec()).unwrap()
}

// ===========================================================================
// Health endpoint tests
// ===========================================================================

#[tokio::test]
async fn test_health_returns_ok() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_health_body_contains_status_ok() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = get_body(resp).await;
    assert!(body.contains("\"status\":\"ok\""), "body was: {}", body);
}

#[tokio::test]
async fn test_health_body_contains_version() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = get_body(resp).await;
    assert!(body.contains("\"version\""), "body was: {}", body);
}

#[tokio::test]
async fn test_health_body_contains_uptime() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = get_body(resp).await;
    assert!(body.contains("\"uptime_seconds\""), "body was: {}", body);
}

// ===========================================================================
// Stats endpoint tests
// ===========================================================================

#[tokio::test]
async fn test_stats_returns_ok() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_stats_body_contains_predictions_total() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = get_body(resp).await;
    assert!(body.contains("\"predictions_total\""), "body was: {}", body);
}

#[tokio::test]
async fn test_stats_body_contains_cache_hits() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = get_body(resp).await;
    assert!(body.contains("\"cache_hits\""), "body was: {}", body);
}

#[tokio::test]
async fn test_stats_initial_values_are_zero() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = get_body(resp).await;
    let stats: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(stats["predictions_total"], 0);
    assert_eq!(stats["submissions_total"], 0);
    assert_eq!(stats["cache_hits"], 0);
    assert_eq!(stats["cache_misses"], 0);
}

// ===========================================================================
// Submit endpoint tests
// ===========================================================================

#[tokio::test]
async fn test_submit_valid_request() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/submit")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"session_id":"test-123","query":"how to configure auth"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = get_body(resp).await;
    assert!(body.contains("cache_verdict"), "body was: {}", body);
}

#[tokio::test]
async fn test_submit_returns_documents() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/submit")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"session_id":"test-docs","query":"how to configure auth"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let body = get_body(resp).await;
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    let docs = json["documents"].as_array().unwrap();
    assert_eq!(docs.len(), 2);
    assert!(docs[0]["text"].as_str().unwrap().contains("configuration"));
}

#[tokio::test]
async fn test_submit_returns_latency() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/submit")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"session_id":"test-latency","query":"test query"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let body = get_body(resp).await;
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(json["latency_ms"].is_number());
}

#[tokio::test]
async fn test_submit_empty_query() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/submit")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"session_id":"test-123","query":""}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_submit_empty_query_error_message() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/submit")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"session_id":"test-123","query":""}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    let body = get_body(resp).await;
    assert!(
        body.contains("query must not be empty"),
        "body was: {}",
        body
    );
}

#[tokio::test]
async fn test_submit_invalid_session_id_semicolon() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/submit")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"session_id":"ab;cd","query":"valid query"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_submit_invalid_session_id_space() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/submit")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"session_id":"ab cd","query":"valid query"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_submit_invalid_session_id_slash() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/submit")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"session_id":"abc/def","query":"valid query"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_submit_empty_session_id() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/submit")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"session_id":"","query":"valid query"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_submit_query_too_long() {
    let long_query = "a".repeat(2001);
    let body = format!(r#"{{"session_id":"test-123","query":"{}"}}"#, long_query);
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/submit")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_submit_query_at_max_length() {
    let max_query = "a".repeat(2000);
    let body = format!(r#"{{"session_id":"test-123","query":"{}"}}"#, max_query);
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/submit")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_submit_missing_content_type() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/submit")
                .body(Body::from(r#"{"session_id":"test-123","query":"test"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    // axum rejects JSON without content-type header
    assert_ne!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_submit_invalid_json() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/submit")
                .header("content-type", "application/json")
                .body(Body::from("not json"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_ne!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_submit_missing_query_field() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/submit")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"session_id":"test-123"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_ne!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_submit_get_method_not_allowed() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/submit")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::METHOD_NOT_ALLOWED);
}

#[tokio::test]
async fn test_submit_verdict_is_miss_on_first_call() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/submit")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"session_id":"fresh-session","query":"completely new query"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let body = get_body(resp).await;
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["cache_verdict"], "Miss");
}

#[tokio::test]
async fn test_submit_valid_session_id_with_hyphens_and_underscores() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/submit")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"session_id":"my-session_123","query":"valid query"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

// ===========================================================================
// Metrics endpoint tests
// ===========================================================================

#[tokio::test]
async fn test_metrics_returns_ok() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_metrics_returns_text() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    // Prometheus metrics endpoint should return text
    assert_eq!(resp.status(), StatusCode::OK);
    // Body may be empty if no metrics recorded yet, but should not error
    let body = get_body(resp).await;
    // Just verify we got a response (possibly empty string for a fresh recorder)
    assert!(
        body.len() < 1024 * 1024,
        "metrics output unexpectedly large"
    );
}

// ===========================================================================
// Ready endpoint tests
// ===========================================================================

#[tokio::test]
async fn test_ready_returns_ok_with_healthy_mocks() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = get_body(resp).await;
    assert!(body.contains("\"status\":\"ready\""), "body was: {}", body);
}

#[tokio::test]
async fn test_ready_returns_503_when_embedder_fails() {
    let app = make_failing_app();
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

// ===========================================================================
// Docs endpoint tests
// ===========================================================================

#[tokio::test]
async fn test_docs_returns_ok() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(Request::builder().uri("/docs").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_docs_contains_openapi() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(Request::builder().uri("/docs").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let body = get_body(resp).await;
    assert!(body.contains("openapi"), "body was: {}", body);
}

#[tokio::test]
async fn test_docs_contains_paths() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(Request::builder().uri("/docs").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let body = get_body(resp).await;
    assert!(body.contains("/health"), "body was: {}", body);
    assert!(body.contains("/submit"), "body was: {}", body);
    assert!(body.contains("/stats"), "body was: {}", body);
}

// ===========================================================================
// 404 for unknown routes
// ===========================================================================

#[tokio::test]
async fn test_unknown_route_returns_404() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/nonexistent")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ===========================================================================
// Auth middleware tests
// ===========================================================================

#[tokio::test]
async fn test_auth_not_required_when_no_key_configured() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_auth_required_when_key_set_missing_header() {
    let app = make_test_app(Some("secret-key".to_string()));
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_required_missing_header_error_body() {
    let app = make_test_app(Some("secret-key".to_string()));
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = get_body(resp).await;
    assert!(body.contains("error"), "body was: {}", body);
}

#[tokio::test]
async fn test_auth_valid_bearer_token() {
    let app = make_test_app(Some("secret-key".to_string()));
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/stats")
                .header("authorization", "Bearer secret-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_auth_invalid_bearer_token() {
    let app = make_test_app(Some("secret-key".to_string()));
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/stats")
                .header("authorization", "Bearer wrong-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_invalid_token_error_body() {
    let app = make_test_app(Some("secret-key".to_string()));
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/stats")
                .header("authorization", "Bearer wrong-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = get_body(resp).await;
    assert!(body.contains("Invalid API key"), "body was: {}", body);
}

#[tokio::test]
async fn test_auth_missing_bearer_prefix() {
    let app = make_test_app(Some("secret-key".to_string()));
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/stats")
                .header("authorization", "secret-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_health_bypasses_auth() {
    let app = make_test_app(Some("secret-key".to_string()));
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_auth_protects_submit() {
    let app = make_test_app(Some("secret-key".to_string()));
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/submit")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"session_id":"test","query":"test query"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_protects_metrics() {
    let app = make_test_app(Some("secret-key".to_string()));
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_protects_ready() {
    let app = make_test_app(Some("secret-key".to_string()));
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_allows_submit_with_valid_key() {
    let app = make_test_app(Some("my-api-key".to_string()));
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/submit")
                .header("content-type", "application/json")
                .header("authorization", "Bearer my-api-key")
                .body(Body::from(r#"{"session_id":"test","query":"test query"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

// ===========================================================================
// Response headers tests
// ===========================================================================

#[tokio::test]
async fn test_response_has_request_id_header() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        resp.headers().contains_key("x-request-id"),
        "Response missing x-request-id header"
    );
}

#[tokio::test]
async fn test_request_id_is_uuid_format() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let request_id = resp
        .headers()
        .get("x-request-id")
        .unwrap()
        .to_str()
        .unwrap();
    // UUID v4 format: 8-4-4-4-12 hex chars
    assert!(
        Uuid::parse_str(request_id).is_ok(),
        "x-request-id is not a valid UUID: {}",
        request_id
    );
}

// ===========================================================================
// WebSocket upgrade test
// ===========================================================================

#[tokio::test]
async fn test_ws_endpoint_without_upgrade_returns_error() {
    // Sending a plain GET to /ws without WebSocket upgrade headers
    // should not return 200 (it requires an upgrade)
    let app = make_test_app(None);
    let resp = app
        .oneshot(Request::builder().uri("/ws").body(Body::empty()).unwrap())
        .await
        .unwrap();
    // Without proper upgrade headers, axum returns a non-success status
    assert_ne!(
        resp.status(),
        StatusCode::OK,
        "WS endpoint should not return 200 without upgrade headers"
    );
}

// ===========================================================================
// Engine integration: submit updates stats
// ===========================================================================

#[tokio::test]
async fn test_submit_updates_engine_stats() {
    // We need to use the same AppState for multiple requests, so we create
    // the state once and build routers from it.
    let embedder = Arc::new(MockEmbedder::new(vec![1.0, 0.0, 0.0]));
    let retriever = Arc::new(MockRetriever::new(vec![make_doc("doc")]));
    let cache = Arc::new(SpeculativeCache::new(60, 5));
    let engine = Arc::new(Engine::new(
        embedder,
        retriever,
        cache,
        EngineConfig::default(),
    ));
    let state = AppState {
        engine,
        prometheus_handle: make_prometheus_handle(),
        start_time: Instant::now(),
        debounce_ms: 50,
        api_key: None,
    };

    // First: submit a query
    let app1 = create_router(state.clone(), test_router_config());
    let resp = app1
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/submit")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"session_id":"stats-test","query":"test query"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Then: check stats
    let app2 = create_router(state, test_router_config());
    let resp = app2
        .oneshot(
            Request::builder()
                .uri("/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = get_body(resp).await;
    let stats: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(stats["submissions_total"], 1);
    assert_eq!(stats["cache_misses"], 1);
}

// ===========================================================================
// Content-type tests
// ===========================================================================

#[tokio::test]
async fn test_health_returns_json_content_type() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let content_type = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(
        content_type.contains("application/json"),
        "content-type was: {}",
        content_type
    );
}

#[tokio::test]
async fn test_stats_returns_json_content_type() {
    let app = make_test_app(None);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let content_type = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(
        content_type.contains("application/json"),
        "content-type was: {}",
        content_type
    );
}
