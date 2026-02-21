//! Structured audit logging for all query events.
//!
//! All events are emitted via the `specai_audit` tracing target, allowing
//! operators to route audit logs independently:
//!
//! ```bash
//! RUST_LOG=specai_audit=info cargo run --release
//! ```

pub fn speculation_started(session_id: &str, query: &str) {
    tracing::info!(
        target: "specai_audit",
        event = "speculation_started",
        session_id = %session_id,
        query = %query,
    );
}

pub fn speculation_complete(session_id: &str, query: &str, num_results: usize, latency_ms: u64) {
    tracing::info!(
        target: "specai_audit",
        event = "speculation_complete",
        session_id = %session_id,
        query = %query,
        num_results = num_results,
        latency_ms = latency_ms,
    );
}

pub fn submission_received(session_id: &str, query: &str) {
    tracing::info!(
        target: "specai_audit",
        event = "submission_received",
        session_id = %session_id,
        query = %query,
    );
}

pub fn submission_result(
    session_id: &str,
    query: &str,
    verdict: &str,
    num_results: usize,
    latency_ms: u64,
) {
    tracing::info!(
        target: "specai_audit",
        event = "submission_result",
        session_id = %session_id,
        query = %query,
        verdict = %verdict,
        num_results = num_results,
        latency_ms = latency_ms,
    );
}
