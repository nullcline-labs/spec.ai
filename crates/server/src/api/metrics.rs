use metrics::{counter, gauge, histogram};
use std::time::Duration;

pub fn record_request(method: &str, path: &str, status: u16, duration: Duration) {
    let labels = [
        ("method", method.to_string()),
        ("path", path.to_string()),
        ("status", status.to_string()),
    ];
    counter!("http_requests_total", &labels).increment(1);
    histogram!("http_request_duration_seconds", &labels).record(duration.as_secs_f64());
}

pub fn update_engine_metrics(stats: &specai_core::types::EngineStats) {
    gauge!("specai_predictions_total").set(stats.predictions_total as f64);
    gauge!("specai_submissions_total").set(stats.submissions_total as f64);
    gauge!("specai_cache_hits").set(stats.cache_hits as f64);
    gauge!("specai_cache_partials").set(stats.cache_partials as f64);
    gauge!("specai_cache_misses").set(stats.cache_misses as f64);
    gauge!("specai_avg_speculation_latency_ms").set(stats.avg_speculation_latency_ms);
    gauge!("specai_avg_submission_latency_ms").set(stats.avg_submission_latency_ms);
    gauge!("specai_active_sessions").set(stats.active_sessions as f64);
    gauge!("specai_cached_entries").set(stats.cached_entries as f64);
    gauge!("specai_cache_hit_rate").set(if stats.submissions_total > 0 {
        stats.cache_hits as f64 / stats.submissions_total as f64
    } else {
        0.0
    });
}
