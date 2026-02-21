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
    gauge!("specai_stale_fallbacks").set(stats.stale_fallbacks as f64);
}

pub fn record_auth_failure() {
    counter!("specai_auth_failures_total").increment(1);
}

pub fn record_speculation_timeout() {
    counter!("specai_speculation_timeouts_total").increment(1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use specai_core::types::EngineStats;

    #[test]
    fn test_record_request() {
        record_request("GET", "/health", 200, Duration::from_millis(5));
        record_request("POST", "/submit", 400, Duration::from_millis(50));
    }

    #[test]
    fn test_update_engine_metrics_with_submissions() {
        let stats = EngineStats {
            predictions_total: 10,
            submissions_total: 5,
            cache_hits: 3,
            cache_partials: 1,
            cache_misses: 1,
            avg_speculation_latency_ms: 15.0,
            avg_submission_latency_ms: 20.0,
            active_sessions: 2,
            cached_entries: 4,
            stale_fallbacks: 0,
        };
        update_engine_metrics(&stats);
    }

    #[test]
    fn test_update_engine_metrics_zero_submissions() {
        let stats = EngineStats {
            predictions_total: 0,
            submissions_total: 0,
            cache_hits: 0,
            cache_partials: 0,
            cache_misses: 0,
            avg_speculation_latency_ms: 0.0,
            avg_submission_latency_ms: 0.0,
            active_sessions: 0,
            cached_entries: 0,
            stale_fallbacks: 0,
        };
        update_engine_metrics(&stats);
    }

    #[test]
    fn test_record_auth_failure() {
        record_auth_failure();
    }

    #[test]
    fn test_record_speculation_timeout() {
        record_speculation_timeout();
    }
}
