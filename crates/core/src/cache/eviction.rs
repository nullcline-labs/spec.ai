use super::SpeculativeCache;
use std::sync::Arc;
use std::time::Duration;

/// Spawn a background task that periodically evicts expired cache entries.
pub fn spawn_eviction_task(
    cache: Arc<SpeculativeCache>,
    interval_secs: u64,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));
        loop {
            interval.tick().await;
            let evicted = cache.evict_expired();
            if evicted > 0 {
                tracing::debug!(
                    evicted = evicted,
                    remaining_sessions = cache.session_count(),
                    "Cache eviction sweep"
                );
            }
        }
    })
}
