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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SpeculativeResult;
    use std::time::Instant;

    #[tokio::test]
    async fn test_eviction_task_removes_expired_entries() {
        // TTL = 0 means entries expire immediately
        let cache = Arc::new(SpeculativeCache::new(0, 5));
        cache.insert(
            &"sess1".to_string(),
            SpeculativeResult {
                query: "test".to_string(),
                embedding: vec![1.0, 0.0],
                documents: vec![],
                created_at: Instant::now(),
            },
        );
        assert_eq!(cache.session_count(), 1);

        let handle = spawn_eviction_task(cache.clone(), 1);
        tokio::time::sleep(Duration::from_secs(2)).await;

        assert_eq!(cache.session_count(), 0);
        handle.abort();
    }
}
