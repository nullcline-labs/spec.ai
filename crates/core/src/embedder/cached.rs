use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use dashmap::DashMap;

use super::{EmbedError, Embedder};
use crate::types::Embedding;

struct CachedEntry {
    embedding: Embedding,
    created_at: Instant,
}

/// An [`Embedder`] wrapper that caches `query_text -> Embedding` mappings.
///
/// When the same text is requested again within the TTL, the cached embedding
/// is returned without calling the underlying embedder. This avoids redundant
/// API calls when multiple sessions or the speculate/submit cycle produce
/// identical queries.
pub struct CachedEmbedder {
    inner: Arc<dyn Embedder>,
    cache: DashMap<String, CachedEntry>,
    max_entries: usize,
    ttl: Duration,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl CachedEmbedder {
    pub fn new(inner: Arc<dyn Embedder>, max_entries: usize, ttl_secs: u64) -> Self {
        Self {
            inner,
            cache: DashMap::new(),
            max_entries,
            ttl: Duration::from_secs(ttl_secs),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    pub fn cache_hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    pub fn cache_misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }

    fn evict_expired(&self) {
        let now = Instant::now();
        let ttl = self.ttl;
        self.cache
            .retain(|_, entry| now.duration_since(entry.created_at) < ttl);
    }
}

#[async_trait]
impl Embedder for CachedEmbedder {
    async fn embed(&self, text: &str) -> Result<Embedding, EmbedError> {
        // Check cache
        if let Some(entry) = self.cache.get(text) {
            if entry.created_at.elapsed() < self.ttl {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(entry.embedding.clone());
            }
            drop(entry);
            self.cache.remove(text);
        }

        // Cache miss
        self.misses.fetch_add(1, Ordering::Relaxed);
        let embedding = self.inner.embed(text).await?;

        // Evict if at capacity
        if self.cache.len() >= self.max_entries {
            self.evict_expired();
        }
        if self.cache.len() < self.max_entries {
            self.cache.insert(
                text.to_string(),
                CachedEntry {
                    embedding: embedding.clone(),
                    created_at: Instant::now(),
                },
            );
        }

        Ok(embedding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    struct CountingEmbedder {
        call_count: AtomicU64,
        should_fail: std::sync::Mutex<bool>,
    }

    impl CountingEmbedder {
        fn new() -> Self {
            Self {
                call_count: AtomicU64::new(0),
                should_fail: std::sync::Mutex::new(false),
            }
        }

        fn calls(&self) -> u64 {
            self.call_count.load(Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl Embedder for CountingEmbedder {
        async fn embed(&self, text: &str) -> Result<Embedding, EmbedError> {
            self.call_count.fetch_add(1, Ordering::Relaxed);
            if *self.should_fail.lock().unwrap() {
                return Err(EmbedError::Api {
                    status: 500,
                    body: "mock error".into(),
                });
            }
            // Return a deterministic embedding based on text length
            Ok(vec![text.len() as f32, 0.0, 0.0])
        }
    }

    #[tokio::test]
    async fn test_cache_hit() {
        let inner = Arc::new(CountingEmbedder::new());
        let cached = CachedEmbedder::new(inner.clone(), 100, 60);

        let emb1 = cached.embed("hello").await.unwrap();
        let emb2 = cached.embed("hello").await.unwrap();

        assert_eq!(emb1, emb2);
        assert_eq!(inner.calls(), 1);
        assert_eq!(cached.cache_hits(), 1);
        assert_eq!(cached.cache_misses(), 1);
    }

    #[tokio::test]
    async fn test_cache_miss_different_text() {
        let inner = Arc::new(CountingEmbedder::new());
        let cached = CachedEmbedder::new(inner.clone(), 100, 60);

        cached.embed("hello").await.unwrap();
        cached.embed("world").await.unwrap();

        assert_eq!(inner.calls(), 2);
        assert_eq!(cached.cache_hits(), 0);
        assert_eq!(cached.cache_misses(), 2);
    }

    #[tokio::test]
    async fn test_ttl_expiry() {
        let inner = Arc::new(CountingEmbedder::new());
        // TTL = 0 means everything expires immediately
        let cached = CachedEmbedder::new(inner.clone(), 100, 0);

        cached.embed("hello").await.unwrap();
        // With TTL=0, the entry is expired on next lookup
        tokio::time::sleep(Duration::from_millis(5)).await;
        cached.embed("hello").await.unwrap();

        assert_eq!(inner.calls(), 2);
    }

    #[tokio::test]
    async fn test_capacity_limit() {
        let inner = Arc::new(CountingEmbedder::new());
        let cached = CachedEmbedder::new(inner.clone(), 2, 60);

        cached.embed("a").await.unwrap();
        cached.embed("b").await.unwrap();
        cached.embed("c").await.unwrap();

        assert!(cached.cache_len() <= 2);
    }

    #[tokio::test]
    async fn test_error_passthrough() {
        let inner = Arc::new(CountingEmbedder::new());
        let cached = CachedEmbedder::new(inner.clone(), 100, 60);

        // First call succeeds and caches
        cached.embed("hello").await.unwrap();

        // Set to fail
        *inner.should_fail.lock().unwrap() = true;

        // Different text triggers miss → error
        let result = cached.embed("world").await;
        assert!(result.is_err());

        // Cached text still works
        let result = cached.embed("hello").await;
        assert!(result.is_ok());
    }
}
