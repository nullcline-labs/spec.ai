pub mod eviction;

use crate::config;
use crate::types::{SessionId, SpeculativeResult};
use dashmap::DashMap;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

struct SessionEntry {
    results: VecDeque<SpeculativeResult>,
    last_active: Instant,
}

/// Concurrent, session-scoped speculative cache.
pub struct SpeculativeCache {
    sessions: DashMap<SessionId, SessionEntry>,
    ttl_secs: u64,
    max_entries_per_session: usize,
    total_entries: AtomicU64,
}

impl SpeculativeCache {
    pub fn new(ttl_secs: u64, max_entries_per_session: usize) -> Self {
        Self {
            sessions: DashMap::new(),
            ttl_secs,
            max_entries_per_session,
            total_entries: AtomicU64::new(0),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(
            config::DEFAULT_CACHE_TTL_SECS,
            config::MAX_ENTRIES_PER_SESSION,
        )
    }

    /// Insert a new speculative result for a session.
    pub fn insert(&self, session_id: &SessionId, result: SpeculativeResult) {
        let mut entry = self
            .sessions
            .entry(session_id.clone())
            .or_insert_with(|| SessionEntry {
                results: VecDeque::with_capacity(self.max_entries_per_session),
                last_active: Instant::now(),
            });

        entry.last_active = Instant::now();

        if entry.results.len() >= self.max_entries_per_session {
            entry.results.pop_front();
        } else {
            self.total_entries.fetch_add(1, Ordering::Relaxed);
        }
        entry.results.push_back(result);
    }

    /// Get the most recent speculative result for a session.
    pub fn get_latest(&self, session_id: &SessionId) -> Option<SpeculativeResult> {
        self.sessions
            .get(session_id)
            .and_then(|entry| entry.results.back().cloned())
    }

    /// Remove a session from the cache.
    pub fn remove_session(&self, session_id: &SessionId) {
        if let Some((_, entry)) = self.sessions.remove(session_id) {
            self.total_entries
                .fetch_sub(entry.results.len() as u64, Ordering::Relaxed);
        }
    }

    /// Evict all sessions older than TTL. Returns count evicted.
    pub fn evict_expired(&self) -> usize {
        let cutoff = Instant::now() - std::time::Duration::from_secs(self.ttl_secs);
        let mut evicted = 0;
        self.sessions.retain(|_, entry| {
            if entry.last_active < cutoff {
                self.total_entries
                    .fetch_sub(entry.results.len() as u64, Ordering::Relaxed);
                evicted += 1;
                false
            } else {
                true
            }
        });
        evicted
    }

    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    pub fn entry_count(&self) -> u64 {
        self.total_entries.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SpeculativeResult;

    fn make_result(query: &str) -> SpeculativeResult {
        SpeculativeResult {
            query: query.to_string(),
            embedding: vec![1.0, 0.0, 0.0],
            documents: vec![],
            created_at: Instant::now(),
        }
    }

    #[test]
    fn test_insert_and_get() {
        let cache = SpeculativeCache::new(60, 3);
        let sid = "session1".to_string();
        cache.insert(&sid, make_result("hello"));
        let latest = cache.get_latest(&sid).unwrap();
        assert_eq!(latest.query, "hello");
        assert_eq!(cache.entry_count(), 1);
    }

    #[test]
    fn test_ring_buffer_overflow() {
        let cache = SpeculativeCache::new(60, 2);
        let sid = "s1".to_string();
        cache.insert(&sid, make_result("a"));
        cache.insert(&sid, make_result("b"));
        cache.insert(&sid, make_result("c"));
        let latest = cache.get_latest(&sid).unwrap();
        assert_eq!(latest.query, "c");
        assert_eq!(cache.entry_count(), 2);
    }

    #[test]
    fn test_remove_session() {
        let cache = SpeculativeCache::new(60, 5);
        let sid = "s1".to_string();
        cache.insert(&sid, make_result("a"));
        cache.insert(&sid, make_result("b"));
        assert_eq!(cache.entry_count(), 2);
        cache.remove_session(&sid);
        assert_eq!(cache.entry_count(), 0);
        assert!(cache.get_latest(&sid).is_none());
    }

    #[test]
    fn test_evict_expired() {
        let cache = SpeculativeCache::new(0, 5); // 0s TTL = everything expires
        let sid = "s1".to_string();
        cache.insert(&sid, make_result("a"));
        std::thread::sleep(std::time::Duration::from_millis(10));
        let evicted = cache.evict_expired();
        assert_eq!(evicted, 1);
        assert_eq!(cache.session_count(), 0);
    }
}
