use crate::cache::SpeculativeCache;
use crate::config;
use crate::embedder::Embedder;
use crate::retriever::Retriever;
use crate::similarity::{SimilarityGate, SimilarityGateConfig};
use crate::types::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Configuration for the speculative engine.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub top_k: usize,
    pub min_query_length: usize,
    pub similarity: SimilarityGateConfig,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            top_k: config::DEFAULT_TOP_K,
            min_query_length: config::MIN_QUERY_LENGTH,
            similarity: SimilarityGateConfig {
                hit_threshold: config::DEFAULT_SIMILARITY_THRESHOLD,
                partial_threshold: config::DEFAULT_PARTIAL_THRESHOLD,
            },
        }
    }
}

/// The result of processing a final submission.
#[derive(Debug, Clone)]
pub struct SubmissionResult {
    pub documents: Vec<Document>,
    pub verdict: CacheVerdict,
    pub latency_ms: u64,
}

/// Errors from the engine.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("Embedding failed: {0}")]
    Embed(#[from] crate::embedder::EmbedError),
    #[error("Retrieval failed: {0}")]
    Retrieve(#[from] crate::retriever::RetrieveError),
}

/// The speculative retrieval engine.
pub struct Engine {
    embedder: Arc<dyn Embedder>,
    retriever: Arc<dyn Retriever>,
    cache: Arc<SpeculativeCache>,
    gate: SimilarityGate,
    config: EngineConfig,
    predictions_total: AtomicU64,
    submissions_total: AtomicU64,
    cache_hits: AtomicU64,
    cache_partials: AtomicU64,
    cache_misses: AtomicU64,
    speculation_latency_sum_ms: AtomicU64,
    submission_latency_sum_ms: AtomicU64,
}

impl Engine {
    pub fn new(
        embedder: Arc<dyn Embedder>,
        retriever: Arc<dyn Retriever>,
        cache: Arc<SpeculativeCache>,
        config: EngineConfig,
    ) -> Self {
        let gate = SimilarityGate::new(config.similarity.clone());
        Self {
            embedder,
            retriever,
            cache,
            gate,
            config,
            predictions_total: AtomicU64::new(0),
            submissions_total: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_partials: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            speculation_latency_sum_ms: AtomicU64::new(0),
            submission_latency_sum_ms: AtomicU64::new(0),
        }
    }

    /// Speculate: embed a partial query and pre-fetch results (background).
    pub async fn speculate(
        &self,
        session_id: &SessionId,
        partial_query: &str,
    ) -> Result<(), EngineError> {
        if partial_query.len() < self.config.min_query_length {
            return Ok(());
        }

        let start = Instant::now();
        self.predictions_total.fetch_add(1, Ordering::Relaxed);

        let embedding = self.embedder.embed(partial_query).await?;
        let documents = self.retriever.search(&embedding, self.config.top_k).await?;

        let result = SpeculativeResult {
            query: partial_query.to_string(),
            embedding,
            documents,
            created_at: Instant::now(),
        };
        self.cache.insert(session_id, result);

        let latency_ms = start.elapsed().as_millis() as u64;
        self.speculation_latency_sum_ms
            .fetch_add(latency_ms, Ordering::Relaxed);

        tracing::debug!(
            session = %session_id,
            query = %partial_query,
            latency_ms = latency_ms,
            "Speculation complete"
        );

        Ok(())
    }

    /// Submit: handle the final user query.
    pub async fn submit(
        &self,
        session_id: &SessionId,
        final_query: &str,
    ) -> Result<SubmissionResult, EngineError> {
        let start = Instant::now();
        self.submissions_total.fetch_add(1, Ordering::Relaxed);

        let final_embedding = self.embedder.embed(final_query).await?;

        let verdict = if let Some(cached) = self.cache.get_latest(session_id) {
            self.gate.evaluate(&cached.embedding, &final_embedding)
        } else {
            CacheVerdict::Miss
        };

        let documents = match verdict {
            CacheVerdict::Hit => {
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                self.cache
                    .get_latest(session_id)
                    .map(|r| r.documents)
                    .unwrap_or_default()
            }
            CacheVerdict::Partial => {
                self.cache_partials.fetch_add(1, Ordering::Relaxed);
                self.retriever
                    .search(&final_embedding, self.config.top_k)
                    .await?
            }
            CacheVerdict::Miss => {
                self.cache_misses.fetch_add(1, Ordering::Relaxed);
                self.retriever
                    .search(&final_embedding, self.config.top_k)
                    .await?
            }
        };

        let result = SpeculativeResult {
            query: final_query.to_string(),
            embedding: final_embedding,
            documents: documents.clone(),
            created_at: Instant::now(),
        };
        self.cache.insert(session_id, result);

        let latency_ms = start.elapsed().as_millis() as u64;
        self.submission_latency_sum_ms
            .fetch_add(latency_ms, Ordering::Relaxed);

        Ok(SubmissionResult {
            documents,
            verdict,
            latency_ms,
        })
    }

    pub fn close_session(&self, session_id: &SessionId) {
        self.cache.remove_session(session_id);
        tracing::debug!(session = %session_id, "Session closed");
    }

    pub fn stats(&self) -> EngineStats {
        let predictions = self.predictions_total.load(Ordering::Relaxed);
        let submissions = self.submissions_total.load(Ordering::Relaxed);
        EngineStats {
            predictions_total: predictions,
            submissions_total: submissions,
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_partials: self.cache_partials.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            avg_speculation_latency_ms: if predictions > 0 {
                self.speculation_latency_sum_ms.load(Ordering::Relaxed) as f64 / predictions as f64
            } else {
                0.0
            },
            avg_submission_latency_ms: if submissions > 0 {
                self.submission_latency_sum_ms.load(Ordering::Relaxed) as f64 / submissions as f64
            } else {
                0.0
            },
            active_sessions: self.cache.session_count(),
            cached_entries: self.cache.entry_count() as usize,
        }
    }

    pub fn cache(&self) -> &Arc<SpeculativeCache> {
        &self.cache
    }
}
