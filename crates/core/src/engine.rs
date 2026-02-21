use crate::cache::SpeculativeCache;
use crate::config;
use crate::embedder::Embedder;
use crate::reranker::Reranker;
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
    reranker: Option<Arc<dyn Reranker>>,
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
        reranker: Option<Arc<dyn Reranker>>,
    ) -> Self {
        let gate = SimilarityGate::new(config.similarity.clone());
        Self {
            embedder,
            retriever,
            cache,
            gate,
            config,
            reranker,
            predictions_total: AtomicU64::new(0),
            submissions_total: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_partials: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            speculation_latency_sum_ms: AtomicU64::new(0),
            submission_latency_sum_ms: AtomicU64::new(0),
        }
    }

    /// Check that both the embedding service and vector database are reachable.
    pub async fn check_readiness(&self) -> Result<(), EngineError> {
        // Attempt a simple embedding to verify the embedding service is reachable
        let embedding = self.embedder.embed("health check").await?;
        // Attempt a search to verify the vector database is reachable
        let _ = self.retriever.search(&embedding, 1).await?;
        Ok(())
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
        crate::audit::speculation_started(session_id, partial_query);

        let embedding = self.embedder.embed(partial_query).await?;
        let documents = self.retriever.search(&embedding, self.config.top_k).await?;
        let documents = self.maybe_rerank(partial_query, documents).await;
        let num_results = documents.len();

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
        crate::audit::speculation_complete(session_id, partial_query, num_results, latency_ms);

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
        crate::audit::submission_received(session_id, final_query);

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
                let docs = self
                    .retriever
                    .search(&final_embedding, self.config.top_k)
                    .await?;
                self.maybe_rerank(final_query, docs).await
            }
            CacheVerdict::Miss => {
                self.cache_misses.fetch_add(1, Ordering::Relaxed);
                let docs = self
                    .retriever
                    .search(&final_embedding, self.config.top_k)
                    .await?;
                self.maybe_rerank(final_query, docs).await
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

        crate::audit::submission_result(
            session_id,
            final_query,
            &verdict.to_string(),
            documents.len(),
            latency_ms,
        );

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

    async fn maybe_rerank(&self, query: &str, documents: Vec<Document>) -> Vec<Document> {
        if let Some(ref reranker) = self.reranker {
            reranker.rerank(query, documents).await
        } else {
            documents
        }
    }

    pub fn cache(&self) -> &Arc<SpeculativeCache> {
        &self.cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::SpeculativeCache;
    use std::sync::Mutex;

    struct MockEmbedder {
        embedding: Mutex<Embedding>,
        should_fail: Mutex<bool>,
    }

    impl MockEmbedder {
        fn new(embedding: Embedding) -> Self {
            Self {
                embedding: Mutex::new(embedding),
                should_fail: Mutex::new(false),
            }
        }

        fn set_failing(&self, fail: bool) {
            *self.should_fail.lock().unwrap() = fail;
        }

        fn set_embedding(&self, emb: Embedding) {
            *self.embedding.lock().unwrap() = emb;
        }
    }

    #[async_trait::async_trait]
    impl crate::embedder::Embedder for MockEmbedder {
        async fn embed(&self, _text: &str) -> Result<Embedding, crate::embedder::EmbedError> {
            if *self.should_fail.lock().unwrap() {
                return Err(crate::embedder::EmbedError::Api {
                    status: 500,
                    body: "mock error".into(),
                });
            }
            Ok(self.embedding.lock().unwrap().clone())
        }
    }

    struct MockRetriever {
        documents: Vec<Document>,
        should_fail: Mutex<bool>,
    }

    impl MockRetriever {
        fn new(documents: Vec<Document>) -> Self {
            Self {
                documents,
                should_fail: Mutex::new(false),
            }
        }

        fn set_failing(&self, fail: bool) {
            *self.should_fail.lock().unwrap() = fail;
        }
    }

    #[async_trait::async_trait]
    impl crate::retriever::Retriever for MockRetriever {
        async fn search(
            &self,
            _embedding: &Embedding,
            _top_k: usize,
        ) -> Result<Vec<Document>, crate::retriever::RetrieveError> {
            if *self.should_fail.lock().unwrap() {
                return Err(crate::retriever::RetrieveError::Api {
                    status: 500,
                    body: "mock error".into(),
                });
            }
            Ok(self.documents.clone())
        }
    }

    fn make_doc(text: &str) -> Document {
        Document {
            id: uuid::Uuid::new_v4(),
            text: text.to_string(),
            score: 0.95,
            metadata: Default::default(),
        }
    }

    fn make_engine(
        embedding: Embedding,
        documents: Vec<Document>,
    ) -> (Arc<Engine>, Arc<MockEmbedder>, Arc<MockRetriever>) {
        let embedder = Arc::new(MockEmbedder::new(embedding));
        let retriever = Arc::new(MockRetriever::new(documents));
        let cache = Arc::new(SpeculativeCache::new(60, 5));
        let config = EngineConfig::default();
        let engine = Arc::new(Engine::new(
            embedder.clone(),
            retriever.clone(),
            cache,
            config,
            None,
        ));
        (engine, embedder, retriever)
    }

    #[tokio::test]
    async fn test_speculate_stores_in_cache() {
        let docs = vec![make_doc("doc1"), make_doc("doc2")];
        let (engine, _, _) = make_engine(vec![1.0, 0.0, 0.0], docs);

        let session = "session1".to_string();
        engine
            .speculate(&session, "how to configure")
            .await
            .unwrap();

        let cached = engine.cache().get_latest(&session);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().query, "how to configure");
    }

    #[tokio::test]
    async fn test_speculate_skips_short_query() {
        let docs = vec![make_doc("doc1")];
        let (engine, _, _) = make_engine(vec![1.0, 0.0, 0.0], docs);

        let session = "session1".to_string();
        // MIN_QUERY_LENGTH is 3, so "ab" should be skipped
        engine.speculate(&session, "ab").await.unwrap();

        let cached = engine.cache().get_latest(&session);
        assert!(cached.is_none());
        assert_eq!(engine.cache().entry_count(), 0);
    }

    #[tokio::test]
    async fn test_submit_cache_hit() {
        let docs = vec![make_doc("result1")];
        let (engine, _, _) = make_engine(vec![1.0, 0.0, 0.0], docs);

        let session = "session1".to_string();
        // Speculate and submit with the same embedding => cosine similarity = 1.0 => Hit
        engine
            .speculate(&session, "how to configure auth")
            .await
            .unwrap();
        let result = engine
            .submit(&session, "how to configure auth")
            .await
            .unwrap();

        assert_eq!(result.verdict, CacheVerdict::Hit);
        assert_eq!(result.documents.len(), 1);
        assert_eq!(result.documents[0].text, "result1");
    }

    #[tokio::test]
    async fn test_submit_cache_miss() {
        let docs = vec![make_doc("result1")];
        let (engine, embedder, _) = make_engine(vec![1.0, 0.0, 0.0], docs);

        let session = "session1".to_string();
        // Speculate with embedding [1,0,0]
        engine
            .speculate(&session, "how to configure auth")
            .await
            .unwrap();

        // Change embedding to orthogonal vector => cosine similarity = 0.0 => Miss
        embedder.set_embedding(vec![0.0, 1.0, 0.0]);
        let result = engine
            .submit(&session, "something completely different")
            .await
            .unwrap();

        assert_eq!(result.verdict, CacheVerdict::Miss);
    }

    #[tokio::test]
    async fn test_submit_no_cache() {
        let docs = vec![make_doc("result1")];
        let (engine, _, _) = make_engine(vec![1.0, 0.0, 0.0], docs);

        let session = "session1".to_string();
        // Submit without prior speculation => Miss
        let result = engine
            .submit(&session, "how to configure auth")
            .await
            .unwrap();

        assert_eq!(result.verdict, CacheVerdict::Miss);
    }

    #[tokio::test]
    async fn test_close_session_removes_cache() {
        let docs = vec![make_doc("doc1")];
        let (engine, _, _) = make_engine(vec![1.0, 0.0, 0.0], docs);

        let session = "session1".to_string();
        engine
            .speculate(&session, "how to configure auth")
            .await
            .unwrap();
        assert!(engine.cache().get_latest(&session).is_some());

        engine.close_session(&session);
        assert!(engine.cache().get_latest(&session).is_none());
        assert_eq!(engine.cache().session_count(), 0);
    }

    #[tokio::test]
    async fn test_stats_counts() {
        let docs = vec![make_doc("doc1")];
        let (engine, _, _) = make_engine(vec![1.0, 0.0, 0.0], docs);

        let session = "session1".to_string();
        engine.speculate(&session, "query one").await.unwrap();
        engine.speculate(&session, "query two").await.unwrap();
        engine.submit(&session, "final query").await.unwrap();

        let stats = engine.stats();
        assert_eq!(stats.predictions_total, 2);
        assert_eq!(stats.submissions_total, 1);
        // Same embedding for both speculate and submit => Hit
        assert_eq!(stats.cache_hits, 1);
    }

    #[tokio::test]
    async fn test_speculate_embed_failure() {
        let docs = vec![make_doc("doc1")];
        let (engine, embedder, _) = make_engine(vec![1.0, 0.0, 0.0], docs);

        embedder.set_failing(true);
        let session = "session1".to_string();
        let result = engine.speculate(&session, "how to configure auth").await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, EngineError::Embed(_)));
    }

    #[tokio::test]
    async fn test_submit_retriever_failure() {
        let docs = vec![make_doc("doc1")];
        let (engine, embedder, retriever) = make_engine(vec![1.0, 0.0, 0.0], docs);

        let session = "session1".to_string();
        // Speculate successfully
        engine
            .speculate(&session, "how to configure auth")
            .await
            .unwrap();

        // Change embedding so we get a Miss (which triggers retriever.search)
        embedder.set_embedding(vec![0.0, 1.0, 0.0]);
        retriever.set_failing(true);

        let result = engine
            .submit(&session, "something completely different")
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, EngineError::Retrieve(_)));
    }
}
