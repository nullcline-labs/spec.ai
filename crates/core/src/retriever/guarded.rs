use std::sync::Arc;

use async_trait::async_trait;

use crate::circuit_breaker::CircuitBreaker;
use crate::types::{Document, Embedding};

use super::{RetrieveError, Retriever};

/// A [`Retriever`] wrapper that guards calls behind a [`CircuitBreaker`].
///
/// When the circuit is open, requests are rejected immediately with a 503
/// error instead of being forwarded to the underlying retriever.
pub struct GuardedRetriever {
    inner: Arc<dyn Retriever>,
    breaker: Arc<CircuitBreaker>,
}

impl GuardedRetriever {
    /// Create a new `GuardedRetriever`.
    pub fn new(inner: Arc<dyn Retriever>, breaker: Arc<CircuitBreaker>) -> Self {
        Self { inner, breaker }
    }
}

#[async_trait]
impl Retriever for GuardedRetriever {
    async fn search(
        &self,
        embedding: &Embedding,
        top_k: usize,
    ) -> Result<Vec<Document>, RetrieveError> {
        self.breaker.check().await.map_err(|_| RetrieveError::Api {
            status: 503,
            body: "Circuit breaker open".to_string(),
        })?;

        match self.inner.search(embedding, top_k).await {
            Ok(docs) => {
                self.breaker.record_success();
                Ok(docs)
            }
            Err(e) => {
                self.breaker.record_failure().await;
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    struct FakeRetriever;

    #[async_trait]
    impl Retriever for FakeRetriever {
        async fn search(
            &self,
            _embedding: &Embedding,
            _top_k: usize,
        ) -> Result<Vec<Document>, RetrieveError> {
            Ok(vec![])
        }
    }

    struct FailingRetriever;

    #[async_trait]
    impl Retriever for FailingRetriever {
        async fn search(
            &self,
            _embedding: &Embedding,
            _top_k: usize,
        ) -> Result<Vec<Document>, RetrieveError> {
            Err(RetrieveError::Api {
                status: 500,
                body: "fail".into(),
            })
        }
    }

    #[tokio::test]
    async fn test_success_path() {
        let breaker = Arc::new(CircuitBreaker::new(3, Duration::from_secs(30)));
        let guarded = GuardedRetriever::new(Arc::new(FakeRetriever), breaker);
        let result = guarded.search(&vec![1.0], 10).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_circuit_open_rejects() {
        let breaker = Arc::new(CircuitBreaker::new(1, Duration::from_secs(60)));
        breaker.record_failure().await;
        let guarded = GuardedRetriever::new(Arc::new(FakeRetriever), breaker);
        let result = guarded.search(&vec![1.0], 10).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            RetrieveError::Api { status, .. } => assert_eq!(status, 503),
            other => panic!("Expected Api error, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_failure_recorded() {
        let breaker = Arc::new(CircuitBreaker::new(3, Duration::from_secs(30)));
        let guarded = GuardedRetriever::new(Arc::new(FailingRetriever), breaker.clone());
        let _ = guarded.search(&vec![1.0], 10).await;
        assert!(breaker.check().await.is_ok());
    }

    #[tokio::test]
    async fn test_health_check_default_impl() {
        let retriever = FakeRetriever;
        assert!(retriever.health_check().await.is_ok());
    }
}
