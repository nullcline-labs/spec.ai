use std::sync::Arc;

use async_trait::async_trait;

use crate::circuit_breaker::CircuitBreaker;
use crate::types::Embedding;

use super::{EmbedError, Embedder};

/// An [`Embedder`] wrapper that guards calls behind a [`CircuitBreaker`].
///
/// When the circuit is open, requests are rejected immediately with a 503
/// error instead of being forwarded to the underlying embedder.
pub struct GuardedEmbedder {
    inner: Arc<dyn Embedder>,
    breaker: Arc<CircuitBreaker>,
}

impl GuardedEmbedder {
    /// Create a new `GuardedEmbedder`.
    pub fn new(inner: Arc<dyn Embedder>, breaker: Arc<CircuitBreaker>) -> Self {
        Self { inner, breaker }
    }
}

#[async_trait]
impl Embedder for GuardedEmbedder {
    async fn embed(&self, text: &str) -> Result<Embedding, EmbedError> {
        self.breaker.check().await.map_err(|_| EmbedError::Api {
            status: 503,
            body: "Circuit breaker open".to_string(),
        })?;

        match self.inner.embed(text).await {
            Ok(embedding) => {
                self.breaker.record_success();
                Ok(embedding)
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

    struct FakeEmbedder;

    #[async_trait]
    impl Embedder for FakeEmbedder {
        async fn embed(&self, _text: &str) -> Result<Embedding, EmbedError> {
            Ok(vec![1.0, 2.0, 3.0])
        }
    }

    struct FailingEmbedder;

    #[async_trait]
    impl Embedder for FailingEmbedder {
        async fn embed(&self, _text: &str) -> Result<Embedding, EmbedError> {
            Err(EmbedError::Api {
                status: 500,
                body: "fail".into(),
            })
        }
    }

    #[tokio::test]
    async fn test_success_path() {
        let breaker = Arc::new(CircuitBreaker::new(3, Duration::from_secs(30)));
        let guarded = GuardedEmbedder::new(Arc::new(FakeEmbedder), breaker);
        let result = guarded.embed("hello").await.unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[tokio::test]
    async fn test_circuit_open_rejects() {
        let breaker = Arc::new(CircuitBreaker::new(1, Duration::from_secs(60)));
        breaker.record_failure().await;
        let guarded = GuardedEmbedder::new(Arc::new(FakeEmbedder), breaker);
        let result = guarded.embed("hello").await;
        assert!(result.is_err());
        match result.unwrap_err() {
            EmbedError::Api { status, .. } => assert_eq!(status, 503),
            other => panic!("Expected Api error, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_failure_recorded() {
        let breaker = Arc::new(CircuitBreaker::new(3, Duration::from_secs(30)));
        let guarded = GuardedEmbedder::new(Arc::new(FailingEmbedder), breaker.clone());
        let _ = guarded.embed("hello").await;
        // Circuit should still be closed (below threshold of 3)
        assert!(breaker.check().await.is_ok());
    }

    #[tokio::test]
    async fn test_health_check_default_impl() {
        let embedder = FakeEmbedder;
        assert!(embedder.health_check().await.is_ok());
    }
}
