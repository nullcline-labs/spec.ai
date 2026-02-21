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
