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
