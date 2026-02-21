pub mod cached;
pub mod guarded;
pub mod http;

use crate::types::Embedding;

#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("API returned error: {status} - {body}")]
    Api { status: u16, body: String },
    #[error("Invalid response format: {0}")]
    InvalidResponse(String),
}

/// Trait for text-to-embedding conversion.
#[async_trait::async_trait]
pub trait Embedder: Send + Sync + 'static {
    async fn embed(&self, text: &str) -> Result<Embedding, EmbedError>;

    /// Health check that verifies the embedder is responsive.
    async fn health_check(&self) -> Result<(), EmbedError> {
        self.embed("health").await.map(|_| ())
    }
}
