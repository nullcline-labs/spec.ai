pub mod vectorsdb;

use crate::types::{Document, Embedding};

#[derive(Debug, thiserror::Error)]
pub enum RetrieveError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("API returned error: {status} - {body}")]
    Api { status: u16, body: String },
    #[error("Invalid response format: {0}")]
    InvalidResponse(String),
}

/// Trait for vector similarity search.
#[async_trait::async_trait]
pub trait Retriever: Send + Sync + 'static {
    async fn search(
        &self,
        embedding: &Embedding,
        top_k: usize,
    ) -> Result<Vec<Document>, RetrieveError>;
}
