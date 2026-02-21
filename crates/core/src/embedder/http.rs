use super::{EmbedError, Embedder};
use crate::types::Embedding;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct HttpEmbedderConfig {
    pub url: String,
    pub model: String,
    pub api_key: Option<String>,
    pub timeout: Duration,
}

pub struct HttpEmbedder {
    client: reqwest::Client,
    config: HttpEmbedderConfig,
}

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a str,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

impl HttpEmbedder {
    pub fn new(config: HttpEmbedderConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .expect("Failed to build HTTP client");
        Self { client, config }
    }
}

#[async_trait::async_trait]
impl Embedder for HttpEmbedder {
    async fn embed(&self, text: &str) -> Result<Embedding, EmbedError> {
        let body = EmbeddingRequest {
            model: &self.config.model,
            input: text,
        };

        let mut req = self.client.post(&self.config.url).json(&body);
        if let Some(ref key) = self.config.api_key {
            req = req.bearer_auth(key);
        }

        let resp = req.send().await?;
        let status = resp.status().as_u16();
        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(EmbedError::Api { status, body });
        }

        let response: EmbeddingResponse = resp.json().await?;
        response
            .data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .ok_or_else(|| EmbedError::InvalidResponse("Empty data array".into()))
    }
}
