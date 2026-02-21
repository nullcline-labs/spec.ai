use super::{RetrieveError, Retriever};
use crate::types::{Document, Embedding};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct VectorsDbRetrieverConfig {
    pub base_url: String,
    pub collection: String,
    pub api_key: Option<String>,
    pub timeout: Duration,
}

pub struct VectorsDbRetriever {
    client: reqwest::Client,
    config: VectorsDbRetrieverConfig,
}

#[derive(Serialize)]
struct SearchRequest<'a> {
    query_embedding: &'a [f32],
    k: usize,
}

#[derive(Deserialize)]
struct SearchResponse {
    results: Vec<SearchResult>,
}

#[derive(Deserialize)]
struct SearchResult {
    id: uuid::Uuid,
    text: String,
    score: f32,
    #[serde(default)]
    metadata: std::collections::HashMap<String, serde_json::Value>,
}

impl VectorsDbRetriever {
    pub fn new(config: VectorsDbRetrieverConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .expect("Failed to build HTTP client");
        Self { client, config }
    }
}

#[async_trait::async_trait]
impl Retriever for VectorsDbRetriever {
    async fn search(
        &self,
        embedding: &Embedding,
        top_k: usize,
    ) -> Result<Vec<Document>, RetrieveError> {
        let url = format!(
            "{}/collections/{}/search",
            self.config.base_url, self.config.collection
        );

        let body = SearchRequest {
            query_embedding: embedding,
            k: top_k,
        };

        let mut req = self.client.post(&url).json(&body);
        if let Some(ref key) = self.config.api_key {
            req = req.bearer_auth(key);
        }

        let resp = req.send().await?;
        let status = resp.status().as_u16();
        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(RetrieveError::Api { status, body });
        }

        let response: SearchResponse = resp.json().await?;
        Ok(response
            .results
            .into_iter()
            .map(|r| Document {
                id: r.id,
                text: r.text,
                score: r.score,
                metadata: r.metadata,
            })
            .collect())
    }
}
