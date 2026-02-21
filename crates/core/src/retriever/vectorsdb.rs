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

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_search_success() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/collections/test_col/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "results": [{
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "text": "hello world",
                    "score": 0.95,
                    "metadata": {}
                }]
            })))
            .mount(&mock_server)
            .await;

        let retriever = VectorsDbRetriever::new(VectorsDbRetrieverConfig {
            base_url: mock_server.uri(),
            collection: "test_col".into(),
            api_key: None,
            timeout: Duration::from_secs(5),
        });
        let results = retriever.search(&vec![0.1, 0.2, 0.3], 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "hello world");
        assert!((results[0].score - 0.95).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_search_api_error() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/collections/test_col/search"))
            .respond_with(ResponseTemplate::new(500).set_body_string("database unavailable"))
            .mount(&mock_server)
            .await;

        let retriever = VectorsDbRetriever::new(VectorsDbRetrieverConfig {
            base_url: mock_server.uri(),
            collection: "test_col".into(),
            api_key: None,
            timeout: Duration::from_secs(5),
        });
        let result = retriever.search(&vec![0.1, 0.2, 0.3], 10).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            RetrieveError::Api { status, body } => {
                assert_eq!(status, 500);
                assert_eq!(body, "database unavailable");
            }
            other => panic!("Expected Api error, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_search_with_api_key() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/collections/test_col/search"))
            .and(header("Authorization", "Bearer my-secret-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "results": [{
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "text": "authenticated result",
                    "score": 0.99,
                    "metadata": {}
                }]
            })))
            .mount(&mock_server)
            .await;

        let retriever = VectorsDbRetriever::new(VectorsDbRetrieverConfig {
            base_url: mock_server.uri(),
            collection: "test_col".into(),
            api_key: Some("my-secret-key".into()),
            timeout: Duration::from_secs(5),
        });
        let results = retriever.search(&vec![0.1, 0.2, 0.3], 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "authenticated result");
    }
}
