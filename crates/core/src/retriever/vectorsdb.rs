use super::{RetrieveError, Retriever};
use crate::types::{Document, Embedding};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct VectorsDbRetrieverConfig {
    pub base_url: String,
    /// One or more collection names to search across.
    pub collections: Vec<String>,
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
    pub fn new(config: VectorsDbRetrieverConfig) -> Result<Self, reqwest::Error> {
        let client = reqwest::Client::builder().timeout(config.timeout).build()?;
        Ok(Self { client, config })
    }

    async fn search_collection(
        &self,
        collection: &str,
        embedding: &Embedding,
        top_k: usize,
    ) -> Result<Vec<Document>, RetrieveError> {
        let url = format!("{}/collections/{}/search", self.config.base_url, collection);

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

#[async_trait::async_trait]
impl Retriever for VectorsDbRetriever {
    async fn search(
        &self,
        embedding: &Embedding,
        top_k: usize,
    ) -> Result<Vec<Document>, RetrieveError> {
        if self.config.collections.len() == 1 {
            return self
                .search_collection(&self.config.collections[0], embedding, top_k)
                .await;
        }

        // Fan out searches to all collections in parallel
        let futures: Vec<_> = self
            .config
            .collections
            .iter()
            .map(|col| self.search_collection(col, embedding, top_k))
            .collect();

        let results = futures::future::try_join_all(futures).await?;

        // Merge all results, sort by score descending, take top_k
        let mut all_docs: Vec<Document> = results.into_iter().flatten().collect();
        all_docs.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_docs.truncate(top_k);

        Ok(all_docs)
    }

    async fn health_check(&self) -> Result<(), RetrieveError> {
        let dummy = vec![0.0_f32];
        let futures: Vec<_> = self
            .config
            .collections
            .iter()
            .map(|col| self.search_collection(col, &dummy, 1))
            .collect();
        futures::future::try_join_all(futures).await?;
        Ok(())
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
            collections: vec!["test_col".into()],
            api_key: None,
            timeout: Duration::from_secs(5),
        })
        .unwrap();
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
            collections: vec!["test_col".into()],
            api_key: None,
            timeout: Duration::from_secs(5),
        })
        .unwrap();
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
            collections: vec!["test_col".into()],
            api_key: Some("my-secret-key".into()),
            timeout: Duration::from_secs(5),
        })
        .unwrap();
        let results = retriever.search(&vec![0.1, 0.2, 0.3], 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "authenticated result");
    }

    #[tokio::test]
    async fn test_multi_collection_merge() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/collections/docs/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "results": [{
                    "id": "550e8400-e29b-41d4-a716-446655440001",
                    "text": "from docs collection",
                    "score": 0.80,
                    "metadata": {}
                }]
            })))
            .mount(&mock_server)
            .await;

        Mock::given(method("POST"))
            .and(path("/collections/faq/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "results": [{
                    "id": "550e8400-e29b-41d4-a716-446655440002",
                    "text": "from faq collection",
                    "score": 0.90,
                    "metadata": {}
                }]
            })))
            .mount(&mock_server)
            .await;

        let retriever = VectorsDbRetriever::new(VectorsDbRetrieverConfig {
            base_url: mock_server.uri(),
            collections: vec!["docs".into(), "faq".into()],
            api_key: None,
            timeout: Duration::from_secs(5),
        })
        .unwrap();

        let results = retriever.search(&vec![0.1, 0.2, 0.3], 10).await.unwrap();
        assert_eq!(results.len(), 2);
        // Should be sorted by score descending
        assert_eq!(results[0].text, "from faq collection");
        assert_eq!(results[1].text, "from docs collection");
    }

    #[tokio::test]
    async fn test_multi_collection_one_fails() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/collections/docs/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "results": [{
                    "id": "550e8400-e29b-41d4-a716-446655440001",
                    "text": "ok result",
                    "score": 0.90,
                    "metadata": {}
                }]
            })))
            .mount(&mock_server)
            .await;

        Mock::given(method("POST"))
            .and(path("/collections/broken/search"))
            .respond_with(ResponseTemplate::new(500).set_body_string("internal error"))
            .mount(&mock_server)
            .await;

        let retriever = VectorsDbRetriever::new(VectorsDbRetrieverConfig {
            base_url: mock_server.uri(),
            collections: vec!["docs".into(), "broken".into()],
            api_key: None,
            timeout: Duration::from_secs(5),
        })
        .unwrap();

        let result = retriever.search(&vec![0.1, 0.2, 0.3], 10).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_multi_collection_truncates_to_top_k() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/collections/col_a/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "results": [
                    {"id": "550e8400-e29b-41d4-a716-446655440001", "text": "a1", "score": 0.95, "metadata": {}},
                    {"id": "550e8400-e29b-41d4-a716-446655440002", "text": "a2", "score": 0.85, "metadata": {}}
                ]
            })))
            .mount(&mock_server)
            .await;

        Mock::given(method("POST"))
            .and(path("/collections/col_b/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "results": [
                    {"id": "550e8400-e29b-41d4-a716-446655440003", "text": "b1", "score": 0.90, "metadata": {}},
                    {"id": "550e8400-e29b-41d4-a716-446655440004", "text": "b2", "score": 0.80, "metadata": {}}
                ]
            })))
            .mount(&mock_server)
            .await;

        let retriever = VectorsDbRetriever::new(VectorsDbRetrieverConfig {
            base_url: mock_server.uri(),
            collections: vec!["col_a".into(), "col_b".into()],
            api_key: None,
            timeout: Duration::from_secs(5),
        })
        .unwrap();

        // top_k=2 should return only the 2 highest scoring docs
        let results = retriever.search(&vec![0.1, 0.2, 0.3], 2).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].text, "a1"); // 0.95
        assert_eq!(results[1].text, "b1"); // 0.90
    }
}
