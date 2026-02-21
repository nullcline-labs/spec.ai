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

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_embed_success() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{"embedding": [0.1, 0.2, 0.3]}]
            })))
            .mount(&mock_server)
            .await;

        let embedder = HttpEmbedder::new(HttpEmbedderConfig {
            url: mock_server.uri(),
            model: "test-model".into(),
            api_key: None,
            timeout: Duration::from_secs(5),
        });
        let result = embedder.embed("hello").await.unwrap();
        assert_eq!(result, vec![0.1_f32, 0.2_f32, 0.3_f32]);
    }

    #[tokio::test]
    async fn test_embed_api_error() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(500).set_body_string("internal server error"))
            .mount(&mock_server)
            .await;

        let embedder = HttpEmbedder::new(HttpEmbedderConfig {
            url: mock_server.uri(),
            model: "test-model".into(),
            api_key: None,
            timeout: Duration::from_secs(5),
        });
        let result = embedder.embed("hello").await;
        assert!(result.is_err());
        match result.unwrap_err() {
            EmbedError::Api { status, body } => {
                assert_eq!(status, 500);
                assert_eq!(body, "internal server error");
            }
            other => panic!("Expected Api error, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_embed_with_api_key() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(header("Authorization", "Bearer test-secret-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{"embedding": [0.5, 0.6]}]
            })))
            .mount(&mock_server)
            .await;

        let embedder = HttpEmbedder::new(HttpEmbedderConfig {
            url: mock_server.uri(),
            model: "test-model".into(),
            api_key: Some("test-secret-key".into()),
            timeout: Duration::from_secs(5),
        });
        let result = embedder.embed("hello").await.unwrap();
        assert_eq!(result, vec![0.5_f32, 0.6_f32]);
    }

    #[tokio::test]
    async fn test_embed_empty_response() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": []
            })))
            .mount(&mock_server)
            .await;

        let embedder = HttpEmbedder::new(HttpEmbedderConfig {
            url: mock_server.uri(),
            model: "test-model".into(),
            api_key: None,
            timeout: Duration::from_secs(5),
        });
        let result = embedder.embed("hello").await;
        assert!(result.is_err());
        match result.unwrap_err() {
            EmbedError::InvalidResponse(msg) => {
                assert!(msg.contains("Empty data array"));
            }
            other => panic!("Expected InvalidResponse error, got: {other:?}"),
        }
    }
}
