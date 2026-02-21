pub mod text_relevance;

use crate::types::Document;

/// Trait for post-retrieval document re-ranking.
#[async_trait::async_trait]
pub trait Reranker: Send + Sync + 'static {
    async fn rerank(&self, query: &str, documents: Vec<Document>) -> Vec<Document>;
}

/// Configuration for the weighted re-ranker.
#[derive(Debug, Clone)]
pub struct WeightedRerankerConfig {
    /// Weight for the original vector similarity score (0.0–1.0).
    /// Text relevance weight is `1.0 - alpha`.
    pub alpha: f32,
}

impl Default for WeightedRerankerConfig {
    fn default() -> Self {
        Self { alpha: 0.7 }
    }
}

/// A re-ranker that combines vector similarity score with Jaccard text relevance.
pub struct WeightedReranker {
    config: WeightedRerankerConfig,
}

impl WeightedReranker {
    pub fn new(config: WeightedRerankerConfig) -> Self {
        Self { config }
    }
}

#[async_trait::async_trait]
impl Reranker for WeightedReranker {
    async fn rerank(&self, query: &str, mut documents: Vec<Document>) -> Vec<Document> {
        let alpha = self.config.alpha;
        for doc in &mut documents {
            let text_score = text_relevance::jaccard_similarity(query, &doc.text);
            doc.score = alpha * doc.score + (1.0 - alpha) * text_score;
        }
        documents.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        documents
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn make_doc(text: &str, score: f32) -> Document {
        Document {
            id: Uuid::new_v4(),
            text: text.to_string(),
            score,
            metadata: Default::default(),
        }
    }

    #[tokio::test]
    async fn test_reranker_reorders() {
        let reranker = WeightedReranker::new(WeightedRerankerConfig { alpha: 0.5 });
        let docs = vec![
            make_doc("unrelated content about nothing", 0.9),
            make_doc("how to configure authentication easily", 0.7),
        ];

        let result = reranker
            .rerank("how to configure authentication", docs)
            .await;

        // The second doc has much higher text relevance for this query
        assert_eq!(result[0].text, "how to configure authentication easily");
    }

    #[tokio::test]
    async fn test_alpha_one_preserves_order() {
        let reranker = WeightedReranker::new(WeightedRerankerConfig { alpha: 1.0 });
        let docs = vec![
            make_doc("first doc", 0.9),
            make_doc("second doc", 0.8),
            make_doc("third doc", 0.7),
        ];

        let result = reranker.rerank("anything", docs).await;

        assert!((result[0].score - 0.9).abs() < 1e-6);
        assert!((result[1].score - 0.8).abs() < 1e-6);
        assert!((result[2].score - 0.7).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_alpha_zero_ranks_by_text() {
        let reranker = WeightedReranker::new(WeightedRerankerConfig { alpha: 0.0 });
        let docs = vec![
            make_doc("unrelated", 0.99),
            make_doc("hello world greeting", 0.5),
        ];

        let result = reranker.rerank("hello world", docs).await;

        // With alpha=0, only text relevance matters
        assert_eq!(result[0].text, "hello world greeting");
    }
}
