use crate::types::{CacheVerdict, Embedding};

/// Configuration for the similarity gate.
#[derive(Debug, Clone)]
pub struct SimilarityGateConfig {
    pub hit_threshold: f32,
    pub partial_threshold: f32,
}

/// Decides whether cached speculative results are usable for a final query.
pub struct SimilarityGate {
    config: SimilarityGateConfig,
}

impl SimilarityGate {
    pub fn new(config: SimilarityGateConfig) -> Self {
        Self { config }
    }

    /// Compute cosine similarity between two embedding vectors.
    pub fn cosine_similarity(a: &Embedding, b: &Embedding) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "Embedding dimensions must match");

        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }

        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom < f32::EPSILON {
            return 0.0;
        }
        dot / denom
    }

    /// Evaluate whether a cached embedding is similar enough to the final query.
    pub fn evaluate(
        &self,
        cached_embedding: &Embedding,
        final_embedding: &Embedding,
    ) -> CacheVerdict {
        let similarity = Self::cosine_similarity(cached_embedding, final_embedding);
        tracing::trace!(similarity = similarity, "Similarity gate evaluation");

        if similarity >= self.config.hit_threshold {
            CacheVerdict::Hit
        } else if similarity >= self.config.partial_threshold {
            CacheVerdict::Partial
        } else {
            CacheVerdict::Miss
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config;

    fn gate() -> SimilarityGate {
        SimilarityGate::new(SimilarityGateConfig {
            hit_threshold: config::DEFAULT_SIMILARITY_THRESHOLD,
            partial_threshold: config::DEFAULT_PARTIAL_THRESHOLD,
        })
    }

    #[test]
    fn test_cosine_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = SimilarityGate::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = SimilarityGate::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let sim = SimilarityGate::cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = SimilarityGate::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_gate_hit() {
        let g = gate();
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.99, 0.1, 0.0]; // very similar
        assert_eq!(g.evaluate(&a, &b), CacheVerdict::Hit);
    }

    #[test]
    fn test_gate_miss() {
        let g = gate();
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0]; // orthogonal
        assert_eq!(g.evaluate(&a, &b), CacheVerdict::Miss);
    }

    #[test]
    fn test_gate_partial() {
        let g = gate();
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.85, 0.5, 0.0]; // somewhat similar
        let verdict = g.evaluate(&a, &b);
        assert_eq!(verdict, CacheVerdict::Partial);
    }
}
