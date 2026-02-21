use std::collections::HashSet;

/// Compute Jaccard similarity between query terms and document terms.
///
/// Tokenizes by whitespace, lowercases, strips punctuation, then computes
/// |intersection| / |union|. Returns a value in `[0.0, 1.0]`.
pub fn jaccard_similarity(query: &str, document: &str) -> f32 {
    let query_terms: HashSet<String> = tokenize(query);
    let doc_terms: HashSet<String> = tokenize(document);

    if query_terms.is_empty() || doc_terms.is_empty() {
        return 0.0;
    }

    let intersection = query_terms.intersection(&doc_terms).count() as f32;
    let union = query_terms.union(&doc_terms).count() as f32;

    intersection / union
}

fn tokenize(text: &str) -> HashSet<String> {
    text.split_whitespace()
        .map(|t| {
            t.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|t| !t.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_strings() {
        let score = jaccard_similarity("hello world", "hello world");
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_no_overlap() {
        let score = jaccard_similarity("hello world", "foo bar");
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_partial_overlap() {
        let score = jaccard_similarity("how to configure", "configure auth settings");
        // intersection: {"configure"} = 1
        // union: {"how", "to", "configure", "auth", "settings"} = 5
        assert!((score - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_case_insensitive() {
        let score = jaccard_similarity("Hello World", "hello world");
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_punctuation_stripped() {
        let score = jaccard_similarity("hello!", "hello.");
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_strings() {
        assert!((jaccard_similarity("", "hello") - 0.0).abs() < 1e-6);
        assert!((jaccard_similarity("hello", "") - 0.0).abs() < 1e-6);
        assert!((jaccard_similarity("", "") - 0.0).abs() < 1e-6);
    }
}
