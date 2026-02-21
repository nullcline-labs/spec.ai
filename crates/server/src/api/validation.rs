use specai_core::config;

#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("session_id must not be empty")]
    EmptySessionId,
    #[error("session_id exceeds maximum length of {0} characters")]
    SessionIdTooLong(usize),
    #[error("session_id contains invalid characters (allowed: alphanumeric, hyphen, underscore)")]
    InvalidSessionIdChars,
    #[error("query must not be empty")]
    EmptyQuery,
    #[error("query exceeds maximum length of {0} characters")]
    QueryTooLong(usize),
}

pub fn validate_session_id(session_id: &str) -> Result<(), ValidationError> {
    if session_id.is_empty() {
        return Err(ValidationError::EmptySessionId);
    }
    if session_id.len() > config::MAX_SESSION_ID_LENGTH {
        return Err(ValidationError::SessionIdTooLong(
            config::MAX_SESSION_ID_LENGTH,
        ));
    }
    if !session_id
        .chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
    {
        return Err(ValidationError::InvalidSessionIdChars);
    }
    Ok(())
}

pub fn validate_query(query: &str) -> Result<(), ValidationError> {
    if query.is_empty() {
        return Err(ValidationError::EmptyQuery);
    }
    if query.len() > config::MAX_QUERY_LENGTH {
        return Err(ValidationError::QueryTooLong(config::MAX_QUERY_LENGTH));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_session_id() {
        assert!(validate_session_id("abc-123_XYZ").is_ok());
    }

    #[test]
    fn test_empty_session_id() {
        assert!(matches!(
            validate_session_id(""),
            Err(ValidationError::EmptySessionId)
        ));
    }

    #[test]
    fn test_too_long_session_id() {
        let long = "a".repeat(config::MAX_SESSION_ID_LENGTH + 1);
        assert!(matches!(
            validate_session_id(&long),
            Err(ValidationError::SessionIdTooLong(_))
        ));
    }

    #[test]
    fn test_session_id_at_max_length() {
        let exact = "a".repeat(config::MAX_SESSION_ID_LENGTH);
        assert!(validate_session_id(&exact).is_ok());
    }

    #[test]
    fn test_session_id_with_special_chars() {
        assert!(matches!(
            validate_session_id("abc def"),
            Err(ValidationError::InvalidSessionIdChars)
        ));
        assert!(matches!(
            validate_session_id("abc;drop"),
            Err(ValidationError::InvalidSessionIdChars)
        ));
        assert!(matches!(
            validate_session_id("abc/def"),
            Err(ValidationError::InvalidSessionIdChars)
        ));
    }

    #[test]
    fn test_valid_query() {
        assert!(validate_query("how to configure auth").is_ok());
    }

    #[test]
    fn test_empty_query() {
        assert!(matches!(
            validate_query(""),
            Err(ValidationError::EmptyQuery)
        ));
    }

    #[test]
    fn test_too_long_query() {
        let long = "a".repeat(config::MAX_QUERY_LENGTH + 1);
        assert!(matches!(
            validate_query(&long),
            Err(ValidationError::QueryTooLong(_))
        ));
    }

    #[test]
    fn test_query_at_max_length() {
        let exact = "a".repeat(config::MAX_QUERY_LENGTH);
        assert!(validate_query(&exact).is_ok());
    }
}
