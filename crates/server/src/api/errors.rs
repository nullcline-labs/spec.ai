use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

#[derive(Debug)]
pub enum ApiError {
    BadRequest(String),
    NotFound(String),
    Internal(String),
    ServiceUnavailable(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            ApiError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            ApiError::ServiceUnavailable(msg) => (StatusCode::SERVICE_UNAVAILABLE, msg),
        };
        let body = axum::Json(json!({ "error": message }));
        (status, body).into_response()
    }
}

impl From<specai_core::engine::EngineError> for ApiError {
    fn from(err: specai_core::engine::EngineError) -> Self {
        tracing::error!("Engine error: {}", err);
        ApiError::Internal("Retrieval engine error".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bad_request_status() {
        let resp = ApiError::BadRequest("bad".into()).into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_not_found_status() {
        let resp = ApiError::NotFound("missing".into()).into_response();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn test_internal_status() {
        let resp = ApiError::Internal("oops".into()).into_response();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_service_unavailable_status() {
        let resp = ApiError::ServiceUnavailable("down".into()).into_response();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn test_from_engine_error() {
        let engine_err = specai_core::engine::EngineError::Embed(
            specai_core::embedder::EmbedError::InvalidResponse("test".into()),
        );
        let api_err: ApiError = engine_err.into();
        let resp = api_err.into_response();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }
}
