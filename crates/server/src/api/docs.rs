use utoipa::OpenApi;

#[derive(OpenApi)]
#[openapi(
    paths(
        super::handlers::health,
        super::handlers::ready,
        super::handlers::stats,
        super::handlers::submit,
        super::handlers::metrics_endpoint,
    ),
    components(schemas(
        super::models::SubmitRequest,
        super::models::SubmitResponse,
        super::models::HealthResponse,
        super::models::ReadyResponse,
        specai_core::types::Document,
        specai_core::types::EngineStats,
    ))
)]
pub struct ApiDoc;
