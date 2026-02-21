use clap::{CommandFactory, FromArgMatches, Parser};
use specai_core::cache::eviction::spawn_eviction_task;
use specai_core::cache::SpeculativeCache;
use specai_core::circuit_breaker::CircuitBreaker;
use specai_core::config;
use specai_core::embedder::cached::CachedEmbedder;
use specai_core::embedder::guarded::GuardedEmbedder;
use specai_core::embedder::http::{HttpEmbedder, HttpEmbedderConfig};
use specai_core::embedder::Embedder;
use specai_core::engine::{Engine, EngineConfig};
use specai_core::retriever::guarded::GuardedRetriever;
use specai_core::retriever::vectorsdb::{VectorsDbRetriever, VectorsDbRetrieverConfig};
use specai_core::similarity::SimilarityGateConfig;
use specai_server::api::create_router;
use specai_server::api::handlers::AppState;
use specai_server::api::RouterConfig;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "spec-ai", about = "Speculative retrieval engine for RAG")]
struct Args {
    /// Path to TOML config file
    #[arg(short = 'c', long)]
    config: Option<std::path::PathBuf>,

    #[arg(short, long, default_value_t = config::DEFAULT_PORT)]
    port: u16,

    #[arg(long, default_value = config::DEFAULT_EMBEDDING_URL)]
    embedding_url: String,

    #[arg(long, default_value = config::DEFAULT_EMBEDDING_MODEL)]
    embedding_model: String,

    #[arg(long)]
    embedding_api_key: Option<String>,

    #[arg(long, default_value = config::DEFAULT_VECTORSDB_URL)]
    vectorsdb_url: String,

    #[arg(long, default_value = config::DEFAULT_COLLECTION, help = "Comma-separated collection names")]
    collection: String,

    #[arg(long)]
    vectorsdb_api_key: Option<String>,

    #[arg(long, default_value_t = config::DEFAULT_SIMILARITY_THRESHOLD)]
    similarity_threshold: f32,

    #[arg(long, default_value_t = config::DEFAULT_PARTIAL_THRESHOLD)]
    partial_threshold: f32,

    #[arg(long, default_value_t = config::DEFAULT_TOP_K)]
    top_k: usize,

    #[arg(long, default_value_t = config::DEFAULT_DEBOUNCE_MS)]
    debounce_ms: u64,

    #[arg(long, default_value_t = config::DEFAULT_CACHE_TTL_SECS)]
    cache_ttl: u64,

    #[arg(long, default_value_t = config::RATE_LIMIT_RPS)]
    rate_limit_rps: u64,

    #[arg(long, default_value_t = config::MAX_CONCURRENT_REQUESTS)]
    max_concurrent: usize,

    #[arg(long, default_value_t = config::REQUEST_TIMEOUT_SECS)]
    request_timeout: u64,

    #[arg(long, default_value_t = config::MAX_REQUEST_BODY_BYTES)]
    max_body_size: usize,

    #[arg(long)]
    api_key: Option<String>,

    #[arg(long, default_value_t = false, help = "Enable result re-ranking")]
    rerank: bool,

    #[arg(long, default_value_t = config::DEFAULT_RERANK_ALPHA)]
    rerank_alpha: f32,

    /// Comma-separated list of allowed WebSocket origins (CSRF protection)
    #[arg(long)]
    allowed_origins: Option<String>,

    /// Metrics update interval in seconds
    #[arg(long, default_value_t = config::DEFAULT_METRICS_INTERVAL_SECS)]
    metrics_interval: u64,

    /// Path to TLS certificate PEM file (enables HTTPS when paired with --tls-key)
    #[arg(long, requires = "tls_key")]
    tls_cert: Option<std::path::PathBuf>,

    /// Path to TLS private key PEM file (enables HTTPS when paired with --tls-cert)
    #[arg(long, requires = "tls_cert")]
    tls_key: Option<std::path::PathBuf>,
}

#[derive(Debug, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct FileConfig {
    port: Option<u16>,
    embedding_url: Option<String>,
    embedding_model: Option<String>,
    embedding_api_key: Option<String>,
    vectorsdb_url: Option<String>,
    collection: Option<String>,
    vectorsdb_api_key: Option<String>,
    similarity_threshold: Option<f32>,
    partial_threshold: Option<f32>,
    top_k: Option<usize>,
    debounce_ms: Option<u64>,
    cache_ttl: Option<u64>,
    rate_limit_rps: Option<u64>,
    max_concurrent: Option<usize>,
    request_timeout: Option<u64>,
    max_body_size: Option<usize>,
    api_key: Option<String>,
    rerank: Option<bool>,
    rerank_alpha: Option<f32>,
    metrics_interval: Option<u64>,
    allowed_origins: Option<String>,
    tls_cert: Option<String>,
    tls_key: Option<String>,
}

fn load_config_file(
    path: &Option<std::path::PathBuf>,
) -> Result<FileConfig, Box<dyn std::error::Error>> {
    let config_path = match path {
        Some(p) => {
            if !p.exists() {
                return Err(format!("Config file not found: {}", p.display()).into());
            }
            p.clone()
        }
        None => {
            let default = std::path::PathBuf::from("specai.toml");
            if default.exists() {
                tracing::info!(path = %default.display(), "Loading config from file");
                default
            } else {
                return Ok(FileConfig::default());
            }
        }
    };
    let content = std::fs::read_to_string(&config_path)?;
    let file_config: FileConfig = toml::from_str(&content)
        .map_err(|e| format!("Failed to parse {}: {}", config_path.display(), e))?;
    tracing::info!(path = %config_path.display(), "Loaded config file");
    Ok(file_config)
}

/// Apply file config values where CLI was not explicitly provided.
fn merge_config(args: &mut Args, matches: &clap::ArgMatches, file_config: FileConfig) {
    use clap::parser::ValueSource;

    macro_rules! apply {
        ($field:ident) => {
            if matches.value_source(stringify!($field)) != Some(ValueSource::CommandLine) {
                if let Some(val) = file_config.$field {
                    args.$field = val;
                }
            }
        };
    }

    apply!(port);
    apply!(embedding_url);
    apply!(embedding_model);
    apply!(vectorsdb_url);
    apply!(collection);
    apply!(similarity_threshold);
    apply!(partial_threshold);
    apply!(top_k);
    apply!(debounce_ms);
    apply!(cache_ttl);
    apply!(rate_limit_rps);
    apply!(max_concurrent);
    apply!(request_timeout);
    apply!(max_body_size);
    apply!(rerank);
    apply!(rerank_alpha);
    apply!(metrics_interval);

    // Option<String> fields: CLI > file config
    macro_rules! apply_opt {
        ($field:ident) => {
            if matches.value_source(stringify!($field)) != Some(ValueSource::CommandLine) {
                if args.$field.is_none() {
                    args.$field = file_config.$field;
                }
            }
        };
    }

    apply_opt!(embedding_api_key);
    apply_opt!(vectorsdb_api_key);
    apply_opt!(api_key);
    apply_opt!(allowed_origins);

    // TLS paths from config file (convert String to PathBuf)
    if matches.value_source("tls_cert") != Some(ValueSource::CommandLine) && args.tls_cert.is_none()
    {
        if let Some(cert) = file_config.tls_cert {
            args.tls_cert = Some(std::path::PathBuf::from(cert));
        }
    }
    if matches.value_source("tls_key") != Some(ValueSource::CommandLine) && args.tls_key.is_none() {
        if let Some(key) = file_config.tls_key {
            args.tls_key = Some(std::path::PathBuf::from(key));
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("specai_server=info".parse()?)
                .add_directive("specai_core=info".parse()?)
                .add_directive("specai_audit=info".parse()?),
        )
        .init();

    let matches = Args::command().get_matches();
    let mut args =
        Args::from_arg_matches(&matches).map_err(|e| format!("Failed to parse arguments: {e}"))?;

    let file_config = load_config_file(&args.config)?;
    merge_config(&mut args, &matches, file_config);

    // Validate threshold ordering
    if args.similarity_threshold < args.partial_threshold {
        return Err(format!(
            "similarity_threshold ({}) must be >= partial_threshold ({})",
            args.similarity_threshold, args.partial_threshold
        )
        .into());
    }

    // Validate rerank_alpha bounds
    if args.rerank && !(0.0..=1.0).contains(&args.rerank_alpha) {
        return Err(format!(
            "rerank_alpha ({}) must be between 0.0 and 1.0",
            args.rerank_alpha
        )
        .into());
    }

    // API key resolution: CLI > env var > config file (already merged above)
    let embedding_api_key = args
        .embedding_api_key
        .or_else(|| std::env::var("SPECAI_EMBEDDING_API_KEY").ok());
    let vectorsdb_api_key = args
        .vectorsdb_api_key
        .or_else(|| std::env::var("SPECAI_VECTORSDB_API_KEY").ok());

    if let Some(ref key) = embedding_api_key {
        if key.trim().is_empty() {
            tracing::warn!("SPECAI_EMBEDDING_API_KEY is set but empty — requests may fail");
        }
    }
    if let Some(ref key) = vectorsdb_api_key {
        if key.trim().is_empty() {
            tracing::warn!("SPECAI_VECTORSDB_API_KEY is set but empty — requests may fail");
        }
    }

    let embedder = Arc::new(HttpEmbedder::new(HttpEmbedderConfig {
        url: args.embedding_url.clone(),
        model: args.embedding_model.clone(),
        api_key: embedding_api_key,
        timeout: Duration::from_secs(config::EXTERNAL_REQUEST_TIMEOUT_SECS),
    })?);

    // Startup embedding validation — non-blocking (service might not be ready in Docker Compose)
    match embedder.embed("startup validation").await {
        Ok(emb) => {
            let dim = emb.len();
            tracing::info!(embedding_dimension = dim, "Embedding service validated");
            if !(config::MIN_EXPECTED_EMBEDDING_DIM..=config::MAX_EXPECTED_EMBEDDING_DIM)
                .contains(&dim)
            {
                tracing::warn!(
                    dim,
                    min = config::MIN_EXPECTED_EMBEDDING_DIM,
                    max = config::MAX_EXPECTED_EMBEDDING_DIM,
                    "Embedding dimension outside expected range"
                );
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, "Embedding validation failed at startup — service may not be ready")
        }
    }

    let collections: Vec<String> = args
        .collection
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let retriever = Arc::new(VectorsDbRetriever::new(VectorsDbRetrieverConfig {
        base_url: args.vectorsdb_url.clone(),
        collections: collections.clone(),
        api_key: vectorsdb_api_key,
        timeout: Duration::from_secs(config::EXTERNAL_REQUEST_TIMEOUT_SECS),
    })?);

    let cache = Arc::new(SpeculativeCache::new(
        args.cache_ttl,
        config::MAX_ENTRIES_PER_SESSION,
    ));

    let engine_config = EngineConfig {
        top_k: args.top_k,
        min_query_length: config::MIN_QUERY_LENGTH,
        similarity: SimilarityGateConfig {
            hit_threshold: args.similarity_threshold,
            partial_threshold: args.partial_threshold,
        },
    };

    let embed_circuit = Arc::new(CircuitBreaker::new(
        config::CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        Duration::from_secs(config::CIRCUIT_BREAKER_RECOVERY_SECS),
    ));
    let retrieve_circuit = Arc::new(CircuitBreaker::new(
        config::CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        Duration::from_secs(config::CIRCUIT_BREAKER_RECOVERY_SECS),
    ));

    let guarded_embedder: Arc<GuardedEmbedder> =
        Arc::new(GuardedEmbedder::new(embedder, embed_circuit));
    let cached_embedder: Arc<CachedEmbedder> = Arc::new(CachedEmbedder::new(
        guarded_embedder,
        config::EMBEDDING_CACHE_MAX_ENTRIES,
        config::EMBEDDING_CACHE_TTL_SECS,
    ));
    let guarded_retriever: Arc<GuardedRetriever> =
        Arc::new(GuardedRetriever::new(retriever, retrieve_circuit));

    let reranker: Option<Arc<dyn specai_core::reranker::Reranker>> = if args.rerank {
        Some(Arc::new(specai_core::reranker::WeightedReranker::new(
            specai_core::reranker::WeightedRerankerConfig {
                alpha: args.rerank_alpha,
            },
        )))
    } else {
        None
    };

    let engine = Arc::new(Engine::new(
        cached_embedder,
        guarded_retriever,
        cache.clone(),
        engine_config,
        reranker,
    ));

    spawn_eviction_task(cache, config::CACHE_EVICTION_INTERVAL_SECS);

    let metrics_engine = engine.clone();
    let metrics_interval_secs = args.metrics_interval;
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(metrics_interval_secs));
        loop {
            interval.tick().await;
            specai_server::api::metrics::update_engine_metrics(&metrics_engine.stats());
        }
    });

    let prometheus_handle =
        metrics_exporter_prometheus::PrometheusBuilder::new().install_recorder()?;

    let allowed_origins = args.allowed_origins.map(|s| {
        s.split(',')
            .map(|o| o.trim().to_string())
            .filter(|o| !o.is_empty())
            .collect::<Vec<_>>()
    });

    let state = AppState {
        engine,
        prometheus_handle,
        start_time: Instant::now(),
        debounce_ms: args.debounce_ms,
        api_key: args.api_key,
        allowed_origins,
        ws_connections_per_ip: Arc::new(dashmap::DashMap::new()),
        auth_failures: Arc::new(dashmap::DashMap::new()),
    };

    let router_config = RouterConfig {
        rate_limit_rps: args.rate_limit_rps,
        max_concurrent: args.max_concurrent,
        request_timeout_secs: args.request_timeout,
        max_body_size: args.max_body_size,
    };

    let app = create_router(state, router_config);
    let addr: std::net::SocketAddr = format!("0.0.0.0:{}", args.port).parse()?;
    let tls_enabled = args.tls_cert.is_some();

    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        port = args.port,
        tls = tls_enabled,
        embedding_url = %args.embedding_url,
        embedding_model = %args.embedding_model,
        vectorsdb_url = %args.vectorsdb_url,
        collections = ?collections,
        similarity_threshold = args.similarity_threshold,
        top_k = args.top_k,
        debounce_ms = args.debounce_ms,
        cache_ttl = args.cache_ttl,
        "spec.ai ready"
    );

    if let (Some(cert_path), Some(key_path)) = (args.tls_cert, args.tls_key) {
        let tls_config =
            axum_server::tls_rustls::RustlsConfig::from_pem_file(&cert_path, &key_path)
                .await
                .map_err(|e| {
                    format!(
                        "Failed to load TLS config (cert={}, key={}): {}",
                        cert_path.display(),
                        key_path.display(),
                        e
                    )
                })?;

        tracing::info!(
            cert = %cert_path.display(),
            key = %key_path.display(),
            "TLS enabled"
        );

        let handle = axum_server::Handle::new();
        let shutdown_handle = handle.clone();
        tokio::spawn(async move {
            wait_for_signal().await;
            tracing::info!(
                timeout_secs = config::GRACEFUL_SHUTDOWN_TIMEOUT_SECS,
                "Draining connections"
            );
            shutdown_handle.graceful_shutdown(Some(Duration::from_secs(
                config::GRACEFUL_SHUTDOWN_TIMEOUT_SECS,
            )));
        });

        axum_server::bind_rustls(addr, tls_config)
            .handle(handle)
            .serve(app.into_make_service_with_connect_info::<std::net::SocketAddr>())
            .await?;
    } else {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .with_graceful_shutdown(wait_for_signal())
        .await?;
    }

    tracing::info!("spec.ai shut down");
    Ok(())
}

async fn wait_for_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("Received SIGINT"),
        _ = terminate => tracing::info!("Received SIGTERM"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_config_full_deserialize() {
        let toml_str = r#"
            port = 8080
            embedding_url = "http://example.com/embed"
            embedding_model = "my-model"
            embedding_api_key = "key1"
            vectorsdb_url = "http://example.com/db"
            collection = "docs,faq"
            vectorsdb_api_key = "key2"
            similarity_threshold = 0.95
            partial_threshold = 0.85
            top_k = 20
            debounce_ms = 500
            cache_ttl = 300
            rate_limit_rps = 100
            max_concurrent = 256
            request_timeout = 60
            max_body_size = 2048
            api_key = "secret"
            rerank = true
            rerank_alpha = 0.5
        "#;
        let config: FileConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.port, Some(8080));
        assert_eq!(
            config.embedding_url.as_deref(),
            Some("http://example.com/embed")
        );
        assert_eq!(config.top_k, Some(20));
        assert_eq!(config.rerank, Some(true));
        assert_eq!(config.api_key.as_deref(), Some("secret"));
    }

    #[test]
    fn test_file_config_partial() {
        let toml_str = r#"port = 9090"#;
        let config: FileConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.port, Some(9090));
        assert!(config.embedding_url.is_none());
        assert!(config.api_key.is_none());
    }

    #[test]
    fn test_file_config_unknown_field_rejected() {
        let toml_str = r#"port = 3040
unknown_field = "bad""#;
        let result: Result<FileConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_config_file_missing_explicit_path() {
        let path = Some(std::path::PathBuf::from("/nonexistent/specai.toml"));
        let result = load_config_file(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_config_file_no_path_no_file() {
        // When no path is given and specai.toml doesn't exist in CWD,
        // should return default (empty) config
        let result = load_config_file(&None);
        // This may or may not find a specai.toml depending on CWD,
        // but should not error
        assert!(result.is_ok());
    }

    #[test]
    fn test_tls_cert_without_key_fails() {
        let result = Args::try_parse_from(["spec-ai", "--tls-cert", "cert.pem"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_tls_both_cert_and_key_succeeds() {
        let result =
            Args::try_parse_from(["spec-ai", "--tls-cert", "cert.pem", "--tls-key", "key.pem"]);
        assert!(result.is_ok());
        let args = result.unwrap();
        assert_eq!(args.tls_cert.unwrap().to_str().unwrap(), "cert.pem");
        assert_eq!(args.tls_key.unwrap().to_str().unwrap(), "key.pem");
    }

    #[test]
    fn test_threshold_validation_rejects_inverted() {
        let args = Args::try_parse_from([
            "spec-ai",
            "--similarity-threshold",
            "0.5",
            "--partial-threshold",
            "0.9",
        ])
        .unwrap();
        // similarity < partial should be rejected
        assert!(args.similarity_threshold < args.partial_threshold);
    }

    #[test]
    fn test_threshold_validation_accepts_valid() {
        let args = Args::try_parse_from([
            "spec-ai",
            "--similarity-threshold",
            "0.92",
            "--partial-threshold",
            "0.80",
        ])
        .unwrap();
        assert!(args.similarity_threshold >= args.partial_threshold);
    }

    #[test]
    fn test_rerank_alpha_bounds() {
        let args = Args::try_parse_from(["spec-ai", "--rerank", "--rerank-alpha", "0.5"]).unwrap();
        assert!((0.0..=1.0).contains(&args.rerank_alpha));
    }

    #[test]
    fn test_allowed_origins_parsed() {
        let args = Args::try_parse_from([
            "spec-ai",
            "--allowed-origins",
            "https://app.example.com,https://staging.example.com",
        ])
        .unwrap();
        assert_eq!(
            args.allowed_origins.as_deref(),
            Some("https://app.example.com,https://staging.example.com")
        );
    }

    #[test]
    fn test_file_config_with_allowed_origins() {
        let toml_str = r#"
            allowed_origins = "https://app.example.com,https://staging.example.com"
        "#;
        let config: FileConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(
            config.allowed_origins.as_deref(),
            Some("https://app.example.com,https://staging.example.com")
        );
    }

    #[test]
    fn test_merge_config_file_overrides_defaults() {
        let matches = Args::command().get_matches_from(["spec-ai"]);
        let mut args = Args::from_arg_matches(&matches).unwrap();

        let file_config = FileConfig {
            port: Some(9999),
            embedding_url: Some("http://custom/v1/embeddings".to_string()),
            embedding_model: Some("custom-model".to_string()),
            vectorsdb_url: Some("http://custom:3030".to_string()),
            collection: Some("custom_col".to_string()),
            similarity_threshold: Some(0.95),
            partial_threshold: Some(0.85),
            top_k: Some(20),
            debounce_ms: Some(500),
            cache_ttl: Some(300),
            rate_limit_rps: Some(100),
            max_concurrent: Some(256),
            request_timeout: Some(60),
            max_body_size: Some(2048),
            api_key: Some("file-key".to_string()),
            rerank: Some(true),
            rerank_alpha: Some(0.5),
            metrics_interval: Some(30),
            embedding_api_key: Some("embed-key".to_string()),
            vectorsdb_api_key: Some("db-key".to_string()),
            allowed_origins: Some("https://example.com".to_string()),
            tls_cert: Some("/path/to/cert.pem".to_string()),
            tls_key: Some("/path/to/key.pem".to_string()),
        };

        merge_config(&mut args, &matches, file_config);

        assert_eq!(args.port, 9999);
        assert_eq!(args.embedding_url, "http://custom/v1/embeddings");
        assert_eq!(args.embedding_model, "custom-model");
        assert_eq!(args.top_k, 20);
        assert_eq!(args.debounce_ms, 500);
        assert!(args.rerank);
        assert_eq!(args.rerank_alpha, 0.5);
        assert_eq!(args.metrics_interval, 30);
        assert_eq!(args.api_key.as_deref(), Some("file-key"));
        assert_eq!(args.embedding_api_key.as_deref(), Some("embed-key"));
        assert_eq!(args.vectorsdb_api_key.as_deref(), Some("db-key"));
        assert_eq!(args.allowed_origins.as_deref(), Some("https://example.com"));
        assert_eq!(
            args.tls_cert.as_ref().unwrap().to_str().unwrap(),
            "/path/to/cert.pem"
        );
        assert_eq!(
            args.tls_key.as_ref().unwrap().to_str().unwrap(),
            "/path/to/key.pem"
        );
    }

    #[test]
    fn test_merge_config_cli_overrides_file() {
        let matches =
            Args::command().get_matches_from(["spec-ai", "--port", "7777", "--top-k", "5"]);
        let mut args = Args::from_arg_matches(&matches).unwrap();

        let file_config = FileConfig {
            port: Some(9999),
            top_k: Some(20),
            ..FileConfig::default()
        };

        merge_config(&mut args, &matches, file_config);

        // CLI values should win
        assert_eq!(args.port, 7777);
        assert_eq!(args.top_k, 5);
    }

    #[test]
    fn test_merge_config_empty_file_keeps_defaults() {
        let matches = Args::command().get_matches_from(["spec-ai"]);
        let mut args = Args::from_arg_matches(&matches).unwrap();
        let default_port = args.port;

        merge_config(&mut args, &matches, FileConfig::default());

        assert_eq!(args.port, default_port);
        assert!(args.api_key.is_none());
        assert!(args.tls_cert.is_none());
    }
}
