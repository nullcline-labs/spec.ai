use clap::Parser;
use specai_core::cache::eviction::spawn_eviction_task;
use specai_core::cache::SpeculativeCache;
use specai_core::config;
use specai_core::embedder::http::{HttpEmbedder, HttpEmbedderConfig};
use specai_core::engine::{Engine, EngineConfig};
use specai_core::retriever::vectorsdb::{VectorsDbRetriever, VectorsDbRetrieverConfig};
use specai_core::similarity::SimilarityGateConfig;
use specai_server::api::create_router;
use specai_server::api::handlers::AppState;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "spec-ai", about = "Speculative retrieval engine for RAG")]
struct Args {
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

    #[arg(long, default_value = config::DEFAULT_COLLECTION)]
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
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("specai_server=info".parse()?)
                .add_directive("specai_core=info".parse()?),
        )
        .init();

    let args = Args::parse();

    let embedding_api_key = args
        .embedding_api_key
        .or_else(|| std::env::var("SPECAI_EMBEDDING_API_KEY").ok());
    let vectorsdb_api_key = args
        .vectorsdb_api_key
        .or_else(|| std::env::var("SPECAI_VECTORSDB_API_KEY").ok());

    let embedder = Arc::new(HttpEmbedder::new(HttpEmbedderConfig {
        url: args.embedding_url.clone(),
        model: args.embedding_model.clone(),
        api_key: embedding_api_key,
        timeout: Duration::from_secs(config::EXTERNAL_REQUEST_TIMEOUT_SECS),
    }));

    let retriever = Arc::new(VectorsDbRetriever::new(VectorsDbRetrieverConfig {
        base_url: args.vectorsdb_url.clone(),
        collection: args.collection.clone(),
        api_key: vectorsdb_api_key,
        timeout: Duration::from_secs(config::EXTERNAL_REQUEST_TIMEOUT_SECS),
    }));

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

    let engine = Arc::new(Engine::new(
        embedder,
        retriever,
        cache.clone(),
        engine_config,
    ));

    spawn_eviction_task(cache, config::CACHE_EVICTION_INTERVAL_SECS);

    let metrics_engine = engine.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(15));
        loop {
            interval.tick().await;
            specai_server::api::metrics::update_engine_metrics(&metrics_engine.stats());
        }
    });

    let prometheus_handle =
        metrics_exporter_prometheus::PrometheusBuilder::new().install_recorder()?;

    let state = AppState {
        engine,
        prometheus_handle,
        start_time: Instant::now(),
    };

    let app = create_router(state);
    let addr = format!("0.0.0.0:{}", args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        port = args.port,
        embedding_url = %args.embedding_url,
        embedding_model = %args.embedding_model,
        vectorsdb_url = %args.vectorsdb_url,
        collection = %args.collection,
        similarity_threshold = args.similarity_threshold,
        top_k = args.top_k,
        debounce_ms = args.debounce_ms,
        cache_ttl = args.cache_ttl,
        "spec.ai ready"
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(wait_for_signal())
        .await?;

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
