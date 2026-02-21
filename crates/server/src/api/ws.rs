use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::IntoResponse;
use dashmap::DashMap;
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use specai_core::config;
use specai_core::engine::Engine;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::task::AbortHandle;
use tokio::time::sleep;

use super::handlers::AppState;

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    #[serde(rename = "keystroke")]
    Keystroke { session_id: String, text: String },
    #[serde(rename = "submit")]
    Submit { session_id: String, query: String },
    #[serde(rename = "close")]
    Close { session_id: String },
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    #[serde(rename = "speculating")]
    Speculating { session_id: String, query: String },
    #[serde(rename = "speculation_ready")]
    SpeculationReady {
        session_id: String,
        query: String,
        num_results: usize,
        latency_ms: u64,
    },
    #[serde(rename = "results")]
    Results {
        session_id: String,
        documents: Vec<specai_core::types::Document>,
        cache_verdict: String,
        latency_ms: u64,
    },
    #[serde(rename = "error")]
    Error { message: String },
}

pub async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state.engine.clone()))
}

async fn handle_socket(socket: WebSocket, engine: Arc<Engine>) {
    let (mut sender, mut receiver) = socket.split();
    let (tx, mut rx) = mpsc::channel::<ServerMessage>(64);

    // Forward server messages to WebSocket
    let send_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            if let Ok(text) = serde_json::to_string(&msg) {
                if sender.send(Message::Text(text)).await.is_err() {
                    break;
                }
            }
        }
    });

    // Per-session debounce tracking
    let debounce_handles: Arc<DashMap<String, AbortHandle>> = Arc::new(DashMap::new());

    while let Some(Ok(msg)) = receiver.next().await {
        let text = match msg {
            Message::Text(t) => t.to_string(),
            Message::Close(_) => break,
            _ => continue,
        };

        let client_msg: ClientMessage = match serde_json::from_str(&text) {
            Ok(m) => m,
            Err(e) => {
                let _ = tx
                    .send(ServerMessage::Error {
                        message: format!("Invalid message: {}", e),
                    })
                    .await;
                continue;
            }
        };

        match client_msg {
            ClientMessage::Keystroke { session_id, text } => {
                // Cancel previous pending speculation for this session
                if let Some(prev) = debounce_handles.get(&session_id) {
                    prev.abort();
                }

                let engine = engine.clone();
                let tx = tx.clone();
                let sid = session_id.clone();
                let query = text;
                let handles = debounce_handles.clone();

                let handle = tokio::spawn(async move {
                    sleep(Duration::from_millis(config::DEFAULT_DEBOUNCE_MS)).await;

                    let _ = tx
                        .send(ServerMessage::Speculating {
                            session_id: sid.clone(),
                            query: query.clone(),
                        })
                        .await;

                    let start = std::time::Instant::now();
                    match engine.speculate(&sid, &query).await {
                        Ok(()) => {
                            let latency_ms = start.elapsed().as_millis() as u64;
                            let num_results = engine
                                .cache()
                                .get_latest(&sid)
                                .map(|r| r.documents.len())
                                .unwrap_or(0);
                            let _ = tx
                                .send(ServerMessage::SpeculationReady {
                                    session_id: sid.clone(),
                                    query,
                                    num_results,
                                    latency_ms,
                                })
                                .await;
                        }
                        Err(e) => {
                            tracing::warn!(session = %sid, "Speculation failed: {}", e);
                        }
                    }
                    handles.remove(&sid);
                });

                debounce_handles.insert(session_id, handle.abort_handle());
            }

            ClientMessage::Submit { session_id, query } => {
                // Cancel any pending speculation
                if let Some(prev) = debounce_handles.get(&session_id) {
                    prev.abort();
                }
                debounce_handles.remove(&session_id);

                let engine = engine.clone();
                let tx = tx.clone();

                tokio::spawn(async move {
                    match engine.submit(&session_id, &query).await {
                        Ok(result) => {
                            let _ = tx
                                .send(ServerMessage::Results {
                                    session_id,
                                    documents: result.documents,
                                    cache_verdict: result.verdict.to_string(),
                                    latency_ms: result.latency_ms,
                                })
                                .await;
                        }
                        Err(e) => {
                            let _ = tx
                                .send(ServerMessage::Error {
                                    message: format!("Submission failed: {}", e),
                                })
                                .await;
                        }
                    }
                });
            }

            ClientMessage::Close { session_id } => {
                if let Some(prev) = debounce_handles.get(&session_id) {
                    prev.abort();
                }
                debounce_handles.remove(&session_id);
                engine.close_session(&session_id);
                break;
            }
        }
    }

    send_task.abort();
    tracing::debug!("WebSocket connection closed");
}
