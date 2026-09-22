mod app;
mod mcp;

use app::App;
use axum::{
    Json, Router,
    body::Body,
    extract::{DefaultBodyLimit, Request, State},
    http::{HeaderValue, StatusCode, header},
    middleware::{self, Next},
    response::{
        IntoResponse, Response, Sse,
        sse::{Event, KeepAlive},
    },
    routing::{get, post},
};
use clap::Parser;
use futures_util::StreamExt;
use rust_embed::RustEmbed;
use serde_json::{Value, json};
use std::{
    convert::Infallible,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::PathBuf,
    sync::Arc,
};
use tokio_stream::wrappers::BroadcastStream;
use torment_nexus::store::Store;
use uuid::Uuid;

#[derive(Parser)]
#[command(
    about = "A local activation-steering playground. Labels are not measures of subjective experience."
)]
struct Args {
    /// Application data directory; defaults to ~/Library/Application Support/Torment Nexus.
    #[arg(long)]
    data_dir: Option<PathBuf>,
    /// Loopback port. Zero asks macOS to choose a free port.
    #[arg(long, default_value_t = 0)]
    port: u16,
    /// Path to the pinned C++ inference worker.
    #[arg(long)]
    worker: Option<PathBuf>,
    /// Path to the installed Codex CLI (existing Codex login is used).
    #[arg(long)]
    codex: Option<PathBuf>,
    #[arg(long)]
    no_open: bool,
}

#[derive(RustEmbed)]
#[folder = "frontend/dist/"]
struct Frontend;

#[derive(Clone)]
struct Server {
    app: Arc<App>,
    token: Arc<String>,
    mcp_token: Arc<String>,
    authority: Arc<String>,
    origin: Arc<String>,
    shutdown: tokio::sync::watch::Receiver<bool>,
}

fn api_error(error: impl std::fmt::Display) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({"error":error.to_string()})),
    )
        .into_response()
}

async fn state(State(server): State<Server>) -> Response {
    match server.app.snapshot().await {
        Ok(mut value) => {
            value["mcp"] =
                json!({"url":format!("{}/mcp",server.origin),"token":server.mcp_token.as_str()});
            Json(value).into_response()
        }
        Err(error) => api_error(format!("{error:#}")),
    }
}

async fn action(State(server): State<Server>, Json(request): Json<Value>) -> Response {
    match server.app.action(request).await {
        Ok(value) => Json(value).into_response(),
        Err(error) => api_error(format!("{error:#}")),
    }
}

async fn events(State(server): State<Server>) -> impl IntoResponse {
    let mut shutdown = server.shutdown.clone();
    let stream = BroadcastStream::new(server.app.events.subscribe())
        .map(|event| {
            let value = event.unwrap_or_else(
            |_| json!({"kind":"resync","data":{"reason":"event consumer lagged; fetch snapshot"}}),
        );
            Ok::<_, Infallible>(Event::default().data(value.to_string()))
        })
        .take_until(async move {
            let _ = shutdown.wait_for(|stopping| *stopping).await;
        });
    Sse::new(stream).keep_alive(KeepAlive::default())
}

fn constant_time_equal(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .fold(0u8, |diff, (a, b)| diff | (a ^ b))
        == 0
}

async fn security(State(server): State<Server>, request: Request, next: Next) -> Response {
    let host = request
        .headers()
        .get(header::HOST)
        .and_then(|h| h.to_str().ok());
    let origin = request
        .headers()
        .get(header::ORIGIN)
        .and_then(|h| h.to_str().ok());
    if host != Some(server.authority.as_str())
        || origin.is_some_and(|origin| origin != server.origin.as_str())
    {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"error":"request origin/host does not match this loopback launch"})),
        )
            .into_response();
    }
    if request.uri().path().starts_with("/api/") || request.uri().path() == "/mcp" {
        let token = if request.uri().path() == "/mcp" {
            &server.mcp_token
        } else {
            &server.token
        };
        let supplied = request
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|h| h.to_str().ok())
            .and_then(|h| h.strip_prefix("Bearer "))
            .unwrap_or("");
        if !constant_time_equal(supplied.as_bytes(), token.as_bytes()) {
            return (
                StatusCode::UNAUTHORIZED,
                Json(
                    json!({"error":"launch token missing or expired; open the current launch URL"}),
                ),
            )
                .into_response();
        }
    }
    let mut response = next.run(request).await;
    for (name, value) in [
        (
            "content-security-policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
        ),
        ("x-content-type-options", "nosniff"),
        ("referrer-policy", "no-referrer"),
        ("cache-control", "no-store"),
        ("cross-origin-resource-policy", "same-origin"),
    ] {
        response
            .headers_mut()
            .insert(name, HeaderValue::from_static(value));
    }
    response
}

async fn static_asset(request: Request) -> Response {
    let requested = request.uri().path().trim_start_matches('/');
    let path = if requested.is_empty() {
        "index.html"
    } else {
        requested
    };
    if let Some(asset) = Frontend::get(path) {
        let mime = mime_guess::from_path(path)
            .first_or_octet_stream()
            .to_string();
        return (
            [(header::CONTENT_TYPE, mime)],
            Body::from(asset.data.into_owned()),
        )
            .into_response();
    }
    StatusCode::NOT_FOUND.into_response()
}

fn router(server: Server) -> Router {
    Router::new()
        .route("/api/state", get(state))
        .route("/api/action", post(action))
        .route("/api/events", get(events))
        .route(
            "/mcp",
            post(mcp::handle)
                .get(|| async { StatusCode::METHOD_NOT_ALLOWED })
                .layer(DefaultBodyLimit::max(64 * 1024)),
        )
        .fallback(static_asset)
        .layer(DefaultBodyLimit::max(384 * 1024 * 1024))
        .layer(middleware::from_fn_with_state(server.clone(), security))
        .with_state(server)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let args = Args::parse();
    let data_dir = args.data_dir.unwrap_or_else(|| {
        PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
            .join("Library/Application Support/Torment Nexus")
    });
    let executable = std::env::current_exe()?;
    let packaged = executable.parent().unwrap().join("torment-engine");
    let worker = args.worker.unwrap_or_else(|| {
        if packaged.is_file() {
            packaged
        } else {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("engine/build/bin/torment-engine")
        }
    });
    let store = Store::open(&data_dir)?;
    let app = App::new(store, worker, args.codex);
    let listener =
        tokio::net::TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), args.port))
            .await?;
    let address = listener.local_addr()?;
    let token = format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple());
    let url = format!("http://{address}/#token={token}");
    let (shutdown_sender, shutdown_receiver) = tokio::sync::watch::channel(false);
    let server = Server {
        app: app.clone(),
        token: Arc::new(token),
        mcp_token: Arc::new(format!(
            "{}{}",
            Uuid::new_v4().simple(),
            Uuid::new_v4().simple()
        )),
        authority: Arc::new(address.to_string()),
        origin: Arc::new(format!("http://{address}")),
        shutdown: shutdown_receiver,
    };
    println!(
        "Torment Nexus — local activation steering\nData: {}\nOpen: {url}",
        data_dir.display()
    );
    if !args.no_open
        && let Err(error) = std::process::Command::new("open").arg(&url).spawn()
    {
        eprintln!("Browser did not open ({error}); use the launch URL above.");
    }
    let shutdown_app = app.clone();
    let shutdown = async move {
        let _ = tokio::signal::ctrl_c().await;
        let _ = shutdown_sender.send(true);
        shutdown_app.shutdown().await;
    };
    let result = axum::serve(listener, router(server))
        .with_graceful_shutdown(shutdown)
        .await;
    app.shutdown().await;
    result?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower::ServiceExt;

    #[tokio::test]
    async fn token_origin_and_dns_rebinding_checks() {
        let directory = tempfile::tempdir().unwrap();
        let app = App::new(
            Store::open(directory.path()).unwrap(),
            PathBuf::from("missing-worker"),
            None,
        );
        let server = Server {
            app,
            token: Arc::new("secret".into()),
            mcp_token: Arc::new("mcp-secret".into()),
            authority: Arc::new("127.0.0.1:7777".into()),
            origin: Arc::new("http://127.0.0.1:7777".into()),
            shutdown: tokio::sync::watch::channel(false).1,
        };
        for (host, origin, token, expected) in [
            ("127.0.0.1:7777", None, None, StatusCode::UNAUTHORIZED),
            (
                "evil.test:7777",
                None,
                Some("secret"),
                StatusCode::FORBIDDEN,
            ),
            (
                "127.0.0.1:7777",
                Some("https://evil.test"),
                Some("secret"),
                StatusCode::FORBIDDEN,
            ),
            (
                "127.0.0.1:7777",
                Some("null"),
                Some("secret"),
                StatusCode::FORBIDDEN,
            ),
            (
                "127.0.0.1:7777",
                Some("http://127.0.0.1:7777"),
                Some("wrong"),
                StatusCode::UNAUTHORIZED,
            ),
            (
                "127.0.0.1:7777",
                Some("http://127.0.0.1:7777"),
                Some("secret"),
                StatusCode::OK,
            ),
        ] {
            let mut request = Request::builder().uri("/api/state").header("host", host);
            if let Some(origin) = origin {
                request = request.header("origin", origin);
            }
            if let Some(token) = token {
                request = request.header("authorization", format!("Bearer {token}"));
            }
            let response = router(server.clone())
                .oneshot(request.body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), expected);
        }
        // MCP credentials are scoped to tools, not the application's full API.
        for (path, token, expected) in [
            ("/api/state", "mcp-secret", StatusCode::UNAUTHORIZED),
            ("/mcp", "secret", StatusCode::UNAUTHORIZED),
            ("/mcp", "mcp-secret", StatusCode::METHOD_NOT_ALLOWED),
        ] {
            let response = router(server.clone())
                .oneshot(
                    Request::builder()
                        .uri(path)
                        .header("host", "127.0.0.1:7777")
                        .header("authorization", format!("Bearer {token}"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), expected);
        }
    }
}
