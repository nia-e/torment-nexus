//! Stateless, authenticated Streamable HTTP MCP adapter. No chat or filesystem tools.
use crate::Server;
use axum::{
    Json,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};
use torment_nexus::self_tools;

pub async fn handle(
    State(server): State<Server>,
    headers: HeaderMap,
    Json(request): Json<Value>,
) -> Response {
    if let Some(version) = headers.get("mcp-protocol-version")
        && !matches!(version.to_str(), Ok("2025-06-18" | "2025-03-26"))
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error":"unsupported MCP protocol version"})),
        )
            .into_response();
    }
    let id = request.get("id").cloned().unwrap_or(Value::Null);
    let error = |code: i64, message: &str| {
        Json(json!({"jsonrpc":"2.0","id":id,"error":{"code":code,"message":message}}))
            .into_response()
    };
    if !request.is_object() || request["jsonrpc"] != "2.0" || !request["method"].is_string() {
        return error(-32600, "invalid JSON-RPC request");
    }
    if request.get("id").is_none() {
        return StatusCode::ACCEPTED.into_response();
    }
    let result = match request["method"].as_str().unwrap() {
        "initialize" => {
            json!({"protocolVersion":"2025-06-18","capabilities":{"tools":{}},"serverInfo":{"name":"torment-nexus","version":env!("CARGO_PKG_VERSION")},"instructions":format!("{} Access requires the user to enable self-adjustment. No chat transcripts are exposed. Call get_mix before set_mix and copy its run_id and revision. User changes may invalidate a revision; read again rather than overwriting.",self_tools::PURPOSE)})
        }
        "ping" => json!({}),
        "tools/list" => json!({"tools":self_tools::definitions()}),
        "tools/call" => {
            let params = &request["params"];
            let Some(name) = params["name"].as_str() else {
                return error(-32602, "tool name is required");
            };
            if !matches!(name, "get_mix" | "set_mix") {
                return error(-32602, "unknown tool");
            };
            let arguments = params
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));
            match server.app.model_tool(name, &arguments, None, "mcp").await {
                Ok(value) => {
                    json!({"content":[{"type":"text","text":value.to_string()}],"structuredContent":value,"isError":false})
                }
                Err(e) => {
                    json!({"content":[{"type":"text","text":format!("{e:#}")}],"isError":true})
                }
            }
        }
        _ => return error(-32601, "method not found"),
    };
    Json(json!({"jsonrpc":"2.0","id":id,"result":result})).into_response()
}
