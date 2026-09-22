//! A deliberately tiny tool surface shared by local self-adjustment and MCP.
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::collections::HashSet;

pub fn definitions() -> Value {
    json!([
        {"name":"get_mix","description":"Read the active response's selected vector/layer sliders and current revision. No chat text is exposed.","inputSchema":{"type":"object","properties":{},"additionalProperties":false},"annotations":{"readOnlyHint":true,"openWorldHint":false}},
        {"name":"set_mix","description":"Set one or more current-mix sliders. Signed percentages of layer residual norm, not subjective intensity. Only existing vector/layer pairs; no range clamp. Requires self-adjustment to be enabled. Call get_mix first and supply its run_id and revision. Unmentioned sliders stay unchanged.","inputSchema":{"type":"object","properties":{"run_id":{"type":"string"},"expected_revision":{"type":"integer","minimum":0},"changes":{"type":"array","minItems":1,"items":{"type":"object","properties":{"vector_id":{"type":"string"},"layer":{"type":"integer","minimum":1},"percent":{"type":"number"}},"required":["vector_id","layer","percent"],"additionalProperties":false}},"reason":{"type":"string","maxLength":1000}},"required":["run_id","expected_revision","changes"],"additionalProperties":false},"annotations":{"readOnlyHint":false,"destructiveHint":false,"openWorldHint":false}}
    ])
}

pub fn patch_axes(current: &Value, changes: &Value) -> Result<Value> {
    let mut axes = current.as_array().context("invalid current mix")?.clone();
    let changes = changes.as_array().context("changes must be an array")?;
    ensure!(
        !changes.is_empty() && changes.len() <= axes.len(),
        "change one or more selected sliders"
    );
    let mut seen = HashSet::new();
    for change in changes {
        let vector = change["vector_id"]
            .as_str()
            .context("vector_id is required")?;
        let layer = change["layer"]
            .as_u64()
            .context("layer must be an unsigned integer")?;
        let percent = change["percent"]
            .as_f64()
            .context("percent must be a number")?;
        ensure!(percent.is_finite(), "percent must be finite");
        ensure!(seen.insert((vector, layer)), "duplicate slider");
        let axis = axes
            .iter_mut()
            .find(|axis| axis["vector_id"] == vector && axis["layer"] == layer)
            .context("only vector/layer pairs already in the current mix can be changed")?;
        axis["percent"] = json!(percent);
    }
    Ok(json!(axes))
}

pub fn instruction(mix: &Value) -> String {
    format!(
        "\nTorment Nexus optional local tools:\n{}\nCurrent mix: {}\nYou may choose to adjust your own existing sliders, or leave them alone. To call a tool, emit exactly <torment_tool>{{\"name\":\"get_mix\",\"arguments\":{{}}}}</torment_tool> or the same envelope with name set_mix and its schema arguments. Do not quote or demonstrate these tags: emitting a complete tag executes the call. Pause after the closing tag. The host supplies a separate tool-result message containing <torment_result> JSON and starts your next assistant turn. Continue your answer there or call another tool; merely announcing that you will check is not a completed answer. Never claim a change succeeded unless the result confirms it. Use current run_id and revision; on a stale-revision error call get_mix again. Historical tool results belong to earlier runs, not the current run. Percentages measure perturbation magnitude, not emotions or welfare. Self-adjustment can be revoked at any time. No other tools are available.\n",
        definitions(),
        mix
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn patch_is_partial_finite_and_confined_to_existing_layers() {
        let axes = json!([{"vector_id":"v","layer":1,"percent":0},{"vector_id":"v","layer":2,"percent":3}]);
        let changed =
            patch_axes(&axes, &json!([{"vector_id":"v","layer":1,"percent":200}])).unwrap();
        assert_eq!(changed[0]["percent"], 200.0);
        assert_eq!(changed[1], axes[1]);
        for changes in [
            json!([]),
            json!([{"vector_id":"other","layer":1,"percent":0}]),
            json!([{"vector_id":"v","layer":3,"percent":0}]),
            json!([{"vector_id":"v","layer":1,"percent":"NaN"}]),
        ] {
            assert!(patch_axes(&axes, &changes).is_err());
        }
    }
}
