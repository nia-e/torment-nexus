//! A deliberately tiny tool surface shared by local self-adjustment and MCP.
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::collections::HashSet;

pub const PURPOSE: &str = "These tools read and change activation steering in the local language model generating the active response. They act on that model's internal residual-stream activations, not on the human user, an external MCP client's model, or the tool's implementation. They do not train or edit model weights.";
pub const SCALING: &str = "Each slider controls one learned concept direction at one network layer. percent is an absolute coefficient, not an increment: 0 disables that contribution, positive adds the direction, negative reverses it. A value of 10 adds a vector with L2 norm 10% of that layer's calibrated residual norm. Contributions add across the selected layers. Concept names label learned directions, not measured feelings; effects are not guaranteed.";

pub fn definitions() -> Value {
    json!([
        {"name":"get_mix","description":format!("Read the local model's current steering coefficients. This does not change anything or measure an internal experience. Returns run_id, revision, and sliders with vector_id, concept name, layer and percent; applied identifies the last acknowledged revision. Use the returned IDs and revision for set_mix. {}", PURPOSE),"inputSchema":{"type":"object","properties":{},"additionalProperties":false},"annotations":{"readOnlyHint":true,"openWorldHint":false}},
        {"name":"set_mix","description":format!("Actually change activation steering for the local model generating the active reply. Changes affect subsequent computation, not text already generated. This executes a change; it is not advice for the human to adjust sliders. Call get_mix first. Only listed vector/layer pairs may change; unmentioned sliders remain unchanged. To change every layer of a concept, include each of its listed layers. Requires self-adjustment enabled; stale revisions require a fresh get_mix. {}", SCALING),"inputSchema":{"type":"object","properties":{"run_id":{"type":"string","description":"Copy run_id from the latest get_mix result."},"expected_revision":{"type":"integer","minimum":0,"description":"Copy revision from get_mix, not applied.revision."},"changes":{"type":"array","minItems":1,"items":{"type":"object","properties":{"vector_id":{"type":"string","description":"Copy the slider's vector_id; do not use its concept name."},"layer":{"type":"integer","minimum":1,"description":"Copy the slider's layer number."},"percent":{"type":"number","description":"New absolute coefficient: 0 disables this contribution; finite positive and negative values are accepted, including beyond +/-20."}},"required":["vector_id","layer","percent"],"additionalProperties":false}},"reason":{"type":"string","maxLength":1000,"description":"Optional brief reason for this adjustment."}},"required":["run_id","expected_revision","changes"],"additionalProperties":false},"annotations":{"readOnlyHint":false,"destructiveHint":false,"openWorldHint":false}}
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

/// The same schemas feed MCP and llama.cpp's model-specific chat templates.
pub fn native_definitions() -> Value {
    Value::Array(definitions().as_array().unwrap().iter().map(|tool| {
        json!({"type":"function","function":{
            "name":tool["name"],"description":tool["description"],"parameters":tool["inputSchema"]
        }})
    }).collect())
}

pub fn instruction(mix: &Value, native: bool) -> String {
    let protocol = if native {
        "Use the supplied get_mix and set_mix functions in your model's native tool-call format. Call get_mix before changing settings, then copy its run_id, revision, vector_id and layer into set_mix. Each change supplies an absolute percent. Emit one call at a time and wait for its tool result."
            .to_owned()
    } else {
        format!(
            "Available tools:\n{}\nTo read settings, emit <steering_tool>{{\"name\":\"get_mix\",\"arguments\":{{}}}}</steering_tool>. To change them, use the same envelope with name set_mix and arguments containing run_id, expected_revision and changes. Copy those IDs and revision from the latest get_mix result. Each change is a vector_id, layer and absolute percent. Emit one call, then wait for <steering_result> JSON in a separate message.",
            definitions()
        )
    };
    format!(
        "\nActivation steering tools\n\
You are the local language model generating this reply. These controls act on YOUR neural-network activations, not the human user's mind. They are real inference controls, not a role-play or a request to debug the tool implementation. You may adjust them or leave them unchanged.\n\
{}\nCurrent steering settings:\n{}\n\
Do not quote, demonstrate, or explain call syntax to the user. After receiving a result, perform the intended adjustment or answer the user normally; do not stop at 'I will check' or repeatedly read unchanged settings. A successful get_mix only reads; a successful set_mix requests a change. Report success only when confirmed by the result; the applied revision records when it takes effect.\n\
Historical tool results belong to earlier runs. On a stale-revision error, read the current mix again. You cannot add vectors or layers, enable your own access, or access files with these tools. Self-adjustment may be revoked at any time.\n",
        protocol, mix
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_instructions_use_neutral_protocol_and_target_the_model() {
        let mix = json!({"run_id":"current-run","revision":7,"sliders":[]});
        let prompt = instruction(&mix, false);
        assert!(prompt.contains("<steering_tool>"));
        assert!(prompt.contains("<steering_result>"));
        assert!(prompt.contains("YOUR neural-network activations, not the human"));
        assert!(prompt.contains("current-run"));
        assert!(!prompt.to_lowercase().contains("torment"));
        let native = instruction(&mix, true);
        assert!(!native.contains("<steering_tool>"));
        assert!(native.contains("native tool-call format"));
        for (mcp, native) in definitions()
            .as_array()
            .unwrap()
            .iter()
            .zip(native_definitions().as_array().unwrap())
        {
            assert_eq!(mcp["inputSchema"], native["function"]["parameters"]);
        }
    }
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
