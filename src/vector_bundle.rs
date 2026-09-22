//! Structural and referential validation of portable vector provenance.
//!
//! These checks establish internal consistency and complete checked references,
//! not authenticity, semantic validity, or a fresh execution of the model.

use crate::{
    artifacts::{ArtifactStore, MAX_ARTIFACT_BYTES, sha256, validate_hash},
    extraction::{Extraction, Method, preview_percentages},
    steering,
};
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::collections::{BTreeSet, HashSet};

const RUNTIME_FIELDS: &[&str] = &[
    "engine_revision",
    "wrapper_sha256",
    "context",
    "batch",
    "microbatch",
    "n_layer",
    "n_embd",
    "cache_k",
    "cache_v",
];

/// Metadata may cross a browser JSON parse/stringify boundary, which turns
/// `1.0` into `1`. Accept that harmless change without equating rounded large
/// integers. Hashes still cover the exact original artifact bytes.
pub fn json_metadata_eq(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => {
            if left == right {
                return true;
            }
            let (integer, float) = match (left.is_f64(), right.is_f64()) {
                (false, true) => (left, right),
                (true, false) => (right, left),
                _ => return false,
            };
            const SAFE: i64 = 1 << 53;
            let safe_integer = integer
                .as_i64()
                .is_some_and(|n| (-SAFE..=SAFE).contains(&n))
                || integer.as_u64().is_some_and(|n| n <= SAFE as u64);
            safe_integer
                && float.as_f64().is_some_and(|f| {
                    f.is_finite()
                        && f.fract() == 0.0
                        && f.abs() <= SAFE as f64
                        && Some(f) == integer.as_f64()
                })
        }
        (Value::Array(left), Value::Array(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| json_metadata_eq(left, right))
        }
        (Value::Object(left), Value::Object(right)) => {
            left.len() == right.len()
                && left.iter().all(|(key, left)| {
                    right
                        .get(key)
                        .is_some_and(|right| json_metadata_eq(left, right))
                })
        }
        _ => left == right,
    }
}

/// `supplied` is the set of hashes physically present in the incoming bundle,
/// not the hashes already available in the application's artifact cache.
pub fn validate_manifest_artifacts(
    artifacts: &ArtifactStore,
    vector: &Value,
    manifest: &Value,
    supplied: &HashSet<String>,
) -> Result<()> {
    let manifest_hash = hash_field(vector, "manifest_hash")?;
    ensure!(
        supplied.contains(manifest_hash),
        "bundle omitted its manifest blob"
    );
    ensure!(
        artifacts.read_json(manifest_hash)? == *manifest,
        "manifest differs from its checked blob"
    );

    let listed = hash_set(
        manifest
            .get("artifacts")
            .context("manifest omitted artifacts")?,
    )?;
    ensure!(
        !listed.contains(manifest_hash),
        "manifest cannot contain itself in its artifact closure"
    );
    ensure!(
        listed.iter().all(|hash| supplied.contains(hash)),
        "bundle omitted an explicitly listed artifact"
    );
    ensure!(
        supplied.len() == listed.len() + 1,
        "bundle contains blobs not declared by its manifest"
    );
    // Extra explicitly listed historical/analysis artifacts are retained, but
    // cannot be used to hide a missing reference in a typed manifest field.
    for hash in &listed {
        artifacts.read_bytes(hash)?;
    }

    let mut required = HashSet::new();
    let dataset_hash = hash_field(manifest, "dataset_hash")?;
    required.insert(dataset_hash.to_owned());
    let recipe = manifest
        .get("recipe")
        .filter(|v| v.is_object())
        .context("manifest omitted recipe object")?;
    let extraction = Extraction::from_record(recipe)?;
    ensure!(
        extraction.algorithm() == manifest["algorithm"],
        "recipe method disagrees with algorithm"
    );
    let preview_coefficients = preview_percentages(recipe)?;
    ensure!(
        hash_field(recipe, "dataset_hash")? == dataset_hash,
        "recipe and manifest dataset hashes disagree"
    );
    let dataset = recipe
        .get("dataset")
        .filter(|v| v.is_array())
        .context("recipe dataset must be a pair array")?;
    ensure!(
        dataset.as_array().unwrap().len() <= 4096,
        "recipe dataset exceeds 4096 pairs"
    );
    let prepared = steering::prepare_dataset(dataset).context("invalid recipe dataset")?;
    ensure!(
        prepared["pairs"] == *dataset,
        "recipe dataset is not the canonical deduplicated family-split dataset"
    );
    let mut expected_jsonl = Vec::new();
    for pair in dataset.as_array().unwrap() {
        serde_json::to_writer(&mut expected_jsonl, pair)?;
        expected_jsonl.push(b'\n');
        ensure!(
            expected_jsonl.len() <= MAX_ARTIFACT_BYTES,
            "dataset JSONL exceeds artifact limit"
        );
    }
    ensure!(
        sha256(&expected_jsonl) == dataset_hash,
        "dataset contents do not match their JSONL hash"
    );
    ensure!(
        artifacts.read_bytes(dataset_hash)? == expected_jsonl,
        "dataset artifact is not the canonical recipe JSONL"
    );

    let stages = recipe
        .get("stages")
        .and_then(Value::as_object)
        .context("recipe omitted stage references")?;
    for (stage, hash) in stages {
        ensure!(!stage.is_empty(), "recipe contains an empty stage name");
        let hash = hash_value(hash)?;
        required.insert(hash.to_owned());
        artifacts
            .read_json(hash)
            .with_context(|| format!("invalid recipe stage artifact {stage}"))?;
    }

    let identity = manifest
        .get("extraction_identity")
        .and_then(Value::as_object)
        .context("manifest omitted extraction identity")?;
    if extraction.method == Method::Paper {
        ensure!(
            identity.get("extraction") == Some(&serde_json::to_value(&extraction)?),
            "paper extraction settings differ from identity"
        );
        ensure!(
            manifest["split"] == crate::paper::fold_assignment(dataset.as_array().unwrap())?,
            "paper fold assignment mismatch"
        );
    }
    ensure!(
        identity.get("dataset_hash").and_then(Value::as_str) == Some(dataset_hash),
        "extraction identity has a different dataset"
    );
    let fingerprint = text(manifest, "model_fingerprint")?;
    validate_hash(fingerprint)?;
    ensure!(
        vector["model_fingerprint"] == fingerprint
            && identity.get("model_fingerprint").and_then(Value::as_str) == Some(fingerprint),
        "extraction identity has a different model fingerprint"
    );
    ensure!(
        identity.get("algorithm") == manifest.get("algorithm"),
        "extraction identity algorithm disagrees with manifest"
    );
    let raw = recipe
        .get("raw")
        .and_then(Value::as_bool)
        .context("recipe omitted raw/chat mode")?;
    ensure!(
        identity.get("raw").and_then(Value::as_bool) == Some(raw),
        "extraction identity raw/chat mode differs from recipe"
    );
    let runtime = identity
        .get("runtime")
        .context("extraction identity omitted runtime")?;
    validate_runtime(runtime)?;
    match_runtime(&manifest["runtime"], runtime)?;
    ensure!(
        identity.get("engine_revision") == runtime.get("engine_revision"),
        "extraction identity engine revision disagrees with runtime"
    );
    let width = integer(runtime, "n_embd")?;
    let depth = integer(runtime, "n_layer")?;
    ensure!(
        integer(&manifest["model_shape"], "n_embd")? == width
            && integer(&manifest["model_shape"], "n_layer")? == depth,
        "manifest model shape disagrees with runtime"
    );
    ensure!(
        json_metadata_eq(&manifest["layers"], &vector["layers"]),
        "vector and manifest layers disagree"
    );
    let layers = manifest["layers"]
        .as_array()
        .context("manifest omitted layers")?;
    ensure!(
        !layers.is_empty() && layers.len() < depth,
        "invalid extracted layer count"
    );
    let mut expected_layers = BTreeSet::new();
    for layer in layers {
        let layer_id = integer(layer, "layer")?;
        steering::legacy_buffer_index(layer_id, depth)?;
        ensure!(
            expected_layers.insert(layer_id),
            "duplicate extracted layer"
        );
        ensure!(
            integer(layer, "width")? == width,
            "layer width differs from model width"
        );
        let direction_hash = hash_field(layer, "direction_hash")?;
        required.insert(direction_hash.to_owned());
        ensure!(
            artifacts.read_f32(direction_hash)?.shape == vec![width],
            "direction tensor shape differs from model width"
        );
        let usable = layer
            .get("usable")
            .and_then(Value::as_bool)
            .context("layer omitted usability flag")?;
        match layer.get("unit_hash").filter(|value| !value.is_null()) {
            Some(hash) => {
                let hash = hash_value(hash)?;
                required.insert(hash.to_owned());
                ensure!(
                    artifacts.read_f32(hash)?.shape == vec![width],
                    "unit tensor shape differs from model width"
                );
            }
            None => ensure!(!usable, "usable layer omitted unit tensor"),
        }
    }
    let identity_layers = identity
        .get("layers")
        .and_then(Value::as_array)
        .context("extraction identity omitted layers")?;
    let identity_layers: Vec<usize> = identity_layers
        .iter()
        .map(|value| {
            usize::try_from(
                value
                    .as_u64()
                    .context("invalid extraction identity layer")?,
            )
            .context("layer overflow")
        })
        .collect::<Result<_>>()?;
    ensure!(
        identity_layers.len() == expected_layers.len()
            && identity_layers.iter().copied().collect::<BTreeSet<_>>() == expected_layers,
        "extraction identity layer set differs from manifest"
    );

    let extractions = manifest
        .get("extractions")
        .and_then(Value::as_object)
        .context("manifest omitted pair/pole extraction mapping")?;
    let pairs = dataset.as_array().unwrap();
    ensure!(
        extractions.len() == pairs.len() * 2,
        "extraction mapping must cover exactly both poles of every pair"
    );
    let mut template_settings: Option<(Value, Value)> = None;
    for pair in pairs {
        let pair_id = text(pair, "id")?;
        for pole in ["positive", "negative"] {
            let key = format!("{pair_id}:{pole}");
            let hash = hash_value(
                extractions
                    .get(&key)
                    .with_context(|| format!("missing extraction mapping for {key}"))?,
            )?;
            required.insert(hash.to_owned());
            let capture = artifacts
                .read_json(hash)
                .with_context(|| format!("invalid extraction artifact {key}"))?;
            let command = extraction.capture(pair, pole, raw)?;
            validate_capture(
                &capture,
                text(&command, "completion")?,
                &expected_layers,
                width,
                runtime,
                &command["messages"],
                command["raw"].as_bool().unwrap(),
            )
            .with_context(|| format!("invalid extraction for {key}"))?;
            let mut settings = capture["settings"].clone();
            if raw {
                // Raw formatting is selected from each example's conversation
                // shape, not a global template change.
                settings.as_object_mut().unwrap().remove("raw_format");
                settings
                    .as_object_mut()
                    .unwrap()
                    .remove("add_generation_prompt");
            }
            let current = (capture["template"].clone(), settings);
            if let Some(expected) = &template_settings {
                ensure!(
                    *expected == current,
                    "extraction template/settings changed between examples"
                );
            } else {
                template_settings = Some(current);
            }
        }
    }

    ensure!(
        json_metadata_eq(&vector["previews"], &manifest["previews"]),
        "vector and manifest previews disagree"
    );
    for preview in manifest["previews"]
        .as_array()
        .context("manifest omitted previews")?
    {
        let hash = hash_field(preview, "artifact_hash")?;
        required.insert(hash.to_owned());
        let artifact = artifacts.read_json(hash)?;
        ensure!(
            artifact["percent"]
                .as_f64()
                .is_some_and(|p| preview_coefficients.contains(&p)),
            "preview coefficient is not allowed by the recipe's preview mode"
        );
        for key in ["prompt", "percent", "output"] {
            ensure!(
                preview
                    .get(key)
                    .zip(artifact.get(key))
                    .is_some_and(|(left, right)| json_metadata_eq(left, right)),
                "preview {key} disagrees with its artifact"
            );
        }
        ensure!(
            artifact["cancelled"] != true,
            "completed preview references a cancelled artifact"
        );
    }
    ensure!(
        required.is_subset(&listed),
        "manifest artifact list omitted a typed dataset/stage/tensor/extraction/preview reference"
    );
    ensure!(
        required.is_subset(supplied),
        "bundle omitted a required provenance artifact"
    );
    Ok(())
}

fn validate_capture(
    capture: &Value,
    completion: &str,
    layers: &BTreeSet<usize>,
    width: usize,
    runtime: &Value,
    messages: &Value,
    raw: bool,
) -> Result<()> {
    ensure!(
        capture["v"] == 1 && capture["event"] == "result" && capture["cancelled"] != true,
        "capture is not a completed engine result"
    );
    match_runtime(&capture["runtime"], runtime)?;
    let prefix = capture
        .get("prefix")
        .and_then(Value::as_str)
        .context("capture omitted rendered prefix")?;
    let rendered = capture
        .get("rendered")
        .and_then(Value::as_str)
        .context("capture omitted rendered input")?;
    ensure!(
        rendered.strip_prefix(prefix) == Some(completion),
        "rendered capture does not end with the exact dataset completion"
    );
    let tokens = capture["token_ids"]
        .as_array()
        .context("capture omitted token IDs")?;
    ensure!(
        !tokens.is_empty() && tokens.len() < integer(runtime, "context")?,
        "capture token count is outside runtime context"
    );
    ensure!(
        tokens
            .iter()
            .all(|token| token.as_u64().is_some_and(|id| id <= i32::MAX as u64)),
        "capture contains invalid token IDs"
    );
    ensure!(
        integer(capture, "capture_position")? == tokens.len() - 1,
        "capture position is not the final content token"
    );
    let template = capture
        .get("template")
        .and_then(Value::as_str)
        .context("capture omitted template")?;
    ensure!(
        if raw {
            template.is_empty()
        } else {
            !template.is_empty()
        },
        "capture template does not match raw/chat mode"
    );
    let settings = &capture["settings"];
    ensure!(
        settings["mode"] == if raw { "raw" } else { "chat" },
        "capture settings raw/chat mode mismatch"
    );
    ensure!(
        settings["capture"] == "last-assistant-content-token" && settings["add_eos"] == false,
        "unsupported content-token capture settings"
    );
    ensure!(
        settings["parse_special"] == true && settings["add_bos"].is_boolean(),
        "capture tokenization/template settings are incomplete or unsupported"
    );
    let messages = messages
        .as_array()
        .context("capture input context must be an array")?;
    ensure!(
        !messages.is_empty() && messages.len() <= 1024,
        "invalid captured conversation size"
    );
    for message in messages {
        ensure!(
            matches!(
                message["role"].as_str(),
                Some("user" | "assistant" | "system")
            ) && message["content"].is_string(),
            "captured conversation contains unsupported roles/content"
        );
    }
    if raw {
        ensure!(
            settings.get("enable_thinking") == Some(&Value::Null)
                && settings.get("template_time_unix") == Some(&Value::Null),
            "raw capture must mark thinking/template time not applicable"
        );
        let verbatim = messages.len() == 1 && messages[0]["role"] == "user";
        let (format, expected_prefix) = if verbatim {
            (
                "single-user-verbatim-v1",
                messages[0]["content"].as_str().unwrap().to_owned(),
            )
        } else {
            let mut rendered = String::new();
            for message in messages {
                rendered.push_str(message["role"].as_str().unwrap());
                rendered.push_str(": ");
                rendered.push_str(message["content"].as_str().unwrap());
                rendered.push_str("\n\n");
            }
            rendered.push_str("assistant: ");
            ("role-labeled-dialogue-v1", rendered)
        };
        ensure!(
            settings["raw_format"] == format && settings["add_generation_prompt"] == !verbatim,
            "raw capture format/prefix setting disagrees with conversation"
        );
        ensure!(
            prefix == expected_prefix,
            "raw capture prefix differs from its recorded conversation"
        );
    } else {
        ensure!(
            settings["enable_thinking"] == false
                && settings["template_time_unix"] == 0
                && settings["add_generation_prompt"] == true
                && settings["raw_format"].is_null(),
            "unsupported chat template/reasoning settings"
        );
    }
    let values = capture["captures"]
        .as_array()
        .context("capture omitted residual tensors")?;
    ensure!(
        values.len() == layers.len(),
        "capture has the wrong layer count"
    );
    let mut seen = BTreeSet::new();
    for tensor in values {
        let layer = integer(tensor, "layer")?;
        ensure!(
            layers.contains(&layer) && seen.insert(layer),
            "capture contains an unexpected or duplicate layer"
        );
        let values = tensor["values"]
            .as_array()
            .context("capture values must be an array")?;
        ensure!(values.len() == width, "capture tensor width mismatch");
        let mut squared = 0f64;
        for value in values {
            let value = value.as_f64().context("capture value must be a number")?;
            ensure!(
                value.is_finite() && (value as f32).is_finite(),
                "capture value is not a finite F32"
            );
            squared += f64::from(value as f32).powi(2);
        }
        let measured = squared.sqrt();
        let norm = tensor["norm"]
            .as_f64()
            .context("capture omitted residual norm")?;
        ensure!(
            norm.is_finite()
                && norm >= 0.0
                && (norm - measured).abs() <= 1e-4 * measured.max(1e-30),
            "capture norm disagrees with residual values"
        );
    }
    Ok(())
}

fn validate_runtime(runtime: &Value) -> Result<()> {
    let revision = text(runtime, "engine_revision")?;
    ensure!(
        revision.len() == 40 && revision.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid engine revision"
    );
    validate_hash(text(runtime, "wrapper_sha256")?)?;
    let (context, batch, microbatch) = (
        integer(runtime, "context")?,
        integer(runtime, "batch")?,
        integer(runtime, "microbatch")?,
    );
    ensure!(
        (128..=262144).contains(&context)
            && (1..=4096).contains(&batch)
            && (1..=batch).contains(&microbatch),
        "invalid runtime context/batch settings"
    );
    ensure!(
        (2..=1024).contains(&integer(runtime, "n_layer")?)
            && (1..=65536).contains(&integer(runtime, "n_embd")?),
        "unsupported runtime model shape"
    );
    ensure!(
        runtime["cache_k"] == "F16" && runtime["cache_v"] == "F16",
        "unsupported runtime cache type"
    );
    Ok(())
}

fn match_runtime(actual: &Value, expected: &Value) -> Result<()> {
    for field in RUNTIME_FIELDS {
        ensure!(
            expected.get(field).is_some() && actual.get(field) == expected.get(field),
            "capture/manifest runtime identity mismatch at {field}"
        );
    }
    Ok(())
}

fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .with_context(|| format!("missing {key}"))
}

fn integer(value: &Value, key: &str) -> Result<usize> {
    usize::try_from(
        value
            .get(key)
            .and_then(Value::as_u64)
            .with_context(|| format!("invalid {key}"))?,
    )
    .with_context(|| format!("{key} overflow"))
}

fn hash_value(value: &Value) -> Result<&str> {
    let hash = value
        .as_str()
        .context("artifact reference must be a hash string")?;
    validate_hash(hash)?;
    Ok(hash)
}

fn hash_field<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    hash_value(value.get(key).with_context(|| format!("missing {key}"))?)
}

fn hash_set(value: &Value) -> Result<HashSet<String>> {
    let mut hashes = HashSet::new();
    for value in value.as_array().context("artifact list must be an array")? {
        ensure!(
            hashes.insert(hash_value(value)?.to_owned()),
            "duplicate manifest artifact hash"
        );
    }
    Ok(hashes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct Fixture {
        _root: tempfile::TempDir,
        artifacts: ArtifactStore,
        manifest: Value,
        vector: Value,
        supplied: HashSet<String>,
    }

    impl Fixture {
        fn new() -> Result<Self> {
            let root = tempfile::tempdir()?;
            let artifacts = ArtifactStore::open(root.path())?;
            let runtime = json!({"engine_revision":"a".repeat(40),"wrapper_sha256":"b".repeat(64),"context":128,"batch":32,"microbatch":8,"n_layer":2,"n_embd":2,"cache_k":"F16","cache_v":"F16"});
            let prepared = steering::prepare_dataset(&json!([
                {"id":"pair-a","family":"garden","messages":[{"role":"user","content":"Describe a garden."}],"positive":"Positive completion.","negative":"Negative completion."},
                {"id":"pair-b","family":"desk","messages":[{"role":"user","content":"Describe a desk."}],"positive":"Another positive.","negative":"Another negative."}
            ]))?;
            let dataset = &prepared["pairs"];
            let dataset_hash = artifacts.put_jsonl(dataset.as_array().unwrap())?;
            let stage_hash = artifacts
                .put_json(&json!({"status":"completed","output":{"interpretation":"test"}}))?;
            let direction = artifacts.put_f32(&[2], &[1., 0.])?;
            let unit = artifacts.put_f32(&[2], &[1., 0.])?;
            let layers = json!([{"layer":1,"width":2,"direction_hash":direction,"unit_hash":unit,"usable":true}]);
            let fingerprint = "c".repeat(64);
            let mut extractions = serde_json::Map::new();
            let mut hashes =
                HashSet::from([dataset_hash.clone(), stage_hash.clone(), direction, unit]);
            for pair in dataset.as_array().unwrap() {
                for pole in ["positive", "negative"] {
                    let prefix = format!(
                        "{} Assistant: ",
                        pair["messages"][0]["content"].as_str().unwrap()
                    );
                    let capture = json!({"v":1,"event":"result","prefix":prefix,"rendered":format!("{}{}",prefix,pair[pole].as_str().unwrap()),"token_ids":[1,2,3],"capture_position":2,"template":"example template","settings":{"mode":"chat","enable_thinking":false,"add_generation_prompt":true,"template_time_unix":0,"add_bos":true,"add_eos":false,"parse_special":true,"capture":"last-assistant-content-token"},"runtime":runtime,"captures":[{"layer":1,"values":[3.,4.],"norm":5.}]});
                    let hash = artifacts.put_json(&capture)?;
                    hashes.insert(hash.clone());
                    extractions.insert(
                        format!("{}:{pole}", pair["id"].as_str().unwrap()),
                        json!(hash),
                    );
                }
            }
            let preview_hash = artifacts.put_json(
                &json!({"prompt":"Preview","percent":0,"output":"text","cancelled":false}),
            )?;
            hashes.insert(preview_hash.clone());
            let previews = json!([{"prompt":"Preview","percent":0,"output":"text","artifact_hash":preview_hash}]);
            let algorithm = "last-assistant-content-token-difference-of-means-v1";
            let manifest = json!({"dataset_hash":dataset_hash,"recipe":{"id":"recipe","dataset_hash":dataset_hash,"dataset":dataset,"stages":{"design":stage_hash},"raw":false},"model_fingerprint":fingerprint,"model_shape":{"n_layer":2,"n_embd":2},"runtime":runtime,"algorithm":algorithm,"layers":layers,"previews":previews,"extractions":extractions,"extraction_identity":{"dataset_hash":dataset_hash,"model_fingerprint":fingerprint,"engine_revision":runtime["engine_revision"],"runtime":runtime,"algorithm":algorithm,"layers":[1],"raw":false},"artifacts":hashes.iter().cloned().collect::<Vec<_>>()});
            let vector = json!({"id":"vector","model_fingerprint":fingerprint,"layers":layers,"previews":previews});
            let mut fixture = Self {
                _root: root,
                artifacts,
                manifest,
                vector,
                supplied: hashes,
            };
            fixture.reseal()?;
            Ok(fixture)
        }

        fn reseal(&mut self) -> Result<()> {
            if let Some(old) = self.vector["manifest_hash"].as_str() {
                self.supplied.remove(old);
            }
            let hash = self.artifacts.put_json(&self.manifest)?;
            self.supplied.insert(hash.clone());
            self.vector["manifest_hash"] = json!(hash);
            Ok(())
        }

        fn validate(&self) -> Result<()> {
            validate_manifest_artifacts(
                &self.artifacts,
                &self.vector,
                &self.manifest,
                &self.supplied,
            )
        }

        fn remove_blob_and_listing(&mut self, hash: &str) -> Result<()> {
            self.supplied.remove(hash);
            self.manifest["artifacts"]
                .as_array_mut()
                .unwrap()
                .retain(|value| value != hash);
            self.reseal()
        }

        fn edit_capture(&mut self, edit: impl FnOnce(&mut Value)) -> Result<()> {
            let key = "pair-a:positive";
            let old = self.manifest["extractions"][key]
                .as_str()
                .unwrap()
                .to_owned();
            let mut capture = self.artifacts.read_json(&old)?;
            edit(&mut capture);
            let hash = self.artifacts.put_json(&capture)?;
            self.supplied.remove(&old);
            self.supplied.insert(hash.clone());
            for listed in self.manifest["artifacts"].as_array_mut().unwrap() {
                if *listed == old {
                    *listed = json!(hash);
                }
            }
            self.manifest["extractions"][key] = json!(hash);
            self.reseal()
        }
    }

    #[test]
    fn valid_bundle_and_explicit_historical_artifact_pass() -> Result<()> {
        let mut fixture = Fixture::new()?;
        fixture.validate()?;
        let hash = fixture
            .artifacts
            .put_json(&json!({"old_attempt":"retained"}))?;
        fixture.manifest["artifacts"]
            .as_array_mut()
            .unwrap()
            .push(json!(hash));
        fixture.supplied.insert(hash);
        fixture.reseal()?;
        fixture.validate()?;
        Ok(())
    }

    #[test]
    fn historical_policy_is_ignored_but_preview_records_must_match_their_declared_mode()
    -> Result<()> {
        let mut f = Fixture::new()?;
        f.manifest["recipe"]["coefficient_policy"] = json!("non_positive");
        f.manifest["recipe"]["preview_mode"] = json!("negative_only");
        f.reseal()?;
        f.validate()?;
        f.manifest["recipe"]["preview_mode"] = json!("none");
        f.reseal()?;
        assert!(f.validate().is_err());
        Ok(())
    }

    #[test]
    fn paper_captures_bind_the_exact_readout_and_folds() -> Result<()> {
        let mut f = Fixture::new()?;
        let extraction = Extraction::default();
        f.manifest["recipe"]["extraction"] = serde_json::to_value(&extraction)?;
        f.vector["extraction"] = serde_json::to_value(&extraction)?;
        f.manifest["algorithm"] = json!(extraction.algorithm());
        f.manifest["extraction_identity"]["algorithm"] = json!(extraction.algorithm());
        f.manifest["extraction_identity"]["extraction"] = serde_json::to_value(&extraction)?;
        f.manifest["split"] =
            crate::paper::fold_assignment(f.manifest["recipe"]["dataset"].as_array().unwrap())?;
        let pairs = f.manifest["recipe"]["dataset"].as_array().unwrap().clone();
        for pair in &pairs {
            for pole in ["positive", "negative"] {
                let key = format!("{}:{pole}", pair["id"].as_str().unwrap());
                let old = f.manifest["extractions"][&key].as_str().unwrap().to_owned();
                let mut capture = f.artifacts.read_json(&old)?;
                let command = extraction.capture(pair, pole, false)?;
                let prefix = command["messages"][0]["content"].as_str().unwrap();
                capture["prefix"] = json!(prefix);
                capture["rendered"] = json!(format!(
                    "{prefix}{}",
                    command["completion"].as_str().unwrap()
                ));
                capture["template"] = json!("");
                capture["settings"]["mode"] = json!("raw");
                capture["settings"]["raw_format"] = json!("single-user-verbatim-v1");
                capture["settings"]["enable_thinking"] = Value::Null;
                capture["settings"]["template_time_unix"] = Value::Null;
                capture["settings"]["add_generation_prompt"] = json!(false);
                let hash = f.artifacts.put_json(&capture)?;
                f.manifest["extractions"][&key] = json!(hash);
                f.supplied.remove(&old);
                f.supplied.insert(hash.clone());
                for value in f.manifest["artifacts"].as_array_mut().unwrap() {
                    if *value == old {
                        *value = json!(hash);
                    }
                }
            }
        }
        f.reseal()?;
        f.validate()?;
        f.manifest["recipe"]["extraction"]["readout_suffix"] = json!("Different:");
        f.reseal()?;
        assert!(f.validate().is_err());
        Ok(())
    }

    #[test]
    fn omitted_typed_blobs_fail_even_when_already_cached() -> Result<()> {
        for kind in ["dataset", "stage", "direction", "preview", "extraction"] {
            let mut fixture = Fixture::new()?;
            let hash = match kind {
                "dataset" => fixture.manifest["dataset_hash"].as_str(),
                "stage" => fixture.manifest["recipe"]["stages"]["design"].as_str(),
                "direction" => fixture.manifest["layers"][0]["direction_hash"].as_str(),
                "preview" => fixture.manifest["previews"][0]["artifact_hash"].as_str(),
                _ => fixture.manifest["extractions"]["pair-a:positive"].as_str(),
            }
            .unwrap()
            .to_owned();
            fixture.remove_blob_and_listing(&hash)?;
            assert!(fixture.validate().is_err(), "accepted omitted {kind} blob");
        }
        Ok(())
    }

    #[test]
    fn dataset_hash_split_and_malformed_pairs_are_rejected() -> Result<()> {
        let mut fixture = Fixture::new()?;
        fixture.manifest["recipe"]["dataset"][0]["positive"] = json!("Changed completion");
        fixture.reseal()?;
        assert!(fixture.validate().is_err());
        let mut fixture = Fixture::new()?;
        fixture.manifest["recipe"]["dataset"][0]["split"] = json!("made-up split");
        fixture.reseal()?;
        assert!(fixture.validate().is_err());
        let mut fixture = Fixture::new()?;
        fixture.manifest["recipe"]["dataset"][0]["messages"] = json!("not an array");
        fixture.reseal()?;
        assert!(fixture.validate().is_err());
        Ok(())
    }

    #[test]
    fn extraction_mapping_requires_exact_two_pole_coverage() -> Result<()> {
        let mut fixture = Fixture::new()?;
        fixture
            .manifest
            .as_object_mut()
            .unwrap()
            .remove("extractions");
        fixture.reseal()?;
        assert!(fixture.validate().is_err());
        let mut fixture = Fixture::new()?;
        fixture.manifest["extractions"]
            .as_object_mut()
            .unwrap()
            .remove("pair-a:negative");
        fixture.reseal()?;
        assert!(fixture.validate().is_err());
        let mut fixture = Fixture::new()?;
        let hash = fixture.manifest["extractions"]
            .as_object_mut()
            .unwrap()
            .remove("pair-a:negative")
            .unwrap();
        fixture.manifest["extractions"]["unknown:negative"] = hash;
        fixture.reseal()?;
        assert!(fixture.validate().is_err());
        Ok(())
    }

    #[test]
    fn malformed_capture_position_rendering_shape_runtime_and_settings_fail() -> Result<()> {
        let changes: &[(&str, Value)] = &[
            ("/capture_position", json!(0)),
            ("/token_ids", json!([1, -1, 3])),
            ("/rendered", json!("not the supplied completion")),
            ("/captures/0/values", json!([1.])),
            ("/captures/0/values/0", json!(1e100)),
            ("/captures/0/layer", json!(0)),
            ("/captures/0/norm", json!(99.)),
            ("/runtime/context", json!(256)),
            ("/runtime/wrapper_sha256", json!("d".repeat(64))),
            ("/settings/add_eos", json!(true)),
            ("/settings/enable_thinking", json!(true)),
        ];
        for (pointer, value) in changes {
            let mut fixture = Fixture::new()?;
            fixture
                .edit_capture(|capture| *capture.pointer_mut(pointer).unwrap() = value.clone())?;
            assert!(
                fixture.validate().is_err(),
                "accepted invalid capture {pointer}"
            );
        }
        Ok(())
    }

    #[test]
    fn raw_capture_formats_follow_each_examples_conversation_shape() -> Result<()> {
        let mut fixture = Fixture::new()?;
        let previous_dataset = fixture.manifest["dataset_hash"]
            .as_str()
            .unwrap()
            .to_owned();
        fixture.manifest["recipe"]["raw"] = json!(true);
        fixture.manifest["extraction_identity"]["raw"] = json!(true);
        fixture.manifest["recipe"]["dataset"][1]["messages"] = json!([
            {"role":"system","content":"Be concise."}, {"role":"user","content":"Describe a desk."}
        ]);
        let pairs = fixture.manifest["recipe"]["dataset"]
            .as_array()
            .unwrap()
            .clone();
        let dataset_hash = fixture.artifacts.put_jsonl(&pairs)?;
        fixture.manifest["recipe"]["dataset_hash"] = json!(dataset_hash);
        fixture.manifest["dataset_hash"] = json!(dataset_hash);
        fixture.manifest["extraction_identity"]["dataset_hash"] = json!(dataset_hash);
        fixture.supplied.remove(&previous_dataset);
        fixture.supplied.insert(dataset_hash.clone());
        for listed in fixture.manifest["artifacts"].as_array_mut().unwrap() {
            if *listed == previous_dataset {
                *listed = json!(dataset_hash);
            }
        }
        for pair in pairs {
            for pole in ["positive", "negative"] {
                let key = format!("{}:{pole}", pair["id"].as_str().unwrap());
                let old_hash = fixture.manifest["extractions"][&key]
                    .as_str()
                    .unwrap()
                    .to_owned();
                let mut capture = fixture.artifacts.read_json(&old_hash)?;
                let messages = pair["messages"].as_array().unwrap();
                let verbatim = messages.len() == 1;
                let prefix = if verbatim {
                    messages[0]["content"].as_str().unwrap().to_owned()
                } else {
                    "system: Be concise.\n\nuser: Describe a desk.\n\nassistant: ".to_owned()
                };
                capture["prefix"] = json!(prefix);
                capture["rendered"] = json!(format!("{prefix}{}", pair[pole].as_str().unwrap()));
                capture["template"] = json!("");
                capture["settings"]["mode"] = json!("raw");
                capture["settings"]["raw_format"] = json!(if verbatim {
                    "single-user-verbatim-v1"
                } else {
                    "role-labeled-dialogue-v1"
                });
                capture["settings"]["enable_thinking"] = Value::Null;
                capture["settings"]["template_time_unix"] = Value::Null;
                capture["settings"]["add_generation_prompt"] = json!(!verbatim);
                let hash = fixture.artifacts.put_json(&capture)?;
                fixture.manifest["extractions"][&key] = json!(hash);
                fixture.supplied.remove(&old_hash);
                fixture.supplied.insert(hash.clone());
                for listed in fixture.manifest["artifacts"].as_array_mut().unwrap() {
                    if *listed == old_hash {
                        *listed = json!(hash);
                    }
                }
            }
        }
        fixture.reseal()?;
        fixture.validate()?;
        fixture.edit_capture(|capture| {
            capture["settings"]
                .as_object_mut()
                .unwrap()
                .remove("raw_format");
        })?;
        assert!(fixture.validate().is_err());
        Ok(())
    }

    #[test]
    fn browser_numeric_metadata_roundtrip_preserves_exact_safe_values_only() -> Result<()> {
        assert!(!json_metadata_eq(
            &json!(9_007_199_254_740_993u64),
            &json!(9_007_199_254_740_992f64)
        ));
        assert!(!json_metadata_eq(
            &json!(9_007_199_254_740_994u64),
            &json!(9_007_199_254_740_994f64)
        ));
        assert!(!json_metadata_eq(
            &json!(-9_007_199_254_740_993i64),
            &json!(-9_007_199_254_740_992f64)
        ));
        assert!(json_metadata_eq(
            &json!({"a":[1,0,-2,9_007_199_254_740_992u64]}),
            &json!({"a":[1.0,0.0,-2.0,9_007_199_254_740_992f64]})
        ));
        assert!(json_metadata_eq(&json!(u64::MAX), &json!(u64::MAX)));
        assert!(!json_metadata_eq(&json!(1), &json!(1.25)));
        assert!(!json_metadata_eq(&json!({"a":1}), &json!({"a":1.0,"b":2})));
        let mut fixture = Fixture::new()?;
        fixture.manifest["layers"][0]["auc"] = json!(1.0);
        fixture.vector["layers"][0]["auc"] = json!(1);
        fixture.manifest["previews"][0]["percent"] = json!(0.0);
        fixture.reseal()?;
        fixture.validate()?;
        Ok(())
    }

    #[test]
    fn real_extraction_norm_survives_checked_manifest_json_roundtrip() -> Result<()> {
        // Real Bonsai manifest /layers/1/raw_norm. serde_json's default fast
        // parser (without float_roundtrip) produces the adjacent lower f64 and
        // rejects the just-written manifest as different from its checked blob.
        let value = f64::from_bits(0x4024_e305_c4ed_6fdd);
        let encoded = serde_json::to_string(&value)?;
        assert_eq!(encoded, "10.443403390869145");
        assert_eq!(
            serde_json::from_str::<f64>(&encoded)?.to_bits(),
            value.to_bits()
        );
        let mut fixture = Fixture::new()?;
        fixture.manifest["layers"][0]["raw_norm"] = json!(value);
        fixture.vector["layers"][0]["raw_norm"] = json!(value);
        fixture.reseal()?;
        fixture.validate()?;
        Ok(())
    }
}
