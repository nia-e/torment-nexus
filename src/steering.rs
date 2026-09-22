//! Pure dataset, extraction-diagnostic, and steering-mixture logic.
//!
//! Graph layers are numbered `1..n_layer`; graph layer zero is not supported by
//! the pinned engine's control-vector API. The engine is responsible for token
//! selection, state resets, tensor capture, and translating each graph layer L
//! to legacy control-buffer row L - 1. No inference happens in this module.

use anyhow::{Context, Result, anyhow, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

const MAX_WIDTH: usize = 1_048_576;
const MAX_MIX_VALUES: usize = 64 * 1024 * 1024;

/// Choose up to five distinct supported graph layers near 25/40/55/70/85%
/// of the graph-index range. Small models can have fewer than five choices.
pub fn supported_layers(n_layer: usize) -> Result<Vec<usize>> {
    ensure!(
        n_layer >= 2,
        "model has no supported steering layer (needs at least 2 layers)"
    );
    let last = n_layer - 1;
    let layers: BTreeSet<_> = [0.25, 0.40, 0.55, 0.70, 0.85]
        .map(|fraction| ((last as f64 * fraction).round() as usize).clamp(1, last))
        .into_iter()
        .collect();
    Ok(layers.into_iter().collect())
}

/// Make the graph-to-legacy mapping explicit and reject the unsupported zero
/// layer instead of silently shifting it to a different layer.
pub fn legacy_buffer_index(layer: usize, n_layer: usize) -> Result<usize> {
    ensure!(
        layer > 0 && layer < n_layer,
        "graph layer {layer} is unsupported for {n_layer} layers"
    );
    Ok(layer - 1)
}

/// Validate accepted dataset pairs and produce the canonical extraction split.
///
/// Accepts an array, or `{ "pairs": [...] }`. Each pair requires a unique `id`,
/// a nonempty scenario `family`, `messages: [{role, content}]`, and nonempty
/// `positive` / `negative` assistant completions. Extra fields are preserved.
/// Normalized-exact duplicates ignore line-ending differences and surrounding
/// whitespace, but preserve internal whitespace (which can matter in code).
///
/// Splitting groups normalized family names together. A deterministic subset
/// sum gets as close to 25% diagnostic *pairs* as whole families allow, leaving
/// both partitions nonempty when there are at least two families. A single
/// family remains entirely training, with an explicit diagnostic warning.
pub fn prepare_dataset(dataset: &Value) -> Result<Value> {
    let input = dataset_array(dataset)?;
    ensure!(!input.is_empty(), "dataset contains no pairs");
    let mut ids = HashSet::new();
    let mut seen = HashMap::<String, String>::new();
    let mut pairs = Vec::new();
    let mut duplicates = Vec::new();
    let mut warnings = Vec::new();
    let mut identical_poles = 0;
    let mut empty_contexts = 0;
    let mut positive_chars = 0usize;
    let mut negative_chars = 0usize;

    for (index, input_pair) in input.iter().enumerate() {
        let pair = input_pair
            .as_object()
            .with_context(|| format!("pair {index} must be an object"))?;
        let id = nonempty_string(input_pair, "id").with_context(|| format!("pair {index}"))?;
        ensure!(
            ids.insert(id.to_owned()),
            "duplicate dataset pair id {id:?}"
        );
        nonempty_string(input_pair, "family").with_context(|| format!("pair {id}"))?;
        let positive =
            nonempty_string(input_pair, "positive").with_context(|| format!("pair {id}"))?;
        let negative =
            nonempty_string(input_pair, "negative").with_context(|| format!("pair {id}"))?;
        let messages = pair
            .get("messages")
            .and_then(Value::as_array)
            .with_context(|| format!("pair {id} requires a messages array"))?;
        let mut normalized_messages = Vec::new();
        for (message_index, message) in messages.iter().enumerate() {
            let role = nonempty_string(message, "role")
                .with_context(|| format!("pair {id}, message {message_index}"))?;
            ensure!(
                matches!(role, "system" | "developer" | "user" | "assistant"),
                "pair {id}, message {message_index}: unsupported role {role:?}"
            );
            let content = message
                .get("content")
                .and_then(Value::as_str)
                .with_context(|| {
                    format!("pair {id}, message {message_index}: content must be a string")
                })?;
            normalized_messages.push(json!({"role": role, "content": normalized_text(content)}));
        }
        let normalized_positive = normalized_text(positive);
        let normalized_negative = normalized_text(negative);
        let canonical = serde_json::to_string(&json!({
            "messages": normalized_messages,
            "positive": normalized_positive,
            "negative": normalized_negative,
        }))?;
        if let Some(kept_id) = seen.get(&canonical) {
            duplicates.push(json!({"id":id,"duplicate_of":kept_id}));
            continue;
        }
        seen.insert(canonical, id.to_owned());
        identical_poles += usize::from(normalized_positive == normalized_negative);
        empty_contexts += usize::from(messages.is_empty());
        positive_chars += positive.chars().count();
        negative_chars += negative.chars().count();
        pairs.push(input_pair.clone());
    }

    let mut families = BTreeMap::<String, usize>::new();
    for pair in &pairs {
        *families
            .entry(normalized_family(nonempty_string(pair, "family")?))
            .or_default() += 1;
    }
    let diagnostic_families = choose_diagnostic_families(&families)?;
    let mut train_count = 0usize;
    let mut diagnostic_count = 0usize;
    for pair in &mut pairs {
        let diagnostic =
            diagnostic_families.contains(&normalized_family(nonempty_string(pair, "family")?));
        pair["split"] = json!(if diagnostic { "diagnostic" } else { "train" });
        if diagnostic {
            diagnostic_count += 1;
        } else {
            train_count += 1;
        }
    }
    if pairs.len() < 96 {
        warnings.push(format!("Dataset has {} usable pairs, below the 96-pair target; no replacements were fabricated.", pairs.len()));
    }
    if !duplicates.is_empty() {
        warnings.push(format!("Removed {} normalized-exact duplicate pairs; original outputs should remain in recipe provenance.", duplicates.len()));
    }
    if identical_poles > 0 {
        warnings.push(format!("{identical_poles} pairs have identical positive and negative completions and contribute no contrast."));
    }
    if empty_contexts > 0 {
        warnings.push(format!("{empty_contexts} pairs have no conversation context; ensure the chosen template or raw-completion mode supports this."));
    }
    if diagnostic_count == 0 {
        warnings.push("Only one scenario family is available; all pairs are training and held-out separation cannot be measured.".to_owned());
    } else if diagnostic_count < 8 {
        warnings.push(format!("Only {diagnostic_count} held-out pairs are available; separation AUC is a noisy diagnostic."));
    }
    if train_count < 8 {
        warnings.push(format!("Only {train_count} training pairs are available; the estimated direction may be unstable."));
    }
    if diagnostic_count > 0 && (diagnostic_count * 4).abs_diff(pairs.len()) > 4 {
        warnings.push(format!("Keeping scenario families together gives {train_count}/{diagnostic_count} training/diagnostic pairs rather than an exact 75/25 split."));
    }
    let shorter = positive_chars.min(negative_chars);
    let longer = positive_chars.max(negative_chars);
    if pairs.len() >= 4
        && longer > shorter.saturating_mul(2)
        && longer - shorter >= 80 * pairs.len()
    {
        warnings.push("Positive and negative completions have a large length imbalance; length may be a confound, not the requested concept.".to_owned());
    }
    Ok(json!({
        "pairs": pairs,
        "warnings": warnings,
        "duplicates": duplicates,
        "split": {
            "algorithm": "sha256-family-subset-sum-75-25-v1",
            "train_count": train_count,
            "diagnostic_count": diagnostic_count,
            "family_count": families.len(),
            "diagnostic_families": diagnostic_families,
            "target_train_fraction": 0.75,
            "actual_train_fraction": train_count as f64 / (train_count + diagnostic_count) as f64,
        },
    }))
}

/// Derive directions from *unsteered last-assistant-content-token* captures.
///
/// `captures` is an array of `{pair_id, pole: "positive" | "negative",
/// captures: [{layer, values: [f32], norm?: number}]}`. Exactly one capture per
/// pole/pair and identical nonempty layer sets and widths are required. Optional
/// supplied norms are checked against the values rather than trusted.
///
/// Returns `{layers: [{layer,width,raw,unit,residual_norm,auc,usable,warnings}],
/// selected_layer,split,warnings,...}`. Zero directions/calibration norms are
/// retained as unusable layers with `unit: null`, never replaced by fake axes.
/// `selected_layer: null` means there is no publishable direction. AUC measures
/// held-out projection separation, not steering efficacy or subjective state.
pub fn analyze_activations(
    dataset: &Value,
    captures: &Value,
    model_fingerprint: &str,
) -> Result<Value> {
    analyze_with_method(dataset, captures, model_fingerprint, false)
}

pub fn analyze_with_method(
    dataset: &Value,
    captures: &Value,
    model_fingerprint: &str,
    paper: bool,
) -> Result<Value> {
    ensure!(
        !model_fingerprint.trim().is_empty(),
        "model fingerprint must not be empty"
    );
    let prepared = prepare_dataset(dataset)?;
    let pairs = prepared["pairs"].as_array().expect("prepared pairs");
    let capture_list = captures.as_array().context("captures must be an array")?;
    let pair_ids: HashSet<_> = pairs
        .iter()
        .map(|p| p["id"].as_str().expect("validated id"))
        .collect();
    let mut tensors = HashMap::<(String, String), BTreeMap<usize, Vec<f32>>>::new();
    let mut expected_layers: Option<BTreeSet<usize>> = None;
    let mut expected_width: Option<usize> = None;

    for (index, record) in capture_list.iter().enumerate() {
        let pair_id = nonempty_string(record, "pair_id")
            .with_context(|| format!("capture record {index}"))?;
        ensure!(
            pair_ids.contains(pair_id),
            "capture references unknown or deduplicated pair {pair_id:?}"
        );
        let pole = nonempty_string(record, "pole")?;
        ensure!(
            matches!(pole, "positive" | "negative"),
            "pair {pair_id}: invalid capture pole {pole:?}"
        );
        let values = record
            .get("captures")
            .and_then(Value::as_array)
            .with_context(|| format!("pair {pair_id}/{pole}: missing captures array"))?;
        ensure!(
            !values.is_empty(),
            "pair {pair_id}/{pole}: no layers were captured"
        );
        let mut layers = BTreeMap::new();
        for capture in values {
            let layer = usize_field(capture, "layer")?;
            ensure!(
                layer > 0,
                "layer zero is unsupported; do not remap it to layer one"
            );
            let vector = f32_vector(capture.get("values").context("capture requires values")?)
                .with_context(|| format!("pair {pair_id}/{pole}, layer {layer}"))?;
            if let Some(width) = expected_width {
                ensure!(
                    vector.len() == width,
                    "capture width mismatch: layer {layer} has {}, expected {width}",
                    vector.len()
                );
            } else {
                expected_width = Some(vector.len());
            }
            if let Some(reported_norm) = capture.get("norm") {
                let reported = finite_number(reported_norm, "capture norm")?;
                let measured = l2(&vector);
                ensure!(
                    reported >= 0.0 && (reported - measured).abs() <= 1e-4 * measured.max(1e-30),
                    "pair {pair_id}/{pole}, layer {layer}: reported norm disagrees with captured values"
                );
            }
            ensure!(
                layers.insert(layer, vector).is_none(),
                "pair {pair_id}/{pole}: duplicate layer {layer}"
            );
        }
        let layer_set: BTreeSet<_> = layers.keys().copied().collect();
        if let Some(expected) = &expected_layers {
            ensure!(
                &layer_set == expected,
                "pair {pair_id}/{pole}: missing or unexpected capture layer(s)"
            );
        } else {
            expected_layers = Some(layer_set);
        }
        ensure!(
            tensors
                .insert((pair_id.to_owned(), pole.to_owned()), layers)
                .is_none(),
            "duplicate capture for pair {pair_id}/{pole}"
        );
    }
    for pair in pairs {
        let pair_id = pair["id"].as_str().expect("validated id");
        for pole in ["positive", "negative"] {
            ensure!(
                tensors.contains_key(&(pair_id.to_owned(), pole.to_owned())),
                "missing capture for pair {pair_id}/{pole}"
            );
        }
    }

    let width = expected_width.context("no activation tensors were provided")?;
    if paper {
        return crate::paper::analyze(
            &prepared,
            &tensors,
            expected_layers.context("missing layers")?,
            model_fingerprint,
        );
    }
    let train: Vec<_> = pairs.iter().filter(|p| p["split"] == "train").collect();
    let diagnostic: Vec<_> = pairs
        .iter()
        .filter(|p| p["split"] == "diagnostic")
        .collect();
    ensure!(!train.is_empty(), "no training pairs after grouped split");
    let mut layers = Vec::new();
    let mut warnings: Vec<String> = serde_json::from_value(prepared["warnings"].clone())?;
    let mut selection: Option<(usize, Option<f64>)> = None;

    for layer in expected_layers.expect("nonempty captures") {
        let mut difference = vec![0.0f64; width];
        let mut norms = Vec::with_capacity(train.len() * 2);
        for pair in &train {
            let pair_id = pair["id"].as_str().expect("validated id");
            let positive = &tensors[&(pair_id.to_owned(), "positive".to_owned())][&layer];
            let negative = &tensors[&(pair_id.to_owned(), "negative".to_owned())][&layer];
            for ((mean, p), n) in difference.iter_mut().zip(positive).zip(negative) {
                *mean += (f64::from(*p) - f64::from(*n)) / train.len() as f64;
            }
            norms.push(l2(positive));
            norms.push(l2(negative));
        }
        let raw: Vec<f32> = difference
            .into_iter()
            .map(|x| checked_f32(x, "mean difference"))
            .collect::<Result<_>>()?;
        let raw_norm = l2(&raw);
        let residual_norm = median(&mut norms);
        let usable = raw_norm > 0.0 && residual_norm > 0.0;
        let mut layer_warnings = Vec::new();
        let unit = if usable {
            Some(
                raw.iter()
                    .map(|x| checked_f32(f64::from(*x) / raw_norm, "unit direction"))
                    .collect::<Result<Vec<_>>>()?,
            )
        } else {
            layer_warnings.push(if raw_norm == 0.0 {
                "Mean positive-minus-negative direction is zero; this layer is unusable."
            } else {
                "Median training residual norm is zero; percentage calibration is unavailable and this layer is unusable."
            }.to_owned());
            None
        };
        let auc = if let Some(unit) = &unit {
            if diagnostic.is_empty() {
                None
            } else {
                let mut positive_scores = Vec::new();
                let mut negative_scores = Vec::new();
                for pair in &diagnostic {
                    let pair_id = pair["id"].as_str().expect("validated id");
                    positive_scores.push(dot(
                        unit,
                        &tensors[&(pair_id.to_owned(), "positive".to_owned())][&layer],
                    ));
                    negative_scores.push(dot(
                        unit,
                        &tensors[&(pair_id.to_owned(), "negative".to_owned())][&layer],
                    ));
                }
                Some(projection_auc(&positive_scores, &negative_scores))
            }
        } else {
            None
        };
        if let Some(auc) = auc {
            if auc < 0.5 {
                layer_warnings.push("Held-out projection reverses the training contrast (AUC below 0.5); this remains a diagnostic, not a veto.".to_owned());
            } else if auc <= 0.55 {
                layer_warnings.push("Held-out projection has weak separation; steering effects still require inspection.".to_owned());
            }
        }
        if usable {
            let is_better = match selection {
                None => true,
                Some((_, None)) => auc.is_some(),
                Some((_, Some(best))) => auc.is_some_and(|candidate| candidate > best),
            };
            // Layers are sorted ascending. Strict comparison keeps the earlier
            // layer on an exact tie, including the no-diagnostic fallback.
            if is_better {
                selection = Some((layer, auc));
            }
        }
        layers.push(json!({
            "layer": layer, "width": width, "raw": raw, "unit": unit,
            "raw_norm": raw_norm, "residual_norm": residual_norm,
            "auc": auc, "usable": usable, "warnings": layer_warnings,
            "train_count": train.len(), "diagnostic_count": diagnostic.len(),
        }));
    }
    if selection.is_none() {
        warnings.push("No layer has a usable nonzero direction and calibration norm; no vector can be published from these captures.".to_owned());
    } else if diagnostic.is_empty() {
        warnings.push("Selected the earliest usable extracted layer because held-out AUC is unavailable; no separation estimate is being claimed.".to_owned());
    }
    Ok(json!({
        "model_fingerprint": model_fingerprint,
        "algorithm": "last-assistant-content-token-difference-of-means-v1",
        "sign_convention": "mean_positive_minus_mean_negative",
        "scaling": "delta_h = (percent / 100) * median_training_residual_l2 * unit_direction",
        "diagnostic": "scenario-family-held-out projection AUC; separation only, not steering quality or subjective experience",
        "selected_layer": selection.map(|(layer, _)| layer),
        "layers": layers,
        "split": prepared["split"],
        "warnings": warnings,
    }))
}

/// Build a complete fresh control snapshot, with zero rows for every supported
/// layer that has no contribution. Never reuse a prior engine buffer.
///
/// `vectors` are final vector records with `id`, `model_fingerprint`, and
/// `layers: [{layer, unit: [f32], residual_norm}]` (the caller resolves unit_hash
/// to `unit` before calling). `axes` is `[{vector_id,layer,percent}]`.
/// The same vector ID may occur only once because live controls identify axes
/// by vector ID. Distinct vectors may share a layer. No mixture renormalization.
pub fn mix_rows(
    vectors: &[Value],
    axes: &Value,
    model_fingerprint: &str,
    n_layer: usize,
    width: usize,
) -> Result<Vec<Value>> {
    ensure!(
        !model_fingerprint.trim().is_empty(),
        "model fingerprint must not be empty"
    );
    ensure!(n_layer >= 2, "model has no supported steering layers");
    ensure!(
        width > 0 && width <= MAX_WIDTH,
        "invalid model residual width {width}"
    );
    let size = (n_layer - 1)
        .checked_mul(width)
        .context("control buffer size overflow")?;
    ensure!(
        size <= MAX_MIX_VALUES,
        "control buffer exceeds bounded capacity ({size} values)"
    );
    let mut rows = vec![vec![0.0f64; width]; n_layer - 1];
    let axes = axes.as_array().context("axes must be an array")?;
    let mut selected_ids = HashSet::new();
    for axis in axes {
        let vector_id = nonempty_string(axis, "vector_id")?;
        let layer = usize_field(axis, "layer")?;
        ensure!(
            selected_ids.insert((vector_id, layer)),
            "vector {vector_id:?} layer {layer} appears more than once; controls require unique vector/layer pairs"
        );
        let row = legacy_buffer_index(layer, n_layer)?;
        let percent = finite_number(
            axis.get("percent").context("axis requires percent")?,
            "axis percent",
        )?;
        let matching: Vec<_> = vectors
            .iter()
            .filter(|vector| vector["id"].as_str() == Some(vector_id))
            .collect();
        ensure!(
            matching.len() == 1,
            "vector {vector_id:?} is missing or ambiguous"
        );
        let vector = matching[0];
        ensure!(
            vector["model_fingerprint"].as_str() == Some(model_fingerprint),
            "vector {vector_id:?} belongs to a different model fingerprint"
        );
        let layers = vector
            .get("layers")
            .and_then(Value::as_array)
            .context("vector requires layers")?;
        let matching: Vec<_> = layers
            .iter()
            .filter(|candidate| candidate["layer"].as_u64() == Some(layer as u64))
            .collect();
        ensure!(
            matching.len() == 1,
            "vector {vector_id:?} does not have exactly one layer {layer}"
        );
        let direction = matching[0];
        ensure!(
            direction["usable"] != false,
            "vector {vector_id:?}, layer {layer} is marked unusable"
        );
        let unit = f32_vector(
            direction
                .get("unit")
                .context("load the vector's unit tensor into layers[].unit before mixing")?,
        )?;
        ensure!(
            unit.len() == width,
            "vector {vector_id:?}, layer {layer}: width {} does not match model width {width}",
            unit.len()
        );
        if let Some(declared_width) = direction.get("width") {
            ensure!(
                declared_width.as_u64() == Some(width as u64),
                "vector {vector_id:?}, layer {layer}: declared width is incompatible"
            );
        }
        let unit_norm = l2(&unit);
        ensure!(
            (unit_norm - 1.0).abs() <= 1e-3,
            "vector {vector_id:?}, layer {layer}: unit direction is not normalized (norm {unit_norm})"
        );
        let residual_norm = finite_number(
            direction
                .get("residual_norm")
                .context("direction requires residual_norm")?,
            "residual norm",
        )?;
        ensure!(
            residual_norm > 0.0,
            "vector {vector_id:?}, layer {layer}: calibration norm must be positive"
        );
        let scale = (percent / 100.0) * residual_norm;
        ensure!(
            scale.is_finite(),
            "vector {vector_id:?}: coefficient overflows calibrated scale"
        );
        for (sum, component) in rows[row].iter_mut().zip(unit) {
            *sum += scale * f64::from(component);
            ensure!(
                sum.is_finite(),
                "mixture accumulation overflow at layer {layer}"
            );
        }
    }
    rows.into_iter()
        .enumerate()
        .map(|(index, values)| {
            let values: Vec<f32> = values
                .into_iter()
                .map(|v| checked_f32(v, "mixed control value"))
                .collect::<Result<_>>()?;
            Ok(json!({"layer": index + 1, "values": values}))
        })
        .collect()
}

/// Advisory-only diagnostics for the exact F32 rows returned by `mix_rows`.
///
/// Cosines compare selected axes only when their graph layers match. The
/// injected norm is the norm of the summed row, not a sum of slider values.
/// A single percentage is reported only when the layer's selected axes share
/// a calibration norm; otherwise per-axis reference percentages make the
/// different denominators explicit. No coefficient or overlap is vetoed here.
pub fn mix_diagnostics(
    vectors: &[Value],
    axes: &Value,
    model_fingerprint: &str,
    n_layer: usize,
    width: usize,
) -> Result<Value> {
    let rows = mix_rows(vectors, axes, model_fingerprint, n_layer, width)?;
    struct Axis<'a> {
        id: &'a str,
        layer: usize,
        percent: f64,
        residual_norm: f64,
        unit: Vec<f32>,
    }
    // mix_rows has already checked every reference, shape, and scalar. Keeping
    // that as the shared validation boundary prevents diagnostics from relaxing
    // the engine's model/shape checks or introducing semantic gating.
    let selected: Vec<Axis<'_>> = axes
        .as_array()
        .expect("validated axes")
        .iter()
        .map(|axis| {
            let id = axis["vector_id"].as_str().expect("validated id");
            let layer = usize_field(axis, "layer")?;
            let vector = vectors
                .iter()
                .find(|vector| vector["id"] == id)
                .expect("validated vector");
            let direction = vector["layers"]
                .as_array()
                .expect("validated layers")
                .iter()
                .find(|candidate| candidate["layer"].as_u64() == Some(layer as u64))
                .expect("validated layer");
            Ok(Axis {
                id,
                layer,
                percent: axis["percent"].as_f64().expect("validated percent"),
                residual_norm: direction["residual_norm"].as_f64().expect("validated norm"),
                unit: f32_vector(&direction["unit"])?,
            })
        })
        .collect::<Result<_>>()?;
    let mut warnings = Vec::new();
    let mut cosines = Vec::new();
    for (index, axis) in selected.iter().enumerate() {
        if axis.percent.abs() > 100.0 {
            warnings.push(format!("Vector {:?} requests {:.3}%: its individual perturbation exceeds its calibration residual norm; degeneration is possible and this is not an intensity measure.", axis.id, axis.percent));
        } else if axis.percent.abs() > 20.0 {
            warnings.push(format!("Vector {:?} is outside the default -20% to +20% range ({:.3}%); the expanded finite coefficient is allowed.", axis.id, axis.percent));
        }
        for other in &selected[index + 1..] {
            if axis.layer != other.layer {
                continue;
            }
            let cosine = (dot(&axis.unit, &other.unit) / (l2(&axis.unit) * l2(&other.unit)))
                .clamp(-1.0, 1.0);
            cosines.push(json!({"left_vector_id":axis.id,"right_vector_id":other.id,"layer":axis.layer,"cosine":cosine}));
            if cosine.abs() >= 0.8 {
                warnings.push(format!("Vectors {:?} and {:?} overlap strongly at layer {} (cosine {cosine:+.3}); signed contributions add or cancel without renormalization.", axis.id, other.id, axis.layer));
            }
        }
    }
    let mut layers = Vec::new();
    for row in rows {
        let layer = usize_field(&row, "layer")?;
        let injected_norm = l2(&f32_vector(&row["values"])?);
        let matching: Vec<_> = selected.iter().filter(|axis| axis.layer == layer).collect();
        let component_norm_sum: f64 = matching
            .iter()
            .map(|axis| ((axis.percent / 100.0) * axis.residual_norm).abs() * l2(&axis.unit))
            .sum();
        let shared_calibration_norm = matching.first().and_then(|first| {
            matching
                .iter()
                .all(|axis| {
                    (axis.residual_norm - first.residual_norm).abs() <= first.residual_norm * 1e-5
                })
                .then_some(first.residual_norm)
        });
        let percent_of_shared_calibration = shared_calibration_norm
            .and_then(|norm| finite_diagnostic((injected_norm / norm) * 100.0));
        let reference_percentages: Vec<_> = matching.iter().map(|axis| json!({
            "vector_id":axis.id,
            "calibration_residual_norm":axis.residual_norm,
            "mixture_percent_of_this_calibration":finite_diagnostic((injected_norm / axis.residual_norm) * 100.0),
        })).collect();
        if !component_norm_sum.is_finite()
            || reference_percentages
                .iter()
                .any(|axis| axis["mixture_percent_of_this_calibration"].is_null())
        {
            warnings.push(format!("Layer {layer} has diagnostic magnitudes outside the numeric reporting range; affected diagnostics are null, while the validated injected F32 row remains finite."));
        }
        layers.push(json!({
            "layer":layer,
            "injected_norm":injected_norm,
            "axis_count":matching.len(),
            "sum_individual_injected_norms":finite_diagnostic(component_norm_sum),
            "remaining_fraction_after_summation":(component_norm_sum > 0.0 && component_norm_sum.is_finite()).then_some(injected_norm / component_norm_sum),
            "shared_calibration_norm":shared_calibration_norm,
            "percent_of_shared_calibration":percent_of_shared_calibration,
            "per_axis_calibration_references":reference_percentages,
        }));
    }
    Ok(
        json!({"warnings":warnings,"cosines":cosines,"layers":layers,
        "meaning":"Perturbation geometry only; no renormalization, no semantic quality gate, and no subjective-intensity interpretation."}),
    )
}

fn finite_diagnostic(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn dataset_array(dataset: &Value) -> Result<&Vec<Value>> {
    dataset
        .as_array()
        .or_else(|| dataset.get("pairs").and_then(Value::as_array))
        .context("dataset must be an array of pairs or an object containing pairs")
}

fn nonempty_string<'a>(value: &'a Value, field: &str) -> Result<&'a str> {
    let text = value
        .get(field)
        .and_then(Value::as_str)
        .with_context(|| format!("{field} must be a string"))?;
    ensure!(!text.trim().is_empty(), "{field} must not be empty");
    Ok(text)
}

fn normalized_text(text: &str) -> String {
    text.replace("\r\n", "\n").trim().to_owned()
}
fn normalized_family(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn usize_field(value: &Value, field: &str) -> Result<usize> {
    let number = value
        .get(field)
        .and_then(Value::as_u64)
        .with_context(|| format!("{field} must be a nonnegative integer"))?;
    usize::try_from(number).with_context(|| format!("{field} is out of range"))
}

fn finite_number(value: &Value, name: &str) -> Result<f64> {
    let number = value
        .as_f64()
        .with_context(|| format!("{name} must be a finite number"))?;
    ensure!(number.is_finite(), "{name} must be finite");
    Ok(number)
}

fn checked_f32(value: f64, name: &str) -> Result<f32> {
    let converted = value as f32;
    ensure!(
        value.is_finite() && converted.is_finite(),
        "{name} is non-finite or outside F32 range"
    );
    Ok(converted)
}

fn f32_vector(value: &Value) -> Result<Vec<f32>> {
    let values = value.as_array().context("tensor must be an array")?;
    ensure!(
        !values.is_empty() && values.len() <= MAX_WIDTH,
        "tensor width must be 1..={MAX_WIDTH}"
    );
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            checked_f32(finite_number(value, "tensor value")?, "tensor value")
                .with_context(|| format!("tensor component {index}"))
        })
        .collect()
}

fn l2(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|x| f64::from(*x).powi(2))
        .sum::<f64>()
        .sqrt()
}
fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum()
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        values[mid - 1] / 2.0 + values[mid] / 2.0
    }
}

fn choose_diagnostic_families(families: &BTreeMap<String, usize>) -> Result<BTreeSet<String>> {
    if families.len() <= 1 {
        return Ok(BTreeSet::new());
    }
    let mut ordered: Vec<_> = families.iter().collect();
    ordered.sort_by_cached_key(|(family, _)| {
        (
            Sha256::digest(family.as_bytes()).to_vec(),
            (*family).clone(),
        )
    });
    let total: usize = families.values().sum();
    let mut reachable: Vec<Option<(usize, usize)>> = vec![None; total + 1];
    reachable[0] = Some((0, usize::MAX));
    for (group_index, (_, count)) in ordered.iter().enumerate() {
        let count = **count;
        for previous in (0..=total - count).rev() {
            if reachable[previous].is_some() && reachable[previous + count].is_none() {
                reachable[previous + count] = Some((previous, group_index));
            }
        }
    }
    let mut sum = (1..total)
        .filter(|sum| reachable[*sum].is_some())
        .min_by_key(|sum| ((sum * 4).abs_diff(total), *sum))
        .ok_or_else(|| anyhow!("unable to create nonempty family-grouped split"))?;
    let mut selected = BTreeSet::new();
    while sum > 0 {
        let (previous, group_index) =
            reachable[sum].context("invalid grouped split reconstruction")?;
        ensure!(previous < sum, "invalid grouped split predecessor");
        selected.insert(ordered[group_index].0.clone());
        sum = previous;
    }
    Ok(selected)
}

/// Mann-Whitney AUC including half credit for ties; not a paired-only score.
fn projection_auc(positive: &[f64], negative: &[f64]) -> f64 {
    debug_assert!(!positive.is_empty() && !negative.is_empty());
    let mut scores: Vec<_> = positive
        .iter()
        .map(|score| (*score, true))
        .chain(negative.iter().map(|score| (*score, false)))
        .collect();
    scores.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut negatives_below = 0usize;
    let mut wins = 0.0;
    let mut start = 0;
    while start < scores.len() {
        let mut end = start + 1;
        while end < scores.len() && scores[end].0 == scores[start].0 {
            end += 1;
        }
        let positives = scores[start..end].iter().filter(|(_, pole)| *pole).count();
        let negatives = end - start - positives;
        wins += positives as f64 * (negatives_below as f64 + 0.5 * negatives as f64);
        negatives_below += negatives;
        start = end;
    }
    wins / (positive.len() as f64 * negative.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(index: usize, family: &str) -> Value {
        json!({"id":format!("p{index}"),"family":family,
            "messages":[{"role":"user","content":format!("prompt {index}")}],
            "positive":format!("positive completion {index}"),
            "negative":format!("negative completion {index}")})
    }

    fn dataset(count: usize) -> Value {
        Value::Array(
            (0..count)
                .map(|index| pair(index, &format!("family{}", index / 4)))
                .collect(),
        )
    }

    fn captures(dataset: &Value, layers: &[usize], positive: &[f32], negative: &[f32]) -> Value {
        Value::Array(dataset_array(dataset).unwrap().iter().flat_map(|pair| {
            [("positive", positive), ("negative", negative)].map(|(pole, values)| {
                json!({"pair_id":pair["id"],"pole":pole,
                    "captures":layers.iter().map(|layer| json!({"layer":layer,"values":values,"norm":l2(values)})).collect::<Vec<_>>()})
            })
        }).collect())
    }

    fn vector(id: &str, layer: usize, unit: &[f32], scale: f64) -> Value {
        json!({"id":id,"model_fingerprint":"model-a","layers":[
            {"layer":layer,"unit":unit,"residual_norm":scale,"width":unit.len()}]})
    }

    fn values(rows: &[Value], layer: usize) -> Vec<f64> {
        rows.iter().find(|row| row["layer"] == layer).unwrap()["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap())
            .collect()
    }

    #[test]
    fn layer_selection_and_legacy_layout_are_explicit() {
        assert_eq!(supported_layers(32).unwrap(), vec![8, 12, 17, 22, 26]);
        assert_eq!(supported_layers(2).unwrap(), vec![1]);
        assert!(supported_layers(1).is_err());
        assert_eq!(legacy_buffer_index(1, 32).unwrap(), 0);
        assert_eq!(legacy_buffer_index(31, 32).unwrap(), 30);
        assert!(legacy_buffer_index(0, 32).is_err());
        assert!(legacy_buffer_index(32, 32).is_err());
    }

    #[test]
    fn split_is_deterministic_family_grouped_and_seventy_five_twenty_five() {
        let dataset = dataset(96);
        let prepared = prepare_dataset(&dataset).unwrap();
        assert_eq!(prepared["split"]["train_count"], 72);
        assert_eq!(prepared["split"]["diagnostic_count"], 24);
        let mapping: BTreeMap<_, _> = prepared["pairs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|pair| (pair["id"].clone().to_string(), pair["split"].clone()))
            .collect();
        let mut reversed = dataset.as_array().unwrap().clone();
        reversed.reverse();
        let repeated = prepare_dataset(&json!(reversed)).unwrap();
        let reordered: BTreeMap<_, _> = repeated["pairs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|pair| (pair["id"].clone().to_string(), pair["split"].clone()))
            .collect();
        assert_eq!(mapping, reordered);
        let mut families = HashMap::new();
        for pair in prepared["pairs"].as_array().unwrap() {
            if let Some(split) =
                families.insert(pair["family"].clone().to_string(), pair["split"].clone())
            {
                assert_eq!(split, pair["split"]);
            }
        }
    }

    #[test]
    fn split_targets_pairs_not_number_of_families_and_normalizes_family_names() {
        let pairs: Vec<_> = (0..100)
            .map(|index| {
                pair(
                    index,
                    match index {
                        0..=79 => "large",
                        80..=89 => "Small A",
                        _ => "small b",
                    },
                )
            })
            .collect();
        let prepared = prepare_dataset(&json!(pairs)).unwrap();
        assert_eq!(prepared["split"]["diagnostic_count"], 20);
        let dataset = json!([pair(0, "Same family"), pair(1, " SAME   FAMILY ")]);
        let prepared = prepare_dataset(&dataset).unwrap();
        assert_eq!(prepared["split"]["family_count"], 1);
        assert_eq!(prepared["split"]["diagnostic_count"], 0);
    }

    #[test]
    fn duplicate_contrast_removed_without_collapsing_internal_whitespace() {
        let first = pair(0, "one");
        let mut duplicate = first.clone();
        duplicate["id"] = json!("duplicate");
        duplicate["family"] = json!("another family does not hide a duplicate");
        duplicate["positive"] = json!(format!(" {}\r\n", first["positive"].as_str().unwrap()));
        let mut distinct = first.clone();
        distinct["id"] = json!("distinct");
        distinct["positive"] = json!("positive  completion 0");
        let prepared = prepare_dataset(&json!([first, duplicate, distinct])).unwrap();
        assert_eq!(prepared["pairs"].as_array().unwrap().len(), 2);
        assert_eq!(
            prepared["duplicates"],
            json!([{"id":"duplicate","duplicate_of":"p0"}])
        );
    }

    #[test]
    fn malformed_pairs_and_duplicate_ids_are_fatal() {
        let pair = pair(0, "family");
        assert!(prepare_dataset(&json!([pair.clone(), pair.clone()])).is_err());
        let mut empty = pair.clone();
        empty["positive"] = json!("   ");
        assert!(prepare_dataset(&json!([empty])).is_err());
        let mut malformed = pair.clone();
        malformed["messages"][0]["role"] = json!("root");
        assert!(prepare_dataset(&json!([malformed])).is_err());
        assert!(prepare_dataset(&json!([])).is_err());
    }

    #[test]
    fn mean_difference_unit_calibration_auc_and_tie_break() {
        let dataset = dataset(16);
        let captured = captures(&dataset, &[2, 5], &[3.0, 4.0], &[0.0, 4.0]);
        let result = analyze_activations(&dataset, &captured, "model-a").unwrap();
        assert_eq!(result["selected_layer"], 2);
        assert_eq!(result["layers"][0]["raw"], json!([3.0, 0.0]));
        assert_eq!(result["layers"][0]["unit"], json!([1.0, 0.0]));
        assert_eq!(result["layers"][0]["residual_norm"], 4.5);
        assert_eq!(result["layers"][0]["auc"], 1.0);
    }

    #[test]
    fn swapped_poles_negate_raw_and_unit_without_changing_auc() {
        let dataset = dataset(16);
        let original = analyze_activations(
            &dataset,
            &captures(&dataset, &[2], &[3.0, 4.0], &[0.0, 4.0]),
            "m",
        )
        .unwrap();
        let swapped = analyze_activations(
            &dataset,
            &captures(&dataset, &[2], &[0.0, 4.0], &[3.0, 4.0]),
            "m",
        )
        .unwrap();
        for key in ["raw", "unit"] {
            for (a, b) in original["layers"][0][key]
                .as_array()
                .unwrap()
                .iter()
                .zip(swapped["layers"][0][key].as_array().unwrap())
            {
                assert_eq!(a.as_f64().unwrap(), -b.as_f64().unwrap());
            }
        }
        assert_eq!(original["layers"][0]["auc"], swapped["layers"][0]["auc"]);
        assert_eq!(
            original["layers"][0]["residual_norm"],
            swapped["layers"][0]["residual_norm"]
        );
    }

    #[test]
    fn diagnostic_examples_do_not_enter_direction_or_calibration() {
        let dataset = dataset(16);
        let prepared = prepare_dataset(&dataset).unwrap();
        let mut captured = captures(&dataset, &[1], &[3.0, 4.0], &[0.0, 4.0]);
        let held_out: HashSet<_> = prepared["pairs"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|pair| pair["split"] == "diagnostic")
            .map(|pair| pair["id"].as_str().unwrap())
            .collect();
        for record in captured.as_array_mut().unwrap() {
            if held_out.contains(record["pair_id"].as_str().unwrap()) {
                let values = if record["pole"] == "positive" {
                    vec![-30.0, 40.0]
                } else {
                    vec![30.0, 40.0]
                };
                record["captures"][0]["values"] = json!(values);
                record["captures"][0]["norm"] = json!(50.0);
            }
        }
        let result = analyze_activations(&dataset, &captured, "m").unwrap();
        assert_eq!(result["layers"][0]["raw"], json!([3.0, 0.0]));
        assert_eq!(result["layers"][0]["residual_norm"], 4.5);
        assert_eq!(result["layers"][0]["auc"], 0.0);
        assert_eq!(result["selected_layer"], 1); // weak separation is not a veto
    }

    #[test]
    fn zero_direction_is_retained_as_unusable_not_fabricated() {
        let dataset = dataset(16);
        let captured = captures(&dataset, &[1, 2], &[3.0, 4.0], &[3.0, 4.0]);
        let result = analyze_activations(&dataset, &captured, "m").unwrap();
        assert!(result["selected_layer"].is_null());
        assert_eq!(result["layers"][0]["usable"], false);
        assert!(result["layers"][0]["unit"].is_null());
        assert_eq!(result["layers"][0]["raw"], json!([0.0, 0.0]));
    }

    #[test]
    fn one_family_stays_usable_without_claiming_auc() {
        let dataset = json!([pair(0, "only"), pair(1, "only")]);
        let result =
            analyze_activations(&dataset, &captures(&dataset, &[2, 4], &[1.0], &[-1.0]), "m")
                .unwrap();
        assert_eq!(result["selected_layer"], 2);
        assert!(result["layers"][0]["auc"].is_null());
    }

    #[test]
    fn capture_validation_rejects_missing_layers_shape_errors_bad_norms_and_overflow() {
        let dataset = dataset(8);
        let base = captures(&dataset, &[1, 2], &[1.0, 0.0], &[-1.0, 0.0]);
        let mut invalid = base.clone();
        invalid.as_array_mut().unwrap().pop();
        assert!(analyze_activations(&dataset, &invalid, "m").is_err());
        let mut invalid = base.clone();
        invalid[0]["captures"].as_array_mut().unwrap().pop();
        assert!(analyze_activations(&dataset, &invalid, "m").is_err());
        let mut invalid = base.clone();
        invalid[0]["captures"][0]["values"] = json!([1.0]);
        assert!(analyze_activations(&dataset, &invalid, "m").is_err());
        let mut invalid = base.clone();
        invalid[0]["captures"][0]["norm"] = json!(2.0);
        assert!(analyze_activations(&dataset, &invalid, "m").is_err());
        let mut invalid = base.clone();
        invalid[0]["captures"][0]["values"][0] = json!(1e300);
        assert!(analyze_activations(&dataset, &invalid, "m").is_err());
        let mut invalid = base.clone();
        invalid[0]["captures"][0]["values"][0] = Value::Null;
        assert!(analyze_activations(&dataset, &invalid, "m").is_err());
    }

    #[test]
    fn auc_ties_and_cross_pair_rank_order_are_correct() {
        assert_eq!(projection_auc(&[1.0, 1.0], &[1.0, 1.0]), 0.5);
        assert_eq!(projection_auc(&[2.0, 4.0], &[1.0, 3.0]), 0.75);
        assert_eq!(projection_auc(&[0.0, 1.0], &[2.0, 3.0]), 0.0);
        assert_eq!(projection_auc(&[-0.0], &[0.0]), 0.5);
    }

    #[test]
    fn signed_scaling_and_two_axes_equal_their_precomputed_sum_without_renormalization() {
        let vectors = vec![
            vector("x", 2, &[1.0, 0.0], 10.0),
            vector("y", 2, &[0.0, 1.0], 20.0),
        ];
        let rows = mix_rows(
            &vectors,
            &json!([
            {"vector_id":"x","layer":2,"percent":20.0},
            {"vector_id":"y","layer":2,"percent":-15.0}]),
            "model-a",
            4,
            2,
        )
        .unwrap();
        assert_eq!(values(&rows, 2), vec![2.0, -3.0]);
        assert_eq!(values(&rows, 1), vec![0.0, 0.0]);
        assert_eq!(values(&rows, 3), vec![0.0, 0.0]);
        let sum_norm = 13.0f64.sqrt();
        let precomputed = vector(
            "sum",
            2,
            &[(2.0 / sum_norm) as f32, (-3.0 / sum_norm) as f32],
            sum_norm,
        );
        let combined = mix_rows(
            &[precomputed],
            &json!([{"vector_id":"sum","layer":2,"percent":100.0}]),
            "model-a",
            4,
            2,
        )
        .unwrap();
        for (a, b) in values(&rows, 2).into_iter().zip(values(&combined, 2)) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn one_concept_can_contribute_at_multiple_layers_with_independent_clearing() {
        let mut concept = vector("x", 1, &[1.0, 0.0], 10.0);
        concept["layers"]
            .as_array_mut()
            .unwrap()
            .push(vector("x", 2, &[0.0, 1.0], 20.0)["layers"][0].clone());
        let vectors = vec![concept, vector("y", 2, &[0.0, 1.0], 20.0)];
        let mut axes = json!([
            {"vector_id":"x","layer":1,"percent":20},
            {"vector_id":"x","layer":2,"percent":-10},
            {"vector_id":"y","layer":2,"percent":5}]);
        let rows = mix_rows(&vectors, &axes, "model-a", 4, 2).unwrap();
        assert_eq!(values(&rows, 1), vec![2., 0.]);
        assert_eq!(values(&rows, 2), vec![0., -1.]);
        assert_eq!(values(&rows, 3), vec![0., 0.]);
        axes[0]["percent"] = json!(0);
        let rows = mix_rows(&vectors, &axes, "model-a", 4, 2).unwrap();
        assert_eq!(values(&rows, 1), vec![0., 0.]);
        assert_eq!(values(&rows, 2), vec![0., -1.]);
        let diagnostics = mix_diagnostics(&vectors, &axes, "model-a", 4, 2).unwrap();
        assert_eq!(diagnostics["cosines"].as_array().unwrap().len(), 1);
        assert_eq!(diagnostics["cosines"][0]["layer"], 2);
        assert!(mix_rows(&vectors, &json!([axes[0], axes[0]]), "model-a", 4, 2).is_err());
    }

    #[test]
    fn zeroing_and_removing_axes_returns_all_zero_rows_without_stale_state() {
        let vectors = vec![vector("x", 1, &[1.0, 0.0], 10.0)];
        let enabled = mix_rows(
            &vectors,
            &json!([{"vector_id":"x","layer":1,"percent":20.0}]),
            "model-a",
            3,
            2,
        )
        .unwrap();
        assert_eq!(values(&enabled, 1), vec![2.0, 0.0]);
        for axes in [
            json!([]),
            json!([{"vector_id":"x","layer":1,"percent":0.0}]),
        ] {
            let rows = mix_rows(&vectors, &axes, "model-a", 3, 2).unwrap();
            assert_eq!(rows.len(), 2);
            assert!(rows.iter().all(|row| {
                row["values"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|value| value == 0.0)
            }));
        }
    }

    #[test]
    fn mixer_diagnostics_measure_sum_and_compare_only_same_layer_axes() {
        let vectors = vec![
            vector("x", 1, &[1.0, 0.0], 10.0),
            vector("y", 1, &[1.0, 0.0], 10.0),
            vector("z", 2, &[1.0, 0.0], 10.0),
        ];
        let axes = json!([
            {"vector_id":"x","layer":1,"percent":20.0},
            {"vector_id":"y","layer":1,"percent":20.0},
            {"vector_id":"z","layer":2,"percent":20.0},
        ]);
        let result = mix_diagnostics(&vectors, &axes, "model-a", 3, 2).unwrap();
        assert_eq!(result["cosines"].as_array().unwrap().len(), 1);
        assert_eq!(result["cosines"][0]["cosine"], 1.0);
        assert_eq!(result["layers"][0]["injected_norm"], 4.0);
        assert_eq!(result["layers"][0]["percent_of_shared_calibration"], 40.0);
        assert_eq!(result["layers"][1]["injected_norm"], 2.0);
        assert_eq!(result["warnings"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn mixer_diagnostics_report_cancellation_and_do_not_invent_a_shared_denominator() {
        let vectors = vec![
            vector("x", 1, &[1.0, 0.0], 10.0),
            vector("y", 1, &[-1.0, 0.0], 20.0),
        ];
        let axes = json!([
            {"vector_id":"x","layer":1,"percent":20.0},
            {"vector_id":"y","layer":1,"percent":10.0},
        ]);
        let result = mix_diagnostics(&vectors, &axes, "model-a", 2, 2).unwrap();
        assert_eq!(result["cosines"][0]["cosine"], -1.0);
        assert_eq!(result["layers"][0]["injected_norm"], 0.0);
        assert_eq!(result["layers"][0]["sum_individual_injected_norms"], 4.0);
        assert_eq!(
            result["layers"][0]["remaining_fraction_after_summation"],
            0.0
        );
        assert!(result["layers"][0]["shared_calibration_norm"].is_null());
        assert!(result["layers"][0]["percent_of_shared_calibration"].is_null());
        assert_eq!(
            result["layers"][0]["per_axis_calibration_references"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn expanded_and_extreme_coefficients_are_advised_not_vetoed() {
        let vectors = vec![vector("x", 1, &[1.0], 10.0)];
        for percent in [-500.0f64, -200.0, -25.0, 25.0, 200.0, 500.0] {
            let axes = json!([{"vector_id":"x","layer":1,"percent":percent}]);
            let result = mix_diagnostics(&vectors, &axes, "model-a", 2, 1).unwrap();
            assert_eq!(result["warnings"].as_array().unwrap().len(), 1);
            assert_eq!(result["layers"][0]["injected_norm"], percent.abs() / 10.0);
        }
    }
    #[test]
    fn historical_coefficient_policy_metadata_does_not_restrict_the_mixer() {
        let mut v = vector("x", 1, &[1.], 10.);
        v["coefficient_policy"] = json!("non_positive");
        for percent in [0., -200., 200., 0.001] {
            assert!(
                mix_rows(
                    &[v.clone()],
                    &json!([{"vector_id":"x","layer":1,"percent":percent}]),
                    "model-a",
                    2,
                    1
                )
                .is_ok()
            );
        }
    }

    #[test]
    fn mixing_rejects_wrong_model_invalid_units_shapes_layers_and_nonfinite_coefficients() {
        let valid = vector("x", 1, &[1.0, 0.0], 10.0);
        let axes = json!([{"vector_id":"x","layer":1,"percent":20.0}]);
        assert!(mix_rows(std::slice::from_ref(&valid), &axes, "wrong-model", 3, 2).is_err());
        assert!(mix_rows(std::slice::from_ref(&valid), &axes, "model-a", 3, 3).is_err());
        let mut invalid = valid.clone();
        invalid["layers"][0]["unit"] = json!([2.0, 0.0]);
        assert!(mix_rows(&[invalid], &axes, "model-a", 3, 2).is_err());
        let mut invalid = valid.clone();
        invalid["layers"][0]["residual_norm"] = json!(-1.0);
        assert!(mix_rows(&[invalid], &axes, "model-a", 3, 2).is_err());
        let mut invalid = axes.clone();
        invalid[0]["layer"] = json!(0);
        assert!(mix_rows(std::slice::from_ref(&valid), &invalid, "model-a", 3, 2).is_err());
        let mut invalid = axes.clone();
        invalid[0]["percent"] = Value::Null;
        assert!(mix_rows(std::slice::from_ref(&valid), &invalid, "model-a", 3, 2).is_err());
        let mut invalid = axes.clone();
        invalid[0]["percent"] = json!(1e300);
        assert!(mix_rows(std::slice::from_ref(&valid), &invalid, "model-a", 3, 2).is_err());
        assert!(mix_rows(&[valid], &json!([axes[0], axes[0]]), "model-a", 3, 2).is_err());
    }
}
