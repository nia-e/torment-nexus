//! Durable concept-generation stages. The caller commits every checkpoint before
//! this module is allowed to advance; successful cloud calls are not repeated on
//! resume. Extraction and previews are deliberately separate local stages.
use crate::codex::{CodexClient, ModelInfo, validate_schema};
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use tokio::sync::mpsc::UnboundedSender;
use uuid::Uuid;

pub const PAIRS_PER_WRITER: usize = 32;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoleAssignments {
    pub designer: String,
    pub writers: Vec<String>,
    pub reviewer: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FactoryOutput {
    pub design: Value,
    pub dataset: Value,
    pub review: Value,
    pub roles: RoleAssignments,
    pub warnings: Vec<String>,
    pub stages: BTreeMap<String, Value>,
}

pub fn default_roles(models: &[ModelInfo]) -> Result<RoleAssignments> {
    let mut candidates: Vec<&ModelInfo> = models
        .iter()
        .filter(|m| {
            !m.hidden
                && (m.input_modalities.is_empty() || m.input_modalities.iter().any(|x| x == "text"))
        })
        .collect();
    candidates.sort_by_key(|m| !m.is_default);
    let mut selected = Vec::new();
    for m in candidates {
        if !selected.contains(&m.model) {
            selected.push(m.model.clone());
        }
        if selected.len() == 3 {
            break;
        }
    }
    ensure!(
        !selected.is_empty(),
        "No text models available for concept generation"
    );
    Ok(RoleAssignments {
        designer: selected[0].clone(),
        writers: (0..3)
            .map(|i| selected[i % selected.len()].clone())
            .collect(),
        reviewer: selected[0].clone(),
    })
}

/// Explicit retry may repair unavailable assignments, but never silently rerun
/// a completed stage under a different model. Record every substitution.
pub fn retry_roles(
    previous: &RoleAssignments,
    stages: &BTreeMap<String, Value>,
    models: &[ModelInfo],
) -> Result<(RoleAssignments, Vec<Value>)> {
    ensure!(
        previous.writers.len() == 3,
        "Exactly three writer roles are required"
    );
    let defaults = default_roles(models)?;
    let mut roles = previous.clone();
    let mut changes = Vec::new();
    let mut repair = |stage: &str, model: &mut String, replacement: &str| -> Result<()> {
        if !models.iter().any(|candidate| candidate.model == *model) {
            ensure!(
                !stages
                    .get(stage)
                    .is_some_and(|saved| saved["status"] == "completed" || saved["edited"] == true),
                "Completed {stage} used unavailable model {model}; keeping its saved output. Edit or fork the stage rather than silently regenerating it."
            );
            changes.push(json!({"stage":stage,"from":model,"to":replacement,"reason":"previous model is not supported by current account and installed catalog"}));
            *model = replacement.to_owned();
        }
        Ok(())
    };
    repair("design", &mut roles.designer, &defaults.designer)?;
    for (index, model) in roles.writers.iter_mut().enumerate() {
        repair(
            &format!("writer_{}", index + 1),
            model,
            &defaults.writers[index],
        )?;
    }
    repair("review", &mut roles.reviewer, &defaults.reviewer)?;
    validate_roles(&roles, models)?;
    Ok((roles, changes))
}

/// `save_stage` must atomically persist the supplied JSON before returning Ok.
/// Checkpoint keys ending `.attempt_<uuid>` preserve every original answer;
/// `<stage>` identifies a reusable, completed output bound to its exact input hash.
/// Manual edits can use {"edited":true,"output":...} after invalidating descendants.
pub async fn run<F>(
    client: &CodexClient,
    concept: &str,
    roles: Option<RoleAssignments>,
    completed_stages: BTreeMap<String, Value>,
    cancel: Arc<AtomicBool>,
    events: UnboundedSender<Value>,
    save_stage: F,
) -> Result<FactoryOutput>
where
    F: Fn(&str, &Value) -> Result<()> + Send + Sync,
{
    run_with_extraction(
        client,
        concept,
        roles,
        completed_stages,
        cancel,
        events,
        save_stage,
        &crate::extraction::Extraction::legacy(),
    )
    .await
}

#[allow(clippy::too_many_arguments)] // Versioned extraction settings bind the cloud prompts.
pub async fn run_with_extraction<F>(
    client: &CodexClient,
    concept: &str,
    roles: Option<RoleAssignments>,
    mut completed_stages: BTreeMap<String, Value>,
    cancel: Arc<AtomicBool>,
    events: UnboundedSender<Value>,
    save_stage: F,
    extraction: &crate::extraction::Extraction,
) -> Result<FactoryOutput>
where
    F: Fn(&str, &Value) -> Result<()> + Send + Sync,
{
    ensure!(
        !concept.trim().is_empty() && concept.len() <= 16_384,
        "Concept must be nonempty and at most 16 KiB"
    );
    let models = client.discover().await?;
    let roles = match roles {
        Some(r) => r,
        None => match completed_stages.get("roles") {
            Some(v) => {
                serde_json::from_value(v.clone()).context("Saved role assignment is malformed")?
            }
            None => default_roles(&models)?,
        },
    };
    validate_roles(&roles, &models)?;
    save(
        &mut completed_stages,
        "roles",
        serde_json::to_value(&roles)?,
        &save_stage,
    )?;
    let distinct: BTreeSet<_> = roles.writers.iter().collect();
    let mut warnings = Vec::new();
    if distinct.len() < 3 {
        warnings.push(format!("Only {} distinct writer model(s) are assigned; the three writer threads remain independent.",distinct.len()));
    }
    let mut design_prompt = format!(
        "Design one operational text contrast for this user-supplied concept: {concept:?}. Choose ONE default interpretation that can be expressed in assistant completions. Describe its positive pole and its negative pole (opposite or absence, explicitly chosen), important confounds to match, 12 diverse named scenario families, and exactly 3 neutral preview prompts. Include alternative interpretations for future forks, but do not mix them into this dataset. Concepts can be unusual; do not replace them with a different familiar axis. Do not claim that a concept label measures subjective experience. Output the schema, with concise fields."
    );
    if extraction.method == crate::extraction::Method::Paper {
        design_prompt.push_str(&format!("\nExtraction uses self-contained raw statements followed by this identical readout suffix: {:?}. It does not encode the common chat context. Design a representation contrast, not just an assistant delivery style. Use diverse positive subtypes and a pooled negative class comprising multiple distinct nuisance/confound controls, not merely a pleasant opposite. Describe and balance those controls in the confounds. Prefer concise naturalistic statements. Neutral preview prompts must be mundane declarative statements, not questions or writing instructions. Never include the readout suffix in generated statements; it is appended locally.", extraction.readout_suffix));
    }
    let design = call_stage(
        client,
        "design",
        &roles.designer,
        design_prompt,
        design_schema(),
        &mut completed_stages,
        cancel.clone(),
        events.clone(),
        &save_stage,
        &|_| Ok(()),
    )
    .await?;
    let mut all_pairs = Vec::new();
    // Independent threads and model assignments, deliberately bounded to one
    // cloud request at a time to avoid surprise account-wide rate-limit bursts.
    if let Some(edited) = completed_stages.get("dataset_input") {
        ensure!(
            edited["edited"] == true,
            "Dataset input must be an explicit edit"
        );
        let mut edited = edited["output"]
            .as_array()
            .context("Edited dataset must be a pair array")?
            .clone();
        ensure!(
            edited.len() <= 4096,
            "Edited dataset exceeds the 4096-pair limit"
        );
        let mut ids = BTreeSet::new();
        for pair in &mut edited {
            // A saved recipe includes a derived split. Recalculate it as a group
            // after an edit; do not send old train/test labels to the reviewer.
            pair.as_object_mut()
                .context("Edited pair must be an object")?
                .remove("split");
            validate_schema(pair, &pair_schema())?;
            validate_pair(pair)?;
            ensure!(
                ids.insert(pair["id"].as_str().unwrap()),
                "Edited dataset has duplicate pair IDs"
            );
        }
        all_pairs = edited;
    } else {
        for (i, model) in roles.writers.iter().enumerate() {
            let stage = format!("writer_{}", i + 1);
            let mut prompt = format!(
                "You are independent writer {} of 3. Create exactly {PAIRS_PER_WRITER} contrastive matched pairs for this design:\n{}\nEach pair has common prior conversation messages and two alternative assistant completions: positive expresses the positive pole, negative the negative pole. The context MUST be identical; only the completion changes. Match topic, factual content, length, formatting, valence, helpfulness and refusal style unless essential to the defined axis. Never name the concept or add pole labels in completions merely to signal the class. Use natural 15-55 word completions; vary everyday scenario details. Use the design's scenario-family names consistently. Cover at least 8 families, with 4 pairs per family when possible. Writer {} should use a distinct set of concrete situations, not merely paraphrase canned examples. IDs will be replaced locally. Do not invent measurement or consciousness claims. Return only the requested structured object.",
                i + 1,
                serde_json::to_string(&design)?,
                i + 1
            );
            if extraction.method == crate::extraction::Method::Paper {
                prompt.push_str("\nFor this paper-style recipe, override the completion length/style instruction above: each pole must be a self-contained 8-25 word naturalistic statement, not conversational advice or a multi-sentence assistant response. Common messages document the writing request only and are NOT decoded during extraction. Spread positives over the design's subtypes and negatives over its distinct confound controls; avoid one dominant comparator. Keep the scenario-family label EXACTLY as supplied by the designer. Do not append a probe suffix or label words.");
            }
            let response = call_stage(
                client,
                &stage,
                model,
                prompt,
                writer_schema(),
                &mut completed_stages,
                cancel.clone(),
                events.clone(),
                &save_stage,
                &|_| Ok(()),
            )
            .await?;
            let pairs = response["pairs"]
                .as_array()
                .context("Writer pairs missing")?;
            if pairs.len() != PAIRS_PER_WRITER {
                warnings.push(format!("Writer {} supplied {} pairs rather than {PAIRS_PER_WRITER}; no filler was fabricated.",i+1,pairs.len()));
            }
            for (j, pair) in pairs.iter().enumerate() {
                let mut pair = pair.clone();
                pair["id"] = json!(format!("w{}-{:03}", i + 1, j + 1));
                all_pairs.push(pair);
            }
        }
    }
    let (unique, duplicates) = deduplicate_pairs(&all_pairs)?;
    save(
        &mut completed_stages,
        "candidates",
        json!({"status":"completed","output":unique}),
        &save_stage,
    )?;
    if duplicates > 0 {
        warnings.push(format!("Removed {duplicates} exact duplicate pair(s) before review; originals remain in writer artifacts."));
    }
    let mut review_prompt = format!(
        "Review this operational concept design and its candidate contrastive dataset. Design:\n{}\nDataset:\n{}\nReturn accepted_ids for pairs usable without edits, flags with specific semantic confounds or duplicate issues, repair_requests for salvageable pairs with targeted instructions, and overall warnings. Match positive/negative sign to the design. Detect near-duplicates as well as superficial lexical shortcuts, length/style confounds, unbalanced refusals and accidental scenario leakage. Scenario families will be split as groups locally; do not assign train/test. Every ID in your output must already exist in the candidate dataset. An accepted pair must not also require repair. Do not reject a concept merely because its effects may be weak or unusual; those are advisory. Do not write replacement pairs in this review.",
        serde_json::to_string(&design)?,
        serde_json::to_string(&unique)?
    );
    if extraction.method == crate::extraction::Method::Paper {
        review_prompt.push_str("\nThis is raw fixed-readout extraction: only each standalone 8-25 word pole statement is decoded, followed by a common suffix. Check that both poles make sense without the messages. Check coverage of multiple positive subtypes and multiple non-concept confound controls, not just a pleasant opposite. Flag label keywords that allow superficial classification; prefer situations that imply the construct without naming it. Do not turn a representation dataset into assistant advice or an answer-style contrast.");
    }
    let review = call_stage(
        client,
        "review",
        &roles.reviewer,
        review_prompt,
        review_schema(),
        &mut completed_stages,
        cancel.clone(),
        events.clone(),
        &save_stage,
        &|value| validate_review(value, &unique),
    )
    .await?;
    validate_review(&review, &unique)?;
    let accepted: BTreeSet<&str> = review["accepted_ids"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_str)
        .collect();
    let mut dataset: Vec<Value> = unique
        .iter()
        .filter(|p| accepted.contains(p["id"].as_str().unwrap_or("")))
        .cloned()
        .collect();
    for warning in review["warnings"].as_array().unwrap() {
        warnings.push(warning.as_str().unwrap().to_owned());
    }
    let requests = review["repair_requests"].as_array().unwrap();
    if !requests.is_empty() {
        let targets: BTreeSet<&str> = requests.iter().filter_map(|r| r["id"].as_str()).collect();
        let originals: Vec<&Value> = unique
            .iter()
            .filter(|p| targets.contains(p["id"].as_str().unwrap_or("")))
            .collect();
        let mut prompt = format!(
            "Make only the targeted repairs requested by the reviewer. Preserve each pair's existing id and family, retain shared context, and keep positive/negative sign aligned with the design. Do not fabricate additional pairs. Return a pairs array. If a repair is not possible, omit that ID; smaller usable datasets are allowed with a warning. Design:\n{}\nOriginal pairs:\n{}\nTargeted requests:\n{}",
            serde_json::to_string(&design)?,
            serde_json::to_string(&originals)?,
            serde_json::to_string(requests)?
        );
        if extraction.method == crate::extraction::Method::Paper {
            prompt.push_str("\nPreserve the paper-style format: self-contained 8-25 word naturalistic pole statements, no readout suffix, no reliance on common messages, no assistant advice. Preserve diversity of nuisance/control classes. Prefer indirect situations over explicit concept-label words.");
        }
        let repairs = call_stage(
            client,
            "repair",
            &roles.reviewer,
            prompt,
            writer_schema(),
            &mut completed_stages,
            cancel.clone(),
            events.clone(),
            &save_stage,
            &|value| validate_repairs(value, &originals),
        )
        .await?;
        let mut repaired_ids = BTreeSet::new();
        for pair in repairs["pairs"].as_array().unwrap() {
            let id = pair["id"].as_str().context("Repair id missing")?;
            ensure!(
                targets.contains(id),
                "Reviewer repair fabricated unknown id {id}"
            );
            ensure!(
                repaired_ids.insert(id.to_owned()),
                "Reviewer repair duplicated id {id}"
            );
            dataset.push(pair.clone());
        }
        if repaired_ids.len() < targets.len() {
            warnings.push(format!("{} requested repairs were not supplied; original and review artifacts were retained.",targets.len()-repaired_ids.len()));
        }
    }
    let (dataset, family_aliases) = canonicalize_families(&dataset, &design)?;
    if !family_aliases.is_empty() {
        warnings.push(format!("Canonicalized {} shortened/case-variant family labels against the designer's declared families before splitting. Original writer/revision outputs remain unchanged.", family_aliases.len()));
    }
    let (dataset, duplicates) = deduplicate_pairs(&dataset)?;
    if duplicates > 0 {
        warnings.push(format!("Removed {duplicates} duplicate repaired pair(s)."));
    }
    ensure!(
        !dataset.is_empty(),
        "Review produced no usable pairs. Edit the dataset or retry review; no replacement data was fabricated."
    );
    if dataset.len() < 3 * PAIRS_PER_WRITER {
        warnings.push(format!("Dataset contains {} usable pairs, below the 96-pair target. Separation estimates may be unstable.",dataset.len()));
    }
    let families: BTreeSet<_> = dataset
        .iter()
        .filter_map(|p| p["family"].as_str())
        .collect();
    if families.len() < 4 {
        warnings.push(format!(
            "Only {} scenario families remain; a grouped held-out diagnostic will be limited.",
            families.len()
        ));
    }
    for pair in &dataset {
        if pair["positive"] == pair["negative"] {
            warnings.push(format!(
                "Pair {} has identical poles; it contributes no directional contrast.",
                pair["id"].as_str().unwrap()
            ));
        }
    }
    // The steering module owns canonical family-grouped split assignment.
    let dataset = Value::Array(dataset);
    let final_stage = json!({"status":"completed","output":dataset,"warnings":warnings,"algorithm":"contrastive-factory-v1","family_normalization":"designer-exact-or-unique-short-name-v1","family_aliases":family_aliases,"target_pairs":96,"sign":"positive minus negative"});
    save(&mut completed_stages, "dataset", final_stage, &save_stage)?;
    let _=events.send(json!({"stage":"dataset","status":"completed","pairs":dataset.as_array().unwrap().len(),"warnings":warnings}));
    Ok(FactoryOutput {
        design,
        dataset,
        review,
        roles,
        warnings,
        stages: completed_stages,
    })
}

/// Resolve a writer's shortened `Name` against a uniquely declared
/// `Name: description` family. Never use arbitrary prefix or semantic matching:
/// ambiguous or genuinely new families must retain their distinct labels.
/// Only grouping metadata changes; original model outputs stay in checkpoints.
pub fn canonicalize_families(
    dataset: &[Value],
    design: &Value,
) -> Result<(Vec<Value>, Vec<Value>)> {
    let declared = design["scenario_families"]
        .as_array()
        .context("design is missing scenario families")?
        .iter()
        .map(|value| value.as_str().context("family must be text"))
        .collect::<Result<Vec<_>>>()?;
    let normalize = |text: &str| {
        text.split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    };
    let mut output = dataset.to_vec();
    let mut changes = Vec::new();
    for pair in &mut output {
        let original = pair["family"]
            .as_str()
            .context("pair family must be text")?
            .to_owned();
        let name = normalize(&original);
        let exact: Vec<_> = declared
            .iter()
            .filter(|family| normalize(family) == name)
            .collect();
        let matches: Vec<_> = if exact.is_empty() {
            declared
                .iter()
                .filter(|family| {
                    family
                        .split_once(':')
                        .is_some_and(|(short, _)| normalize(short) == name)
                })
                .collect()
        } else {
            exact
        };
        if matches.len() == 1 && original != **matches[0] {
            let canonical = *matches[0];
            changes.push(json!({"pair_id":pair["id"],"from":original,"to":canonical}));
            pair["family"] = json!(canonical);
        }
    }
    Ok((output, changes))
}

#[allow(clippy::too_many_arguments)] // Explicit stage inputs are bound into the durable hash.
async fn call_stage<F>(
    client: &CodexClient,
    stage: &str,
    model: &str,
    prompt: String,
    schema: Value,
    stages: &mut BTreeMap<String, Value>,
    cancel: Arc<AtomicBool>,
    events: UnboundedSender<Value>,
    save_stage: &F,
    validate_content: &(dyn Fn(&Value) -> Result<()> + Sync),
) -> Result<Value>
where
    F: Fn(&str, &Value) -> Result<()> + Send + Sync,
{
    let input_hash = hex::encode(Sha256::digest(serde_json::to_vec(
        &json!({"factory_version":1,"stage":stage,"model":model,"prompt":prompt,"schema":schema}),
    )?));
    if let Some(checkpoint) = stages.get(stage)
        && (checkpoint["edited"] == true
            || (checkpoint["status"] == "completed" && checkpoint["input_hash"] == input_hash))
    {
        let output = checkpoint["output"].clone();
        validate_schema(&output, &schema)
            .with_context(|| format!("Saved {stage} stage no longer validates"))?;
        validate_stage_semantics(stage, &output)?;
        validate_content(&output)?;
        let _ = events.send(json!({"stage":stage,"status":"reused","model":model}));
        return Ok(output);
    }
    let mut current_prompt = prompt.clone();
    let mut attempt_ids = Vec::new();
    for attempt in 0..=2 {
        ensure!(
            !cancel.load(Ordering::Relaxed),
            "Dataset generation cancelled before {stage}"
        );
        let attempt_id = format!("{stage}.attempt_{}", Uuid::new_v4());
        let mut started = json!({"status":"running","input_hash":input_hash,"stage":stage,"model":model,"prompt":current_prompt,"schema":schema,"repair_attempt":attempt});
        save(stages, &attempt_id, started.clone(), save_stage)?;
        let _ = events
            .send(json!({"stage":stage,"status":"running","model":model,"repair_attempt":attempt}));
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Value>();
        let mut generation =
            Box::pin(client.run_json(model, &current_prompt, schema.clone(), cancel.clone(), tx));
        let mut partial = String::new();
        let mut persisted_bytes = 0;
        let mut persisted_at = std::time::Instant::now();
        let result = loop {
            tokio::select! {
                result=&mut generation => break result,
                Some(mut event)=rx.recv() => {
                    if let Some(delta)=event["delta"].as_str() {
                        partial.push_str(delta);
                        if partial.len()-persisted_bytes>=4096 || persisted_at.elapsed().as_secs()>=2 {
                            started["partial_raw"]=json!(partial);
                            save(stages,&attempt_id,started.clone(),save_stage)?;
                            persisted_bytes=partial.len();persisted_at=std::time::Instant::now();
                        }
                    }
                    event["stage"]=json!(stage);
                    let _=events.send(event);
                }
            }
        };
        drop(generation);
        // Preserve the latest partial text on normal failure/cancellation too.
        if !partial.is_empty() {
            started["partial_raw"] = json!(partial);
        }
        let output = match result {
            Ok(output) => output,
            Err(error) => {
                let mut failed = started;
                failed["status"] = json!("failed");
                failed["error"] = json!(error.to_string());
                save(stages, &attempt_id, failed, save_stage)?;
                return Err(error).with_context(|| {
                    format!("{stage} failed; saved stages can be reused by explicit retry")
                });
            }
        };
        let mut artifact = started;
        artifact["status"] = json!(output.status);
        artifact["agent"] = serde_json::to_value(&output)?;
        save(stages, &attempt_id, artifact, save_stage)?;
        attempt_ids.push(attempt_id);
        ensure!(
            output.status == "completed",
            "{stage} {}: {}",
            output.status,
            output
                .error
                .as_deref()
                .unwrap_or("Codex turn did not complete")
        );
        let validation_error = output.validation_error.or_else(|| {
            validate_stage_semantics(stage, &output.output)
                .and_then(|()| validate_content(&output.output))
                .err()
                .map(|e| e.to_string())
        });
        if let Some(error) = validation_error {
            if attempt == 2 {
                bail!(
                    "{stage} returned malformed output after two repair attempts: {error}. Originals are saved; edit or retry this stage."
                );
            }
            let _=events.send(json!({"stage":stage,"status":"repairing_schema","error":error,"repair_attempt":attempt+1}));
            current_prompt = format!(
                "{prompt}\n\nYour previous structured answer failed validation: {error}. Repair its shape/content to satisfy the provided schema. Preserve valid content. Previous answer:\n{}",
                output.raw
            );
        } else {
            let checkpoint = json!({"status":"completed","input_hash":input_hash,"model":model,"output":output.output,"attempt_ids":attempt_ids,"runtime":output.runtime});
            save(stages, stage, checkpoint, save_stage)?;
            let _ = events.send(json!({"stage":stage,"status":"completed","model":model}));
            return Ok(output.output);
        }
    }
    unreachable!()
}
fn save<F>(
    stages: &mut BTreeMap<String, Value>,
    name: &str,
    value: Value,
    callback: &F,
) -> Result<()>
where
    F: Fn(&str, &Value) -> Result<()>,
{
    callback(name, &value)
        .with_context(|| format!("Persist {name} checkpoint before advancing"))?;
    stages.insert(name.into(), value);
    Ok(())
}
fn validate_roles(roles: &RoleAssignments, models: &[ModelInfo]) -> Result<()> {
    ensure!(
        roles.writers.len() == 3,
        "Exactly three independent writer roles are required"
    );
    for model in std::iter::once(&roles.designer)
        .chain(roles.writers.iter())
        .chain(std::iter::once(&roles.reviewer))
    {
        ensure!(
            models.iter().any(|m| &m.model == model || &m.id == model),
            "Assigned Codex model {model} is not currently listed; edit role assignments or restore model access"
        );
    }
    Ok(())
}
fn validate_stage_semantics(stage: &str, output: &Value) -> Result<()> {
    if stage.starts_with("writer_") || stage == "repair" {
        for pair in output["pairs"].as_array().context("Missing pair array")? {
            validate_pair(pair)?;
        }
    }
    if stage == "design" {
        ensure!(
            output["neutral_prompts"]
                .as_array()
                .is_some_and(|a| a.len() == 3),
            "Exactly three neutral preview prompts are required"
        );
    }
    Ok(())
}
fn validate_pair(pair: &Value) -> Result<()> {
    for k in ["id", "family", "positive", "negative"] {
        ensure!(
            pair[k].as_str().is_some_and(|s| !s.trim().is_empty()),
            "Pair {k} must not be blank"
        );
    }
    let messages = pair["messages"]
        .as_array()
        .context("Pair messages missing")?;
    ensure!(
        !messages.is_empty(),
        "Pair needs common conversation context"
    );
    ensure!(
        messages.last().is_some_and(|m| m["role"] == "user"),
        "Pair context must end with a user message, before the assistant completion"
    );
    for m in messages {
        ensure!(
            m["content"].as_str().is_some_and(|s| !s.trim().is_empty()),
            "Context message must not be blank"
        );
    }
    Ok(())
}
fn validate_review(review: &Value, pairs: &[Value]) -> Result<()> {
    let ids: BTreeSet<&str> = pairs.iter().filter_map(|p| p["id"].as_str()).collect();
    let mut accepted = BTreeSet::new();
    for id in review["accepted_ids"]
        .as_array()
        .context("Accepted ids missing")?
    {
        let id = id.as_str().context("Accepted id not string")?;
        ensure!(ids.contains(id), "Review invented unknown accepted id {id}");
        ensure!(accepted.insert(id), "Review duplicated accepted id {id}");
    }
    let mut repair_ids = BTreeSet::new();
    for r in review["repair_requests"]
        .as_array()
        .context("Repair requests missing")?
    {
        let id = r["id"].as_str().context("Repair id missing")?;
        ensure!(ids.contains(id), "Review invented repair id {id}");
        ensure!(
            !accepted.contains(id),
            "Review both accepted and requested repair for {id}"
        );
        ensure!(
            repair_ids.insert(id),
            "Review duplicated repair request {id}"
        );
    }
    for r in review["flags"].as_array().context("Review flags missing")? {
        ensure!(
            ids.contains(r["id"].as_str().unwrap_or("")),
            "Review flag refers to unknown id"
        );
    }
    Ok(())
}
fn validate_repairs(output: &Value, originals: &[&Value]) -> Result<()> {
    let mut ids = BTreeSet::new();
    for pair in output["pairs"].as_array().context("Repair pairs missing")? {
        let id = pair["id"].as_str().context("Repair id missing")?;
        let original = originals
            .iter()
            .find(|p| p["id"] == id)
            .with_context(|| format!("Repair invented unrequested ID {id}"))?;
        ensure!(ids.insert(id), "Repair duplicated ID {id}");
        ensure!(
            pair["family"] == original["family"],
            "Repair changed scenario family for {id}"
        );
    }
    Ok(())
}
fn deduplicate_pairs(pairs: &[Value]) -> Result<(Vec<Value>, usize)> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    let mut removed = 0;
    for pair in pairs {
        validate_pair(pair)?;
        let key = serde_json::to_string(
            &json!({"messages":pair["messages"],"positive":pair["positive"],"negative":pair["negative"]}),
        )?;
        if seen.insert(key) {
            out.push(pair.clone())
        } else {
            removed += 1;
        }
    }
    Ok((out, removed))
}
fn string() -> Value {
    json!({"type":"string","minLength":1,"maxLength":8192})
}
fn strings(max: usize) -> Value {
    json!({"type":"array","items":string(),"maxItems":max})
}
fn object(properties: Value) -> Value {
    let required: Vec<String> = properties.as_object().unwrap().keys().cloned().collect();
    json!({"type":"object","properties":properties,"required":required,"additionalProperties":false})
}
pub fn design_schema() -> Value {
    object(
        json!({"interpretation":string(),"positive_pole":string(),"negative_pole":string(),"confounds":strings(32),"scenario_families":strings(32),"neutral_prompts":{"type":"array","items":string(),"minItems":3,"maxItems":3},"alternatives":strings(16)}),
    )
}
pub fn pair_schema() -> Value {
    object(
        json!({"id":string(),"family":string(),"messages":{"type":"array","minItems":1,"maxItems":32,"items":object(json!({"role":{"type":"string","enum":["system","user","assistant"]},"content":string()}))},"positive":string(),"negative":string()}),
    )
}
pub fn writer_schema() -> Value {
    object(json!({"pairs":{"type":"array","items":pair_schema(),"maxItems":128}}))
}
pub fn review_schema() -> Value {
    object(
        json!({"accepted_ids":strings(128),"flags":{"type":"array","items":object(json!({"id":string(),"issue":string()})),"maxItems":256},"repair_requests":{"type":"array","items":object(json!({"id":string(),"instruction":string()})),"maxItems":128},"warnings":strings(64)}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn model(id: &str, default: bool) -> ModelInfo {
        ModelInfo {
            id: id.into(),
            model: id.into(),
            display_name: id.into(),
            is_default: default,
            input_modalities: vec!["text".into()],
            hidden: false,
            default_reasoning_effort: None,
        }
    }
    fn pair(id: &str) -> Value {
        json!({"id":id,"family":"recipe","messages":[{"role":"user","content":"Make soup."}],"positive":"Yes! Soup!","negative":"No soup."})
    }
    #[test]
    fn assignments_start_with_default_then_provider_order() {
        let r = default_roles(&[
            model("other", false),
            model("default", true),
            model("third", false),
            model("fourth", false),
        ])
        .unwrap();
        assert_eq!(r.writers, ["default", "other", "third"]);
        assert_eq!(r.designer, "default");
        let r = default_roles(&[model("one", true)]).unwrap();
        assert_eq!(r.writers, ["one", "one", "one"]);
    }
    #[test]
    fn retry_only_replaces_unavailable_unfinished_roles() {
        let previous = RoleAssignments {
            designer: "available".into(),
            writers: vec![
                "available".into(),
                "missing-sol".into(),
                "missing-luna".into(),
            ],
            reviewer: "available".into(),
        };
        let mut stages = BTreeMap::from([
            ("design".into(), json!({"status":"completed"})),
            ("writer_1".into(), json!({"status":"completed"})),
        ]);
        let models = [
            model("available", true),
            model("second", false),
            model("third", false),
        ];
        let (roles, changes) = retry_roles(&previous, &stages, &models).unwrap();
        assert_eq!(roles.designer, previous.designer);
        assert_eq!(roles.writers, ["available", "second", "third"]);
        assert_eq!(roles.reviewer, previous.reviewer);
        assert_eq!(changes.len(), 2);
        stages.insert("writer_2".into(), json!({"status":"completed"}));
        assert!(
            retry_roles(&previous, &stages, &models)
                .unwrap_err()
                .to_string()
                .contains("keeping its saved output")
        );
    }
    #[test]
    fn exact_dedup_preserves_original_first_id() {
        let p = pair("b");
        let (out, n) = deduplicate_pairs(&[pair("a"), p]).unwrap();
        assert_eq!(n, 1);
        assert_eq!(out[0]["id"], "a");
    }
    #[test]
    fn short_family_aliases_cannot_leak_across_the_grouped_split() {
        let design = json!({"scenario_families":["Ankle: strain versus stiffness", "Letters: grief versus recollection"]});
        let input: Vec<_> = [
            "Ankle",
            "Ankle: strain versus stiffness",
            "Letters",
            "Letters: grief versus recollection",
        ]
        .iter()
        .enumerate()
        .map(|(index, family)| {
            let mut p = pair(&index.to_string());
            p["family"] = json!(family);
            p["positive"] = json!(format!("Different completion {index}"));
            p
        })
        .collect();
        let (canonical, aliases) = canonicalize_families(&input, &design).unwrap();
        assert_eq!(aliases.len(), 2);
        assert_eq!(input[0]["family"], "Ankle");
        let prepared = crate::steering::prepare_dataset(&json!(canonical)).unwrap();
        let pairs = prepared["pairs"].as_array().unwrap();
        assert_eq!(pairs[0]["split"], pairs[1]["split"]);
        assert_eq!(pairs[2]["split"], pairs[3]["split"]);
        for (before, after) in input.iter().zip(&canonical) {
            for field in ["id", "messages", "positive", "negative"] {
                assert_eq!(before[field], after[field]);
            }
        }
    }
    #[test]
    fn ambiguous_family_aliases_and_new_families_are_not_guessed() {
        let design = json!({"scenario_families":["Room: quiet", "Room: noisy"]});
        for family in ["Room", "Roommate", "Room: different"] {
            let mut p = pair("a");
            p["family"] = json!(family);
            let (out, changes) = canonicalize_families(&[p.clone()], &design).unwrap();
            assert_eq!(out, [p]);
            assert!(changes.is_empty());
        }
    }
    #[test]
    fn family_canonicalization_is_idempotent_and_whitespace_insensitive() {
        let design = json!({"scenario_families":["A Room: an ordinary scene"]});
        let mut p = pair("a");
        p["family"] = json!(" A  ROOM ");
        let (once, changes) = canonicalize_families(&[p], &design).unwrap();
        assert_eq!(changes.len(), 1);
        let (twice, changes) = canonicalize_families(&once, &design).unwrap();
        assert_eq!(once, twice);
        assert!(changes.is_empty());
    }
    #[test]
    fn dedup_does_not_erase_case_or_whitespace_concepts() {
        let mut p = pair("b");
        p["positive"] = json!("YES! Soup!");
        let mut q = pair("c");
        q["positive"] = json!("Yes!  Soup!");
        assert_eq!(deduplicate_pairs(&[pair("a"), p, q]).unwrap().0.len(), 3);
    }
    #[test]
    fn opposite_poles_not_deduplicated() {
        let mut p = pair("b");
        let a = p["positive"].clone();
        p["positive"] = p["negative"].clone();
        p["negative"] = a;
        assert_eq!(deduplicate_pairs(&[pair("a"), p]).unwrap().0.len(), 2);
    }
    #[test]
    fn malformed_messages_fail() {
        let mut p = pair("a");
        p["messages"][0]["role"] = json!("assistant");
        assert!(validate_pair(&p).is_err());
    }
    #[test]
    fn invented_or_conflicting_review_ids_fail() {
        let valid = json!({"accepted_ids":["a"],"flags":[],"repair_requests":[],"warnings":[]});
        assert!(validate_review(&valid, &[pair("a")]).is_ok());
        let mut bad = valid.clone();
        bad["accepted_ids"] = json!(["fiction"]);
        assert!(validate_review(&bad, &[pair("a")]).is_err());
        bad = valid;
        bad["repair_requests"] = json!([{"id":"a","instruction":"fix"}]);
        assert!(validate_review(&bad, &[pair("a")]).is_err());
    }
    #[test]
    fn persistence_failure_does_not_mark_complete() {
        let mut stages = BTreeMap::new();
        assert!(
            save(&mut stages, "design", json!({"output":"x"}), &|_, _| bail!(
                "disk full"
            ))
            .is_err()
        );
        assert!(stages.is_empty());
    }
    #[test]
    fn schemas_reject_malformed_but_allow_smaller_datasets() {
        assert!(validate_schema(&json!({"pairs":[pair("a")]}), &writer_schema()).is_ok());
        assert!(validate_schema(&json!({"pairs":[{"id":"a"}]}), &writer_schema()).is_err());
    }
    #[cfg(unix)]
    #[tokio::test]
    async fn authentication_failure_is_durable_and_only_explicit_retry_recovers() {
        use crate::store::Store;

        fn persist(store: &Store, name: &str, value: &Value) -> Result<()> {
            let hash = store.artifacts().put_json(value)?;
            store.put(
                "jobs",
                name,
                &json!({"id":name,"status":value["status"],"checkpoint_hash":hash}),
            )
        }

        let (directory, client) = crate::codex::test_client("auth_recovery");
        let store_path = directory.path().join("checkpoints");
        let store = Store::open(&store_path).unwrap();
        let mut checkpoints = BTreeMap::new();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let schema = json!({"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false});
        let error = call_stage(
            &client,
            "test",
            "fake-model",
            "Test".into(),
            schema.clone(),
            &mut checkpoints,
            Arc::new(AtomicBool::new(false)),
            tx.clone(),
            &|name, value| persist(&store, name, value),
            &|_| Ok(()),
        )
        .await
        .unwrap_err();
        let error = format!("{error:#}");
        assert!(error.contains("401"));
        assert!(error.contains("explicit retry"));
        assert!(!checkpoints.contains_key("test"));
        assert_eq!(checkpoints.len(), 1);
        let (failed_id, failed) = checkpoints.iter().next().unwrap();
        let (failed_id, failed) = (failed_id.clone(), failed.clone());
        assert!(failed_id.starts_with("test.attempt_"));
        assert_eq!(failed["status"], "failed");
        assert_eq!(failed["repair_attempt"], 0);
        assert!(
            failed["error"]
                .as_str()
                .unwrap()
                .contains("expired authentication")
        );
        while let Ok(event) = rx.try_recv() {
            assert_ne!(event["status"], "repairing_schema");
            assert_ne!(event["status"], "completed");
        }
        let methods = crate::codex::test_request_methods(directory.path());
        assert_eq!(methods.iter().filter(|m| *m == "turn/start").count(), 1);

        // Restore from the real SQLite/artifact store, not the in-memory map.
        drop(checkpoints);
        drop(store);
        let store = Store::open(&store_path).unwrap();
        let mut checkpoints: BTreeMap<String, Value> = store
            .list("jobs")
            .unwrap()
            .into_iter()
            .map(|record| {
                (
                    record["id"].as_str().unwrap().to_owned(),
                    store
                        .artifacts()
                        .read_json(record["checkpoint_hash"].as_str().unwrap())
                        .unwrap(),
                )
            })
            .collect();
        assert_eq!(checkpoints.len(), 1);
        assert_eq!(checkpoints[&failed_id], failed);
        assert_eq!(
            crate::codex::test_request_methods(directory.path()),
            methods
        );

        // A test-local marker simulates restored credentials; no real auth changes.
        std::fs::write(directory.path().join("auth-recovered"), b"yes").unwrap();
        let output = call_stage(
            &client,
            "test",
            "fake-model",
            "Test".into(),
            schema.clone(),
            &mut checkpoints,
            Arc::new(AtomicBool::new(false)),
            tx.clone(),
            &|name, value| persist(&store, name, value),
            &|_| Ok(()),
        )
        .await
        .unwrap();
        assert_eq!(output, json!({"ok":true}));
        assert_eq!(checkpoints[&failed_id], failed);
        assert_eq!(checkpoints.len(), 3); // Old failure, new attempt, completed stage.
        let successful_id = checkpoints["test"]["attempt_ids"][0].as_str().unwrap();
        assert_ne!(successful_id, failed_id);
        assert_eq!(
            checkpoints["test"]["attempt_ids"].as_array().unwrap().len(),
            1
        );
        assert_eq!(checkpoints[successful_id]["repair_attempt"], 0);
        let methods = crate::codex::test_request_methods(directory.path());
        assert_eq!(methods.iter().filter(|m| *m == "turn/start").count(), 2);

        let completed = checkpoints.clone();
        call_stage(
            &client,
            "test",
            "fake-model",
            "Test".into(),
            schema,
            &mut checkpoints,
            Arc::new(AtomicBool::new(false)),
            tx,
            &|name, value| persist(&store, name, value),
            &|_| Ok(()),
        )
        .await
        .unwrap();
        assert_eq!(checkpoints, completed);
        assert_eq!(
            crate::codex::test_request_methods(directory.path()),
            methods
        );
    }
    #[cfg(unix)]
    #[tokio::test]
    async fn malformed_stage_repairs_twice_and_resume_reuses_without_cloud_call() {
        let (directory, client) = crate::codex::test_client("two_repairs");
        let mut checkpoints = BTreeMap::new();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let schema = json!({"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false});
        let saves = std::sync::Mutex::new(Vec::new());
        let save = |name: &str, value: &Value| {
            saves.lock().unwrap().push((name.to_owned(), value.clone()));
            Ok(())
        };
        let output = call_stage(
            &client,
            "test",
            "fake-model",
            "Test".into(),
            schema.clone(),
            &mut checkpoints,
            Arc::new(AtomicBool::new(false)),
            tx.clone(),
            &save,
            &|_| Ok(()),
        )
        .await
        .unwrap();
        assert_eq!(output, json!({"ok":true}));
        assert_eq!(
            std::fs::read_to_string(directory.path().join("counter")).unwrap(),
            "3"
        );
        assert_eq!(
            checkpoints["test"]["attempt_ids"].as_array().unwrap().len(),
            3
        );
        let restored = checkpoints.clone();
        let output = call_stage(
            &client,
            "test",
            "fake-model",
            "Test".into(),
            schema,
            &mut checkpoints,
            Arc::new(AtomicBool::new(false)),
            tx,
            &save,
            &|_| Ok(()),
        )
        .await
        .unwrap();
        assert_eq!(output, json!({"ok":true}));
        assert_eq!(checkpoints, restored);
        assert_eq!(
            std::fs::read_to_string(directory.path().join("counter")).unwrap(),
            "3"
        );
    }
}
