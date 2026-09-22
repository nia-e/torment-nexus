use anyhow::{Context, Result, bail, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::PathBuf,
    sync::{
        Arc, Mutex as StdMutex,
        atomic::{AtomicBool, Ordering},
    },
};
use tokio::sync::{Mutex, OwnedSemaphorePermit, RwLock, Semaphore, broadcast, mpsc};
use torment_nexus::{
    codex::CodexClient,
    download,
    engine::Engine,
    extraction::{Extraction, Method, preview_percentages},
    factory, self_tools, steering,
    store::{Store, now_seconds},
    vector_bundle::{json_metadata_eq, validate_manifest_artifacts},
};
use uuid::Uuid;

const ENGINE_REVISION: &str = "922be44aa6ac81b46f092716351cddff1c1733a7";
fn id() -> String {
    Uuid::new_v4().to_string()
}
fn field<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value[key]
        .as_str()
        .filter(|s| !s.is_empty())
        .with_context(|| format!("missing {key}"))
}

fn run_transcript(run: &Value) -> Result<Vec<Value>> {
    let mut messages = run["messages"]
        .as_array()
        .context("invalid run messages")?
        .clone();
    if let Some(reply) = run["reply_messages"]
        .as_array()
        .filter(|reply| !reply.is_empty())
    {
        messages.extend(reply.iter().cloned());
    } else {
        messages
            .push(json!({"role":"assistant","content":run["output"].as_str().unwrap_or_default()}));
    }
    Ok(messages)
}

/// Apply a complete coefficient snapshot to the frozen (vector, layer) members.
/// Old clients may omit the layer only when the vector has exactly one member.
fn control_axes(frozen: &Value, coefficients: &[Value]) -> Result<Value> {
    let mut axes = frozen.as_array().context("invalid frozen axes")?.clone();
    ensure!(
        coefficients.len() == axes.len(),
        "controls must contain a complete snapshot of frozen vector/layer pairs"
    );
    let mut seen = HashSet::new();
    for coefficient in coefficients {
        let vector_id = field(coefficient, "vector_id")?;
        let layer = match coefficient.get("layer") {
            Some(value) => value
                .as_u64()
                .context("coefficient layer must be an unsigned integer")?,
            None => {
                let candidates: Vec<_> = axes
                    .iter()
                    .filter(|axis| axis["vector_id"] == vector_id)
                    .collect();
                ensure!(
                    !candidates.is_empty(),
                    "vector selection is frozen for this response"
                );
                ensure!(
                    candidates.len() == 1,
                    "layer is required for a multi-layer concept"
                );
                candidates[0]["layer"]
                    .as_u64()
                    .context("invalid frozen layer")?
            }
        };
        ensure!(
            seen.insert((vector_id, layer)),
            "duplicate coefficient for vector/layer pair"
        );
        let axis = axes
            .iter_mut()
            .find(|axis| axis["vector_id"] == vector_id && axis["layer"].as_u64() == Some(layer))
            .context("vector/layer selection is frozen for this response")?;
        ensure!(
            coefficient["percent"].as_f64().is_some_and(f64::is_finite),
            "coefficient must be finite"
        );
        axis["percent"] = coefficient["percent"].clone();
    }
    Ok(json!(axes))
}

fn validate_vector_manifest(vector: &Value, manifest: &Value, fingerprint: &str) -> Result<()> {
    ensure!(
        manifest["format"] == "torment-vector-v1",
        "unsupported vector format"
    );
    ensure!(
        manifest["runtime"]["engine_revision"] == ENGINE_REVISION,
        "vector was extracted by an unverified engine revision"
    );
    ensure!(
        matches!(
            manifest["algorithm"].as_str(),
            Some(
                torment_nexus::extraction::LEGACY_ALGORITHM
                    | torment_nexus::extraction::PAPER_ALGORITHM
            )
        ) && manifest["sign_convention"] == "mean_positive_minus_mean_negative",
        "unsupported extraction algorithm/sign convention"
    );
    ensure!(
        Extraction::from_record(&manifest["recipe"])?.algorithm() == manifest["algorithm"],
        "recipe extraction method differs from manifest"
    );
    ensure!(
        Extraction::from_record(vector)? == Extraction::from_record(&manifest["recipe"])?,
        "vector extraction settings differ from recipe"
    );
    ensure!(
        vector["preview_mode"].as_str().unwrap_or("standard")
            == manifest["recipe"]["preview_mode"]
                .as_str()
                .unwrap_or("standard"),
        "vector preview mode differs from recipe"
    );
    ensure!(
        manifest["model_fingerprint"] == fingerprint
            && json_metadata_eq(&manifest["layers"], &vector["layers"])
            && manifest["selected_layer"] == vector["selected_layer"],
        "vector manifest disagrees with index"
    );
    let layers = vector["layers"]
        .as_array()
        .context("missing layer metadata")?;
    ensure!(
        layers
            .iter()
            .any(|layer| layer["layer"] == vector["selected_layer"] && layer["usable"] != false),
        "default layer is absent or unusable"
    );
    Ok(())
}

fn runtime_identity(runtime: &Value) -> Result<Value> {
    let keys = [
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
    let mut identity = json!({});
    for key in keys {
        ensure!(
            !runtime[key].is_null(),
            "worker omitted stable runtime identity field {key}; rebuild the worker"
        );
        identity[key] = runtime[key].clone();
    }
    Ok(identity)
}

fn neutral_prompts(recipe: &Value) -> Result<Vec<String>> {
    let prompts = recipe["design"]
        .get("neutral_prompts")
        .or_else(|| recipe["design"].get("preview_prompts"))
        .and_then(Value::as_array)
        .context("designer did not supply neutral preview prompts")?;
    ensure!(
        prompts.len() == 3,
        "exactly three neutral preview prompts are required"
    );
    prompts
        .iter()
        .map(|p| {
            p.as_str()
                .filter(|s| !s.trim().is_empty())
                .map(str::to_owned)
                .context("neutral preview prompt must be nonempty text")
        })
        .collect()
}

fn new_concept_settings(mut request: Value) -> Result<Value> {
    if request.get("extraction").is_none_or(Value::is_null) {
        request["extraction"] = serde_json::to_value(Extraction::default())?;
    }
    if request.get("preview_mode").is_none_or(Value::is_null) {
        request["preview_mode"] = json!("none");
    }
    Extraction::from_record(&request)?;
    preview_percentages(&request)?;
    Ok(request)
}

impl App {
    async fn start_generation(self: &Arc<Self>, mut request: Value) -> Result<Value> {
        let permit = self.local_permit()?;
        if let Some(baseline_id) = request["baseline_of"].as_str() {
            let previous = self.record("runs", baseline_id)?;
            request["model_id"] = previous["model_id"].clone();
            request["messages"] = previous["messages"].clone();
            request["sampling"] = previous["sampling"].clone();
            request["raw"] = previous["raw"].clone();
            request["axes"] = json!([]);
            request["self_modification"] = json!(false);
            // A baseline is not a new turn in the original conversation.
            request["conversation_id"] = Value::Null;
        }
        let model = self.record("models", field(&request, "model_id")?)?;
        let messages = request["messages"]
            .as_array()
            .context("messages must be an array")?;
        ensure!(
            !messages.is_empty() && messages.len() <= 1024,
            "invalid message count"
        );
        for message in messages {
            ensure!(
                matches!(
                    message["role"].as_str(),
                    Some("system" | "user" | "assistant")
                ),
                "unsupported role"
            );
            ensure!(
                message["content"].is_string(),
                "message content must be text"
            );
        }
        let axes = request.get("axes").cloned().unwrap_or_else(|| json!([]));
        self.load_vectors(&axes, field(&model, "fingerprint")?)?;
        if let Some(conversation) = request["conversation_id"].as_str() {
            self.record("conversations", conversation)?;
            ensure!(
                !self.record_deleted("conversations", conversation)?,
                "conversation was deleted; start a new conversation"
            );
        }
        let mut sampling = json!({"seed":42,"temperature":0.7,"top_p":0.95,"max_tokens":256});
        if let Some(settings) = request["sampling"].as_object() {
            for (key, value) in settings {
                sampling[key] = value.clone();
            }
        }
        ensure!(
            sampling.get("unbounded").is_none_or(Value::is_boolean),
            "unbounded must be a boolean"
        );
        ensure!(
            request
                .get("self_modification")
                .is_none_or(Value::is_boolean),
            "self_modification must be a boolean"
        );
        ensure!(
            sampling["temperature"]
                .as_f64()
                .is_some_and(|v| v.is_finite() && (0.0..=100.0).contains(&v)),
            "invalid temperature"
        );
        ensure!(
            sampling["top_p"]
                .as_f64()
                .is_some_and(|v| v.is_finite() && v > 0.0 && v <= 1.0),
            "invalid top_p"
        );
        ensure!(
            sampling["seed"]
                .as_u64()
                .is_some_and(|v| v <= u32::MAX as u64),
            "seed must be an unsigned 32-bit integer"
        );
        ensure!(
            sampling["max_tokens"]
                .as_u64()
                .is_some_and(|v| (1..=32768).contains(&v)),
            "max_tokens must be 1..32768"
        );
        let run_id = id();
        let job_id = self.create_job("generate", &request)?;
        let coefficients: Vec<Value> = axes
            .as_array()
            .unwrap()
            .iter()
            .map(|axis| json!({"vector_id":axis["vector_id"],"layer":axis["layer"],"percent":axis["percent"]}))
            .collect();
        let run = json!({"id":run_id,"created_at":now_seconds(),"job_id":job_id,"model_id":model["id"],"model_fingerprint":model["fingerprint"],
            "messages":messages,"axes":axes,"sampling":sampling,"raw":request["raw"].as_bool().unwrap_or(false),"output":"","output_token_count":0,
            "self_modification":request["self_modification"].as_bool().unwrap_or(false),"self_tools_available":request["self_modification"].as_bool().unwrap_or(false),"tool_calls":[],"context_rollovers":[],"permission_events":[],
            "status":"queued","requested_controls":[{"revision":0,"coefficients":coefficients,"source":"user","requested_at":now_seconds()}],"applied_controls":[],
            "conversation_id":request["conversation_id"],"baseline_of":request["baseline_of"],"error":null});
        self.store.put("runs", &run_id, &run)?;
        self.update("jobs", &job_id, |job| job["run_id"] = json!(run_id))?;
        self.start_job(job_id.clone(), Some(permit))?;
        Ok(json!({"run_id":run_id,"job_id":job_id}))
    }

    async fn run_generation(
        &self,
        job_id: &str,
        run_id: &str,
        cancellation: Arc<AtomicBool>,
    ) -> Result<()> {
        let run = self.record("runs", run_id)?;
        let result=async {
            let capabilities=self.ensure_model(job_id,field(&run,"model_id")?).await?;
            ensure!(run["self_tools_available"]!=true || capabilities["self_tools"]==true,"this worker lacks self-adjustment tools; rebuild or update the packaged engine");
            ensure!(run["sampling"]["unbounded"]!=true || capabilities["unbounded_output"]==true,"this worker lacks context rollover; rebuild or update the packaged engine");
            let fingerprint=field(&run,"model_fingerprint")?;
            let vectors=self.load_vectors(&run["axes"],fingerprint)?;
            let (depth,width)=(capabilities["n_layer"].as_u64().unwrap()as usize,capabilities["n_embd"].as_u64().unwrap()as usize);
            let rows=steering::mix_rows(&vectors,&run["axes"],fingerprint,depth,width)?;
            let diagnostics=steering::mix_diagnostics(&vectors,&run["axes"],fingerprint,depth,width)?;
            ensure!(!cancellation.load(Ordering::Acquire),"cancelled before generation");
            self.update("runs",run_id,|run|{
                run["status"]=json!("running");
                run["initial_mix_diagnostics"]=diagnostics.clone();
                run["requested_controls"][0]["diagnostics"]=diagnostics;
            })?;
            self.update("jobs",job_id,|job|job["engine_target"]=json!(run_id))?;
            self.progress(job_id,"generating",0.1)?;
            self.engine_state.write().await["status"]=json!("running");
            let engine=self.engine().await?;
            let mut messages=run["messages"].as_array().unwrap().clone();
            if run["self_tools_available"] == true {
                let instruction=self_tools::instruction(&self.mix_view(run_id)?);
                if messages[0]["role"] == "system" {
                    messages[0]["content"]=json!(format!("{}{}", messages[0]["content"].as_str().unwrap(),instruction));
                } else { messages.insert(0,json!({"role":"system","content":instruction})); }
            }
            let (_,mut events)=engine.send_cancellable(json!({"id":run_id,"op":"generate","messages":messages,"raw":run["raw"],"sampling":run["sampling"],"tools_enabled":run["self_tools_available"]==true,"controls":{"revision":0,"rows":rows}}), &cancellation).await?;
            let mut terminal=false;
            while let Some(event)=events.recv().await {
                match event["event"].as_str(){
                    Some("rendered")=>{
                        self.observe_runtime(&event["runtime"]).await;
                        let hash=self.store.artifacts().put_json(&event)?;
                        self.update("runs",run_id,|run|{run["prompt_hash"]=json!(hash);run["runtime"]=event["runtime"].clone();})?;
                    }
                    Some("token")=>{
                        self.update("runs",run_id,|run|{
                            let mut text=run["output"].as_str().unwrap_or_default().to_owned();text.push_str(event["text"].as_str().unwrap_or_default());
                            run["output"]=json!(text);run["output_token_count"]=json!(event["index"].as_u64().unwrap_or(0)+1);
                        })?;
                        self.emit("token",Some(job_id),Some(run_id),event.clone());
                    }
                    Some("applied")=>{
                        self.update("runs",run_id,|run|{
                            let diagnostics=run["requested_controls"].as_array().unwrap().iter()
                                .find(|request|request["revision"]==event["revision"])
                                .map(|request|request["diagnostics"].clone()).unwrap_or(Value::Null);
                            run["mix_diagnostics"]=json!({"revision":event["revision"],"first_token_index":event["first_token_index"],"geometry":diagnostics});
                            run["applied_controls"].as_array_mut().unwrap().push(event.clone());
                        })?;
                        self.emit("applied",Some(job_id),Some(run_id),event.clone());
                    }
                    Some("tool_call")=>{
                        let request=serde_json::from_str::<Value>(event["json"].as_str().unwrap_or_default());
                        let result=match request {
                            Ok(call)=>self.model_tool(call["name"].as_str().unwrap_or_default(),&call["arguments"],Some(run_id),"model").await,
                            Err(error)=>Err(error.into()),
                        };
                        let response=match result {Ok(value)=>json!({"ok":true,"result":value}),Err(error)=>json!({"ok":false,"error":format!("{error:#}")})};
                        let entry=json!({"id":event["tool_id"],"input":event["json"],"result":response,"after_token_index":event["after_token_index"],"created_at":now_seconds()});
                        self.update("runs",run_id,|run|run["tool_calls"].as_array_mut().unwrap().push(entry.clone()))?;
                        self.emit("tool_call",Some(job_id),Some(run_id),entry);
                        engine.execute(json!({"op":"tool_result","target":run_id,"tool_id":event["tool_id"],"result":response})).await?;
                    }
                    Some("context_rollover")=>{
                        self.update("runs",run_id,|run|run["context_rollovers"].as_array_mut().unwrap().push(event.clone()))?;
                        self.emit("context_rollover",Some(job_id),Some(run_id),event.clone());
                    }
                    Some("tool_continuation")=>{
                        self.update("runs",run_id,|run|{
                            if !run["tool_continuations"].is_array() {run["tool_continuations"]=json!([]);}
                            run["tool_continuations"].as_array_mut().unwrap().push(event.clone());
                        })?;
                    }
                    Some("done")=>{
                        terminal=true;
                        self.observe_runtime(&event["runtime"]).await;
                        if event["cancelled"]==true{cancellation.store(true,Ordering::Release);}
                        self.update("runs",run_id,|run|{run["output"]=event["output"].clone();run["output_token_count"]=event["tokens"].clone();run["runtime"]=event["runtime"].clone();run["reply_messages"]=event["reply_messages"].clone();})?;
                    }
                    Some("error")=>bail!("{}",event["error"].as_str().unwrap_or("inference error")),
                    _=>{}
                }
            }
            ensure!(terminal,"worker ended without a completion event");
            Ok(())
        }.await;
        if result.is_err() {
            // A failed artifact/database write can leave a decode command in flight.
            // Do not release the inference reservation with that command still alive.
            self.discard_worker().await;
            return result;
        }
        let mut state = self.engine_state.write().await;
        if state["model_id"].is_string() {
            state["status"] = json!("idle");
        }
        result
    }

    fn finalize_job(&self, job_id: &str, status: &str, error: Option<&str>) -> Result<()> {
        let _lock = self
            .writes
            .lock()
            .map_err(|_| anyhow::anyhow!("record lock poisoned"))?;
        let mut job = self.record("jobs", job_id)?;
        job["status"] = json!(status);
        job["error"] = json!(error);
        job["finished_at"] = json!(now_seconds());
        if status == "completed" {
            job["progress"] = json!(1.0);
            job["stage"] = json!("completed");
        }
        let mut records = vec![("jobs", job_id.to_owned(), job.clone())];
        if let Some(run_id) = job["run_id"].as_str() {
            let mut run = self.record("runs", run_id)?;
            run["status"] = json!(status);
            run["error"] = json!(error);
            run["finished_at"] = json!(now_seconds());
            run.as_object_mut().unwrap().remove("manifest_hash");
            // Artifact before terminal database references. Run, job and chat advance together.
            run["manifest_hash"] = json!(self.store.artifacts().put_json(&run)?);
            if let Some(conversation_id) = run["conversation_id"].as_str()
                && (!run["output"].as_str().unwrap_or_default().is_empty()
                    || run["reply_messages"]
                        .as_array()
                        .is_some_and(|messages| !messages.is_empty()))
            {
                let mut conversation = self.record("conversations", conversation_id)?;
                conversation["messages"] = json!(run_transcript(&run)?);
                let ids = conversation["run_ids"]
                    .as_array_mut()
                    .context("invalid conversation run IDs")?;
                if !ids.iter().any(|id| id == run_id) {
                    ids.push(json!(run_id));
                }
                records.push(("conversations", conversation_id.to_owned(), conversation));
            }
            records.push(("runs", run_id.to_owned(), run));
        }
        let rows: Vec<_> = records
            .iter()
            .map(|(kind, id, value)| (*kind, id.as_str(), value.clone()))
            .collect();
        self.store.put_batch(&rows)?;
        if let Some(run_id) = job["run_id"].as_str() {
            self.emit(
                "completed",
                Some(job_id),
                Some(run_id),
                json!({"status":status,"error":error}),
            );
        }
        Ok(())
    }

    async fn controls(&self, request: &Value) -> Result<Value> {
        let _serial = self.controls_lock.lock().await;
        self.controls_inner(request, "user").await
    }

    async fn controls_inner(&self, request: &Value, source: &str) -> Result<Value> {
        let run_id = field(request, "run_id")?;
        let run = self.record("runs", run_id)?;
        ensure!(
            run["status"] == "running",
            "run is not currently generating"
        );
        let revision = request["revision"]
            .as_u64()
            .context("revision must be an unsigned integer")?;
        if let Some(base) = request.get("base_revision") {
            ensure!(
                *base
                    == run["requested_controls"]
                        .as_array()
                        .unwrap()
                        .last()
                        .unwrap()["revision"],
                "mix changed concurrently; sliders were refreshed, try your adjustment again"
            );
        }
        ensure!(
            revision <= i64::MAX as u64
                && revision
                    > run["requested_controls"]
                        .as_array()
                        .unwrap()
                        .last()
                        .unwrap()["revision"]
                        .as_u64()
                        .unwrap(),
            "control revision must increase"
        );
        let coefficients = request["coefficients"]
            .as_array()
            .context("coefficients must be an array")?;
        let axes = control_axes(&run["axes"], coefficients)?;
        // Persist explicit layer identity even when accepting a legacy single-layer
        // client. Never guess a layer if the same concept has multiple frozen rows.
        let coefficients: Vec<_> = axes.as_array().unwrap().iter().map(|axis|
            json!({"vector_id":axis["vector_id"],"layer":axis["layer"],"percent":axis["percent"]})
        ).collect();
        let state = self.engine_state.read().await.clone();
        let caps = &state["capabilities"];
        ensure!(state["model_id"] == run["model_id"], "loaded model changed");
        let fingerprint = field(&run, "model_fingerprint")?;
        let vectors = self.load_vectors(&axes, fingerprint)?;
        let rows = steering::mix_rows(
            &vectors,
            &axes,
            fingerprint,
            caps["n_layer"].as_u64().context("missing engine depth")? as usize,
            caps["n_embd"].as_u64().context("missing width")? as usize,
        )?;
        let diagnostics = steering::mix_diagnostics(
            &vectors,
            &axes,
            fingerprint,
            caps["n_layer"].as_u64().context("missing engine depth")? as usize,
            caps["n_embd"].as_u64().context("missing width")? as usize,
        )?;
        self.update("runs",run_id,|run|run["requested_controls"].as_array_mut().unwrap().push(json!({"revision":revision,"coefficients":coefficients,"source":source,"diagnostics":diagnostics,"requested_at":now_seconds()})))?;
        self.emit(
            "controls_requested",
            None,
            Some(run_id),
            json!({"revision":revision,"coefficients":coefficients,"source":source}),
        );
        let result = async {
            self.engine()
                .await?
                .execute(json!({"op":"controls","target":run_id,"revision":revision,"rows":rows}))
                .await
        }
        .await;
        if let Err(error) = &result {
            self.update("runs", run_id, |run| {
                if let Some(events) = run["requested_controls"].as_array_mut()
                    && let Some(event) = events
                        .iter_mut()
                        .find(|event| event["revision"] == revision)
                {
                    event["error"] = json!(format!("{error:#}"));
                }
            })?;
        }
        result
    }

    fn mix_view(&self, run_id: &str) -> Result<Value> {
        let run = self.record("runs", run_id)?;
        let latest = run["requested_controls"]
            .as_array()
            .context("missing controls")?
            .last()
            .context("missing initial controls")?;
        let axes = control_axes(
            &run["axes"],
            latest["coefficients"]
                .as_array()
                .context("invalid controls")?,
        )?;
        let mut sliders = axes.as_array().unwrap().clone();
        for axis in &mut sliders {
            let id = field(axis, "vector_id")?;
            let vector = self.record("vectors", id)?;
            let metadata = self
                .store
                .get("record_metadata", &format!("vectors:{id}"))?;
            axis["name"] = metadata
                .as_ref()
                .and_then(|m| m.get("name"))
                .unwrap_or(&vector["name"])
                .clone();
        }
        Ok(
            json!({"run_id":run_id,"revision":latest["revision"],"self_modification":run["self_modification"]==true,"sliders":sliders,"applied":run["applied_controls"].as_array().and_then(|events|events.last()),"meaning":"percent of each layer's calibrated residual norm; finite signed values, no intensity claim"}),
        )
    }

    pub async fn model_tool(
        &self,
        name: &str,
        arguments: &Value,
        local_run: Option<&str>,
        source: &str,
    ) -> Result<Value> {
        let _serial = self.controls_lock.lock().await;
        let arguments_object = arguments
            .as_object()
            .context("tool arguments must be an object")?;
        ensure!(
            arguments_object.keys().all(|key| name == "set_mix"
                && matches!(
                    key.as_str(),
                    "run_id" | "expected_revision" | "changes" | "reason"
                )),
            "unexpected tool argument"
        );
        let run_id = if let Some(id) = local_run {
            id.to_owned()
        } else {
            self.store.list("runs")?.into_iter().find(|run|run["status"]=="running")
                .context("no response is currently generating; start a response with self-adjustment enabled")?["id"].as_str().unwrap().to_owned()
        };
        let run = self.record("runs", &run_id)?;
        ensure!(run["status"] == "running", "response is not generating");
        ensure!(
            run["self_modification"] == true,
            "self-adjustment is disabled"
        );
        match name {
            "get_mix" => self.mix_view(&run_id),
            "set_mix" => {
                ensure!(
                    arguments["run_id"] == run_id,
                    "run changed; call get_mix again"
                );
                let latest = run["requested_controls"]
                    .as_array()
                    .unwrap()
                    .last()
                    .unwrap();
                ensure!(
                    arguments["expected_revision"] == latest["revision"],
                    "mix changed; call get_mix again before editing"
                );
                ensure!(
                    arguments.get("reason").is_none_or(|reason| reason
                        .as_str()
                        .is_some_and(|reason| reason.chars().count() <= 1000)),
                    "reason is too long"
                );
                let current =
                    control_axes(&run["axes"], latest["coefficients"].as_array().unwrap())?;
                let axes = self_tools::patch_axes(&current, &arguments["changes"])?;
                let revision = latest["revision"].as_u64().unwrap() + 1;
                let applied = self
                    .controls_inner(
                        &json!({"run_id":run_id,"revision":revision,"coefficients":axes}),
                        source,
                    )
                    .await?;
                self.update("runs", &run_id, |run| {
                    let event = run["requested_controls"]
                        .as_array_mut()
                        .unwrap()
                        .last_mut()
                        .unwrap();
                    event["reason"] = arguments["reason"].clone();
                })?;
                Ok(json!({"mix":self.mix_view(&run_id)?,"applied":applied}))
            }
            _ => bail!("unknown tool {name}"),
        }
    }

    fn export_vector(&self, vector_id: &str) -> Result<Value> {
        let vector = self.record("vectors", vector_id)?;
        let manifest_hash = field(&vector, "manifest_hash")?;
        let manifest = self.store.artifacts().read_json(manifest_hash)?;
        let mut hashes: Vec<String> = manifest["artifacts"]
            .as_array()
            .context("manifest has no artifacts")?
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_owned)
                    .context("invalid artifact hash")
            })
            .collect::<Result<_>>()?;
        hashes.push(manifest_hash.to_owned());
        hashes.sort();
        hashes.dedup();
        validate_vector_manifest(&vector, &manifest, field(&vector, "model_fingerprint")?)?;
        validate_manifest_artifacts(
            self.store.artifacts(),
            &vector,
            &manifest,
            &hashes.iter().cloned().collect(),
        )?;
        let mut blobs = Vec::new();
        let mut total = 0;
        for hash in hashes {
            let bytes = self.store.artifacts().read_bytes(&hash)?;
            total += bytes.len();
            ensure!(
                total <= 256 * 1024 * 1024,
                "export exceeds 256 MiB decoded artifact limit"
            );
            blobs.push(json!({"hash":hash,"data":STANDARD.encode(bytes)}));
        }
        Ok(json!({"format":"torment-vector-v1","vector":vector,"blobs":blobs}))
    }

    fn import_vector(&self, bundle: &Value) -> Result<Value> {
        ensure!(
            bundle["format"] == "torment-vector-v1",
            "unsupported vector bundle format"
        );
        let mut vector = bundle["vector"].clone();
        let fingerprint = field(&vector, "model_fingerprint")?.to_owned();
        let model = self
            .store
            .list("models")?
            .into_iter()
            .find(|model| model["fingerprint"] == fingerprint)
            .context("import the matching model first; vector model fingerprint is incompatible")?;
        let blobs = bundle["blobs"].as_array().context("bundle blobs missing")?;
        ensure!(blobs.len() <= 4096, "too many bundle blobs");
        let mut total = 0;
        let mut hashes = HashSet::new();
        for blob in blobs {
            let hash = field(blob, "hash")?;
            ensure!(hashes.insert(hash.to_owned()), "duplicate blob hash");
            let bytes = STANDARD
                .decode(field(blob, "data")?)
                .context("invalid blob base64")?;
            total += bytes.len();
            ensure!(
                total <= 256 * 1024 * 1024,
                "bundle exceeds 256 MiB decoded artifact limit"
            );
            self.store.artifacts().import_bytes(hash, &bytes)?;
        }
        let manifest_hash = field(&vector, "manifest_hash")?;
        ensure!(hashes.contains(manifest_hash), "bundle omitted manifest");
        let manifest = self.store.artifacts().read_json(manifest_hash)?;
        validate_vector_manifest(&vector, &manifest, &fingerprint)?;
        validate_manifest_artifacts(self.store.artifacts(), &vector, &manifest, &hashes)?;
        for hash in manifest["artifacts"]
            .as_array()
            .context("manifest has no artifact list")?
        {
            ensure!(
                hashes.contains(hash.as_str().context("invalid artifact reference")?),
                "bundle omitted referenced artifact"
            );
        }
        let depth = manifest["model_shape"]["n_layer"]
            .as_u64()
            .context("missing model depth")? as usize;
        let width = manifest["model_shape"]["n_embd"]
            .as_u64()
            .context("missing width")? as usize;
        let mut hydrated = vector.clone();
        for layer in hydrated["layers"]
            .as_array_mut()
            .context("invalid layer metadata")?
        {
            let direction = self
                .store
                .artifacts()
                .read_f32(field(layer, "direction_hash")?)?;
            ensure!(direction.shape == vec![width], "direction shape mismatch");
            if layer["usable"] != false {
                let unit = self
                    .store
                    .artifacts()
                    .read_f32(field(layer, "unit_hash")?)?;
                ensure!(unit.shape == vec![width], "unit shape mismatch");
                let norm = direction
                    .values
                    .iter()
                    .map(|v| f64::from(*v).powi(2))
                    .sum::<f64>()
                    .sqrt();
                ensure!(
                    norm.is_finite() && norm > 0.0,
                    "usable direction has invalid norm"
                );
                ensure!(
                    direction
                        .values
                        .iter()
                        .zip(&unit.values)
                        .all(
                            |(raw, unit)| (f64::from(*raw) / norm - f64::from(*unit)).abs() <= 1e-5
                        ),
                    "unit tensor does not normalize its raw direction"
                );
                layer["unit"] = json!(unit.values);
                steering::legacy_buffer_index(
                    layer["layer"].as_u64().context("invalid layer")? as usize,
                    depth,
                )?;
            }
        }
        let validate_axes: Vec<Value> = hydrated["layers"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|layer| layer["usable"] != false)
            .map(|layer| json!({"vector_id":vector["id"],"layer":layer["layer"],"percent":0}))
            .collect();
        for axis in validate_axes {
            steering::mix_rows(
                &[hydrated.clone()],
                &json!([axis]),
                &fingerprint,
                depth,
                width,
            )?;
        }
        let original_id = vector["id"].clone();
        let vector_id = id();
        vector["id"] = json!(vector_id);
        vector["imported_from_id"] = original_id;
        vector["model_id"] = model["id"].clone();
        vector["created_at"] = json!(now_seconds());
        let mut recipe = manifest["recipe"].clone();
        let recipe_id = id();
        recipe["imported_from_id"] = recipe["id"].clone();
        recipe["id"] = json!(recipe_id);
        recipe["model_id"] = model["id"].clone();
        recipe["created_at"] = json!(now_seconds());
        vector["recipe_id"] = json!(recipe_id);
        self.store.put_batch(&[
            ("recipes", recipe_id.as_str(), recipe),
            ("vectors", vector_id.as_str(), vector),
        ])?;
        Ok(json!({"id":vector_id}))
    }
}

pub struct App {
    pub store: Store,
    pub events: broadcast::Sender<Value>,
    worker_path: PathBuf,
    worker: Mutex<Option<Engine>>,
    engine_state: RwLock<Value>,
    local: Arc<Semaphore>,
    codex: CodexClient,
    codex_state: RwLock<Value>,
    writes: StdMutex<()>,
    cancellations: StdMutex<HashMap<String, Arc<AtomicBool>>>,
    controls_lock: Mutex<()>,
}

impl App {
    pub fn new(store: Store, worker_path: PathBuf, codex_path: Option<PathBuf>) -> Arc<Self> {
        let mut codex = CodexClient::new(store.root().join("codex-worker"));
        if let Some(path) = codex_path {
            codex = codex.with_executable(path);
        }
        let (events, _) = broadcast::channel(4096);
        Arc::new(Self {
            store,
            events,
            worker_path,
            worker: Mutex::new(None),
            engine_state: RwLock::new(
                json!({"status":"unloaded","model_id":null,"capabilities":null}),
            ),
            local: Arc::new(Semaphore::new(1)),
            codex,
            codex_state: RwLock::new(json!({"models":[],"error":null})),
            writes: StdMutex::new(()),
            cancellations: StdMutex::new(HashMap::new()),
            controls_lock: Mutex::new(()),
        })
    }

    fn emit(&self, kind: &str, job_id: Option<&str>, run_id: Option<&str>, data: Value) {
        let _ = self
            .events
            .send(json!({"kind":kind,"job_id":job_id,"run_id":run_id,"data":data}));
    }

    fn record(&self, kind: &str, record_id: &str) -> Result<Value> {
        self.store
            .get(kind, record_id)?
            .with_context(|| format!("{kind} record {record_id} not found"))
    }

    fn update<F>(&self, kind: &str, record_id: &str, edit: F) -> Result<Value>
    where
        F: FnOnce(&mut Value),
    {
        let _lock = self
            .writes
            .lock()
            .map_err(|_| anyhow::anyhow!("record lock poisoned"))?;
        let mut value = self.record(kind, record_id)?;
        edit(&mut value);
        value["updated_at"] = json!(now_seconds());
        self.store.put(kind, record_id, &value)?;
        Ok(value)
    }

    pub async fn snapshot(&self) -> Result<Value> {
        let mut snapshot = self.store.snapshot()?;
        // Mutable labels/tombstones never rewrite recipes, vectors, or provenance.
        let metadata = snapshot["record_metadata"].as_array().unwrap().clone();
        for entry in metadata {
            let Some(kind) = entry["kind"].as_str() else {
                continue;
            };
            if let Some(records) = snapshot.get_mut(kind).and_then(Value::as_array_mut)
                && let Some(record) = records
                    .iter_mut()
                    .find(|record| record["id"] == entry["record_id"])
            {
                if let Some(name) = entry["name"].as_str() {
                    record["display_name"] = json!(name);
                    if matches!(kind, "vectors" | "presets") {
                        record["name"] = json!(name);
                    }
                    if kind == "conversations" {
                        record["title"] = json!(name);
                    }
                }
                record["deleted"] = json!(entry["deleted"] == true);
            }
        }
        let deleted_chats: HashSet<_> = snapshot["conversations"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|record| record["deleted"] == true)
            .filter_map(|record| record["id"].as_str().map(str::to_owned))
            .collect();
        for run in snapshot["runs"].as_array_mut().unwrap() {
            if run["conversation_id"]
                .as_str()
                .is_some_and(|id| deleted_chats.contains(id))
            {
                run["deleted"] = json!(true);
            }
        }
        snapshot["engine"] = self.engine_state.read().await.clone();
        snapshot["codex"] = self.codex_state.read().await.clone();
        Ok(snapshot)
    }

    fn record_deleted(&self, kind: &str, record_id: &str) -> Result<bool> {
        Ok(self
            .store
            .get("record_metadata", &format!("{kind}:{record_id}"))?
            .is_some_and(|entry| entry["deleted"] == true))
    }

    fn new_conversation(&self, request: &Value) -> Result<Value> {
        let _lock = self
            .writes
            .lock()
            .map_err(|_| anyhow::anyhow!("record lock poisoned"))?;
        let conversation_id = id();
        let mut conversation = json!({"id":conversation_id,"created_at":now_seconds(),
            "title":request["title"].as_str().unwrap_or("New conversation"),"messages":[],"run_ids":[]});
        if let Some(source_id) = request.get("from_run_id").filter(|value| !value.is_null()) {
            let source_id = source_id.as_str().context("from_run_id must be a run ID")?;
            let run = self.record("runs", source_id)?;
            ensure!(
                !matches!(run["status"].as_str(), Some("queued" | "running")),
                "stop the response before forking it"
            );
            conversation["messages"] = json!(run_transcript(&run)?);
            conversation["forked_from_run_id"] = json!(source_id);
            let mut run_ids = vec![];
            if let Some(parent_id) = run["conversation_id"].as_str() {
                let parent = self.record("conversations", parent_id)?;
                conversation["forked_from_conversation_id"] = json!(parent_id);
                if let Some(ids) = parent["run_ids"].as_array()
                    && let Some(index) = ids.iter().position(|id| id == source_id)
                {
                    run_ids.extend_from_slice(&ids[..index]);
                }
            }
            // Share immutable runs, never move them or copy later turns from the parent.
            run_ids.push(json!(source_id));
            conversation["run_ids"] = json!(run_ids);
        }
        self.store
            .put("conversations", &conversation_id, &conversation)?;
        Ok(conversation)
    }

    fn manage_record(&self, request: &Value, deleting: bool) -> Result<Value> {
        let kind = field(request, "kind")?;
        ensure!(
            matches!(kind, "vectors" | "recipes" | "presets" | "conversations"),
            "this record kind cannot be renamed or deleted"
        );
        let record_id = field(request, "id")?;
        self.record(kind, record_id)?;
        // Delete cannot race a new inference turn. It never alters engine buffers.
        let _permit = if deleting {
            Some(self.local_permit()?)
        } else {
            None
        };
        let _lock = self
            .writes
            .lock()
            .map_err(|_| anyhow::anyhow!("record lock poisoned"))?;
        let metadata_id = format!("{kind}:{record_id}");
        let mut entry = self
            .store
            .get("record_metadata", &metadata_id)?
            .unwrap_or_else(|| json!({"id":metadata_id,"kind":kind,"record_id":record_id}));
        if deleting {
            entry["deleted"] = json!(true);
        } else {
            ensure!(entry["deleted"] != true, "record was deleted");
            let name = field(request, "name")?.trim();
            ensure!(
                !name.is_empty() && name.chars().count() <= 200,
                "name must contain 1–200 characters"
            );
            entry["name"] = json!(name);
        }
        entry["updated_at"] = json!(now_seconds());
        self.store.put("record_metadata", &metadata_id, &entry)?;
        Ok(entry)
    }

    async fn engine(&self) -> Result<Engine> {
        let mut slot = self.worker.lock().await;
        if slot.as_ref().is_none_or(|worker| !worker.is_alive()) {
            if let Some(old) = slot.take() {
                old.shutdown().await;
            }
            *slot = Some(Engine::start(&self.worker_path, &self.store.root().join("logs")).await?);
        }
        Ok(slot.as_ref().unwrap().clone())
    }

    pub async fn shutdown(&self) {
        for flag in self.cancellations.lock().unwrap().values() {
            flag.store(true, Ordering::Release);
        }
        if let Some(worker) = self.worker.lock().await.as_ref() {
            worker.shutdown().await;
        }
    }

    fn create_job(&self, kind: &str, request: &Value) -> Result<String> {
        let job_id = id();
        let job = json!({"id":job_id,"created_at":now_seconds(),"kind":kind,"status":"queued",
            "stage":"queued","progress":0.0,"error":null,"request":request,
            "model_id":request["model_id"],"recipe_id":request["recipe_id"],"details":{"stages":{},"extractions":{},"previews":{}}});
        self.store.put("jobs", &job_id, &job)?;
        Ok(job_id)
    }

    fn checkpoint(&self, job_id: &str, name: &str, value: &Value) -> Result<String> {
        let hash = self.store.artifacts().put_json(value)?;
        self.update("jobs", job_id, |job| {
            job["details"]["stages"][name] = json!(hash);
        })?;
        Ok(hash)
    }

    fn progress(&self, job_id: &str, stage: &str, progress: f64) -> Result<()> {
        self.update("jobs", job_id, |job| {
            job["status"] = json!("running");
            job["stage"] = json!(stage);
            job["progress"] = json!(progress);
        })?;
        self.emit(
            "progress",
            Some(job_id),
            None,
            json!({"stage":stage,"progress":progress}),
        );
        Ok(())
    }

    fn start_job(
        self: &Arc<Self>,
        job_id: String,
        permit: Option<OwnedSemaphorePermit>,
    ) -> Result<()> {
        let cancellation = Arc::new(AtomicBool::new(false));
        {
            let mut jobs = self.cancellations.lock().unwrap();
            ensure!(!jobs.contains_key(&job_id), "job is already running");
            jobs.insert(job_id.clone(), cancellation.clone());
        }
        let app = self.clone();
        tokio::spawn(async move {
            let result = app.run_job(&job_id, cancellation.clone(), permit).await;
            // Live-control requests and their eventual errors must be included in
            // the terminal manifest, never appended after its durable snapshot.
            let _controls = app.controls_lock.lock().await;
            let cancelled = cancellation.load(Ordering::Acquire);
            let mut status = if cancelled {
                "cancelled"
            } else if result.is_ok() {
                "completed"
            } else {
                "failed"
            };
            let mut error = result.err().map(|e| format!("{e:#}"));
            if let Err(failure) = app.finalize_job(&job_id, status, error.as_deref()) {
                status = "failed";
                error = Some(format!(
                    "durable finalization failed: {failure:#}; prior output is retained"
                ));
                // Best effort when the persistence medium itself has failed. Never emit completed.
                let record = app.update("jobs", &job_id, |job| {
                    job["status"] = json!(status);
                    job["error"] = json!(error);
                });
                if let Ok(job) = record
                    && let Some(run_id) = job["run_id"].as_str()
                {
                    let _ = app.update("runs", run_id, |run| {
                        run["status"] = json!(status);
                        run["error"] = json!(error);
                    });
                }
                eprintln!("Job {job_id}: {}", error.as_deref().unwrap());
            }
            app.cancellations.lock().unwrap().remove(&job_id);
            app.emit(
                if status == "failed" {
                    "error"
                } else {
                    "completed"
                },
                Some(&job_id),
                None,
                json!({"status":status,"error":error}),
            );
        });
        Ok(())
    }

    fn local_permit(&self) -> Result<OwnedSemaphorePermit> {
        self.local
            .clone()
            .try_acquire_owned()
            .context("local engine is busy; stop or wait for the current job")
    }

    pub async fn action(self: &Arc<Self>, request: Value) -> Result<Value> {
        match field(&request, "action")? {
            "rename_record" => self.manage_record(&request, false),
            "delete_record" => self.manage_record(&request, true),
            "import_model" => {
                let path = PathBuf::from(field(&request, "path")?);
                let name = request["name"].as_str().map(str::to_owned);
                let model = tokio::task::spawn_blocking(move || {
                    download::import_model(&path, name.as_deref())
                })
                .await??;
                self.store.put("models", field(&model, "id")?, &model)?;
                Ok(json!({"id":model["id"],"model":model}))
            }
            "browse_hf" => download::browse_hf(field(&request, "repo")?).await,
            "download_model" => {
                field(&request, "repo")?;
                field(&request, "file")?;
                let job_id = self.create_job("download", &request)?;
                self.start_job(job_id.clone(), None)?;
                Ok(json!({"job_id":job_id}))
            }
            "load_model" => {
                self.record("models", field(&request, "model_id")?)?;
                let permit = self.local_permit()?;
                let job_id = self.create_job("load", &request)?;
                self.start_job(job_id.clone(), Some(permit))?;
                Ok(json!({"job_id":job_id}))
            }
            "unload_model" => {
                let _permit = self.local_permit()?;
                self.engine().await?.execute(json!({"op":"unload"})).await?;
                *self.engine_state.write().await =
                    json!({"status":"unloaded","model_id":null,"capabilities":null});
                Ok(json!({"ok":true}))
            }
            "discover_codex" => match self.codex.discover().await {
                Ok(models) => {
                    let roles = factory::default_roles(&models)?;
                    let result = json!({"models":models,"assignments":roles,"error":null});
                    *self.codex_state.write().await = result.clone();
                    Ok(result)
                }
                Err(error) => {
                    *self.codex_state.write().await =
                        json!({"models":[],"error":format!("{error:#}")});
                    Err(error)
                }
            },
            "create_concept" => {
                let request = new_concept_settings(request)?;
                ensure!(
                    field(&request, "concept")?.len() <= 8000,
                    "concept is too long"
                );
                self.record("models", field(&request, "model_id")?)?;
                let job_id = self.create_job("factory", &request)?;
                self.start_job(job_id.clone(), None)?;
                Ok(json!({"job_id":job_id}))
            }
            "extract_recipe" => {
                self.record("recipes", field(&request, "recipe_id")?)?;
                self.record("models", field(&request, "model_id")?)?;
                let job_id = self.create_job("extract", &request)?;
                self.start_job(job_id.clone(), None)?;
                Ok(json!({"job_id":job_id}))
            }
            "edit_recipe" => self.edit_recipe(&request),
            "edit_job_stage" => self.edit_job_stage(&request),
            "retry_job" => {
                let job_id = field(&request, "job_id")?;
                let job = self.record("jobs", job_id)?;
                ensure!(
                    matches!(
                        job["status"].as_str(),
                        Some("failed" | "cancelled" | "interrupted")
                    ),
                    "job is not retryable"
                );
                ensure!(
                    job["kind"] != "generate",
                    "interrupted inference is saved, not exactly resumable; duplicate the run instead"
                );
                let mut reassignment = None;
                if job["kind"] == "factory" && job["details"]["recipe_published"] != true {
                    let stages = self.read_stages(&job)?;
                    let previous = job["request"]
                        .get("roles")
                        .filter(|value| !value.is_null())
                        .or_else(|| stages.get("roles"));
                    if let Some(previous) = previous {
                        let models = self.codex.discover().await?;
                        let (roles, changes) = factory::retry_roles(
                            &serde_json::from_value(previous.clone())?,
                            &stages,
                            &models,
                        )?;
                        if !changes.is_empty() {
                            reassignment = Some((
                                serde_json::to_value(roles)?,
                                json!({
                                    "at":now_seconds(),"previous_roles":previous,"changes":changes
                                }),
                            ));
                        }
                    }
                }
                self.update("jobs", job_id, |job| {
                    if let Some((roles, record)) = reassignment {
                        job["request"]["roles"] = roles;
                        let mut history = job["details"]["role_reassignments"]
                            .as_array()
                            .cloned()
                            .unwrap_or_default();
                        history.push(record);
                        job["details"]["role_reassignments"] = json!(history);
                    }
                    job["status"] = json!("queued");
                    job["error"] = Value::Null;
                    job["cancel_requested"] = json!(false);
                })?;
                self.start_job(job_id.to_owned(), None)?;
                Ok(json!({"job_id":job_id}))
            }
            "cancel_job" => {
                self.cancel_job(field(&request, "job_id")?).await?;
                Ok(json!({"ok":true}))
            }
            "cancel_run" => {
                let run = self.record("runs", field(&request, "run_id")?)?;
                self.cancel_job(field(&run, "job_id")?).await?;
                Ok(json!({"ok":true}))
            }
            "generate" => self.start_generation(request).await,
            "controls" => self.controls(&request).await,
            "set_self_modification" => {
                let _serial = self.controls_lock.lock().await;
                let run_id = field(&request, "run_id")?;
                let enabled = request["enabled"]
                    .as_bool()
                    .context("enabled must be a boolean")?;
                let run = self.record("runs", run_id)?;
                ensure!(
                    matches!(run["status"].as_str(), Some("running" | "queued")),
                    "response is no longer active"
                );
                ensure!(
                    !enabled || run["self_tools_available"] == true,
                    "enable self-adjustment before starting the response"
                );
                self.update("runs", run_id, |run| {
                    run["self_modification"] = json!(enabled);
                    run["permission_events"]
                        .as_array_mut()
                        .unwrap()
                        .push(json!({"enabled":enabled,"at":now_seconds()}));
                })?;
                self.emit(
                    "state",
                    None,
                    Some(run_id),
                    json!({"self_modification":enabled}),
                );
                Ok(json!({"enabled":enabled}))
            }
            "save_mix" => {
                let model = self.record("models", field(&request, "model_id")?)?;
                ensure!(request["axes"].is_array(), "axes must be an array");
                self.load_vectors(&request["axes"], field(&model, "fingerprint")?)?;
                let preset_id = id();
                let preset = json!({"id":preset_id,"created_at":now_seconds(),"name":field(&request,"name")?,"model_id":model["id"],"model_fingerprint":model["fingerprint"],"axes":request["axes"]});
                self.store.put("presets", &preset_id, &preset)?;
                Ok(json!({"id":preset_id}))
            }
            "new_conversation" => self.new_conversation(&request),
            "artifact" => Ok(self.store.artifacts().read_json(field(&request, "hash")?)?),
            "export_vector" => self.export_vector(field(&request, "vector_id")?),
            "import_vector" => self.import_vector(&request["bundle"]),
            "set_vector_archived" => {
                let vector_id = field(&request, "vector_id")?;
                let archived = request["archived"]
                    .as_bool()
                    .context("archived must be a boolean")?;
                self.record("vectors", vector_id)?;
                // Library visibility is mutable UI metadata, not part of the
                // immutable vector or anything its saved runs/presets reference.
                let visibility =
                    json!({"id":vector_id,"archived":archived,"updated_at":now_seconds()});
                self.store
                    .put("vector_visibility", vector_id, &visibility)?;
                Ok(visibility)
            }
            other => bail!("unknown action {other}"),
        }
    }

    async fn cancel_job(&self, job_id: &str) -> Result<()> {
        if let Some(flag) = self.cancellations.lock().unwrap().get(job_id) {
            flag.store(true, Ordering::Release);
        }
        // Read the target only after publishing cancellation. The enqueue path
        // rechecks the same flag once the command has reached its private pipe.
        let job = self.update("jobs", job_id, |job| {
            job["cancel_requested"] = json!(true);
        })?;
        if let Some(target) = job["engine_target"].as_str()
            && let Some(worker) = self.worker.lock().await.as_ref()
        {
            worker.cancel(target).await?;
        }
        Ok(())
    }

    async fn discard_worker(&self) {
        if let Some(worker) = self.worker.lock().await.take() {
            worker.shutdown().await;
        }
        *self.engine_state.write().await = json!({
            "status":"unloaded","model_id":null,"capabilities":null,
            "error":"Local operation failed; its worker was stopped. The next request will reload the model."
        });
    }

    async fn observe_runtime(&self, runtime: &Value) {
        if let Some(bytes) = runtime["peak_rss_bytes"].as_u64() {
            let mut state = self.engine_state.write().await;
            state["memory_bytes"] = json!(bytes);
            state["memory_kind"] = json!("peak_worker_rss");
        }
    }

    async fn load_model(&self, job_id: &str, request: &Value) -> Result<Value> {
        let model_id = field(request, "model_id")?;
        let model = self.record("models", model_id)?;
        self.progress(job_id, "verify-model", 0.02)?;
        *self.engine_state.write().await =
            json!({"status":"loading","model_id":model_id,"capabilities":null});
        let (path, fingerprint) = (
            PathBuf::from(field(&model, "path")?),
            field(&model, "fingerprint")?.to_owned(),
        );
        let verified =
            tokio::task::spawn_blocking(move || download::verify_model(&path, &fingerprint))
                .await?;
        let result = async {
            verified?;
            self.progress(job_id,"loading",0.05)?;
            let engine=self.engine().await?;
            engine.execute(json!({"op":"load","path":model["path"],"context":request["context"].as_u64().unwrap_or(8192),
                "gpu_layers":request["gpu_layers"].as_i64().unwrap_or(99),"batch":512,"microbatch":128})).await
        }.await;
        match result {
            Ok(capabilities) => {
                *self.engine_state.write().await =
                    json!({"status":"idle","model_id":model_id,"capabilities":capabilities});
                self.observe_runtime(&capabilities["runtime"]).await;
                self.update("models", model_id, |model| {
                    model["capabilities"] = capabilities.clone();
                })?;
                Ok(capabilities)
            }
            Err(error) => {
                *self.engine_state.write().await = json!({"status":"error","model_id":null,"error":format!("{error:#}"),"capabilities":null});
                Err(error)
            }
        }
    }

    async fn ensure_model(&self, job_id: &str, model_id: &str) -> Result<Value> {
        let state = self.engine_state.read().await.clone();
        let alive = self
            .worker
            .lock()
            .await
            .as_ref()
            .is_some_and(Engine::is_alive);
        if state["model_id"] == model_id && !state["capabilities"].is_null() && alive {
            return Ok(state["capabilities"].clone());
        }
        self.load_model(job_id, &json!({"model_id":model_id})).await
    }

    fn load_vectors(&self, axes: &Value, fingerprint: &str) -> Result<Vec<Value>> {
        let axes = axes.as_array().context("axes must be an array")?;
        ensure!(axes.len() <= 256, "too many axes");
        let mut result: Vec<Value> = Vec::new();
        let mut seen = HashSet::new();
        for axis in axes {
            let vector_id = field(axis, "vector_id")?;
            let layer_id = axis["layer"]
                .as_u64()
                .context("layer must be an unsigned integer")?;
            ensure!(
                seen.insert((vector_id, layer_id)),
                "duplicate vector/layer selection"
            );
            ensure!(
                axis["percent"].as_f64().is_some_and(f64::is_finite),
                "coefficient must be finite"
            );
            if let Some(vector) = result.iter().find(|vector| vector["id"] == vector_id) {
                ensure!(
                    vector["layers"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|layer| layer["layer"] == axis["layer"] && layer["usable"] != false),
                    "selected layer missing or unusable"
                );
                continue;
            }
            let mut vector = self.record("vectors", vector_id)?;
            ensure!(
                vector["model_fingerprint"] == fingerprint,
                "vector belongs to a different model fingerprint"
            );
            let manifest = self
                .store
                .artifacts()
                .read_json(field(&vector, "manifest_hash")?)?;
            validate_vector_manifest(&vector, &manifest, fingerprint)?;
            for layer in vector["layers"]
                .as_array_mut()
                .context("invalid vector layers")?
            {
                if let Some(hash) = layer["unit_hash"].as_str() {
                    let tensor = self.store.artifacts().read_f32(hash)?;
                    ensure!(
                        tensor.shape
                            == vec![layer["width"].as_u64().context("missing width")? as usize],
                        "unit tensor shape mismatch"
                    );
                    layer["unit"] = json!(tensor.values);
                }
            }
            ensure!(
                vector["layers"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|layer| layer["layer"] == axis["layer"] && layer["usable"] != false),
                "selected layer missing or unusable"
            );
            result.push(vector);
        }
        Ok(result)
    }
}

impl App {
    async fn run_job(
        self: &Arc<Self>,
        job_id: &str,
        cancellation: Arc<AtomicBool>,
        permit: Option<OwnedSemaphorePermit>,
    ) -> Result<()> {
        let job = self.record("jobs", job_id)?;
        let request = &job["request"];
        self.progress(job_id, "starting", 0.0)?;
        match field(&job, "kind")? {
            "download" => {
                let app = self.clone();
                let jid = job_id.to_owned();
                let progress = Arc::new(move |progress: download::DownloadProgress| {
                    let fraction = progress
                        .total
                        .filter(|n| *n > 0)
                        .map_or(0.0, |n| progress.downloaded as f64 / n as f64);
                    let _ = app.update("jobs", &jid, |job| {
                        job["stage"] = json!(progress.phase);
                        job["progress"] = json!(fraction);
                        job["details"]["download"] = json!(progress);
                    });
                    app.emit("progress", Some(&jid), None, json!(progress));
                });
                let revision = request["revision"]
                    .as_str()
                    .or_else(|| job["details"]["download"]["revision"].as_str());
                let model = download::download_model(
                    &self.store.root().join("models"),
                    field(request, "repo")?,
                    field(request, "file")?,
                    revision,
                    cancellation,
                    progress,
                )
                .await?;
                self.store.put("models", field(&model, "id")?, &model)?;
                self.update("jobs", job_id, |job| {
                    job["model_id"] = model["id"].clone();
                })?;
            }
            "load" => {
                let _permit = match permit {
                    Some(p) => p,
                    None => self.local_permit()?,
                };
                self.load_model(job_id, request).await?;
            }
            "factory" => {
                if job["details"]["recipe_published"] == true {
                    let recipe = self.record("recipes", field(&job, "recipe_id")?)?;
                    return self
                        .extract_recipe(job_id, &recipe, field(request, "model_id")?, cancellation)
                        .await;
                }
                let stages = self.read_stages(&job)?;
                let roles = request
                    .get("roles")
                    .filter(|v| !v.is_null())
                    .cloned()
                    .map(serde_json::from_value)
                    .transpose()?;
                let (send, mut receive) = mpsc::unbounded_channel::<Value>();
                let app = self.clone();
                let jid = job_id.to_owned();
                let event_task = tokio::spawn(async move {
                    while let Some(event) = receive.recv().await {
                        let stage = event["stage"].as_str().unwrap_or("agents");
                        let _ = app.update("jobs", &jid, |job| {
                            job["stage"] = json!(stage);
                            job["details"]["agent_event"] = event.clone();
                        });
                        app.emit("progress", Some(&jid), None, event);
                    }
                });
                let app = self.clone();
                let jid = job_id.to_owned();
                let extraction = Extraction::from_record(request)?;
                let output = factory::run_with_extraction(
                    &self.codex,
                    field(request, "concept")?,
                    roles,
                    stages,
                    cancellation.clone(),
                    send,
                    move |stage, value| {
                        app.checkpoint(&jid, stage, value)?;
                        Ok(())
                    },
                    &extraction,
                )
                .await;
                event_task.abort();
                let output = output?;
                ensure!(!cancellation.load(Ordering::Acquire), "cancelled");
                let latest = self.record("jobs", job_id)?;
                let recipe = if let Some(existing) = latest["recipe_id"]
                    .as_str()
                    .filter(|_| latest["details"]["recipe_published"] == true)
                {
                    self.record("recipes", existing)?
                } else {
                    let recipe_id = id();
                    let prepared = steering::prepare_dataset(&output.dataset)?;
                    let dataset_hash = self.store.artifacts().put_jsonl(
                        prepared["pairs"]
                            .as_array()
                            .context("invalid prepared dataset")?,
                    )?;
                    let parent = request["parent_recipe_id"]
                        .as_str()
                        .map(|pid| self.record("recipes", pid))
                        .transpose()?;
                    let recipe = json!({"id":recipe_id,"created_at":now_seconds(),"concept":request["concept"],"model_id":request["model_id"],
                        "parent_id":request["parent_recipe_id"],"version":parent.as_ref().and_then(|p|p["version"].as_u64()).unwrap_or(0)+1,
                        "design":output.design,"dataset":prepared["pairs"],"dataset_hash":dataset_hash,"review":output.review,
                        "roles":output.roles,"stages":latest["details"]["stages"],"warnings":output.warnings,
                        "dataset_warnings":prepared["warnings"],"raw":request["raw"].as_bool().unwrap_or(false),
                        "extraction":request["extraction"],"preview_mode":request["preview_mode"].as_str().unwrap_or("standard")});
                    let _lock = self
                        .writes
                        .lock()
                        .map_err(|_| anyhow::anyhow!("record lock poisoned"))?;
                    let mut published_job = self.record("jobs", job_id)?;
                    published_job["recipe_id"] = json!(recipe_id);
                    published_job["details"]["recipe_published"] = json!(true);
                    self.store.put_batch(&[
                        ("recipes", recipe_id.as_str(), recipe.clone()),
                        ("jobs", job_id, published_job),
                    ])?;
                    recipe
                };
                self.extract_recipe(job_id, &recipe, field(request, "model_id")?, cancellation)
                    .await?;
            }
            "extract" => {
                let recipe = self.recipe_for_extraction(job_id, request)?;
                self.extract_recipe(job_id, &recipe, field(request, "model_id")?, cancellation)
                    .await?;
            }
            "generate" => {
                let _permit = permit.context("generation lost its exclusive engine reservation")?;
                self.run_generation(job_id, field(&job, "run_id")?, cancellation)
                    .await?;
            }
            other => bail!("unknown job kind {other}"),
        }
        Ok(())
    }

    fn read_stages(&self, job: &Value) -> Result<BTreeMap<String, Value>> {
        let mut result = BTreeMap::new();
        if let Some(stages) = job["details"]["stages"].as_object() {
            for (name, hash) in stages {
                result.insert(
                    name.clone(),
                    self.store
                        .artifacts()
                        .read_json(hash.as_str().context("invalid stage hash")?)?,
                );
            }
        }
        Ok(result)
    }

    fn recipe_for_extraction(&self, job_id: &str, request: &Value) -> Result<Value> {
        let _lock = self
            .writes
            .lock()
            .map_err(|_| anyhow::anyhow!("record lock poisoned"))?;
        let mut job = self.record("jobs", job_id)?;
        if let Some(recipe_id) = job["details"]["effective_recipe_id"].as_str() {
            return self.record("recipes", recipe_id);
        }
        let recipe = self.record("recipes", field(request, "recipe_id")?)?;
        ensure!(
            recipe["draft"] != true,
            "this recipe has unfinished downstream stages; wait for its factory job"
        );
        let raw = if let Some(value) = request.get("raw") {
            value.as_bool().context("raw must be a boolean")?
        } else {
            recipe["raw"].as_bool().unwrap_or(false)
        };
        let extraction = if request.get("extraction").is_some() {
            Extraction::from_record(request)?
        } else {
            Extraction::from_record(&recipe)?
        };
        if raw == recipe["raw"].as_bool().unwrap_or(false)
            && recipe["model_id"] == request["model_id"]
            && extraction == Extraction::from_record(&recipe)?
        {
            return Ok(recipe);
        }
        let mut version = recipe.clone();
        let recipe_id = id();
        version["id"] = json!(recipe_id);
        version["parent_id"] = recipe["id"].clone();
        version["created_at"] = json!(now_seconds());
        version["version"] = json!(recipe["version"].as_u64().unwrap_or(1) + 1);
        version["raw"] = json!(raw);
        version["extraction"] = serde_json::to_value(extraction)?;
        version["model_id"] = request["model_id"].clone();
        version["edited_stage"] = json!("extraction-settings");
        job["recipe_id"] = json!(recipe_id);
        job["details"]["effective_recipe_id"] = json!(recipe_id);
        self.store.put_batch(&[
            ("recipes", recipe_id.as_str(), version.clone()),
            ("jobs", job_id, job),
        ])?;
        Ok(version)
    }

    async fn wait_local(
        &self,
        job_id: &str,
        cancellation: &AtomicBool,
    ) -> Result<OwnedSemaphorePermit> {
        self.progress(job_id, "waiting-for-local-engine", 0.1)?;
        loop {
            ensure!(
                !cancellation.load(Ordering::Acquire),
                "cancelled while waiting for engine"
            );
            if let Ok(permit) = self.local.clone().try_acquire_owned() {
                return Ok(permit);
            }
            tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        }
    }

    async fn extract_recipe(
        &self,
        job_id: &str,
        recipe: &Value,
        model_id: &str,
        cancellation: Arc<AtomicBool>,
    ) -> Result<()> {
        let _permit = self.wait_local(job_id, &cancellation).await?;
        let result = self
            .extract_inner(job_id, recipe, model_id, cancellation)
            .await;
        if result.is_err() {
            self.discard_worker().await;
            return result;
        }
        let mut state = self.engine_state.write().await;
        if state["model_id"].is_string() {
            state["status"] = json!("idle");
        }
        result
    }

    async fn extract_inner(
        &self,
        job_id: &str,
        recipe: &Value,
        model_id: &str,
        cancellation: Arc<AtomicBool>,
    ) -> Result<()> {
        ensure!(
            recipe["draft"] != true,
            "this edited recipe has unfinished downstream stages; wait for its factory job"
        );
        let capabilities = self.ensure_model(job_id, model_id).await?;
        let model = self.record("models", model_id)?;
        let fingerprint = field(&model, "fingerprint")?;
        let layers = steering::supported_layers(
            capabilities["n_layer"].as_u64().context("missing depth")? as usize,
        )?;
        let prepared = steering::prepare_dataset(&recipe["dataset"])?;
        let pairs = prepared["pairs"]
            .as_array()
            .context("invalid prepared dataset")?;
        let dataset_hash = self.store.artifacts().put_jsonl(pairs)?;
        let raw = recipe["raw"].as_bool().unwrap_or(false);
        let extraction = Extraction::from_record(recipe)?;
        let preview_coefficients = preview_percentages(recipe)?;
        let identity = json!({"dataset_hash":dataset_hash,"model_fingerprint":fingerprint,"engine_revision":ENGINE_REVISION,
            "runtime":runtime_identity(&capabilities["runtime"])?,"layers":layers,"raw":raw,"algorithm":extraction.algorithm()});
        let mut identity = identity;
        if extraction.method == Method::Paper {
            identity["extraction"] = serde_json::to_value(&extraction)?;
        }
        let job = self.record("jobs", job_id)?;
        if !job["details"]["extraction_identity"].is_null() {
            ensure!(
                job["details"]["extraction_identity"] == identity,
                "extraction checkpoint belongs to a different dataset/model/runtime"
            );
        }
        self.update("jobs", job_id, |job| {
            job["details"]["extraction_identity"] = identity.clone()
        })?;
        self.engine_state.write().await["status"] = json!("running");
        let engine = self.engine().await?;
        let mut captures = Vec::new();
        let mut capture_hashes = Vec::new();
        for (index, pair) in pairs.iter().enumerate() {
            for (pole_index, pole) in ["positive", "negative"].iter().enumerate() {
                ensure!(
                    !cancellation.load(Ordering::Acquire),
                    "cancelled; completed captures retained"
                );
                self.progress(
                    job_id,
                    &format!("extract {}/{} {pole}", index + 1, pairs.len()),
                    0.15 + 0.6 * ((index * 2 + pole_index) as f64 / (pairs.len() * 2) as f64),
                )?;
                let key = format!("{}:{pole}", field(pair, "id")?);
                let current = self.record("jobs", job_id)?;
                let (captured, hash) =
                    if let Some(hash) = current["details"]["extractions"][&key].as_str() {
                        (self.store.artifacts().read_json(hash)?, hash.to_owned())
                    } else {
                        let target = id();
                        self.update("jobs", job_id, |job| job["engine_target"] = json!(target))?;
                        let mut command = extraction.capture(pair, pole, raw)?;
                        command["id"] = json!(target);
                        command["op"] = json!("extract");
                        command["layers"] = json!(layers);
                        let event = engine.execute_cancellable(command, &cancellation).await?;
                        self.observe_runtime(&event["runtime"]).await;
                        ensure!(
                            event["cancelled"] != true,
                            "cancelled; incomplete capture discarded"
                        );
                        let hash = self.store.artifacts().put_json(&event)?;
                        self.update("jobs", job_id, |job| {
                            job["details"]["extractions"][&key] = json!(hash);
                            job["engine_target"] = Value::Null;
                        })?;
                        (event, hash)
                    };
                ensure!(
                    runtime_identity(&captured["runtime"])? == identity["runtime"],
                    "saved capture has a different worker/configuration identity"
                );
                captures.push(
                    json!({"pair_id":pair["id"],"pole":pole,"captures":captured["captures"]}),
                );
                capture_hashes.push(hash);
            }
        }
        self.progress(job_id, "separation-and-calibration", 0.76)?;
        let analysis_dataset = prepared.clone();
        let analysis_captures = json!(captures);
        let analysis_fingerprint = fingerprint.to_owned();
        let paper = extraction.method == Method::Paper;
        let analysis = tokio::task::spawn_blocking(move || {
            steering::analyze_with_method(
                &analysis_dataset,
                &analysis_captures,
                &analysis_fingerprint,
                paper,
            )
        })
        .await
        .context("extraction analysis task failed")??;
        self.checkpoint(job_id, "analysis", &analysis)?;
        ensure!(
            !cancellation.load(Ordering::Acquire),
            "cancelled; captures and completed analysis retained"
        );
        ensure!(
            analysis["selected_layer"].is_number(),
            "all extracted directions have zero norm; captures and dataset retained, but no usable vector can be published"
        );
        let mut stored_layers = analysis["layers"].clone();
        let mut artifact_hashes = capture_hashes;
        artifact_hashes.push(dataset_hash.clone());
        for layer in stored_layers.as_array_mut().unwrap() {
            let width = layer["width"].as_u64().context("missing direction width")? as usize;
            let raw: Vec<f32> = serde_json::from_value(layer["raw"].clone())?;
            let raw_hash = self.store.artifacts().put_f32(&[width], &raw)?;
            layer["direction_hash"] = json!(raw_hash);
            artifact_hashes.push(raw_hash);
            if layer["unit"].is_array() {
                let unit: Vec<f32> = serde_json::from_value(layer["unit"].clone())?;
                let hash = self.store.artifacts().put_f32(&[width], &unit)?;
                layer["unit_hash"] = json!(hash);
                artifact_hashes.push(hash);
            }
            layer.as_object_mut().unwrap().remove("raw");
            layer.as_object_mut().unwrap().remove("unit");
        }
        let vector_id = self.record("jobs", job_id)?["details"]["vector_id"]
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(id);
        self.update("jobs", job_id, |job| {
            job["details"]["vector_id"] = json!(vector_id)
        })?;
        let mut draft = json!({"id":vector_id,"created_at":now_seconds(),"name":recipe["concept"],"recipe_id":recipe["id"],"model_id":model_id,
            "model_fingerprint":fingerprint,"selected_layer":analysis["selected_layer"],"layers":stored_layers,"warnings":analysis["warnings"],
            "extraction":recipe["extraction"],"preview_mode":recipe["preview_mode"].as_str().unwrap_or("standard")});
        // Use in-memory directions for preview. Index publication happens only after
        // preview artifacts and the complete provenance manifest are durable.
        let hydrated =
            json!({"id":vector_id,"model_fingerprint":fingerprint,"layers":analysis["layers"]});
        let prompts = if preview_coefficients.is_empty() {
            Vec::new()
        } else {
            neutral_prompts(recipe)?
        };
        let mut previews = Vec::new();
        for prompt in &prompts {
            let prompt = if extraction.method == Method::Paper {
                format!("{} {}", prompt.trim_end(), extraction.readout_suffix)
            } else {
                prompt.clone()
            };
            let raw = extraction.method == Method::Paper || raw;
            for percent in &preview_coefficients {
                ensure!(
                    !cancellation.load(Ordering::Acquire),
                    "cancelled; extracted directions and completed previews retained"
                );
                self.progress(
                    job_id,
                    "previews",
                    0.78 + 0.2
                        * (previews.len() as f64
                            / (prompts.len() * preview_coefficients.len()) as f64),
                )?;
                let sampling = json!({"seed":42,"temperature":0.7,"top_p":0.95,"max_tokens":64});
                let key = torment_nexus::artifacts::sha256(&serde_json::to_vec(
                    &json!({"prompt":prompt,"percent":percent,"sampling":sampling,
                    "extraction_identity":identity,"selected_layer":analysis["selected_layer"],"layers":stored_layers}),
                )?);
                let current = self.record("jobs", job_id)?;
                let (preview, hash) = if let Some(hash) =
                    current["details"]["previews"][&key].as_str()
                {
                    (self.store.artifacts().read_json(hash)?, hash.to_owned())
                } else {
                    let rows = steering::mix_rows(
                        std::slice::from_ref(&hydrated),
                        &json!([{"vector_id":vector_id,"layer":analysis["selected_layer"],"percent":percent}]),
                        fingerprint,
                        capabilities["n_layer"].as_u64().unwrap() as usize,
                        capabilities["n_embd"].as_u64().unwrap() as usize,
                    )?;
                    let target = id();
                    self.update("jobs", job_id, |job| job["engine_target"] = json!(target))?;
                    let (_,mut events)=engine.send_cancellable(json!({"id":target,"op":"generate","messages":[{"role":"user","content":prompt}],"raw":raw,"sampling":sampling,"controls":{"revision":0,"rows":rows}}), &cancellation).await?;
                    let mut history = Vec::new();
                    let mut done = None;
                    let mut partial = String::new();
                    let mut last_saved = std::time::Instant::now();
                    while let Some(event) = events.recv().await {
                        if event["event"] == "token" {
                            partial.push_str(event["text"].as_str().unwrap_or_default());
                        }
                        history.push(event.clone());
                        if last_saved.elapsed().as_secs() >= 1 || event["event"] == "error" {
                            self.checkpoint(job_id,&format!("partial-preview-{key}"),&json!({"prompt":prompt,"percent":percent,
                                "output":partial,"sampling":sampling,"events":history,"partial":true}))?;
                            last_saved = std::time::Instant::now();
                        }
                        if event["event"] == "error" {
                            bail!("preview failed: {}", event["error"]);
                        }
                        if event["event"] == "done" {
                            self.observe_runtime(&event["runtime"]).await;
                            done = Some(event.clone());
                        }
                    }
                    if done.is_none() {
                        self.checkpoint(job_id,&format!("partial-preview-{key}"),&json!({"prompt":prompt,"percent":percent,"output":partial,"sampling":sampling,"events":history,"partial":true}))?;
                    }
                    let done =
                        done.context("preview ended without completion; partial output saved")?;
                    let preview = json!({"prompt":prompt,"percent":percent,"output":done["output"],"sampling":sampling,"events":history,"cancelled":done["cancelled"]});
                    let hash = self.store.artifacts().put_json(&preview)?;
                    // Partial outputs are preserved, but not mistaken for completed previews.
                    if done["cancelled"] == true {
                        self.checkpoint(job_id, &format!("partial-preview-{key}"), &preview)?;
                        bail!("preview cancelled");
                    }
                    self.update("jobs", job_id, |job| {
                        job["details"]["previews"][&key] = json!(hash);
                        job["engine_target"] = Value::Null;
                    })?;
                    (preview, hash)
                };
                previews.push(json!({"prompt":preview["prompt"],"percent":percent,"output":preview["output"],"artifact_hash":hash}));
                artifact_hashes.push(hash);
            }
        }
        let current = self.record("jobs", job_id)?;
        for hash in current["details"]["stages"]
            .as_object()
            .into_iter()
            .flat_map(|x| x.values())
        {
            if let Some(h) = hash.as_str() {
                artifact_hashes.push(h.to_owned());
            }
        }
        for hash in recipe["stages"]
            .as_object()
            .into_iter()
            .flat_map(|x| x.values())
        {
            if let Some(h) = hash.as_str() {
                artifact_hashes.push(h.to_owned());
            }
        }
        artifact_hashes.sort();
        artifact_hashes.dedup();
        draft["previews"] = json!(previews);
        let manifest = json!({"format":"torment-vector-v1","model_fingerprint":fingerprint,"source":model["source"],"runtime":capabilities["runtime"],
            "model_shape":{"n_layer":capabilities["n_layer"],"n_embd":capabilities["n_embd"]},"recipe":recipe,"dataset_hash":dataset_hash,
            "extraction_identity":identity,"extractions":current["details"]["extractions"],
            "algorithm":analysis["algorithm"],"sign_convention":analysis["sign_convention"],"scaling":analysis["scaling"],"split":analysis["split"],
            "selected_layer":analysis["selected_layer"],"layers":stored_layers,"warnings":analysis["warnings"],"previews":previews,"artifacts":artifact_hashes});
        draft["manifest_hash"] = json!(self.store.artifacts().put_json(&manifest)?);
        let mut supplied: HashSet<String> = artifact_hashes.into_iter().collect();
        supplied.insert(field(&draft, "manifest_hash")?.to_owned());
        validate_vector_manifest(&draft, &manifest, fingerprint)?;
        validate_manifest_artifacts(self.store.artifacts(), &draft, &manifest, &supplied)?;
        // Retry after publication reuses the existing immutable record.
        if self.store.get("vectors", &vector_id)?.is_none() {
            self.store.put("vectors", &vector_id, &draft)?;
        }
        self.update("jobs", job_id, |job| job["vector_id"] = json!(vector_id))?;
        Ok(())
    }

    fn edit_recipe(self: &Arc<Self>, request: &Value) -> Result<Value> {
        let old = self.record("recipes", field(request, "recipe_id")?)?;
        self.fork_stage(&old, request)
    }

    fn edit_job_stage(self: &Arc<Self>, request: &Value) -> Result<Value> {
        let job = self.record("jobs", field(request, "job_id")?)?;
        ensure!(
            job["kind"] == "factory",
            "only factory stages can be forked"
        );
        let checkpoints = self.read_stages(&job)?;
        let source = if let Some(recipe_id) = job["recipe_id"].as_str() {
            self.record("recipes", recipe_id)?
        } else {
            json!({"id":null,"version":0,"model_id":job["request"]["model_id"],
                "concept":job["request"]["concept"],"raw":job["request"]["raw"].as_bool().unwrap_or(false),
                "extraction":job["request"]["extraction"],"preview_mode":job["request"]["preview_mode"]})
        };
        let mut snapshot = source;
        snapshot["source_job_id"] = job["id"].clone();
        snapshot["stages"] = job["details"]["stages"].clone();
        snapshot["roles"] = checkpoints
            .get("roles")
            .cloned()
            .unwrap_or(job["request"]["roles"].clone());
        for name in ["design", "review", "dataset"] {
            if let Some(checkpoint) = checkpoints.get(name) {
                snapshot[name] = checkpoint["output"].clone();
            }
        }
        // The source job remains untouched, even if it is still running.
        self.fork_stage(&snapshot, request)
    }

    fn fork_stage(self: &Arc<Self>, old: &Value, request: &Value) -> Result<Value> {
        let stage = field(request, "stage")?;
        let value = &request["value"];
        ensure!(
            matches!(
                stage,
                "design" | "writer_1" | "writer_2" | "writer_3" | "dataset" | "review" | "repair"
            ),
            "stage must be design, writer_1/2/3, review, repair, or dataset"
        );
        let previous = self.store.artifacts().read_json(
            old["stages"][stage]
                .as_str()
                .context("this stage has no completed checkpoint")?,
        )?;
        ensure!(
            previous["status"] == "completed" || previous["edited"] == true,
            "only completed or explicitly edited stages can be forked"
        );
        let recipe_id = id();
        let mut edited = old.clone();
        edited["id"] = json!(recipe_id);
        edited["parent_id"] = old["id"].clone();
        edited["version"] = json!(old["version"].as_u64().unwrap_or(1) + 1);
        edited["created_at"] = json!(now_seconds());
        edited[stage] = value.clone();
        edited["edited_stage"] = json!(stage);
        edited["draft"] = json!(true);
        let mut stages = old["stages"].as_object().cloned().unwrap_or_default();
        // Job-stage forks may also contain completed local products. Preserve
        // their source job, but never label its old analysis as this version's.
        stages.retain(|name, _| name != "analysis" && !name.starts_with("partial-preview-"));
        for key in invalidated_stages(stage) {
            stages.remove(*key);
        }
        if stage != "dataset" {
            edited["dataset"] = json!([]);
            edited["dataset_hash"] = Value::Null;
        }
        if stage == "design" || stage.starts_with("writer_") {
            edited["review"] = Value::Null;
        }
        if stage == "dataset" {
            let prepared = steering::prepare_dataset(value)?;
            edited["dataset"] = prepared["pairs"].clone();
            edited["review"] = Value::Null;
            edited["dataset_hash"] = json!(
                self.store
                    .artifacts()
                    .put_jsonl(edited["dataset"].as_array().unwrap())?
            );
        }
        let checkpoint = if stage == "dataset" {
            "dataset_input"
        } else {
            stage
        };
        let override_value = json!({"edited":true,"output":edited[stage]});
        stages.insert(
            checkpoint.to_owned(),
            json!(self.store.artifacts().put_json(&override_value)?),
        );
        edited["stages"] = json!(stages);
        let factory_request = json!({"concept":old["concept"],"model_id":old["model_id"],"roles":old["roles"],"raw":old["raw"],"parent_recipe_id":recipe_id,
            "extraction":old["extraction"],"preview_mode":old["preview_mode"]});
        let job_id = id();
        let job = json!({"id":job_id,"created_at":now_seconds(),"kind":"factory","status":"queued",
            "stage":"queued","progress":0.0,"error":null,"request":factory_request,
            "model_id":old["model_id"],"details":{"stages":stages,"extractions":{},"previews":{}}});
        self.store.put_batch(&[
            ("recipes", recipe_id.as_str(), edited),
            ("jobs", job_id.as_str(), job),
        ])?;
        self.start_job(job_id.clone(), None)?;
        Ok(json!({"id":recipe_id,"job_id":job_id}))
    }
}

fn invalidated_stages(stage: &str) -> &'static [&'static str] {
    match stage {
        "design" => &[
            "writer_1",
            "writer_2",
            "writer_3",
            "candidates",
            "review",
            "repair",
            "dataset",
            "dataset_input",
        ],
        "writer_1" | "writer_2" | "writer_3" => {
            &["candidates", "review", "repair", "dataset", "dataset_input"]
        }
        "dataset" => &["review", "repair", "dataset", "candidates"],
        "review" => &["repair", "dataset"],
        "repair" => &["dataset"],
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn controls_identify_layers_and_reject_ambiguous_or_changed_membership() {
        let frozen = json!([
            {"vector_id":"a","layer":1,"percent":0},
            {"vector_id":"a","layer":2,"percent":0},
            {"vector_id":"b","layer":1,"percent":0}]);
        let coefficients = vec![
            json!({"vector_id":"b","percent":3}),
            json!({"vector_id":"a","layer":2,"percent":-200}),
            json!({"vector_id":"a","layer":1,"percent":200}),
        ];
        let changed = control_axes(&frozen, &coefficients).unwrap();
        assert_eq!(changed[0]["percent"], 200);
        assert_eq!(changed[1]["percent"], -200);
        assert_eq!(changed[2]["percent"], 3);
        assert_eq!(frozen[0]["percent"], 0); // no mutation of the saved initial axes
        for bad in [
            json!({"vector_id":"a","percent":1}), // ambiguous old-client input
            json!({"vector_id":"a","layer":3,"percent":1}), // not frozen
            json!({"vector_id":"a","layer":1,"percent":1}), // duplicate pair
            json!({"vector_id":"a","layer":null,"percent":1}),
            json!({"vector_id":"a","layer":2,"percent":null}),
        ] {
            let mut invalid = coefficients.clone();
            invalid[1] = bad;
            assert!(control_axes(&frozen, &invalid).is_err());
        }
        assert!(control_axes(&frozen, &coefficients[..2]).is_err());
        let legacy = json!([{"vector_id":"a","layer":7,"percent":0}]);
        assert_eq!(
            control_axes(&legacy, &[json!({"vector_id":"a","percent":-4})]).unwrap()[0],
            json!({"vector_id":"a","layer":7,"percent":-4})
        );
    }

    #[tokio::test]
    async fn multi_layer_vectors_load_once_and_presets_and_events_keep_layer_identity() {
        let (_directory, app) = app();
        let unit_hash = app.store.artifacts().put_f32(&[1], &[1.]).unwrap();
        let layers = json!([
            {"layer":1,"width":1,"unit_hash":unit_hash,"residual_norm":10.,"usable":true},
            {"layer":2,"width":1,"unit_hash":unit_hash,"residual_norm":20.,"usable":true}]);
        let manifest = json!({"format":"torment-vector-v1","runtime":{"engine_revision":ENGINE_REVISION},
            "algorithm":torment_nexus::extraction::LEGACY_ALGORITHM,"sign_convention":"mean_positive_minus_mean_negative",
            "model_fingerprint":"model","selected_layer":1,"layers":layers});
        let manifest_hash = app.store.artifacts().put_json(&manifest).unwrap();
        app.store.put("vectors", "vector", &json!({"id":"vector","model_fingerprint":"model","selected_layer":1,"layers":layers,"manifest_hash":manifest_hash})).unwrap();
        app.store
            .put(
                "models",
                "model",
                &json!({"id":"model","fingerprint":"model"}),
            )
            .unwrap();
        let axes = json!([{"vector_id":"vector","layer":1,"percent":10}, {"vector_id":"vector","layer":2,"percent":-5}]);
        let hydrated = app.load_vectors(&axes, "model").unwrap();
        assert_eq!(hydrated.len(), 1);
        assert_eq!(
            steering::mix_rows(&hydrated, &axes, "model", 3, 1).unwrap(),
            vec![
                json!({"layer":1,"values":[1.]}),
                json!({"layer":2,"values":[-1.]})
            ]
        );
        assert!(
            app.load_vectors(&json!([axes[0], axes[0]]), "model")
                .is_err()
        );
        assert!(
            app.load_vectors(
                &json!([axes[0], {"vector_id":"vector","layer":3,"percent":0}]),
                "model"
            )
            .is_err()
        );
        let preset = app
            .action(json!({"action":"save_mix","model_id":"model","name":"two layers","axes":axes}))
            .await
            .unwrap();
        assert_eq!(
            app.record("presets", preset["id"].as_str().unwrap())
                .unwrap()["axes"],
            axes
        );
        *app.engine_state.write().await =
            json!({"model_id":"model","capabilities":{"n_layer":3,"n_embd":1}});
        app.store.put("runs", "run", &json!({"id":"run","status":"running","model_id":"model","model_fingerprint":"model",
            "axes":axes,"requested_controls":[{"revision":0}]})).unwrap();
        let coefficients = json!([{"vector_id":"vector","layer":2,"percent":0}, {"vector_id":"vector","layer":1,"percent":-3}]);
        // Missing worker is intentional; the accepted request must still persist
        // an unambiguous failed event, not disappear or overwrite another layer.
        assert!(
            app.controls(&json!({"run_id":"run","revision":1,"coefficients":coefficients}))
                .await
                .is_err()
        );
        let run = app.record("runs", "run").unwrap();
        assert_eq!(run["axes"], axes);
        assert_eq!(
            run["requested_controls"][1]["coefficients"],
            json!([
            {"vector_id":"vector","layer":1,"percent":-3}, {"vector_id":"vector","layer":2,"percent":0}])
        );
        assert!(run["requested_controls"][1]["error"].is_string());
    }

    fn app() -> (tempfile::TempDir, Arc<App>) {
        let directory = tempfile::tempdir().unwrap();
        let app = App::new(
            Store::open(directory.path()).unwrap(),
            directory.path().join("nonexistent-worker"),
            Some(directory.path().join("nonexistent-codex")),
        );
        (directory, app)
    }

    #[test]
    fn preview_prompts_use_the_designer_contract() {
        let recipe = json!({"design":{"neutral_prompts":["One", "Two", "Three"],"preview_prompts":["obsolete"]}});
        assert_eq!(neutral_prompts(&recipe).unwrap(), ["One", "Two", "Three"]);
        assert!(neutral_prompts(&json!({"design":{}})).is_err());
        assert!(
            neutral_prompts(&json!({"design":{"neutral_prompts":["One","", "Three"]}})).is_err()
        );
    }

    #[tokio::test]
    async fn self_tools_are_opt_in_revocable_and_revision_scoped() {
        let (_directory, app) = app();
        app.store
            .put("vectors", "v", &json!({"id":"v","name":"Direction"}))
            .unwrap();
        let axes = json!([{"vector_id":"v","layer":1,"percent":0}]);
        let run = json!({"id":"r","status":"running","axes":axes,"requested_controls":[{"revision":0,"coefficients":axes}],"applied_controls":[],"self_modification":false,"self_tools_available":true,"permission_events":[],"messages":[{"role":"user","content":"private transcript"}]});
        app.store.put("runs", "r", &run).unwrap();
        assert!(
            app.model_tool("get_mix", &json!({}), None, "mcp")
                .await
                .is_err()
        );
        app.action(json!({"action":"set_self_modification","run_id":"r","enabled":true}))
            .await
            .unwrap();
        let view = app
            .model_tool("get_mix", &json!({}), None, "mcp")
            .await
            .unwrap();
        assert_eq!(view["sliders"][0]["name"], "Direction");
        assert!(!view.to_string().contains("private transcript"));
        for args in [
            json!({"run_id":"different","expected_revision":0}),
            json!({"run_id":"r","expected_revision":1}),
            json!({"run_id":"r","expected_revision":0,"changes":[{"vector_id":"not-selected","layer":1,"percent":4}]}),
        ] {
            assert!(app.model_tool("set_mix", &args, None, "mcp").await.is_err());
        }
        app.action(json!({"action":"set_self_modification","run_id":"r","enabled":false}))
            .await
            .unwrap();
        assert!(
            app.model_tool("get_mix", &json!({}), Some("r"), "model")
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn library_names_and_deletions_preserve_original_records() {
        let (_directory, app) = app();
        for kind in ["vectors", "recipes", "presets", "conversations"] {
            let original =
                json!({"id":"entry","name":"original","title":"original","concept":"original"});
            app.store.put(kind, "entry", &original).unwrap();
            app.action(
                json!({"action":"rename_record","kind":kind,"id":"entry","name":"  new name  "}),
            )
            .await
            .unwrap();
            assert_eq!(
                app.snapshot().await.unwrap()[kind][0]["display_name"],
                "new name"
            );
            assert!(
                app.action(json!({"action":"rename_record","kind":kind,"id":"entry","name":" "}))
                    .await
                    .is_err()
            );
            let permit = app.local_permit().unwrap();
            assert!(
                app.action(json!({"action":"delete_record","kind":kind,"id":"entry"}))
                    .await
                    .is_err()
            );
            drop(permit);
            app.action(json!({"action":"delete_record","kind":kind,"id":"entry"}))
                .await
                .unwrap();
            assert_eq!(app.snapshot().await.unwrap()[kind][0]["deleted"], true);
            assert_eq!(app.record(kind, "entry").unwrap(), original);
            assert!(
                app.action(
                    json!({"action":"rename_record","kind":kind,"id":"entry","name":"resurrect"})
                )
                .await
                .is_err()
            );
        }
        app.store
            .put(
                "runs",
                "turn",
                &json!({"id":"turn","conversation_id":"entry"}),
            )
            .unwrap();
        assert_eq!(app.snapshot().await.unwrap()["runs"][0]["deleted"], true);
    }

    #[tokio::test]
    async fn archive_and_restore_change_only_library_visibility() {
        let (_directory, app) = app();
        let manifest = json!({"fixture":"immutable artifact"});
        let manifest_hash = app.store.artifacts().put_json(&manifest).unwrap();
        let vector = json!({"id":"vector","name":"Test concept","manifest_hash":manifest_hash});
        let preset = json!({"id":"preset","axes":[{"vector_id":"vector","percent":0}]});
        let run =
            json!({"id":"run","status":"completed","axes":[{"vector_id":"vector","percent":0}]});
        app.store.put("vectors", "vector", &vector).unwrap();
        app.store.put("presets", "preset", &preset).unwrap();
        app.store.put("runs", "run", &run).unwrap();
        assert_eq!(
            app.snapshot().await.unwrap()["vector_visibility"],
            json!([])
        );
        for archived in [true, true, false] {
            let visibility = app
                .action(json!({"action":"set_vector_archived","vector_id":"vector","archived":archived}))
                .await
                .unwrap();
            assert_eq!(visibility["id"], "vector");
            assert_eq!(visibility["archived"], archived);
            assert!(visibility["updated_at"].is_i64());
            let snapshot = app.snapshot().await.unwrap();
            assert_eq!(snapshot["vector_visibility"], json!([visibility]));
            assert_eq!(snapshot["vectors"], json!([vector.clone()]));
            assert_eq!(snapshot["presets"], json!([preset.clone()]));
            assert_eq!(snapshot["runs"], json!([run.clone()]));
            assert_eq!(
                app.store.artifacts().read_json(&manifest_hash).unwrap(),
                manifest
            );
        }
    }

    #[tokio::test]
    async fn archive_requires_an_existing_vector_and_an_explicit_boolean() {
        let (_directory, app) = app();
        app.store
            .put("vectors", "vector", &json!({"id":"vector"}))
            .unwrap();
        for invalid in [Value::Null, json!(0), json!("false"), json!([]), json!({})] {
            let error = app
                .action(
                    json!({"action":"set_vector_archived","vector_id":"vector","archived":invalid}),
                )
                .await
                .unwrap_err();
            assert!(error.to_string().contains("archived must be a boolean"));
        }
        let error = app
            .action(json!({"action":"set_vector_archived","vector_id":"vector"}))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("archived must be a boolean"));
        let error = app
            .action(json!({"action":"set_vector_archived","vector_id":"missing","archived":true}))
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("vectors record missing not found")
        );
        assert!(
            app.action(json!({"action":"set_vector_archived","archived":true}))
                .await
                .is_err()
        );
        assert!(app.store.list("vector_visibility").unwrap().is_empty());
    }

    #[test]
    fn stable_runtime_identity_excludes_observations_not_engine_settings() {
        let mut runtime = json!({"engine_revision":ENGINE_REVISION,"wrapper_sha256":"wrapper-one",
            "context":8192,"batch":512,"microbatch":128,"n_layer":64,"n_embd":4096,
            "cache_k":"F16","cache_v":"F16","resident_bytes":1,"build":"today"});
        let identity = runtime_identity(&runtime).unwrap();
        runtime["resident_bytes"] = json!(200);
        runtime["build"] = json!("tomorrow");
        assert_eq!(runtime_identity(&runtime).unwrap(), identity);
        for key in [
            "wrapper_sha256",
            "context",
            "batch",
            "microbatch",
            "cache_k",
        ] {
            let mut changed = runtime.clone();
            changed[key] = json!("changed");
            assert_ne!(runtime_identity(&changed).unwrap(), identity, "{key}");
            changed.as_object_mut().unwrap().remove(key);
            assert!(runtime_identity(&changed).is_err(), "{key}");
        }
    }

    #[test]
    fn index_must_agree_with_the_checked_vector_manifest() {
        let vector = json!({"selected_layer":1,"layers":[{"layer":1,"usable":true,"auc":1}]});
        let manifest = json!({"format":"torment-vector-v1","runtime":{"engine_revision":ENGINE_REVISION},
            "algorithm":"last-assistant-content-token-difference-of-means-v1","sign_convention":"mean_positive_minus_mean_negative",
            "model_fingerprint":"model","selected_layer":1,"layers":[{"layer":1,"usable":true,"auc":1.0}]});
        validate_vector_manifest(&vector, &manifest, "model").unwrap();
        assert!(validate_vector_manifest(&vector, &manifest, "other").is_err());
        for (key, value) in [
            ("selected_layer", json!(2)),
            ("sign_convention", json!("reversed")),
        ] {
            let mut changed = manifest.clone();
            changed[key] = value;
            assert!(validate_vector_manifest(&vector, &changed, "model").is_err());
        }
        let mut changed = manifest;
        changed["runtime"]["engine_revision"] = json!("unverified");
        assert!(validate_vector_manifest(&vector, &changed, "model").is_err());
    }

    #[tokio::test]
    async fn historical_sign_metadata_does_not_gate_server_loading_presets_or_live_snapshots() {
        // Pure temporary artifacts and an intentionally nonexistent worker: this
        // tests request acceptance without running any model or concept.
        let (_directory, app) = app();
        let unit_hash = app.store.artifacts().put_f32(&[1], &[1.]).unwrap();
        let layers =
            json!([{"layer":1,"width":1,"unit_hash":unit_hash,"residual_norm":10.,"usable":true}]);
        let manifest = json!({"format":"torment-vector-v1","runtime":{"engine_revision":ENGINE_REVISION},
            "algorithm":torment_nexus::extraction::LEGACY_ALGORITHM,"sign_convention":"mean_positive_minus_mean_negative",
            "model_fingerprint":"model","selected_layer":1,"layers":layers,"recipe":{"coefficient_policy":"non_positive"}});
        let manifest_hash = app.store.artifacts().put_json(&manifest).unwrap();
        app.store.put("vectors", "vector", &json!({"id":"vector","model_fingerprint":"model","selected_layer":1,"layers":layers,
            "manifest_hash":manifest_hash,"coefficient_policy":"non_positive"})).unwrap();
        app.store
            .put(
                "models",
                "model",
                &json!({"id":"model","fingerprint":"model"}),
            )
            .unwrap();
        for percent in [200., -200.] {
            let axes = json!([{"vector_id":"vector","layer":1,"percent":percent}]);
            assert_eq!(app.load_vectors(&axes, "model").unwrap().len(), 1);
            app.action(
                json!({"action":"save_mix","model_id":"model","name":"fixture","axes":axes}),
            )
            .await
            .unwrap();
        }
        *app.engine_state.write().await =
            json!({"model_id":"model","capabilities":{"n_layer":2,"n_embd":1}});
        app.store.put("runs", "run", &json!({"id":"run","status":"running","model_id":"model","model_fingerprint":"model",
            "axes":[{"vector_id":"vector","layer":1,"percent":0}],"requested_controls":[{"revision":0}]})).unwrap();
        assert!(app.controls(&json!({"run_id":"run","revision":1,"coefficients":[{"vector_id":"vector","percent":200}]})).await.is_err());
        let run = app.record("runs", "run").unwrap();
        assert_eq!(
            run["requested_controls"][1]["coefficients"][0]["percent"],
            200
        );
        assert!(run["requested_controls"][1]["error"].is_string()); // nonexistent worker, not a sign gate
    }

    fn seed_run(app: &App) {
        app.store
            .put(
                "jobs",
                "job",
                &json!({"id":"job","run_id":"run","status":"running"}),
            )
            .unwrap();
        app.store
            .put(
                "runs",
                "run",
                &json!({"id":"run","status":"running","conversation_id":"chat",
            "messages":[{"role":"user","content":"hello"}],"output":"partial response"}),
            )
            .unwrap();
        app.store
            .put(
                "conversations",
                "chat",
                &json!({"id":"chat","messages":[],"run_ids":[]}),
            )
            .unwrap();
    }

    #[test]
    fn chat_forks_keep_the_exact_reply_and_prefix_without_changing_the_source() {
        let (directory, app) = app();
        seed_run(&app);
        let request = json!({"from_run_id":"run","title":"Fork"});
        assert!(app.new_conversation(&request).is_err());
        app.update("runs", "run", |run| {
            run["status"] = json!("completed");
            run["reply_messages"] = json!([
                {"role":"assistant","content":"Checking the mix."},
                {"role":"user","content":"Tool result: zero."},
                {"role":"assistant","content":"The mix is zero."}]);
        })
        .unwrap();
        app.update("conversations", "chat", |chat| {
            chat["run_ids"] = json!(["earlier", "run", "later"]);
        })
        .unwrap();
        let source = app.record("runs", "run").unwrap();
        let parent = app.record("conversations", "chat").unwrap();
        let fork = app.new_conversation(&request).unwrap();
        assert_eq!(fork["run_ids"], json!(["earlier", "run"]));
        assert_eq!(fork["messages"], json!(run_transcript(&source).unwrap()));
        assert_eq!(fork["messages"][3]["content"], "The mix is zero.");
        assert_eq!(app.record("runs", "run").unwrap(), source);
        assert_eq!(app.record("conversations", "chat").unwrap(), parent);
        let mut child = source.clone();
        child["id"] = json!("child");
        child["conversation_id"] = fork["id"].clone();
        child["messages"] = fork["messages"].clone();
        child["reply_messages"] = json!([{ "role":"assistant","content":"A new branch." }]);
        app.store.put("runs", "child", &child).unwrap();
        app.update("jobs", "job", |job| job["run_id"] = json!("child"))
            .unwrap();
        app.finalize_job("job", "completed", None).unwrap();
        let fork_id = fork["id"].as_str().unwrap();
        let saved = app.record("conversations", fork_id).unwrap();
        assert_eq!(saved["run_ids"], json!(["earlier", "run", "child"]));
        assert_eq!(app.record("conversations", "chat").unwrap(), parent);
        assert_eq!(
            app.new_conversation(&json!({"from_run_id":"child"}))
                .unwrap()["run_ids"],
            saved["run_ids"]
        );

        // Scratchpad runs and older runs without explicit tool messages also fork.
        child.as_object_mut().unwrap().remove("conversation_id");
        child.as_object_mut().unwrap().remove("reply_messages");
        app.store.put("runs", "child", &child).unwrap();
        let scratch = app
            .new_conversation(&json!({"from_run_id":"child"}))
            .unwrap();
        assert_eq!(scratch["run_ids"], json!(["child"]));
        assert_eq!(
            scratch["messages"].as_array().unwrap().last().unwrap()["content"],
            child["output"]
        );
        assert_eq!(
            app.new_conversation(&json!({})).unwrap()["messages"],
            json!([])
        );
        drop(app);
        assert_eq!(
            Store::open(directory.path())
                .unwrap()
                .get("conversations", fork_id)
                .unwrap()
                .unwrap(),
            saved
        );
    }

    #[test]
    fn durable_finalization_commits_run_job_and_chat_together() {
        let (_directory, app) = app();
        seed_run(&app);
        app.finalize_job("job", "cancelled", None).unwrap();
        let run = app.record("runs", "run").unwrap();
        assert_eq!(run["status"], "cancelled");
        let artifact = app
            .store
            .artifacts()
            .read_json(run["manifest_hash"].as_str().unwrap())
            .unwrap();
        assert_eq!(artifact["output"], "partial response");
        assert_eq!(artifact["status"], "cancelled");
        assert_eq!(app.record("jobs", "job").unwrap()["status"], "cancelled");
        let chat = app.record("conversations", "chat").unwrap();
        assert_eq!(chat["run_ids"], json!(["run"]));
        assert_eq!(chat["messages"][1]["content"], "partial response");
        app.finalize_job("job", "cancelled", None).unwrap();
        assert_eq!(
            app.record("conversations", "chat").unwrap()["run_ids"],
            json!(["run"])
        );
    }

    #[test]
    fn failed_manifest_write_cannot_claim_completion_or_advance_chat() {
        let (directory, app) = app();
        seed_run(&app);
        let root = app.store.artifacts().root();
        std::fs::rename(root, directory.path().join("saved-artifacts")).unwrap();
        std::fs::write(root, b"not a directory").unwrap();
        assert!(app.finalize_job("job", "completed", None).is_err());
        assert_eq!(app.record("jobs", "job").unwrap()["status"], "running");
        assert_eq!(app.record("runs", "run").unwrap()["status"], "running");
        assert_eq!(
            app.record("conversations", "chat").unwrap()["messages"],
            json!([])
        );
    }

    #[test]
    fn raw_mode_overrides_create_one_reusable_immutable_version() {
        let (_directory, app) = app();
        let recipe = json!({"id":"recipe","version":1,"model_id":"model","raw":false});
        app.store.put("recipes", "recipe", &recipe).unwrap();
        let request = json!({"recipe_id":"recipe","model_id":"model","raw":true});
        let job_id = app.create_job("extract", &request).unwrap();
        let changed = app.recipe_for_extraction(&job_id, &request).unwrap();
        assert_eq!(changed["raw"], true);
        assert_eq!(changed["parent_id"], "recipe");
        assert_eq!(
            app.recipe_for_extraction(&job_id, &request).unwrap(),
            changed
        );
        assert_eq!(app.record("recipes", "recipe").unwrap(), recipe);
        assert_eq!(app.store.list("recipes").unwrap().len(), 2);
        let draft = json!({"id":"draft","model_id":"model","draft":true});
        app.store.put("recipes", "draft", &draft).unwrap();
        let request = json!({"recipe_id":"draft","model_id":"model"});
        let job_id = app.create_job("extract", &request).unwrap();
        assert!(app.recipe_for_extraction(&job_id, &request).is_err());
    }

    #[test]
    fn paper_override_versions_the_recipe_and_retains_historical_metadata() {
        let (_directory, app) = app();
        let recipe = json!({"id":"recipe","version":1,"model_id":"model","coefficient_policy":"non_positive","preview_mode":"none"});
        app.store.put("recipes", "recipe", &recipe).unwrap();
        let request = json!({"recipe_id":"recipe","model_id":"model","extraction":Extraction::default(),"coefficient_policy":"unrestricted"});
        let job_id = app.create_job("extract", &request).unwrap();
        let changed = app.recipe_for_extraction(&job_id, &request).unwrap();
        assert_eq!(
            Extraction::from_record(&changed).unwrap(),
            Extraction::default()
        );
        assert_eq!(changed["coefficient_policy"], "non_positive");
        assert_eq!(changed["preview_mode"], "none");
        assert_eq!(changed["version"], 2);
        assert_eq!(
            app.recipe_for_extraction(&job_id, &request).unwrap(),
            changed
        );
        assert_eq!(app.record("recipes", "recipe").unwrap(), recipe);
        assert_eq!(
            Extraction::from_record(&recipe).unwrap(),
            Extraction::legacy()
        );
    }

    #[test]
    fn new_concepts_default_to_paper_with_previews_off_but_explicit_sets_are_allowed() {
        let request = new_concept_settings(json!({})).unwrap();
        assert_eq!(
            Extraction::from_record(&request).unwrap(),
            Extraction::default()
        );
        assert!(preview_percentages(&request).unwrap().is_empty());
        let explicit = new_concept_settings(
            json!({"preview_mode":"standard","coefficient_policy":"non_positive"}),
        )
        .unwrap();
        assert_eq!(preview_percentages(&explicit).unwrap(), [-10., 0., 10.]);
    }

    #[tokio::test]
    async fn a_published_factory_recipe_resumes_locally_without_codex() {
        let (_directory, app) = app();
        app.store
            .put("recipes", "recipe", &json!({"id":"recipe"}))
            .unwrap();
        let job_id = app
            .create_job("factory", &json!({"model_id":"missing-model"}))
            .unwrap();
        app.update("jobs", &job_id, |job| {
            job["recipe_id"] = json!("recipe");
            job["details"]["recipe_published"] = json!(true);
        })
        .unwrap();
        let error = app
            .run_job(&job_id, Arc::new(AtomicBool::new(false)), None)
            .await
            .unwrap_err();
        // Reaches local model validation, not the deliberately nonexistent Codex binary.
        assert!(
            error
                .to_string()
                .contains("models record missing-model not found"),
            "{error:#}"
        );
    }

    #[tokio::test]
    async fn terminal_manifest_waits_for_inflight_control_outcomes() {
        let (_directory, app) = app();
        seed_run(&app);
        app.update("jobs", "job", |job| {
            job["kind"] = json!("deliberately-invalid-job-kind");
            job["request"] = json!({});
        })
        .unwrap();
        let control = app.controls_lock.lock().await;
        app.start_job("job".into(), None).unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        assert_eq!(app.record("runs", "run").unwrap()["status"], "running");
        app.update("runs", "run", |run| {
            run["requested_controls"] =
                json!([{"revision":1,"error":"finished before application"}]);
        })
        .unwrap();
        drop(control);
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while app.record("runs", "run").unwrap()["status"] == "running" {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        let run = app.record("runs", "run").unwrap();
        let saved = app
            .store
            .artifacts()
            .read_json(run["manifest_hash"].as_str().unwrap())
            .unwrap();
        assert_eq!(saved["requested_controls"], run["requested_controls"]);
    }

    #[tokio::test]
    async fn editing_a_completed_writer_forks_without_mutating_the_running_source() {
        let (_directory, app) = app();
        let mut stages = json!({});
        for name in [
            "design",
            "writer_1",
            "writer_2",
            "writer_3",
            "candidates",
            "review",
            "repair",
            "dataset",
            "analysis",
        ] {
            stages[name] = json!(
                app.store
                    .artifacts()
                    .put_json(&json!({"status":"completed","output":{"original":name}}))
                    .unwrap()
            );
        }
        let original = json!({"id":"source","kind":"factory","status":"running",
            "request":{"concept":"dry humor","model_id":"model","raw":false},
            "details":{"stages":stages,"extractions":{"old":"must-not-be-reused"}}});
        app.store.put("jobs", "source", &original).unwrap();
        let result = app
            .edit_job_stage(&json!({"job_id":"source","stage":"writer_2","value":{"pairs":[]}}))
            .unwrap();
        assert_eq!(app.record("jobs", "source").unwrap(), original);
        let recipe = app
            .record("recipes", result["id"].as_str().unwrap())
            .unwrap();
        assert_eq!(recipe["source_job_id"], "source");
        assert_eq!(recipe["draft"], true);
        for unaffected in ["design", "writer_1", "writer_3"] {
            assert_eq!(recipe["stages"][unaffected], stages[unaffected]);
        }
        for removed in ["candidates", "review", "repair", "dataset", "analysis"] {
            assert!(recipe["stages"].get(removed).is_none(), "{removed}");
        }
        let edited = app
            .store
            .artifacts()
            .read_json(recipe["stages"]["writer_2"].as_str().unwrap())
            .unwrap();
        assert_eq!(edited, json!({"edited":true,"output":{"pairs":[]}}));
        let job = app
            .record("jobs", result["job_id"].as_str().unwrap())
            .unwrap();
        assert_eq!(job["details"]["extractions"], json!({}));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn persistence_failure_stops_inflight_worker_before_releasing_reservation() {
        use std::os::unix::fs::PermissionsExt;
        let (directory, app) = app();
        let fixture = directory.path().join("generation-worker");
        std::fs::write(
            &fixture,
            include_bytes!("../tests/fixtures/generation_worker.py"),
        )
        .unwrap();
        std::fs::set_permissions(&fixture, std::fs::Permissions::from_mode(0o700)).unwrap();
        let worker = Engine::start(&fixture, &directory.path().join("logs"))
            .await
            .unwrap();
        *app.worker.lock().await = Some(worker.clone());
        *app.engine_state.write().await = json!({"status":"idle","model_id":"model",
            "capabilities":{"n_layer":2,"n_embd":2}});
        seed_run(&app);
        app.update("jobs", "job", |job| {
            job["kind"] = json!("generate");
            job["request"] = json!({});
        })
        .unwrap();
        app.update("runs", "run", |run| {
            run["model_id"] = json!("model");
            run["model_fingerprint"] = json!("test-model");
            run["axes"] = json!([]);
            run["requested_controls"] = json!([{"revision":0}]);
        })
        .unwrap();
        let root = app.store.artifacts().root();
        std::fs::rename(root, directory.path().join("saved-artifacts")).unwrap();
        std::fs::write(root, b"unwritable artifact directory").unwrap();
        let permit = app.local_permit().unwrap();
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(3),
            app.run_job("job", Arc::new(AtomicBool::new(false)), Some(permit)),
        )
        .await
        .unwrap();
        assert!(result.is_err());
        assert!(!worker.is_alive());
        assert!(app.worker.lock().await.is_none());
        assert_eq!(app.engine_state.read().await["status"], "unloaded");
        assert!(app.local_permit().is_ok());
    }
}
