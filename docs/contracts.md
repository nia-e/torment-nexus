# v1 integration contracts

This is an implementation coordination document. The user plan remains authoritative.

## Local browser API

All API requests send `Authorization: Bearer <launch token>`. Browser obtains token
from the launch URL fragment, removes it from the URL and keeps it in sessionStorage.
Origin/Host must match the loopback listener. `GET /api/state` returns a JSON snapshot:

```
{models:[], recipes:[], vectors:[], vector_visibility:[], jobs:[], runs:[], presets:[], conversations:[],
 engine:{status:"unloaded",model_id:null,capabilities:null},codex:{models:[],error:null}}
```

Records have `id`, `created_at` (epoch seconds), and the fields below. Snapshot lists
contain full records (large activation tensors and raw agent output remain artifacts).

`POST /api/action` accepts `{action: <name>, ...fields}` and returns JSON, usually
`{id}` or `{job_id}`. Errors are non-2xx `{error: "actionable explanation"}`.

- `import_model`: `path`, optional `name`; record: name,path,fingerprint,source,size_bytes.
- `browse_hf`: `repo`; response: `{repo,revision,files:[{name,size_bytes}]}`.
- `download_model`: `repo`, `file`, optional `revision`; creates resumable job.
  Bonsai preset: prism-ml/Ternary-Bonsai-2-27B-gguf / Ternary-Bonsai-2-27B-PQ2_0.gguf.
- `load_model`: `model_id`, optional `context` (8192), `gpu_layers` (99).
- `unload_model`: no fields. One engine job at once; busy is visible, not raced.
- `discover_codex`: response `{models:[{id,displayName,isDefault,...}],assignments:...}`.
- `create_concept`: `concept`, `model_id`, optional `roles:{designer, writers:[3],reviewer}`,
  optional `raw` false; optional `extraction:{method:"paper",readout_suffix:"I feel:"}`
  (the default for new requests), or `{method:"completion",readout_suffix:""}`.
  Optional `preview_mode:"standard"|"negative_only"|"none"` (default `none` for new requests).
  Historical `coefficient_policy` metadata does not restrict coefficient signs.
  Agent content alone leaves the machine, not local chat.
- `edit_recipe`: `recipe_id`, `stage` (design/writer_1/writer_2/writer_3/review/repair/dataset), `value` (JSON).
  Creates a new immutable recipe version; descendants are not reused incorrectly.
- `edit_job_stage`: `job_id`, `stage`, `value`; fork any completed factory checkpoint
  into a new draft recipe and job, without altering the running original.
- `retry_job`: `job_id`; resumes only incomplete stages; no silent automatic retry on launch.
- `extract_recipe`: `recipe_id`, `model_id`, optional `raw` and `extraction` overrides.
  Omitted settings preserve the recipe, including historical completion extraction.
  Changed settings/model create an immutable recipe version; retry reuses that version.
  Preview settings and historical metadata are inherited by this action.
- `cancel_job`: `job_id`.
- `generate`: `model_id`, `messages:[{role,content}]`, `axes:[{vector_id,layer,percent}]`,
  `sampling:{seed:42,temperature:0.7,top_p:0.95,max_tokens:256}`, optional `raw` false,
  optional `conversation_id`, optional `baseline_of`. Returns `{run_id,job_id}`.
- `controls`: `run_id`, `revision` (monotone int), `coefficients:[{vector_id,layer,percent}]`.
  Full snapshot for frozen axes, identified by the unique **(vector_id, layer)** pair.
  One concept may contribute at multiple layers. Layer and vector membership cannot
  change mid-response. Legacy clients may omit `layer` only for a vector used at
  exactly one frozen layer; ambiguous requests fail instead of choosing a layer.
  New persisted coefficient events always include explicit layer IDs.
- `cancel_run`: `run_id`.
- `save_mix`: `name`, `model_id`, `axes` as above. Returns preset id.
- `new_conversation`: optional `title` and `from_run_id`; returns the conversation,
  including `id`. A source run seeds the full transcript through its reply (including
  tool exchanges) and shares only that prefix's run IDs, without changing the source.
  Active responses must be stopped before forking. Later turns stay in their own branch.
- `export_vector`: `vector_id`; returns portable checked artifact bundle JSON.
- `import_vector`: `bundle` as exported; validates all hashes, shapes, model binding.
- `set_vector_archived`: `vector_id`, required boolean `archived`; returns
  `{id:vector_id,archived,updated_at}`. This separate mutable `vector_visibility`
  record only hides the library card. All vector/recipe records and artifacts remain
  immutable; saved mix/run references and current coefficients are unchanged.
- `artifact`: `hash`; returns parsed JSON for a stored artifact (editor/provenance).

Vector fields: id,name,recipe_id,model_id,model_fingerprint,selected_layer,
layers:[{layer,auc,residual_norm,direction_hash,unit_hash,width}],warnings,manifest_hash,previews.
New vectors also carry `extraction` and `preview_mode` bound to the immutable
recipe. Historical `coefficient_policy` fields are provenance only, not an active
control constraint. Layers fitted with the adapted method retain `fold_aucs` and `removed_control_pcs`.
Recipe fields: id,concept,model_id,parent_id,version,design,dataset,review,stages,roles,warnings.
Dataset pair fields: id,family,messages:[{role,content}],positive,negative,split (train/diagnostic).
That pair-level split is a legacy diagnostic annotation, not the fit mask for the
[Tagliabue et al. adaptation](paper-extraction.md). This method stores its family-fold
assignment in the manifest's `split` and fits all pairs; common messages are
provenance only, not decoded by this method.
Job fields: id,kind,status (queued/running/completed/failed/interrupted/cancelled),stage,
progress (0..1),error,recipe_id,model_id,run_id,details (including per-agent progress).
Run fields: id,model_id,messages,axes,sampling,output,status,requested_controls,
applied_controls,conversation_id,baseline_of,manifest_hash,error.
Applied `mix_diagnostics` is `{revision,first_token_index,geometry}`; each requested
control event also retains its own geometry. Pending geometry is not called applied.

`GET /api/events` is an authenticated SSE stream, consumed via streaming fetch (no token
in a query string). Each data payload: `{kind,run_id?,job_id?,data}`. Kinds include
state, token, applied, progress, error, completed. `token.data={text,index,revision}`;
`applied.data={revision,first_token_index}`. Snapshot polling is a recovery fallback.

## Engine JSONL v1

Stdin commands: `{v:1,id:<string>,op:<operation>,...}`.
Stdout events: `{v:1,id:<command id>,event:<kind>,...}`. Final kinds: result, done, error.
Stderr logs never contain protocol JSON. One owner thread mutates inference state.

- `capabilities`, `load` {path,context:8192,gpu_layers:99,batch:512,microbatch:128}, `unload`.
- `extract` {messages,completion,layers:[graph ids],raw:false,prefix_chunk:512}.
  result: rendered,prefix,token_ids,capture_position,template,settings,captures:
  [{layer,values:[f32],norm}], runtime. Steering always disabled and all memory reset.
- `generate` {messages,raw:false,sampling,controls:{revision:0,rows:[{layer,values:[f32]}]}}.
  events: rendered,applied,token,done. Full zero-filled legacy buffer replaces controls.
- `controls` {target:<generate id>,revision,rows}; ack via generate event, errors via command id.
- `cancel` {target:<job id>}. At a decode boundary; no inference mutation on reader thread.

Graph layers are 1..n_layer-1. Legacy buffer row for graph layer L is L-1.

## Rust module ownership/interfaces

Root owns engine/, src/engine.rs, src/main.rs, src/app.rs, integration, launcher/package.
Storage worker owns src/store.rs, src/artifacts.rs, src/download.rs.
Codex worker owns src/codex.rs, src/factory.rs plus own tests/docs.
Steering worker owns src/steering.rs plus own tests/docs.
Frontend worker owns frontend/ (including prebuilt dist) and browser tests.
Do not edit another worker's files without coordination. Use anyhow::Result and serde_json
at module boundaries; agree exact exported Rust signatures with root before integration.

## Opt-in self-adjustment and continuation

`generate` also accepts `self_modification:bool` (default false) and
`sampling.unbounded:bool` (default false). Baselines force self-modification off.
`set_self_modification` takes `run_id,enabled`; revocation is live, serialized with
control writes. Enabling mid-run requires that the run started with tools available.
`controls` may send `base_revision` for compare-and-set; stale snapshots are rejected.
New requested-control events include trusted `source:user|model|mcp` and may include
`reason`. SSE `controls_requested` mirrors complete accepted snapshots into the UI.

Run provenance includes `self_tools_available`, current `self_modification`,
`permission_events`, `tool_calls` (input, result, after_token_index), and
`context_rollovers` (boundary, retained-prefix/recent counts, dropped-token count).
The original user messages are kept separate from the rendered tool instructions;
the rendered prompt artifact contains exactly what the worker received.

The engine advertises `self_tools` and `unbounded_output`. Opt-in `generate` accepts
`tools_enabled`; a generated `torment_tool` envelope produces `tool_call` and pauses.
The host sends `tool_result {target,tool_id,result}`; only the matching active call
can be answered. Cancellation and complete control updates remain available while
paused. Tool feedback is decoded as plain text, without special-token parsing.
`context_rollover` events disclose full memory resets followed by prefix/tail replay.
Sampling has no output-token cap when unbounded, but EOS and cancellation still end it.

`POST /mcp` exposes `get_mix` and `set_mix` through stateless Streamable HTTP. It has
its own launch token (shown in authenticated `/api/state` as `mcp:{url,token}`). That
token never authenticates `/api/*`. The same loopback Host/Origin checks apply, with
a 64 KiB MCP request limit. Only the active, opted-in response is visible to tools;
there are no transcript, model-loading, filesystem or permission-granting tools.
