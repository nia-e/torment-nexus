# Steering math and dataset boundary

`src/steering.rs` contains no inference. Its checks establish data and algebra
invariants; engine tests must independently establish correct token capture,
state resets, graph layers, and live-update timing.

## Dataset and split

New concepts default to [extraction adapted from Tagliabue et al. (2026)](paper-extraction.md). Missing
settings in historical artifacts mean completion contrast, not the new default.

`prepare_dataset(dataset)` accepts a pair array or `{pairs: [...]}`. Pairs require
`id`, `family`, `messages: [{role, content}]`, `positive`, and `negative`. It returns
`{pairs, warnings, duplicates, split}`. Messages are common context, not generated
completions. Positive and negative assistant completions must be nonempty.

The function rejects malformed records and repeated IDs. It removes repeated
contrasts with the same context and completions, ignoring only surrounding
whitespace and CRLF/LF differences. Internal whitespace remains meaningful.
Original source outputs belong in immutable recipe provenance.

Family names are lowercased and whitespace-normalized for splitting. Families
from the automatic factory first resolve exact or uniquely shortened labels
against the designer's declared `Name: description` entries. Those metadata-only
corrections are recorded; raw writer outputs are retained. Ambiguous labels and
arbitrary semantic paraphrases are not automatically merged.
For completion contrast, families are ordered by SHA-256, then a deterministic subset-sum selects the closest
available 25% diagnostic pair count without splitting a family. Ties prefer the
smaller diagnostic partition. One-family datasets remain entirely training;
the missing held-out diagnostic is explicit. Small datasets, identical poles,
and large completion-length imbalances produce warnings, not invented examples.
These checks do not establish the absence of semantic confounds. The adapted
analysis instead uses whole-family CV folds and an all-pair final fit; the legacy
pair-level split annotation is ignored. Old artifacts are never regrouped silently.

## Captures and outputs

`analyze_activations(dataset, captures, model_fingerprint)` accepts captures as:

```json
[
  {"pair_id":"pair-1","pole":"positive","captures":[{"layer":8,"values":[1.0,2.0]}]},
  {"pair_id":"pair-1","pole":"negative","captures":[{"layer":8,"values":[0.0,2.0]}]}
]
```

There must be exactly one record per accepted pair and pole. Every record must
contain the same supported graph layers and residual width. A provided `norm`
is checked against the captured values. The caller must supply unsteered captures
from independently reset inference state: final raw readout token for the adapted method,
last assistant content token for completion contrast. The artifact validator binds
the rendered text, suffix, token position and template settings to the recipe.

For completion contrast, training captures determine the raw mean-positive-minus-mean-
negative direction, its unit direction, and the median L2 norm across **both**
training poles. Diagnostic captures never enter these estimates. AUC uses all
held-out positive/negative projection comparisons, with half credit for ties.
Highest AUC wins; equal AUC chooses the earlier graph layer. With no held-out
families, the earliest usable layer is selected and AUC remains `null`.

Output `layers[]` contains `layer`, `width`, `raw`, `unit`, `raw_norm`,
`residual_norm`, `auc`, `usable`, and per-layer warnings. A zero direction or
calibration norm is retained as an unusable layer with `unit: null`. If every
layer is unusable, `selected_layer` is `null`: retain the diagnostics, but do not
publish a fake direction. AUC means **separation**, not established steering
quality or a measure of subjective experience.

The adapted method uses control-centered PCA in `src/paper.rs`, fitted separately in each
training fold, removing the leading components covering 50% of control variance.
It selects by mean fold AUC (earlier layer wins ties), then refits all pairs.
Calibration is the median residual norm over both poles of **all** accepted pairs.
No independent final test is implied. Control-side denoising is asymmetric:
swapping the poles need not exactly negate the resulting direction. The exact pole-
reversal invariant applies to plain completion difference-of-means only.

## Mixing

`mix_rows(vectors, axes, model_fingerprint, n_layer, width)` accepts final vector
records with unit tensors loaded into `layers[].unit`. Axes are
`{vector_id, layer, percent}`. For each layer it adds, without renormalizing:

```text
delta = sum((percent / 100) * residual_norm * unit)
```

Positive, zero, and negative finite percentages are supported. Different model
fingerprints, invalid shapes, non-unit directions, unusable layers, duplicate
selected **(vector ID, layer)** pairs, nonfinite data, and F32 overflow are errors.
The same concept may contribute independently at multiple layers; its artifact
is hydrated only once. Each call
constructs every supported layer row afresh, including zeros. Removing an axis
therefore cannot leave its previous contribution behind.

New library selections expose every extracted usable layer at zero. Old saved
single-layer mixes preserve their existing values and gain zero-valued rows only
before the next response. Frozen runs never acquire new members mid-generation.

There is no per-vector sign restriction or arbitrary percentage ceiling, including
200%. Historical `coefficient_policy` fields are retained as provenance only.
Numeric validity and representable F32 buffers remain required.

`mix_diagnostics` takes the same arguments and returns `warnings`, `cosines`, and
`layers`. It compares only same-layer directions. Layer diagnostics include the
actual summed F32 injection norm, the sum of individual contribution norms,
and their ratio (showing cancellation). A single calibrated percentage is
reported only if all selected axes on that layer have equal calibration norms
within relative tolerance `1e-5`; otherwise each calibration reference is shown
separately. Expanded coefficients and strong overlap are advisory, never vetoes.

Graph layers are `1` through `n_layer-1`; graph layer zero is unsupported. The explicit
legacy buffer index is `graph_layer - 1`. `supported_layers` chooses distinct
nearest graph indices to 25%, 40%, 55%, 70%, and 85% of `n_layer-1`, clamped to
the supported range. Small models can provide fewer than five distinct layers.

## Self-adjustment and MCP

**Allow self-adjustment** is off by default. Enable it before starting a response.
The model can call `get_mix` and `set_mix` for that response's selected vector/layer
pairs only. The same tools are available at the local `/mcp` endpoint. The **MCP
connection** section of the mixer shows its URL and a separate launch-scoped bearer
token; that token cannot access the browser API or chat transcripts. No client is
configured automatically, and dataset-generation agents remain tool-isolated.

The worker uses llama.cpp's native tool templates and parsers when the model's
GGUF template supports them (tested with Gemma 4 and Bonsai). Tools are supplied
as function schemas; calls and results use the model's own syntax. No adapter
selection is needed. Template-free or unsupported models retain the
`<steering_tool>{"name":...,"arguments":...}</steering_tool>` text bridge and
`<steering_result>` replies; historical envelopes remain accepted there.

Native calls execute only at a completed model turn, never from partial streamed
arguments. Rust validates each request using the same permissions and revision
checks as MCP. The worker appends the template-framed result to its existing
state, without replaying earlier tokens under new coefficients. Result data is
tokenized as plain text; only template boundaries parse special tokens. If a
native template cannot safely append its result, generation fails visibly rather
than silently resetting state. Calls are separate from prose and saved as
structured assistant/tool messages for subsequent turns and forks. Malformed
or incomplete calls never bypass validation. A model may still choose not to
call a tool; that is not a successful adjustment.

`get_mix` returns the active run ID, current control revision, slider IDs/layers,
names, percentages, and the last applied revision. These tools target the local
model generating the reply, not the human or an external MCP client. Percentages
are absolute coefficients: zero disables a contribution and negative values
reverse its direction. Changing a whole concept requires listing each layer. `set_mix` takes `run_id`,
`expected_revision`, and a partial `changes` array of `{vector_id,layer,percent}`;
optional `reason` is recorded. Unmentioned axes remain unchanged. It cannot add
vectors/layers, modify recipes, read chat, start inference, or grant itself access.
Finite signed values are accepted, including values outside ±20%. The host expands
changes to a complete engine buffer. Stale revisions fail rather than overwriting a
newer mix. Accepted model/MCP changes update the browser controls without echoing
back as manual writes. Manual edits can supersede them.

Uncheck **Allow self-adjustment** during generation to revoke access. Revocation
serializes with accepted control writes: once its acknowledgement returns, no later
tool request can change the mix. It does not undo an already applied change; use
**Zero all** or your own coefficients for that. A response begun with tools disabled
cannot enable them halfway through; enable them for the next response instead.
Baseline comparisons always disable tools as well as steering.

MCP is stateless Streamable HTTP using the
[2025-06-18 transport contract](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports)
and [tools interface](https://modelcontextprotocol.io/specification/2025-06-18/server/tools).
POST JSON-RPC to `/mcp` with `Authorization: Bearer <MCP token>`, `Content-Type:
application/json`, and `Accept: application/json, text/event-stream`. Initialize,
then use `tools/list` and `tools/call`. GET returns 405 (no server-push SSE stream).
This surface controls the **active response**, not an idle browser-local draft.

## Unbounded output

**Unbounded output** removes the output-token cap. Natural end-of-generation and
**Stop** still end the response; it does not force endless speech. At a full context
window, the worker clears both attention and recurrent state and replays the first
quarter-window (or shorter original prefix) plus the most recent half-window under
the current mix. Earlier middle context is dropped, not summarized. The saved
output is not erased, and each rollover records its token boundary and retained/
dropped counts. Long initial inputs use the same explicit prefix/tail policy.

This is bounded-context continuation, not unlimited memory or exact recall.
Ordinary prompt-size, record-size, available-memory and disk limits still apply.
All token counts, tool calls, requested/applied control events, and permission
changes remain in the run's provenance.
