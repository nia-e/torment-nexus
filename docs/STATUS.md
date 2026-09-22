# Verified implementation status

## Build and scope

Torment Nexus 0.1.0 is implemented as a local Rust/Axum application with an embedded
React UI and a separate pinned Prism inference worker. Verification below was run
on 21 September 2026, Apple M4 Max / 36 GiB, macOS arm64, Rust 1.97.1 and
Codex CLI 0.155.1. It is a local unsigned build, not a notarized app bundle.

- Engine: `922be44aa6ac81b46f092716351cddff1c1733a7`.
- Wrapper source SHA-256: `b116e97ad07f68a62dcd4afdf035f304663e4981e783db78b71359e6f9af93be`.
- Bonsai: `prism-ml/Ternary-Bonsai-2-27B-gguf`, revision
  `6ed5e12bf84b7a63069882c91dd9e9218647d17b`, PQ2_0 file, 7,206,168,928 bytes.
- Bonsai file SHA-256: `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`.
- Tiny GGUF SHA-256: `270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d`.

## Real engine checks — passed

These used Metal inference, not a mock. Both tiny GGUF and genuine Bonsai passed:

| Check | Tiny | Bonsai |
| --- | ---: | ---: |
| Zero/disabled capture maximum absolute difference | 0 | 0 |
| A → B → A after complete state resets | 0 | 0 |
| Signed final-token injected delta maximum error | 2.98e-8 | 5.96e-8 |
| Removed control rows, maximum residual difference | 0 | 0 |
| Prefix-chunk variation, maximum absolute difference | 0.001884 | 0.005654 |
| Zero/disabled generated token sequence | identical | identical |

Capture position/token IDs and exact layer shapes were stable across prefix chunks
1, 7 and 512. Chunk-dependent Metal arithmetic is **not** bitwise identical; the
measured differences above passed the explicit numerical bound in the harness.
Tiny logits also matched exactly for zero/disabled controls. Live changes were
acknowledged at output token 3, prior tokens matched baseline, cancellation worked,
and a model switch during generation was rejected. Explicit raw-mode single/multi-
message formatting, control-token rejection and recovery also passed on tiny GGUF.

## Real application loop — passed

The app itself downloaded and SHA-verified Bonsai from the immutable revision.
Using the existing Codex login, two independent factories ran designer/reviewer
Astra and writer threads Astra, Sol and Terra:

- **Playful dry humor versus earnest literal delivery**: 96 accepted/repaired pairs,
  grouped 72 extraction / 24 diagnostic. Selected graph layer 25, held-out AUC 0.97049.
- **Sensory concrete detail versus abstract generality**: 96 pairs, grouped 72 / 24.
  Selected graph layer 16, held-out AUC 0.99132.

Each has five extracted layers, complete two-pole captures, nine negative/zero/positive
previews, immutable recipe/stage artifacts and model/runtime provenance. These AUCs
measure separation, **not causal steering quality or subjective experience**. The
reviewer's confound warnings remain attached; length, punctuation and recurring
imagery were explicitly flagged rather than hidden.

A separate NumPy audit recomputed mean-of-means vectors from the real saved captures,
unit normalization, calibration medians, held-out AUCs, pole reversal and signed
scaling. Largest raw-vector disagreement was 1.20e-7; families were disjoint and
layer selection reproduced. This is independent arithmetic, not another call to
the production steering module.

The real app exercised two-axis live mixing (+8%, −6%), an acknowledgement at token
13, same-input/seed/settings zero baseline, two-turn chat, saved mix, complete bundle
export/import and browser JSON round-trip import. Corrupt blobs, wrong-model mixes/
imports, and a dataset reference hidden from a manifest's artifact list were rejected.

Native Firefox UI exercised signed mixing (+4%, −3% → −5%, −3%), frozen vector/layer
membership, revision 1 at token 310, reset-to-zero revision 2 at token 736, Stop with
960 saved tokens, and side-by-side baseline (1024 tokens, correct baseline_of link).
The actual five-layer diagnostics and nine-preview gallery were inspected. The
native window became unavailable before the final screenshot; backend completion
was checked independently. Browser fixture tests do not substitute for these runs.

## Recovery and defensive checks

- SIGKILL during cloud review/repair: both jobs reopened as interrupted, with all
  completed designer/writer/reviewer checkpoint hashes unchanged. Startup made no
  automatic cloud request. Explicit retry reused the completed work.
- With the Codex executable deliberately unavailable, published recipes resumed
  local extraction/publication using **every** previous stage/capture/preview hash
  unchanged. No cloud job was repeated.
- Exact float round-tripping is required: a real calibration/direction norm exposed
  a one-ULP default JSON parse error. `serde_json/float_roundtrip` and a regression
  protect checked manifests. Browser integer/float metadata normalization is handled
  without accepting large-integer precision loss.
- Cancellation enqueue races, late controls versus manifest finalization, and local
  worker cleanup after persistence errors have explicit handling and regressions.
- Artifact writes precede database references; terminal job/run/chat updates commit
  together. Hash/shape/non-finite corruption, partial downloads, explicit retries,
  immutable versions, and WAL reopen behavior have regression coverage.
- Actual installed Codex requests for all three writer models exposed zero tools,
  including the additional_tools registry. Current login discovery and a real
  schema-constrained request passed. Authentication/protocol/429 failure paths use
  deterministic fixtures; the user's real account was not signed out or rate-limited.
- Loopback Host/Origin/token checks and same-tab token rotation are tested.

Explicit authentication regressions verify signed-out discovery never starts a
thread/turn, expired authentication makes exactly one turn attempt, and the factory
persists that failed attempt without treating it as malformed output. An explicit
retry can recover after restart; completed stages are reused. No live account or
credentials were changed for these fixtures.

Current automated suite: **98 Rust tests passed**, two explicit live Codex tests
excluded from the ordinary run (exercised separately); Clippy with warnings denied
and formatting checks pass. **20 frontend unit tests and 18 browser contract tests passed**, including live
revision geometry, same-tab token rotation and in-flight stage forks.
The added range test verifies +200% submission, live −200%, reload, and slider
expansion beyond ±160%. Extreme finite values remain allowed, with diagnostics.

## Tagliabue et al. adaptation and bounded pain comparison

New concepts default to [extraction adapted from Tagliabue et al. (2026)](paper-extraction.md):
raw fixed-readout extraction, control-PCA denoising,
family-grouped CV and all-pair final fitting. Completion contrast remains available
for response-style axes and historical recipes. Changed extraction settings create
immutable versions. New automatic previews default off; explicit standard or
negative-only preview sets remain available. Coefficient signs are unrestricted.
Historical `coefficient_policy` metadata is retained but is no longer enforced,
per the user's subsequent instruction. Shape, checksum and non-finite validation
remain in force; they are not semantic/welfare judgements.

- The same frozen 96 generated pairs transferred better under the adapted method:
  external S2 AUC **0.5659 → 0.8305**, at independently internally selected layers.
- A fresh real Astra/Sol/Terra factory using the new instructions produced 96 pairs
  across 12 families, completed 192 Bonsai captures and published/exported/imported
  the adapted-method vector with no automatic previews. Selected layer 16: CV AUC 0.8987,
  external AUC **0.7848**, reference cosine 0.2070. Do not substitute the best
  external-test layer for the internally selected result.
- Independent NumPy SVD reproduced PCA component counts, fold AUCs, calibration
  and selection in both runs (maximum unit-coordinate discrepancy <7.1e-9).
- The earlier negative sweep stopped at −5% on repetition (20 responses, 1,044
  tokens). The fresh adapted-method **zero baseline** itself repeated and was cancelled
  at 57 tokens. No fresh negative/live pain response followed that stop.

These are exploratory representation results, not a clean reproduction of
Tagliabue et al.'s behavioral findings or evidence of suffering/relief. Control coverage and
lexical confounds remain documented. The core recipe changes are implemented, but
the study's full layer/perspective/injection protocol is not reproduced. See the
[full comparison](pain-benchmark.md) and [method differences](paper-extraction.md).

Browser contracts additionally cover extraction/preview defaults, legacy sign
metadata accepting ±200%, explicit re-extraction without rewriting old recipes,
and long concept names at desktop/mobile sizes. Positive-sign acceptance is tested
with synthetic fixtures, not additional positive pain steering.

The final unrestricted-sign package was restarted against the populated lab:
recipe/vector hashes, imported copy, cancelled baseline and every cloud checkpoint
were unchanged. Checked export passed again and Bonsai was reloaded. A real
Chromium session verified PCA/CV diagnostics, explicit disabled previews, both-sign
controls without policy UI, adapted-method/no-preview creation defaults and zero-mix reload,
with no new generations or concepts and no page errors. Screenshots were inspected.
The relocated final archive also launched under a system-only PATH, served the new
embedded UI and discovered six Codex models; that separate smoke-test instance was
stopped. Evidence: `work/real-paper-ui-report.json`, `work/paper-relocated-proof.json`,
and `work/benchmarks/pain-paper-v3/app-report.json`.

The unrestricted-sign application also repeated an ordinary **non-pain** two-axis
response, a same-input/settings baseline and two-turn chat with the existing humor
and sensory-detail vectors. The live update was acknowledged from token 7; incomplete,
unknown-member and duplicate-member snapshots were rejected without changing the
frozen selection. No new cloud concepts were generated. Evidence:
`work/final-package-smoke.json`. Subsequent archive support only adds library metadata;
it does not change the inference/controls path.

## Final packaging/restart checks

**Passed.** The archive was unpacked into a separate `/private/tmp` directory and
launched with only system directories in PATH (no Node or Rust). The embedded UI
served correctly, existing models/vectors/mixes/chat/history reloaded, and the
absolute installed Codex CLI discovered all six account models. Final package
inference and browser rendering passed again, including measured peak worker RSS
(8,129,576,960 bytes at the last sample, explicitly not a live RAM gauge). Both binaries link
only Apple/system libraries; Metal resources are embedded. Raw/tiny/Bonsai engine
conformance was rerun against the **relocated packaged worker** and passed with the
same measured tolerances.

The app was SIGKILLed during a genuine 2048-token-request stream after observing
21 tokens. On reopening, 22 durably written tokens were retained, the run was marked
interrupted, and exact-resume requests were rejected. Completed runs retained their
requested/applied revisions and saved manifests.

The final embedded UI was then exercised in a real Chromium browser, without an
API fixture: edited an actual completed repair through the JSON editor, created an
immutable recipe version, ran all 192 local captures plus nine previews, published
the new vector, and reloaded it. The original recipe remained unchanged; all
completed cloud-stage hashes and generation-attempt IDs were reused. No additional
cloud generation was performed. No browser page errors occurred. Actual library,
editor and reload screenshots are retained in `work/real-ui-*.png`.

Package: `dist/Torment-Nexus-macos-arm64.tar.gz` (about 7.4 MiB compressed / 23 MiB
unpacked, excluding models/data). Launcher: `./torment`. The populated development
lab opens with `./torment --data-dir ./work/e2e-data`.

## Library cleanup — passed

The user's populated library now shows the three canonical concepts: humor,
sensory detail and pain using the adapted method. Three exact import-test copies and the
browser-edit sensory test version are archived, not deleted. **Show archived**
reveals all four with Restore controls. Separate mutable visibility records leave
all existing vectors, recipes, runs, jobs, saved mixes and artifacts unchanged;
their complete record hashes were compared before and after cleanup and actual
browser restore/rearchive. A consistent SQLite backup is retained locally.

The updated package was rebuilt and launched under system-only PATH, with Bonsai
loaded. The real browser verified three visible cards, four archived cards,
restore/rearchive and reload without errors or new inference/cloud jobs. Historical
run/preset references and active coefficients remain resolvable independently of
library visibility; browser contracts exercise this during streaming too.
Evidence: `work/library-cleanup/`; screenshot visually inspected. The original-scope
[completion audit](completion-audit.md) records coverage and explicit limitations.

## Multi-layer controls and conversation-first UI — passed

Each concept now exposes an independent slider/numeric input for every extracted
usable layer. Selection freezes unique `(vector_id, layer)` members, not a single
layer per concept. New selections start entirely at zero; loading an old mix keeps
its coefficients and adds zero-valued missing layers before the next response.
Legacy live clients may omit the layer only when unambiguous. Same-layer duplicate
pairs, incomplete snapshots, changed membership and non-finite values still fail.

Regression tests cover per-layer calibrated sums and independent clearing,
single hydration of shared artifacts, saved presets, explicit persisted control
identities, ambiguous old requests, live reload, and old-mix migration. The real
packaged browser used the existing humor concept's five layer controls on Bonsai:
L16 started at −1%, L25 at +1%, with all other rows zero; live L25 changed to −2%
at token 1, then L16 cleared at token 22. Applied geometry confirmed only L25
remained nonzero with the expected calibrated norm. The check deliberately stopped
after 25 output tokens, retained the run, and verified per-layer reload. No cloud
concepts or pain inference were run; existing vectors, recipes, presets and archive
metadata were unchanged. Evidence: `work/multilayer-real-report.json`.

The large decorative heading was replaced with a compact conversation header,
sidebars narrowed, model output enlarged to 15 px, and explanatory chrome collapsed.
At 1440 px viewport width, conversation occupies 944 px; Focus chat hides both panels
without changing coefficients. Mobile places conversation first. Desktop, mobile,
and actual populated-app screenshots were inspected. Package rebuilt and relaunched
with Bonsai loaded; the four archived test entries remain hidden.

## Evidence and reproduction

Scripts: `tests/engine_conformance.py`, `tests/engine_raw.py`, `tests/app_e2e.py`,
`tests/lab_conformance.py` (development-only NumPy), `frontend/tests/workbench.spec.ts`,
and the opt-in real-lab editor/reload test `frontend/tests/real-workbench.mjs`.
The current workspace retains detailed reports/logs in `work/`, including
`app-e2e-report.json`, `recovery-proof.json`, `inference-recovery-proof.json`,
`lab-numerical-conformance.json`, engine conformance reports and test logs. Reports
contain run/artifact IDs; the launch token is confined to the current local launch log.
The populated lab is `work/e2e-data`; it is separate from the application archive.

## Explicit limits

Only the pinned Bonsai model and existing tiny test GGUF were exercised. Other GGUFs
must expose the supported exact residual hook and a compatible chat template, or
use explicitly selected raw mode. No inference-quality guarantee follows from the
AUCs or mechanical tests. Interrupted inference is saved, not exactly resumable.
MLX, vision, fine-tuning, remote serving, speculative decoding and native/notarized
packaging remain out of scope. The Codex isolation adapter intentionally fails closed
for unverified CLI versions.
