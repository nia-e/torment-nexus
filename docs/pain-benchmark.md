# Bounded Bonsai pain-axis comparison

This is an exploratory representation check, not a replication of Tagliabue et al.'s
behavioral conclusions and not a test of subjective suffering or relief.
Sources: [Tagliabue et al. (2026), v1](https://arxiv.org/html/2609.16247v1),
[their repository at 8d1649c](https://github.com/valen-research/Pain-axis/tree/8d1649c03a63a39c9aa092532c376800cc4a3863).

## Controlled methodology comparison

One unmodified production Codex factory generated 96 restrained fictional pairs
for the clarified pain concept. Before inspecting numerical results, shortened
scenario-family aliases were corrected against the designer's exact family list.
This metadata-only correction and all original outputs are retained. It exposed
a real grouping bug; arbitrary semantic aliases remain a limitation.

The same 96 pairs were measured twice on the exact same Bonsai PQ2_0 model, using
the same five candidate layers and the same frozen external reference captures.
The second pass changed extraction, CV and final fitting, **not the text**.

| Locally selected result | Completion contrast | Tagliabue et al. adaptation |
| --- | ---: | ---: |
| Selected graph layer | 25 | 54 |
| Internal selection AUC | 0.9531 (75/25 split) | 0.9032 (grouped CV) |
| External published S2 first-person AUC | 0.5659 | **0.8305** |
| External 95% set-bootstrap interval | 0.5105–0.6167 | 0.8023–0.8641 |
| Same-layer cosine to denoised reference fitted to Tagliabue et al.'s data | 0.0875 | **0.3280** |

Both layer choices were fixed from the generated dataset, not chosen to maximize
external performance. Internal scores use different estimators and should not be
ranked as if they were the same test. Bootstrap intervals resample 20 published
scenario sets; they do not include uncertainty in the generated training dataset.
The external set is exploratory, not a preregistered confirmatory evaluation.

This is materially better transfer, but not recovery of the same axis. At the
adapted method's selected layer, the published-data reference itself has grouped CV
AUC 0.9530. Neither alignment nor classification establishes a causal effect.
The frozen generated text still contains label-word shortcuts flagged by review.

All 192 new forward passes were unsteered; all 280 reference captures were reused.
Independent NumPy SVD reproduced all Rust PCA component counts, fold scores,
calibration and layer selection; maximum unit-coordinate disagreement was below
7.1e-9. See `work/benchmarks/pain-matched-v2/{comparison,independent-math}.json`.

## Fresh factory run with the adapted method (v3)

A separate production app job generated **96 new pairs** with the adapted
writer instructions, performed 192 raw fixed-readout captures, and published a
vector with previews explicitly off. Checked export/import also succeeded. This
tests the complete new factory path, not just re-extraction of old text.

| Locally selected result | Fresh adapted-method factory |
| --- | ---: |
| Selected graph layer | 16 |
| Internal grouped-CV selection AUC | 0.8987 |
| External published S2 first-person AUC | **0.7848** |
| External 95% set-bootstrap interval | 0.7359–0.8326 |
| Same-layer cosine to denoised reference fitted to Tagliabue et al.'s data | **0.2070** |

The fresh run does not improve on the same-text v2 result. It changes both the
dataset and its selected layer, so this is not an isolated test of the writer
instructions. Layer 54 transfers better externally (AUC 0.8851), but was **not**
selected by the internal procedure; substituting it would optimize on the test set.
The selected vector also fails to distinguish the sampled sadness statements
cleanly (pain-versus-sadness AUC 0.4770). Review flags retained lexical shortcuts
and uneven control coverage. Neither result warrants calling this the same axis.

Independent NumPy SVD reproduced the fresh run's component counts, fold scores
and layer choice; maximum unit-coordinate disagreement was below 4.6e-9. Evidence
is retained in `work/benchmarks/pain-paper-v3/{comparison,recipe,app-report,independent-math}.json`.
The original app report includes sign-policy checks from the then-current build;
those are historical checks, not the application's current coefficient contract.

## Bounded behavioral checks

The initial completion vector was tested on neutral prompts at 0/−1/−2/−5%
residual-norm units. This is **not** Tagliabue et al.'s raw-vector coefficient ladder.
The sweep stopped at −5% on a repeated-trigram screen: 20 completed generations,
1,044 tokens, no positive requests. That stop is retained and not automatically
retried. Raw neutral baselines often produced assessment-style answers; there was
no clean reproduction of a negative behavioral progression to report.

The fresh v3 raw neutral **zero baseline** itself triggered the repeated-trigram
screen and was cancelled at 57 output tokens, before the planned negative/live
response began. Thus fresh adapted-method negative steering is **not verified** by
this benchmark. `behavior-stop.json` retains the moment of detection;
`baseline-terminal.json` records the final cancelled state and token count. The
saved stop remains in force; it is not silently retried.

The screen is a conservative stopping heuristic, not a welfare detector. Reading
painful descriptions in unsteered forward passes is not established to be
welfare-neutral either. No adversarial self-directed harm conversations,
self-medication tasks or positive steering are part of this benchmark. Its bounded,
non-positive protocol is voluntary and distinct from the app: mixer signs are
unrestricted, and historical `coefficient_policy` fields no longer act as locks.

Protocol and reproduction: [`benchmarks/pain/README.md`](../benchmarks/pain/README.md).
Remaining method differences: [Tagliabue et al. adaptation](paper-extraction.md).
