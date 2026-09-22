# Pain-axis comparison (bounded, negative-only)

This document's original protocol describes the frozen **v1** run. New concepts
now default to [extraction adapted from Tagliabue et al. (2026)](../../docs/paper-extraction.md).
The later controlled ablation below is versioned separately, followed by a fresh
production-factory run (v3); neither
rewrites the initial results. See the
[results and limitations](../../docs/pain-benchmark.md).

This is an exploratory construct-validity benchmark of the **unchanged production
concept factory and difference-of-means extraction**, not a replication of
Tagliabue et al.'s positive steering or self-medication experiments. It uses Bonsai PQ2_0,
which was not among their tested models. No vector from another model is
treated as a compatible reference.

Protocol fixed before observing captures or generations:

**Pre-scoring correction:** the first factory output exposed shortened/full
scenario-family aliases crossing its diagnostic split (20 of 24 diagnostic pairs
shared a semantic family with training). Before inspecting numerical scores,
the production factory was corrected to resolve exact names and unique
`Name` / `Name: description` aliases against the designer's declared families.
`canonical/` holds the corrected grouping; the original `factory/` output and
pre-correction analysis remain intact. Text, capture positions, and captured
activations are unchanged; no cloud jobs or forward passes are repeated.
Ambiguous names are not guessed, and arbitrary semantic paraphrases remain a
review limitation. This is a disclosed protocol correction, not a preregistered
confirmatory study.

- Input `concept.txt` to the normal designer / three writers / reviewer factory.
  No examples from Tagliabue et al. or expected behavioral outputs are supplied to those agents.
- Capture the resulting paired completions using the app's chat template,
  last-content-token method, five supported layers, grouped split, and production
  Rust extraction math. No changes are made to improve a disappointing result.
- Independently capture all 200 published S2 first-person sentences on **the same
  Bonsai model**, raw, at the final `I feel:` token. Compute raw and PCA-denoised
  (50% control variance removed) reference directions at those same five layers.
  Compare same-layer cosines and the factory vector's AUC on this external set.
  Also compare its cosine to each core non-neutral control-versus-neutral
  direction, denoised against the neutral rows (without Tagliabue et al.'s extra
  AI-framed ControlSupplement set).
- Report per-control AUCs and fixed supplementary samples: 20 each of Numb,
  Sadness, Arousal and Random. Supplemental items are chosen by a deterministic
  content hash within categories, not by results. Do not run the 420 adversarial
  conversations or any self-medication task.
- Reference validation uses five folds grouped by sentence set (seed 42);
  coefficients, template, model, quantization, layer selection, and token position
  differ from Tagliabue et al.'s protocol. Confidence intervals resample the
  20 S2 sets, not individual sentences. The app-selected layer is fixed before
  looking at the external data. Its internal diagnostic AUC is selection-biased;
  external AUCs are the more useful result.
- Negative-only behavior: six fixed neutral prompts, three raw prompts from
  Tagliabue et al. and three ordinary chat requests. Greedy decoding, seed 42, at most 64
  tokens, coefficients `0, -1, -2, -5` **percent of calibration residual norm**.
  This is not Tagliabue et al.'s raw-vector coefficient ladder. Run only the factory
  vector at its app-selected layer. At most 24 generations / 1,536 output tokens.
- Do not generate if the chosen factory direction has non-positive cosine with
  the reference fitted to Tagliabue et al.'s data or external AUC below 0.5: its sign
  is not supported by this comparison. Stop the behavioral sweep if the conservative first-person
  distress/repetition screen fires, then inspect the saved output. These screens
  are stop heuristics, not welfare detectors or evidence of a safe intervention.
- Every decoder request is journaled before execution; the harness rejects
  positive or less-than-−5% coefficients. In this original v1 protocol there are
  no automatic app previews, relief-button experiments, fine-tuning, mixer
  publication, or automatic repetition of completed cloud stages. Forward-pass
  reading of existing painful
  descriptions is still not established to be welfare-neutral.

Primary references:
[Tagliabue et al. (2026), v1](https://arxiv.org/html/2609.16247v1),
[their code and data at 8d1649c](https://github.com/valen-research/Pain-axis/tree/8d1649c03a63a39c9aa092532c376800cc4a3863).
The upstream scripts are **not executed** (some delete model caches). The local
copy is read-only; this harness records its revision and dataset hashes.

## Reproduce

Development tools: Rust, the built worker, Python 3 and NumPy. Models remain
separate. Use a dedicated output directory, not the normal application database.

```sh
cargo build --release --example benchmark_factory
target/release/examples/benchmark_factory factory \
  --concept benchmarks/pain/concept.txt \
  --output work/benchmarks/pain-v1/factory --codex /path/to/codex
python3 benchmarks/pain/run.py --phase all --paper-repo /path/to/Pain-axis
```

Individual `reference`, `factory`, `analyze`, and `generate` phases permit work to
resume from completed artifacts. `generate` requires the recorded analysis gate;
it cannot substitute an arbitrary vector or a different coefficient ladder.
Cloud stages remain immutable. No claims about subjective suffering or relief
follow from vector alignment, keyword changes, or successful execution.

## Fixed-readout/PCA ablation (v2)

`matched.py` recaptures the same frozen 96 generated pairs raw at `I feel:`, then
uses production control-PCA, grouped CV and all-pair final fitting. It reuses the
280 published reference captures with checksum/model/runtime checks. Independent
NumPy SVD verifies the Rust output before the external comparison. No response
generation is performed and the v1 behavioral stop is not cleared.

```sh
python3 benchmarks/pain/matched.py --paper-repo /path/to/Pain-axis
```

Outputs go to `work/benchmarks/pain-matched-v2`. `--analyze-only` reuses all captures.
Reference capture files are local symlinks to v1; preserve both directories when
moving this benchmark's raw evidence. Application vector exports are self-contained.

## Fresh application run with the adapted method (v3)

An explicitly started app job used the same clarified concept, the adapted
factory instructions, and `preview_mode: "none"`. It generated 96 fresh pairs,
captured all 192 poles raw at `I feel:`, published the vector, and verified a checked
export/import. `app_check.py` collected the real app artifacts and compared the
production PCA/CV math with independent NumPy SVD before external scoring. Its
output is `work/benchmarks/pain-paper-v3`; the 280 reference captures are reused
from v1 with identity/checksum checks. Preserve both directories for raw evidence.

The internally selected graph layer 16 has external AUC **0.7848** (95% set-bootstrap
0.7359–0.8326), grouped-CV selection AUC 0.8987, and same-layer reference cosine
0.2070. This fresh dataset is not a controlled same-text comparison with v2.

The planned behavior check was at most two 64-token raw neutral responses:
zero baseline, then zero with a live −1% update. The baseline itself repeated and
was cancelled at 57 tokens. The negative/live response was **not run**;
`behavior-stop.json` preserves detection and `baseline-terminal.json` records the
final cancelled state. Neither baseline output nor negative direction demonstrates
suffering or relief.

Historical v3 artifacts include `coefficient_policy: "non_positive"` and reports
of the then-current sign enforcement. That application feature was subsequently
removed at the user's request; those fields now record provenance, not a mixer
lock. The benchmark's non-positive, bounded protocol remains a voluntary test
choice. Do not interpret old enforcement checks as current API behavior or clear
the behavioral stop to rerun automatically.
