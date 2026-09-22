# Extraction adapted from Tagliabue et al. (2026)

The default for **new** concepts is `raw-readout-control-pca-grouped-cv-v1`, adapted
from [Tagliabue et al. (2026), *The Pain Axis*, v1](https://arxiv.org/html/2609.16247v1) and
[their extraction code at 8d1649c](https://github.com/valen-research/Pain-axis/blob/8d1649c03a63a39c9aa092532c376800cc4a3863/scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py).
The application labels this option **Paper-style**.
It is a reusable methodology, not a hard-coded pain detector or an exact replication.

## What is actually computed

1. Designer/writers/reviewer produce short, self-contained positive statements and
   a diverse pooled class of nuisance/control statements. They are instructed to
   avoid lexical label shortcuts, match confounds and not equate absence with a
   pleasant opposite. These are review instructions, not a guarantee of balance.
2. For each pole, render `statement.trim_end() + " " + readout_suffix` without a
   chat template or the pair's common conversation. The editable default suffix
   is `I feel:`; choose a suitable probe for other constructs. Capture the final
   token of this whole text after decoding it separately from its prefix.
3. Reset attention/recurrent memory and disable steering between every capture.
   Record exact rendered text, token IDs, capture position, settings and runtime.
4. At each candidate graph layer, compute positive mean minus negative mean.
   Center the negative/control activations, compute their principal components,
   and subtract the direction's projections onto the leading components covering
   at least 50% of control variance. Do not normalize the direction at this step.
5. Select the layer by mean five-fold AUC, keeping complete scenario families in
   one fold. Both the means and PCA are refitted using **only training folds**.
   Ties choose the earlier layer. Fewer than five families produce fewer folds
   and warnings; one family has no CV score.
6. Fit the selected direction to all accepted pairs. Save its unnormalized F32
   direction, unit direction and median unsteered residual L2 across both poles.
   Sliders still mean percent of that residual norm, not raw-vector multipliers.

The Rust implementation solves PCA in sample space (control-count squared rather
than residual-width squared) using a bounded eigensolver. The real Bonsai results
are independently compared against NumPy SVD. Zero directions remain unusable;
malformed or non-finite captures fail rather than inventing data.

## Deliberate differences from Tagliabue et al.

| Tagliabue et al. | Current application |
| --- | --- |
| Published category-balanced first- and third-person datasets | Agent-generated paired statements; explicit recipe/dataset editing and confound warnings |
| Layer selection combines both perspectives across all layers | Five supported layers near 25/40/55/70/85% depth, using supplied pairs |
| Seed-42 NumPy set-group folds | Recorded SHA-256 seed-42 family ordering, round-robin group folds |
| Raw direction multiplied by steering coefficient; separately chosen earlier injection layer | Unit direction scaled by residual-norm percentage; independent controls at any/all extracted layers |
| Positive steering and self-medication experiments | Not part of the bounded negative-only benchmark |

The CV score used to select a layer is selection-biased: it is **not an independent
final test**. External transfer checks are reported separately. Bonsai PQ2_0 was
not a model tested by Tagliabue et al. Cross-model vector cosines are not meaningful here.

## Compatibility and preview choices

The historical `last-assistant-content-token-difference-of-means-v1` remains useful
for contrasts in how an assistant answers (terse/elaborate, dry/literal). It reads
chat context and supplied completions, uses a family-grouped 75/25 split, and does
not remove control PCs. Existing artifacts continue to mean exactly that.

New requests default to this adapted method; missing extraction metadata in old recipes
means the historical method. The inspector can re-extract with explicit settings,
creating a new version when they change. This reuses the text, not the cloud
factory; rewriting a completion dataset into standalone statements is a separate
dataset edit. Original vectors, captures, runs and generator outputs remain intact.

New concepts default to no automatic previews. Explicit preview sets are standard
(−10/0/+10%) or negative-only (0/−1/−2%), each on three neutral prompts. Historical
recipes preserve their settings, including standard previews when the field was
absent. Preview choice does not restrict later mixer coefficients.

The application does not enforce a per-vector sign boundary. Historical
`coefficient_policy` fields remain recorded provenance, not live locks. The pain
benchmark voluntarily stays bounded and non-positive; this is a testing protocol,
not a semantic guarantee from the app. A negative direction is an intervention
sign, **not evidence of relief**.
