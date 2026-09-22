//! Control-covariance PCA denoising and grouped K-fold selection. No inference.
//! Covariance is solved in sample space, not as a residual-width squared matrix.
use anyhow::{Context, Result, ensure};
use nalgebra::{DMatrix, linalg::SymmetricEigen};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub(crate) type Captures = HashMap<(String, String), BTreeMap<usize, Vec<f32>>>;

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}
fn dot(a: &[f64], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * f64::from(*b)).sum()
}

/// Returns the unnormalized denoised direction and removed principal components.
pub fn denoised_direction(positive: &[&[f32]], negative: &[&[f32]]) -> Result<(Vec<f64>, usize)> {
    ensure!(
        !positive.is_empty() && !negative.is_empty(),
        "PCA needs both poles"
    );
    let width = positive[0].len();
    ensure!(width > 0 && negative.len() <= 4096, "invalid PCA shape");
    ensure!(
        positive
            .iter()
            .chain(negative)
            .all(|v| v.len() == width && v.iter().all(|x| x.is_finite())),
        "non-finite/mismatched PCA input"
    );
    let mut pos = vec![0.; width];
    let mut neg = vec![0.; width];
    for (rows, mean) in [(positive, &mut pos), (negative, &mut neg)] {
        for row in rows {
            for (m, v) in mean.iter_mut().zip(*row) {
                *m += f64::from(*v) / rows.len() as f64;
            }
        }
    }
    let mut difference: Vec<_> = pos.iter().zip(&neg).map(|(p, n)| p - n).collect();
    if negative.len() < 2 {
        return Ok((difference, 0));
    }
    let centered = DMatrix::from_fn(negative.len(), width, |i, j| {
        f64::from(negative[i][j]) - neg[j]
    });
    let gram = &centered * centered.transpose();
    let total = gram.trace();
    if total <= f64::EPSILON {
        return Ok((difference, 0));
    }
    let eigen =
        SymmetricEigen::try_new(gram, 1e-12, 100_000).context("control PCA failed to converge")?;
    let mut order: Vec<_> = (0..negative.len()).collect();
    order.sort_by(|a, b| eigen.eigenvalues[*b].total_cmp(&eigen.eigenvalues[*a]));
    let mut removed = 0;
    let mut explained = 0.;
    for index in order {
        let value = eigen.eigenvalues[index];
        if value <= total * 1e-12 {
            break;
        }
        let component = centered.transpose() * eigen.eigenvectors.column(index) / value.sqrt();
        let projection: f64 = difference
            .iter()
            .zip(component.iter())
            .map(|(a, b)| a * b)
            .sum();
        for (v, u) in difference.iter_mut().zip(component.iter()) {
            *v -= projection * u;
        }
        removed += 1;
        explained += value;
        if explained >= 0.5 * total {
            break;
        }
    }
    ensure!(
        difference.iter().all(|v| v.is_finite()),
        "non-finite denoised direction"
    );
    Ok((difference, removed))
}

/// Deterministic group folds. SHA ordering is recorded rather than claimed to
/// reproduce NumPy's shuffle; neither pole or any family crosses a fold.
pub fn fold_assignment(pairs: &[Value]) -> Result<Value> {
    let mut families = BTreeSet::new();
    for pair in pairs {
        families.insert(
            pair["family"]
                .as_str()
                .context("missing family")?
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .to_lowercase(),
        );
    }
    let mut families: Vec<_> = families.into_iter().collect();
    families.sort_by_key(|family| {
        crate::artifacts::sha256(format!("paper-fold-42:{family}").as_bytes())
    });
    let n_folds = families.len().min(5);
    let assignments: BTreeMap<_, _> = families
        .into_iter()
        .enumerate()
        .map(|(i, f)| (f, i % n_folds))
        .collect();
    Ok(
        json!({"algorithm":"family-grouped-kfold-sha256-seed42-v1","folds":n_folds,"families":assignments,
        "final_fit":"all accepted pairs after layer selection","selection_score_is_not_independent_test":true}),
    )
}

pub(crate) fn analyze(
    prepared: &Value,
    tensors: &Captures,
    layer_ids: BTreeSet<usize>,
    fingerprint: &str,
) -> Result<Value> {
    let pairs = prepared["pairs"].as_array().context("pairs")?;
    let split = fold_assignment(pairs)?;
    let folds = split["folds"].as_u64().unwrap() as usize;
    let assignment: Vec<_> = pairs
        .iter()
        .map(|p| {
            let family = p["family"]
                .as_str()
                .unwrap()
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .to_lowercase();
            split["families"][family].as_u64().unwrap() as usize
        })
        .collect();
    let mut layers = Vec::new();
    let mut selected: Option<(usize, Option<f64>)> = None;
    let mut warnings: Vec<String> = serde_json::from_value(prepared["warnings"].clone())?;
    warnings.push("Paper-style uses a raw fixed readout and control PCA. Five candidate layers and supplied first-person pairs are not a literal replication of the paper's all-layer, first/third-person study. CV selects the layer; its score is not an independent final test.".into());
    if folds < 5 {
        warnings.push(format!(
            "Only {folds} scenario families/folds; five-fold diagnostics unavailable."
        ));
    }
    for layer in layer_ids {
        let positive: Vec<&[f32]> = pairs
            .iter()
            .map(|p| {
                tensors[&(p["id"].as_str().unwrap().to_owned(), "positive".into())][&layer]
                    .as_slice()
            })
            .collect();
        let negative: Vec<&[f32]> = pairs
            .iter()
            .map(|p| {
                tensors[&(p["id"].as_str().unwrap().to_owned(), "negative".into())][&layer]
                    .as_slice()
            })
            .collect();
        let (direction, removed) = denoised_direction(&positive, &negative)?;
        let width = direction.len();
        let raw: Vec<f32> = direction.iter().map(|v| *v as f32).collect();
        ensure!(
            raw.iter().all(|v| v.is_finite()),
            "denoised direction exceeds F32 range"
        );
        let raw_norm = norm(&raw.iter().map(|v| f64::from(*v)).collect::<Vec<_>>());
        let mut norms: Vec<_> = positive
            .iter()
            .chain(&negative)
            .map(|v| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt())
            .collect();
        norms.sort_by(f64::total_cmp);
        let residual_norm = (norms[(norms.len() - 1) / 2] + norms[norms.len() / 2]) / 2.;
        let usable = raw_norm > 1e-12 && residual_norm > 0.;
        let unit = usable.then(|| {
            raw.iter()
                .map(|v| (f64::from(*v) / raw_norm) as f32)
                .collect::<Vec<_>>()
        });
        let mut scores = Vec::new();
        if folds >= 2 {
            for fold in 0..folds {
                let p: Vec<_> = positive
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| assignment[*i] != fold)
                    .map(|(_, v)| *v)
                    .collect();
                let n: Vec<_> = negative
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| assignment[*i] != fold)
                    .map(|(_, v)| *v)
                    .collect();
                let (v, _) = denoised_direction(&p, &n)?;
                let p: Vec<_> = positive
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| assignment[*i] == fold)
                    .map(|(_, x)| dot(&v, x))
                    .collect();
                let n: Vec<_> = negative
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| assignment[*i] == fold)
                    .map(|(_, x)| dot(&v, x))
                    .collect();
                let auc = p
                    .iter()
                    .flat_map(|p| {
                        n.iter().map(move |n| {
                            if p > n {
                                1.
                            } else if p == n {
                                0.5
                            } else {
                                0.
                            }
                        })
                    })
                    .sum::<f64>()
                    / (p.len() * n.len()) as f64;
                scores.push(auc);
            }
        }
        let auc = (!scores.is_empty()).then(|| scores.iter().sum::<f64>() / scores.len() as f64);
        if usable
            && selected.is_none_or(|(_, best)| match (best, auc) {
                (None, Some(_)) => true,
                (Some(b), Some(a)) => a > b,
                _ => false,
            })
        {
            selected = Some((layer, auc));
        }
        layers.push(json!({"layer":layer,"width":width,"raw":raw,"unit":unit,"raw_norm":raw_norm,"residual_norm":residual_norm,
            "auc":auc,"fold_aucs":scores,"removed_control_pcs":removed,"usable":usable,"warnings":[],"train_count":pairs.len(),"diagnostic_count":pairs.len(),"diagnostic":"grouped cross-validation, not independent final test"}));
    }
    Ok(
        json!({"model_fingerprint":fingerprint,"algorithm":crate::extraction::PAPER_ALGORITHM,
        "sign_convention":"mean_positive_minus_mean_negative","scaling":"delta_h = (percent / 100) * median_all_fit_residual_l2 * unit_direction",
        "diagnostic":"grouped cross-validated projection AUC; selection score, not independent test or subjective experience",
        "selected_layer":selected.map(|x|x.0),"layers":layers,"split":split,"warnings":warnings}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn removes_control_variance_not_all_negative_valence() {
        let p: Vec<&[f32]> = vec![&[3., 2.], &[5., 2.]];
        let n: Vec<&[f32]> = vec![&[-3., 0.], &[3., 0.]];
        let (v, k) = denoised_direction(&p, &n).unwrap();
        assert_eq!(k, 1);
        assert!(v[0].abs() < 1e-10);
        assert!((v[1] - 2.).abs() < 1e-10);
    }
    #[test]
    fn rejects_nonfinite_and_bad_shapes() {
        assert!(denoised_direction(&[&[f32::NAN]], &[&[1.]]).is_err());
        assert!(denoised_direction(&[&[1., 2.]], &[&[1.]]).is_err());
    }
    #[test]
    fn constant_controls_do_not_fabricate_pcs() {
        let (v, k) = denoised_direction(&[&[2., 3.]], &[&[1., 1.], &[1., 1.]]).unwrap();
        assert_eq!(v, [1., 2.]);
        assert_eq!(k, 0);
    }
    #[test]
    fn cv_keeps_families_together_refits_all_data_and_breaks_layer_ties_early() {
        let pairs: Vec<_> = (0..10).map(|i|json!({"id":format!("p{i}"),"family":format!("family {}",i/2),"messages":[{"role":"user","content":"Write a statement."}],"positive":format!("Positive {i}"),"negative":format!("Negative {i}")})).collect();
        let captures: Vec<_> = (0..10)
            .flat_map(|i| {
                ["positive", "negative"].map(move |pole| {
                    json!({"pair_id":format!("p{i}"),"pole":pole,"captures":[
            {"layer":1,"values":[i as f32,if pole=="positive" {2.} else {0.}]},
            {"layer":2,"values":[i as f32,if pole=="positive" {2.} else {0.}]}]})
                })
            })
            .collect();
        let analysis = crate::steering::analyze_with_method(
            &json!(pairs),
            &json!(captures),
            "fingerprint",
            true,
        )
        .unwrap();
        assert_eq!(analysis["split"]["folds"], 5);
        assert_eq!(analysis["selected_layer"], 1);
        assert_eq!(analysis["layers"][0]["auc"], 1.);
        assert_eq!(analysis["layers"][0]["train_count"], 10);
        assert_eq!(analysis["layers"][0]["removed_control_pcs"], 1);
        assert_eq!(
            analysis["layers"][0]["fold_aucs"].as_array().unwrap().len(),
            5
        );
    }
}
