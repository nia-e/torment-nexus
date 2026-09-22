#!/usr/bin/env python3
"""Readout/PCA ablation using the frozen v1 generated text, with no generations.

The external published captures are reused byte-for-byte. Only the generated
examples are recaptured, raw at `I feel:`, using the production inference worker.
The Rust result is independently checked against NumPy SVD before comparison.
"""
import argparse
import fcntl
from pathlib import Path
import subprocess

import numpy as np

from run import Bench, ROOT, auc, digest, direction, encoded, file_hash, read, require, write


def audit(out):
    prepared = read(out / "canonical/prepared.json")
    analysis = read(out / "factory-analysis.json")
    captures = {(r["pair_id"], r["pole"]): r["captures"]
                for r in read(out / "factory-captures.json")}
    pairs = prepared["pairs"]
    split = analysis["split"]
    family = lambda p: " ".join(p["family"].split()).lower()
    folds = np.array([split["families"][family(p)] for p in pairs])
    checks = []
    for layer in analysis["layers"]:
        arrays = [np.array([next(c["values"] for c in captures[(p["id"], pole)]
                                 if c["layer"] == layer["layer"]) for p in pairs], dtype=np.float64)
                  for pole in ("positive", "negative")]
        x = np.concatenate(arrays)
        labels = np.array([True] * len(pairs) + [False] * len(pairs))
        unit, removed = direction(x, labels)
        error = float(np.max(np.abs(unit - np.asarray(layer["unit"])) ))
        require(error < 1e-6 and removed == layer["removed_control_pcs"], "Rust/SVD direction disagreement")
        require(abs(float(np.median(np.linalg.norm(x, axis=1))) - layer["residual_norm"]) < 1e-7,
                "calibration disagreement")
        fold_scores = []
        repeated = np.concatenate([folds, folds])
        for fold in range(split["folds"]):
            mask = repeated == fold
            v, _ = direction(x[~mask], labels[~mask])
            scores = x[mask] @ v
            fold_scores.append(auc(scores[labels[mask]], scores[~labels[mask]]))
        require(np.allclose(fold_scores, layer["fold_aucs"], rtol=0, atol=1e-10), "CV disagreement")
        checks.append({"layer": layer["layer"], "unit_max_abs_error": error,
                       "removed_control_pcs": removed, "fold_aucs": fold_scores})
    selected = max(analysis["layers"], key=lambda layer: (layer["auc"], -layer["layer"]))["layer"]
    require(selected == analysis["selected_layer"], "selection disagreement")
    write(out / "independent-math.json", {"checks": checks, "selected_layer": selected,
                                          "method": "NumPy SVD versus production Rust sample-space eigendecomposition"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-repo", required=True)
    parser.add_argument("--source", default=str(ROOT / "work/benchmarks/pain-v1"))
    parser.add_argument("--output", default=str(ROOT / "work/benchmarks/pain-matched-v2"))
    parser.add_argument("--model", default=str(ROOT / "work/models/Ternary-Bonsai-2-27B-PQ2_0.gguf"))
    parser.add_argument("--binary", default=str(ROOT / "engine/build/bin/torment-engine"))
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    args.protocol = "pain-fixed-readout-pca-ablation-v2-no-generation"
    source, out = Path(args.source).resolve(), Path(args.output).resolve()
    require(source != out, "do not overwrite the historical benchmark")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "benchmark.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bench = Bench(args)
        try:
            old_identity = read(source / "identity.json")
            for field in ("model_sha256", "worker_sha256", "reference_hashes", "paper_revision"):
                require(bench.identity[field] == old_identity[field], f"incompatible reference {field}")
            references = read(source / "reference.json")
            (out / "captures").mkdir(exist_ok=True)
            for reference in references:
                path = source / "captures" / f"{reference['key']}.json"
                saved = read(path)
                require(saved["sha256"] == digest(encoded(saved["result"])), "reference corruption")
                target = out / "captures" / path.name
                if not target.exists():
                    target.symlink_to(path)
            write(out / "reference.json", references)
            prepared = read(source / "canonical/prepared.json")
            write(out / "canonical/prepared.json", prepared)
            write(out / "protocol.json", {
                "source": str(source), "source_identity_sha256": file_hash(source / "identity.json"),
                "frozen_dataset_sha256": file_hash(source / "canonical/prepared.json"),
                "reference_captures_reused": len(references), "readout_suffix": "I feel:",
                "changed": ["raw fixed readout", "control PCA removing 50% variance", "grouped CV", "all-pair final fit"],
                "unchanged": ["generated text", "five candidate layers", "model", "external reference"],
                "generations_authorized": 0,
                "limitations": ["old writers were not instructed to avoid direct pain keywords", "no third-person condition", "not all-layer selection"]})
            if not args.analyze_only:
                bench.start()
                records = []
                for index, pair in enumerate(prepared["pairs"]):
                    for pole in ("positive", "negative"):
                        rendered = pair[pole].rstrip() + " I feel:"
                        result, key = bench.call({"op": "extract", "messages": [{"role": "user", "content": rendered[:-1]}],
                                                 "completion": ":", "raw": True, "layers": bench.layers})
                        require(result["rendered"] == rendered, "raw readout changed")
                        records.append({"pair_id": pair["id"], "pole": pole, "key": key, "captures": result["captures"]})
                    if (index + 1) % 8 == 0:
                        print(f"fixed readout {index + 1}/{len(prepared['pairs'])} pairs", flush=True)
                write(out / "factory-captures.json", records)
            bench.close()
            bench.worker = None
            subprocess.run([str(ROOT / "target/release/examples/benchmark_factory"), "analyze", "--paper",
                            "--dataset", str(out / "canonical/prepared.json"), "--captures", str(out / "factory-captures.json"),
                            "--fingerprint", bench.identity["model_sha256"], "--output", str(out / "factory-analysis.json")], check=True)
            audit(out)
            bench.analyze()
        finally:
            bench.close()


if __name__ == "__main__":
    main()
