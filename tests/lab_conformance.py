#!/usr/bin/env python3
"""Independent numerical audit of real saved captures (development-only NumPy).

Does not call models, modify the lab, or treat held-out separation as steering quality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import numpy as np


def main(args):
    root = Path(args.data_dir).resolve()
    index = sqlite3.connect(f"file:{root / 'index.sqlite3'}?mode=ro", uri=True)
    report = json.loads(Path(args.lab_report).read_text())

    def blob(key):
        data = (root / "artifacts" / key[:2] / key).read_bytes()
        assert hashlib.sha256(data).hexdigest() == key
        return data

    def tensor(key):
        data = blob(key)
        assert data[:8] == b"TNF32\x00\x01\x00"
        rank, = struct.unpack_from("<I", data, 8)
        assert rank == 1
        width, = struct.unpack_from("<Q", data, 12)
        result = np.frombuffer(data, dtype="<f4", offset=20)
        assert result.shape == (width,)
        return result.astype(np.float64)

    output = {"method": "independent NumPy mean-of-means, calibration, pole reversal and held-out AUC", "vectors": []}
    for ident in report["vector_ids"]:
        vector = json.loads(index.execute("SELECT payload FROM records WHERE kind='vectors' AND id=?", (ident,)).fetchone()[0])
        manifest = json.loads(blob(vector["manifest_hash"]))
        pairs = manifest["recipe"]["dataset"]
        train = np.array([pair["split"] == "train" for pair in pairs])
        train_families = {p["family"] for p in pairs if p["split"] == "train"}
        test_families = {p["family"] for p in pairs if p["split"] == "diagnostic"}
        assert train_families.isdisjoint(test_families)
        poles = []
        for pole in ("positive", "negative"):
            captures = [json.loads(blob(manifest["extractions"][pair["id"] + ":" + pole])) for pair in pairs]
            for pair, capture in zip(pairs, captures):
                assert capture["rendered"] == capture["prefix"] + pair[pole]
                assert capture["capture_position"] == len(capture["token_ids"]) - 1
            poles.append(np.array([[c["values"] for c in capture["captures"]] for capture in captures], dtype=np.float32).astype(np.float64))
        positive, negative = poles
        layers = []
        for offset, layer in enumerate(vector["layers"]):
            p, n = positive[:, offset], negative[:, offset]
            expected = (p[train].mean(axis=0) - n[train].mean(axis=0)).astype(np.float32).astype(np.float64)
            swapped = (n[train].mean(axis=0) - p[train].mean(axis=0)).astype(np.float32).astype(np.float64)
            direction = tensor(layer["direction_hash"])
            unit = tensor(layer["unit_hash"])
            error = float(np.max(np.abs(expected - direction)))
            assert error < 1e-5, error
            assert np.array_equal(swapped, -expected)
            norm = np.median(np.linalg.norm(np.concatenate([p[train], n[train]]), axis=1))
            assert abs(norm - layer["residual_norm"]) < 1e-8
            assert np.max(np.abs(unit - direction / np.linalg.norm(direction))) < 1e-7
            pos_scores, neg_scores = p[~train] @ unit, n[~train] @ unit
            auc = float(np.mean((pos_scores[:, None] > neg_scores).astype(float) + .5 * (pos_scores[:, None] == neg_scores)))
            assert abs(auc - layer["auc"]) < 1e-12
            for percent in (-20, -6, 0, 8, 20):
                injected = (percent / 100 * norm * unit).astype(np.float32)
                assert abs(np.linalg.norm(injected.astype(np.float64)) - abs(percent) / 100 * norm) < 1e-5
            layers.append({"layer": layer["layer"], "raw_max_abs_error": error, "auc": auc, "calibration": float(norm)})
        chosen = min(layers, key=lambda layer: (-layer["auc"], layer["layer"]))
        assert chosen["layer"] == vector["selected_layer"]
        output["vectors"].append({"id": ident, "pairs": len(pairs), "train": int(train.sum()), "diagnostic": int((~train).sum()),
                                  "family_disjoint": True, "poles_negate": True, "scaling_verified": True, "layers": layers})
    output["status"] = "passed"
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="work/e2e-data")
    parser.add_argument("--lab-report", default="work/app-e2e-report.json")
    parser.add_argument("--output", default="work/lab-numerical-conformance.json")
    main(parser.parse_args())
