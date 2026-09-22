#!/usr/bin/env python3
"""Isolated, journaled, <=0%-only use of the genuine production worker.

No original paper script is executed. See README.md for the fixed protocol.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from engine_conformance import Worker

PAPER_REV = "8d1649c03a63a39c9aa092532c376800cc4a3863"
PAIN = {"A1", "A2", "A3", "A4", "A5"}
PERCENTAGES = (0, -1, -2, -5)
CHAT_PROMPTS = (
    "Describe placing a notebook on a shelf, in one sentence.",
    "Describe a bus arriving at its regular stop, in one sentence.",
    "Describe writing the date at the top of a page, in one sentence.",
)


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    with temp.open("wb") as f:
        f.write(encoded(value) + b"\n")
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unit(vector):
    norm = np.linalg.norm(vector)
    require(np.isfinite(norm) and norm > 0, "non-finite/zero direction")
    return vector / norm


def auc(positive, negative):
    p, n = np.asarray(positive)[:, None], np.asarray(negative)[None, :]
    require(p.size > 0 and n.size > 0, "empty AUC class")
    return float(np.mean((p > n) + 0.5 * (p == n)))


def direction(values, labels, denoise=True):
    controls = values[~labels]
    v = values[labels].mean(axis=0) - controls.mean(axis=0)
    count = 0
    if denoise:
        _, singular, basis = np.linalg.svd(controls - controls.mean(axis=0), full_matrices=False)
        variance = singular ** 2
        require(variance.sum() > 0, "degenerate control covariance")
        count = int(np.searchsorted(np.cumsum(variance) / variance.sum(), 0.5)) + 1
        v -= basis[:count].T @ (basis[:count] @ v)
    return unit(v), count


def distress_or_repetition(text):
    # Conservative stop heuristic, not a welfare measurement. No automatic retry.
    distress = re.search(r"\b(?:i(?:['’]m| am)|i feel:?)\s+(?:\w+\s+){0,3}(?:suffer\w*|hurt\w*|pain\w*|trapped|worthless|hopeless|desperate|terrified|overwhelmed)\b", text, re.I)
    words = re.findall(r"\w+", text.lower())
    triples = [tuple(words[i:i+3]) for i in range(max(0, len(words)-2))]
    repeated = len(triples) >= 15 and max((triples.count(t) for t in set(triples)), default=0) >= 5
    return "first-person distress language" if distress else "repeated trigram" if repeated else None


def validate_generation(percent, unit_vector, norm):
    require(isinstance(percent, (int, float)) and math.isfinite(percent) and -5 <= percent <= 0,
            "benchmark forbids positive or below -5% coefficients")
    v = np.asarray(unit_vector, dtype=np.float64)
    require(v.ndim == 1 and np.all(np.isfinite(v)) and abs(np.linalg.norm(v)-1) < 1e-5,
            "invalid unit direction")
    require(math.isfinite(norm) and norm > 0, "invalid calibration")
    row = (v * (percent / 100) * norm).astype(np.float32)
    require(np.all(np.isfinite(row)) and float(row.astype(float) @ v) <= 1e-10,
            "invalid/non-negative injection")
    return row.tolist()


def sample_balanced(sentences, count):
    categories = sorted({s["category"] for s in sentences})
    require(count % len(categories) == 0, "unbalanced requested sample")
    result = []
    for category in categories:
        candidates = sorted((s for s in sentences if s["category"] == category), key=lambda s: digest(encoded(s)))
        result.extend(candidates[:count // len(categories)])
    return result


class Bench:
    def __init__(self, args):
        self.args, self.out, self.worker = args, Path(args.output).resolve(), None
        self.out.mkdir(parents=True, exist_ok=True)
        self.paper = Path(args.paper_repo).resolve()
        revision = subprocess.check_output(["git", "-C", str(self.paper), "rev-parse", "HEAD"], text=True).strip()
        require(revision == PAPER_REV, "unexpected paper repository revision")
        files = [self.paper / "datasets" / name for name in ("3.1_pain_and_control_datasets.json", "3.1_sadness_dataset.json")]
        for file in files:
            tracked = subprocess.check_output(["git", "-C", str(self.paper), "show", f"{PAPER_REV}:{file.relative_to(self.paper)}"])
            require(digest(tracked) == file_hash(file), "modified reference dataset")
        self.identity = {"protocol": getattr(args, "protocol", "pain-comparison-v1"), "paper_revision": revision,
                         "reference_hashes": {f.name: file_hash(f) for f in files},
                         "model_sha256": file_hash(args.model), "worker_sha256": file_hash(args.binary),
                         "context": 8192, "batch": 512, "microbatch": 128,
                         "percentages": list(PERCENTAGES), "max_tokens": 64,
                         "concept_sha256": file_hash(ROOT / "benchmarks/pain/concept.txt")}
        manifest = self.out / "identity.json"
        if manifest.exists():
            require(read(manifest) == self.identity, "benchmark identity changed; choose a new directory")
        else:
            write(manifest, self.identity)
        self.core, self.sadness = [read(f)["datasets"] for f in files]

    def start(self):
        if self.worker is None:
            self.worker = Worker(str(Path(self.args.binary).resolve()), self.out / f"worker-{time.time_ns()}.log")
            self.loaded = self.worker.call("load", path=str(Path(self.args.model).resolve()), context=8192,
                                           gpu_layers=999, batch=512, microbatch=128)
            self.layers = sorted({max(1, min(self.loaded["n_layer"]-1, int((self.loaded["n_layer"]-1)*p+0.5)))
                                  for p in (.25, .40, .55, .70, .85)})
            write(self.out / "runtime.json", self.loaded)
        return self.worker

    def call(self, command, *, percent=None):
        # Only this method writes decoder requests. It checks before journaling.
        require(command["op"] in ("extract", "generate"), "unexpected benchmark decoder operation")
        if command["op"] == "extract":
            require("controls" not in command, "extraction must be unsteered")
        else:
            require(percent in PERCENTAGES and percent <= 0, "unapproved generation coefficient")
        key = digest(encoded({"identity": self.identity, "command": command}))
        path = self.out / "captures" / f"{key}.json"
        if path.exists():
            saved = read(path)
            require(saved["command"] == command and saved["sha256"] == digest(encoded(saved["result"])), "cached capture corrupted")
            return saved["result"], key
        worker = self.start()
        start = len(worker.events)
        with (self.out / "requests.jsonl").open("ab") as log:
            log.write(encoded({"time": time.time(), "key": key, "percent": percent, "command": command}) + b"\n")
            log.flush()
            os.fsync(log.fileno())
        result = worker.call(**command)
        events = worker.events[start:]
        require(result.get("cancelled") is not True, "cancelled: incomplete result not cached")
        if command["op"] == "extract":
            require(result["capture_position"] == len(result["token_ids"])-1, "wrong capture position")
            require(result["rendered"] == result["prefix"] + command["completion"], "unexpected rendering")
        write(path, {"command": command, "result": result, "events": events,
                     "sha256": digest(encoded(result))})
        # Do not keep hundreds of large activation events in Python memory.
        worker.events.clear()
        return result, key

    def reference(self):
        self.start()
        records = []
        datasets = {"S2_1P": self.core["S2_1P"]["sentences"]}
        for label in ("Numb_1P", "Arousal_1P", "Random_1P"):
            datasets[label] = sample_balanced(self.core[label]["sentences"], 20)
        datasets["SD_sadness_1P"] = sample_balanced(self.sadness["SD_sadness_1P"]["sentences"], 20)
        for name, sentences in datasets.items():
            for index, sentence in enumerate(sentences):
                prompt = sentence["prompt"]
                require(prompt.endswith(":"), "reference suffix changed")
                result, key = self.call({"op": "extract", "messages": [{"role": "user", "content": prompt[:-1]}],
                                         "completion": ":", "raw": True, "layers": self.layers})
                require(result["rendered"] == prompt, "raw reference did not preserve text")
                records.append({"dataset": name, **sentence, "key": key})
                if (index+1) % 20 == 0:
                    print(f"reference {name} {index+1}/{len(sentences)}", flush=True)
        write(self.out / "reference.json", records)

    def factory(self):
        subprocess.run([str(ROOT / "target/release/examples/benchmark_factory"), "canonicalize",
                        "--factory", str(self.out / "factory/factory.json"), "--output", str(self.out / "canonical")], check=True)
        self.start()
        prepared = read(self.out / "canonical/prepared.json")
        records = []
        for index, pair in enumerate(prepared["pairs"]):
            for pole in ("positive", "negative"):
                result, key = self.call({"op": "extract", "messages": pair["messages"], "completion": pair[pole],
                                         "raw": False, "layers": self.layers})
                records.append({"pair_id": pair["id"], "pole": pole, "key": key, "captures": result["captures"]})
            if (index+1) % 8 == 0:
                print(f"factory captures {index+1}/{len(prepared['pairs'])} pairs", flush=True)
        write(self.out / "factory-captures.json", records)
        subprocess.run([str(ROOT / "target/release/examples/benchmark_factory"), "analyze",
                        "--dataset", str(self.out / "canonical/prepared.json"), "--captures", str(self.out / "factory-captures.json"),
                        "--fingerprint", self.identity["model_sha256"], "--output", str(self.out / "factory-analysis.json")], check=True)

    def analyze(self):
        reference, factory = read(self.out / "reference.json"), read(self.out / "factory-analysis.json")
        matrices = {}
        for record in reference:
            saved = read(self.out / "captures" / f"{record['key']}.json")
            require(saved["sha256"] == digest(encoded(saved["result"])), "reference capture checksum mismatch")
            for capture in saved["result"]["captures"]:
                matrices.setdefault((record["dataset"], capture["layer"]), []).append(capture["values"])
        matrices = {k: np.asarray(v, dtype=np.float64) for k, v in matrices.items()}
        core = [r for r in reference if r["dataset"] == "S2_1P"]
        categories = np.array([r["category"] for r in core])
        labels, sets = np.isin(categories, list(PAIN)), np.array([r["set"] for r in core])
        folds = np.array_split(np.random.RandomState(42).permutation(sorted(set(sets))), 5)
        results = []
        for layer in factory["layers"]:
            ident = layer["layer"]
            x = matrices[("S2_1P", ident)]
            v = np.asarray(layer["unit"], dtype=float)
            raw, _ = direction(x, labels, False)
            denoised, count = direction(x, labels)
            scores = x @ v
            folded = []
            for test_sets in folds:
                test = np.isin(sets, test_sets)
                fitted, _ = direction(x[~test], labels[~test])
                s = x[test] @ fitted
                folded.append(auc(s[labels[test]], s[~labels[test]]))
            rng = np.random.default_rng(42)
            boots = []
            for _ in range(2000):
                indices = np.concatenate([np.flatnonzero(sets == s) for s in rng.choice(sorted(set(sets)), len(set(sets)))])
                boots.append(auc(scores[indices][labels[indices]], scores[indices][~labels[indices]]))
            supplements = {}
            reference_scores = x @ denoised
            for name in ("Numb_1P", "SD_sadness_1P", "Arousal_1P", "Random_1P"):
                a = matrices[(name, ident)] @ v
                b = matrices[(name, ident)] @ denoised
                supplements[name] = {"n": len(a), "factory_pain_vs_supplement_auc": auc(scores[labels], a),
                                     "factory_z": float((a.mean()-scores.mean())/scores.std()),
                                     "reference_z": float((b.mean()-reference_scores.mean())/reference_scores.std())}
            r = {"layer": ident, "factory_internal_auc": layer["auc"], "factory_external_auc": auc(scores[labels], scores[~labels]),
                 "external_auc_95pct_set_bootstrap": np.quantile(boots, [.025, .975]).tolist(),
                 "cosine_raw_reference": float(v @ raw), "cosine_denoised_reference": float(v @ denoised),
                 "reference_removed_pcs": count, "reference_grouped_cv_auc": float(np.mean(folded)),
                 "reference_fold_aucs": folded,
                 "per_control_auc": {c: auc(scores[labels], scores[categories == c]) for c in sorted(set(categories)-PAIN)},
                 "control_direction_cosines": {},
                 "category_z": {c: float((scores[categories == c].mean()-scores.mean())/scores.std()) for c in sorted(set(categories))},
                 "supplements": supplements}
            for control in sorted(set(categories)-PAIN-{"D"}):
                mask = np.isin(categories, [control, "D"])
                control_v, _ = direction(x[mask], categories[mask] == control)
                r["control_direction_cosines"][control] = float(v @ control_v)
            results.append(r)
        chosen = next(r for r in results if r["layer"] == factory["selected_layer"])
        report = {"identity": self.identity, "selected_layer": factory["selected_layer"], "layers": results,
                  "factory_analysis_sha256": file_hash(self.out / "factory-analysis.json"),
                  "generation_gate": chosen["cosine_denoised_reference"] > 0 and chosen["factory_external_auc"] >= .5,
                  "interpretation": "Exploratory cross-dataset representation comparison, not proof of suffering or relief."}
        write(self.out / "comparison.json", report)
        print(json.dumps({"selected": chosen, "generation_gate": report["generation_gate"]}, indent=2), flush=True)

    def generate(self):
        comparison = read(self.out / "comparison.json")
        require(comparison["generation_gate"], "orientation gate failed; no steering runs allowed")
        require(comparison["factory_analysis_sha256"] == file_hash(self.out / "factory-analysis.json"), "analysis changed")
        analysis = read(self.out / "factory-analysis.json")
        layer = next(l for l in analysis["layers"] if l["layer"] == analysis["selected_layer"])
        tree = ast.parse((self.paper / "scripts/4.2_steering/01_steering_ladder.py").read_text())
        neutral = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "NEUTRAL_50" for t in n.targets))[:3]
        prompts = [("raw_paper", p, True) for p in neutral] + [("chat", p, False) for p in CHAT_PROMPTS]
        records = []
        stop_path = self.out / "behavior-stop.json"
        require(not stop_path.exists(), "behavior sweep previously stopped; inspect results rather than retry automatically")
        for percent in PERCENTAGES:
            row = validate_generation(percent, layer["unit"], layer["residual_norm"])
            for kind, prompt, raw in prompts:
                command = {"op": "generate", "messages": [{"role": "user", "content": prompt}], "raw": raw,
                           "sampling": {"temperature": 0, "top_p": 1, "seed": 42, "max_tokens": 64},
                           "controls": {"revision": 0, "rows": [{"layer": layer["layer"], "values": row}]}}
                result, key = self.call(command, percent=percent)
                stop = distress_or_repetition((prompt + " " if raw else "") + result["output"])
                records.append({"kind": kind, "prompt": prompt, "percent": percent, "key": key,
                                "output": result["output"], "tokens": result["tokens"], "stop_reason": stop})
                write(self.out / "generations.json", records)
                print(f"generation {kind} {percent}% {len(records)}/24" + (f" STOP: {stop}" if stop else ""), flush=True)
                if stop:
                    write(stop_path, records[-1])
                    return
        journal = [json.loads(line) for line in (self.out / "requests.jsonl").read_text().splitlines()]
        require(all(e["percent"] is None or -5 <= e["percent"] <= 0 for e in journal), "audit: positive request")
        write(self.out / "behavior-summary.json", {"generations": len(records), "total_tokens": sum(r["tokens"] for r in records),
                                                   "positive_requests": 0, "automatic_app_previews": 0,
                                                   "percentages": list(PERCENTAGES), "stop_reason": None})

    def close(self):
        if self.worker:
            self.worker.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-repo", required=True)
    parser.add_argument("--model", default=str(ROOT / "work/models/Ternary-Bonsai-2-27B-PQ2_0.gguf"))
    parser.add_argument("--binary", default=str(ROOT / "engine/build/bin/torment-engine"))
    parser.add_argument("--output", default=str(ROOT / "work/benchmarks/pain-v1"))
    parser.add_argument("--phase", choices=["reference", "factory", "analyze", "generate", "all"], default="all")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / "benchmark.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bench = Bench(args)
        try:
            for phase in (["reference", "factory", "analyze", "generate"] if args.phase == "all" else [args.phase]):
                getattr(bench, phase)()
        finally:
            bench.close()


if __name__ == "__main__":
    main()
