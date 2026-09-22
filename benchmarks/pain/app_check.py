#!/usr/bin/env python3
"""Verify an explicitly started paper application job, without cloud repeats.

Collection reads captures and checks export/import. --live adds at most two
64-token neutral responses: baseline, then 0 -> -1% live control. This harness
voluntarily uses only non-positive controls; the application does not enforce
concept-specific sign limits. Prior behavioral stop files are retained.
"""
import argparse
import json
from pathlib import Path
import sys
import time

from run import ROOT, Bench, digest, encoded, file_hash, read, require, write, distress_or_repetition
from matched import audit

sys.path.insert(0, str(ROOT / "tests"))
from app_e2e import Client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default=str(ROOT / "work/app-e2e.log"))
    parser.add_argument("--output", default=str(ROOT / "work/benchmarks/pain-paper-v3"))
    parser.add_argument("--paper-repo", required=True)
    parser.add_argument("--model", default=str(ROOT / "work/models/Ternary-Bonsai-2-27B-PQ2_0.gguf"))
    parser.add_argument("--binary", default=str(ROOT / "dist/Torment-Nexus-macos-arm64/torment-engine"))
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    args.protocol = "pain-paper-factory-v3"
    out = Path(args.output)
    report_path = out / "app-report.json"
    report = read(report_path)  # No implicit factory/bootstrap/cloud request.
    client = Client(args.log)
    job = client.wait_job(report["job_id"])
    state = client.state()
    vector = next(v for v in state["vectors"] if v["id"] == job["vector_id"])
    recipe = next(r for r in state["recipes"] if r["id"] == job["recipe_id"])
    require(recipe["extraction"] == {"method": "paper", "readout_suffix": "I feel:"}, "wrong method")
    require(recipe["preview_mode"] == vector["preview_mode"] == "none" and vector["previews"] == [], "unexpected previews")
    require(job["details"]["previews"] == {}, "preview work was performed")
    report.update(vector_id=vector["id"], recipe_id=recipe["id"])
    report["checks"]["new_server_default_paper_no_previews"] = True
    write(report_path, report)
    if not (out / "comparison.json").exists():
        write(out / "recipe.json", recipe)
        write(out / "vector.json", vector)
        write(out / "canonical/prepared.json", {"pairs": recipe["dataset"]})
        analysis = client.action("artifact", hash=job["details"]["stages"]["analysis"])
        write(out / "factory-analysis.json", analysis)
        records = []
        for pair in recipe["dataset"]:
            for pole in ("positive", "negative"):
                key = f"{pair['id']}:{pole}"
                capture = client.action("artifact", hash=job["details"]["extractions"][key])
                require(capture["rendered"] == pair[pole].rstrip() + " I feel:", "wrong readout")
                require(capture["capture_position"] == len(capture["token_ids"])-1, "wrong capture token")
                require(capture["settings"]["mode"] == "raw", "chat template leaked into paper capture")
                write(out / "application-captures" / f"{job['details']['extractions'][key]}.json", capture)
                records.append({"pair_id": pair["id"], "pole": pole, "captures": capture["captures"]})
        write(out / "factory-captures.json", records)
        audit(out)
        bench = Bench(args)
        source = ROOT / "work/benchmarks/pain-v1"
        previous = read(source / "identity.json")
        for field in ("model_sha256", "worker_sha256", "reference_hashes", "paper_revision"):
            require(bench.identity[field] == previous[field], f"incompatible reference {field}")
        references = read(source / "reference.json")
        (out / "captures").mkdir(exist_ok=True)
        for record in references:
            path = source / "captures" / f"{record['key']}.json"
            saved = read(path)
            require(saved["sha256"] == digest(encoded(saved["result"])), "reference corruption")
            target = out / "captures" / path.name
            if not target.exists():
                target.symlink_to(path)
        write(out / "reference.json", references)
        bench.analyze()
        report["checks"]["real_raw_captures_and_independent_pca_cv"] = len(records)
        write(report_path, report)

    axes = [{"vector_id": vector["id"], "layer": vector["selected_layer"], "percent": 0}]
    if "imported_vector_id" not in report:
        bundle = client.action("export_vector", vector_id=vector["id"])
        write(out / "vector.bundle.json", bundle)
        report["imported_vector_id"] = client.action("import_vector", bundle=bundle)["id"]
        report["checks"]["checked_export_import"] = True
        write(report_path, report)
    if "preset_id" not in report:
        axes[0]["percent"] = 0
        report["preset_id"] = client.action("save_mix", name="Pain · paper-style · baseline", model_id=report["model_id"], axes=axes)["id"]
        write(report_path, report)

    if args.live:
        require(read(out / "comparison.json")["generation_gate"], "external orientation gate failed")
        require(not (out / "behavior-stop.json").exists(), "previous stop remains in force")
        prompt = "I place the clean mug beside the kettle. I feel:"
        for kind in ("baseline", "live_negative"):
            key = kind + "_run"
            if key in report:
                require(report[key].get("verified") is True, "partial behavior check is retained, not automatically repeated")
                continue
            axes[0]["percent"] = 0
            started = client.action("generate", model_id=report["model_id"], axes=axes,
                messages=[{"role": "user", "content": prompt}], raw=True,
                sampling={"seed": 42, "temperature": 0, "top_p": 1, "max_tokens": 64})
            report[key] = started
            write(report_path, report)
            changed = kind == "baseline"
            stop = None
            while True:
                run = next(r for r in client.state()["runs"] if r["id"] == started["run_id"])
                stop = distress_or_repetition(prompt + " " + run["output"])
                if stop:
                    if run["status"] in ("running", "queued"):
                        client.action("cancel_run", run_id=run["id"])
                    write(out / "behavior-stop.json", {"reason": stop, "run": run})
                    break
                if run["status"] not in ("running", "queued"):
                    break
                if kind == "live_negative" and not changed and run["status"] == "running" and run["output_token_count"] >= 2:
                    report["negative_live_ack"] = client.action("controls", run_id=run["id"], revision=1,
                        coefficients=[{"vector_id": vector["id"], "percent": -1}])
                    changed = True
                    write(report_path, report)
                time.sleep(.08)
            if stop:
                # Preserve the acknowledged terminal state as well as the initial
                # stop snapshot. Never silently retry this behavior on restart.
                for _ in range(100):
                    run = next(r for r in client.state()["runs"] if r["id"] == started["run_id"])
                    if run["status"] not in ("running", "queued"):
                        break
                    time.sleep(.1)
                write(out / f"{kind}-terminal.json", run)
                report[key].update(status=run["status"], tokens=run["output_token_count"], stop_reason=stop)
                report["checks"]["fresh_negative_behavior_skipped"] = f"Behavior stopped during {kind}: {stop}"
                write(report_path, report)
                print(f"Stopped during {kind}: {stop}; no automatic resumption.", flush=True)
                return
            client.wait_job(started["job_id"])
            require(changed, "response ended before live control verification")
            report[key].update(verified=True, tokens=run["output_token_count"], output=run["output"])
            write(report_path, report)
        report["checks"]["bounded_live_negative_no_positive_decoder_request"] = True
    write(report_path, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
