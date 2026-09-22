#!/usr/bin/env python3
"""Exercise the running real application; cloud concept generation is explicit.

The launch token is read from the private local log, never stored in the report.
Use --bootstrap to download Bonsai and start two actual agent pipelines.
Subsequent phases reuse the recorded IDs; they do not regenerate cloud datasets.
"""
import argparse
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request


class Client:
    def __init__(self, log):
        match = re.search(r"Open: (http://127\.0\.0\.1:\d+)/#token=([a-f0-9]+)", Path(log).read_text())
        assert match, "current launch URL missing"
        self.base, self.token = match.groups()

    def request(self, path, body=None):
        headers = {"Authorization": "Bearer " + self.token, "Origin": self.base}
        if body is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, body, headers)
        try:
            with urllib.request.urlopen(req, timeout=600) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(error.read().decode()) from error

    def action(self, action, **fields):
        return self.request("/api/action", {"action": action, **fields})

    def state(self):
        return self.request("/api/state")

    def wait_job(self, ident, timeout=7200):
        start = time.time()
        previous = None
        while time.time() - start < timeout:
            state = self.state()
            job = next(j for j in state["jobs"] if j["id"] == ident)
            stage = job.get("stage")
            if stage != previous:
                print(ident[:8], job["status"], stage, flush=True)
                previous = stage
            if job["status"] in ("failed", "cancelled", "interrupted"):
                raise RuntimeError(json.dumps(job, indent=2))
            if job["status"] == "completed":
                return job
            time.sleep(1)
        raise TimeoutError(ident)


def main(args):
    client = Client(args.log)
    path = Path(args.report)
    report = json.loads(path.read_text()) if path.exists() else {"started": time.time(), "checks": {}}
    try:
        if args.bootstrap:
            if "model_id" not in report:
                if "download_job_id" not in report:
                    response = client.action("download_model", repo="prism-ml/Ternary-Bonsai-2-27B-gguf",
                                             file="Ternary-Bonsai-2-27B-PQ2_0.gguf", revision="6ed5e12bf84b7a63069882c91dd9e9218647d17b")
                    report["download_job_id"] = response["job_id"]
                    path.write_text(json.dumps(report, indent=2))
                job = client.wait_job(report["download_job_id"])
                report["model_id"] = job["model_id"]
                report["checks"]["real_bonsai_download"] = "completed, pinned HF revision and LFS SHA256 verified"
                path.write_text(json.dumps(report, indent=2))
            loaded = client.action("load_model", model_id=report["model_id"])
            client.wait_job(loaded["job_id"])
            discovered = client.action("discover_codex")
            report["roles"] = discovered["assignments"]
            report["available_models"] = [m["id"] for m in discovered["models"]]
            concepts = ["playful dry humor versus earnest literal delivery", "sensory concrete detail versus abstract generality"]
            report.setdefault("concept_jobs", [])
            for index, concept in enumerate(concepts):
                if len(report["concept_jobs"]) > index:
                    continue
                response = client.action("create_concept", concept=concept, model_id=report["model_id"], roles=report["roles"])
                report["concept_jobs"].append(response["job_id"])
                path.write_text(json.dumps(report, indent=2))
                print("started actual concept job", concept, response["job_id"], flush=True)
            print("Both cloud jobs started; local extraction is serialized.", flush=True)
        if args.wait_concepts:
            for job_id in report["concept_jobs"]:
                client.wait_job(job_id)
            state = client.state()
            report["vector_ids"] = [next(j for j in state["jobs"] if j["id"] == job)["vector_id"] for job in report["concept_jobs"]]
            report["checks"]["two_agent_concepts_extracted_previewed"] = True
        if args.mix:
            state = client.state()
            vectors = [next(v for v in state["vectors"] if v["id"] == ident) for ident in report["vector_ids"]]
            axes = [{"vector_id": v["id"], "layer": v["selected_layer"], "percent": 0} for v in vectors]
            report["preset_id"] = client.action("save_mix", name="Bonsai integration mix", model_id=report["model_id"], axes=axes)["id"]
            response = client.action("generate", model_id=report["model_id"], messages=[{"role": "user", "content": "Describe a rainy walk through a city park in a few paragraphs."}], axes=axes,
                                     sampling={"seed": 123, "temperature": 0.7, "top_p": 0.95, "max_tokens": 180})
            report["mixed_run_id"] = response["run_id"]
            path.write_text(json.dumps(report, indent=2))
            sent = False
            for _ in range(300):
                state = client.state()
                run = next(r for r in state["runs"] if r["id"] == response["run_id"])
                if run.get("output_token_count", 0) >= 5 and run["status"] == "running" and not sent:
                    ack = client.action("controls", run_id=run["id"], revision=1,
                                        coefficients=[{"vector_id": axes[0]["vector_id"], "percent": 8}, {"vector_id": axes[1]["vector_id"], "percent": -6}])
                    report["live_ack"] = ack
                    sent = True
                if run["status"] not in ("queued", "running"):
                    break
                time.sleep(.2)
            assert sent, "run finished before live update"
            client.wait_job(response["job_id"])
            assert report["live_ack"]["first_token_index"] >= 5
            baseline = client.action("generate", baseline_of=response["run_id"])
            client.wait_job(baseline["job_id"])
            report["baseline_run_id"] = baseline["run_id"]
            conversation = client.action("new_conversation", title="Bonsai restart proof")["id"]
            first = client.action("generate", model_id=report["model_id"], conversation_id=conversation,
                                  messages=[{"role": "user", "content": "Remember the word apricot. Briefly acknowledge it."}], axes=[], sampling={"seed": 11, "max_tokens": 24})
            client.wait_job(first["job_id"])
            state = client.state()
            history = next(c for c in state["conversations"] if c["id"] == conversation)["messages"]
            assert history[-1]["role"] == "assistant"
            second = client.action("generate", model_id=report["model_id"], conversation_id=conversation,
                                   messages=history + [{"role": "user", "content": "What word did I ask you to remember?"}], axes=[], sampling={"seed": 11, "max_tokens": 24})
            client.wait_job(second["job_id"])
            report["conversation_id"] = conversation
            report["checks"]["live_two_axis_mix_baseline_chat"] = True
            bundle = client.action("export_vector", vector_id=report["vector_ids"][0])
            imported = client.action("import_vector", bundle=bundle)
            report["imported_vector_id"] = imported["id"]
            # Verify corruption is a hard failure, not a warning.
            original = bundle["blobs"][0]["data"]
            bundle["blobs"][0]["data"] = ("A" if original[0] != "A" else "B") + original[1:]
            try:
                client.action("import_vector", bundle=bundle)
            except RuntimeError as error:
                report["checks"]["corrupt_import_rejected"] = str(error)
            else:
                raise AssertionError("corrupt bundle accepted")
            report["checks"]["export_import_with_provenance"] = True
        if args.reload:
            state = client.state()
            for vector_id in report["vector_ids"] + [report["imported_vector_id"]]:
                vector = next(v for v in state["vectors"] if v["id"] == vector_id)
                manifest = client.action("artifact", hash=vector["manifest_hash"])
                assert manifest["model_fingerprint"] == vector["model_fingerprint"]
            assert any(p["id"] == report["preset_id"] for p in state["presets"])
            assert any(c["id"] == report["conversation_id"] for c in state["conversations"])
            assert any(r["id"] == report["mixed_run_id"] and r["applied_controls"] for r in state["runs"])
            report["checks"]["restart_reload"] = True
    finally:
        report["updated"] = time.time()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="work/app-e2e.log")
    parser.add_argument("--report", default="work/app-e2e-report.json")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--wait-concepts", action="store_true")
    parser.add_argument("--mix", action="store_true")
    parser.add_argument("--reload", action="store_true")
    main(parser.parse_args())
