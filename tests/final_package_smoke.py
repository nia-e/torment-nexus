#!/usr/bin/env python3
"""Opt-in real final-package smoke check using existing NON-PAIN concepts.

No cloud generation. At most 304 output tokens across a two-axis response,
same-input baseline, and two short chat turns. Does not resume stopped benchmarks.
"""
import json
from pathlib import Path
import time

from app_e2e import Client


def main():
    destination = Path("work/final-package-smoke.json")
    assert not destination.exists(), "retain previous evidence; do not silently repeat inference"
    client = Client("work/app-e2e.log")
    original = json.loads(Path("work/app-e2e-report.json").read_text())
    state = client.state()
    assert not any(j["status"] in ("queued", "running") for j in state["jobs"]), "local app busy"
    vectors = [next(v for v in state["vectors"] if v["id"] == key) for key in original["vector_ids"]]
    assert len(vectors) == 2
    assert all("pain" not in v["name"].lower() for v in vectors)
    report = {"model_id": original["model_id"], "vector_ids": original["vector_ids"],
              "concepts": [v["name"] for v in vectors], "checks": {}, "new_cloud_concepts": 0}

    def save():
        destination.write_text(json.dumps(report, indent=2))

    axes = [{"vector_id": v["id"], "layer": v["selected_layer"], "percent": 0} for v in vectors]
    sampling = {"seed": 73, "temperature": 0, "top_p": 1, "max_tokens": 128}
    messages = [{"role": "user", "content": "Describe how to organize a desk into three tidy areas, in one short paragraph of about 100 words."}]
    started = client.action("generate", model_id=original["model_id"], axes=axes, messages=messages, sampling=sampling)
    report["mixed"] = started
    save()
    changed = False
    for _ in range(1200):
        run = next(r for r in client.state()["runs"] if r["id"] == started["run_id"])
        if run["status"] not in ("queued", "running"):
            break
        if run["status"] == "running" and run["output_token_count"] >= 2 and not changed:
            snapshots = {
                "incomplete": [{"vector_id": vectors[0]["id"], "percent": -1}],
                "unknown": [{"vector_id": vectors[0]["id"], "percent": -1}, {"vector_id": "not-a-frozen-axis", "percent": -1}],
                "duplicate": [{"vector_id": vectors[0]["id"], "percent": -1}] * 2,
            }
            for name, coefficients in snapshots.items():
                try:
                    client.action("controls", run_id=run["id"], revision=1, coefficients=coefficients)
                except RuntimeError as error:
                    assert any(word in str(error) for word in ("complete snapshot", "frozen", "duplicate")), error
                    report["checks"][name + "_snapshot_rejected"] = str(error)
                else:
                    raise AssertionError(name + " malformed snapshot accepted")
            report["live_ack"] = client.action("controls", run_id=run["id"], revision=1,
                coefficients=[{"vector_id": v["id"], "percent": -i-1} for i, v in enumerate(vectors)])
            assert report["live_ack"]["first_token_index"] >= run["output_token_count"]
            changed = True
            save()
        time.sleep(.05)
    assert changed, "response ended before live check"
    client.wait_job(started["job_id"])
    run = next(r for r in client.state()["runs"] if r["id"] == started["run_id"])
    assert run["status"] == "completed" and len(run["requested_controls"]) == 2
    assert any(event["revision"] == 1 for event in run["applied_controls"])
    report["mixed_tokens"] = run["output_token_count"]
    baseline = client.action("generate", baseline_of=run["id"])
    report["baseline"] = baseline
    save()
    client.wait_job(baseline["job_id"])
    baseline = next(r for r in client.state()["runs"] if r["id"] == baseline["run_id"])
    assert baseline["axes"] == [] and baseline["messages"] == run["messages"] and baseline["sampling"] == run["sampling"]
    report["baseline_tokens"] = baseline["output_token_count"]
    report["conversation_id"] = client.action("new_conversation", title="Final package chat check")["id"]
    save()
    turns = []
    for prompt in ("Remember the word marigold. Briefly acknowledge it.", "What word did I ask you to remember? Answer with only that word."):
        conversation = next(c for c in client.state()["conversations"] if c["id"] == report["conversation_id"])
        messages = conversation["messages"] + [{"role": "user", "content": prompt}]
        result = client.action("generate", model_id=original["model_id"], conversation_id=report["conversation_id"], messages=messages,
                               axes=[], sampling={"seed": 73, "temperature": 0, "max_tokens": 24})
        turns.append(result)
        report["turns"] = turns
        save()
        client.wait_job(result["job_id"])
    final = client.state()
    conversation = next(c for c in final["conversations"] if c["id"] == report["conversation_id"])
    assert len(conversation["messages"]) == 4 and len(conversation["run_ids"]) == 2
    assert "marigold" in conversation["messages"][-1]["content"].lower()
    report["checks"]["final_binary_live_mix_same_input_baseline_two_turn_chat"] = True
    report["status"] = "passed"
    save()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
