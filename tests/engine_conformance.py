#!/usr/bin/env python3
"""Real pinned-engine tests, never a mocked inference worker.

python3 tests/engine_conformance.py --model work/models/stories260K.gguf
python3 tests/engine_conformance.py --model work/models/Ternary-Bonsai-2-27B-PQ2_0.gguf --bonsai
"""
import argparse
import json
import math
from pathlib import Path
import queue
import subprocess
import threading
import time


class Worker:
    def __init__(self, binary, log):
        self.log = open(log, "w")
        self.proc = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.log, text=True, bufsize=1)
        self.inbox = queue.Queue()
        self.sequence = 0
        self.events = []

        def reader():
            for line in self.proc.stdout:
                try:
                    self.inbox.put(json.loads(line))
                except Exception as error:
                    self.inbox.put({"event": "bad_json", "error": str(error), "line": line})
            self.inbox.put({"event": "eof"})

        threading.Thread(target=reader, daemon=True).start()

    def send(self, op, **fields):
        self.sequence += 1
        ident = str(self.sequence)
        self.proc.stdin.write(json.dumps({"v": 1, "id": ident, "op": op, **fields}) + "\n")
        self.proc.stdin.flush()
        return ident

    def recv(self):
        result = self.inbox.get(timeout=600)
        assert result["event"] not in ("eof", "bad_json"), result
        self.events.append(result)
        return result

    def call(self, op, **fields):
        ident = self.send(op, **fields)
        while True:
            event = self.recv()
            if event.get("id") == ident and event["event"] in ("result", "done", "error"):
                assert event["event"] != "error", event
                return event

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=60)
        assert self.proc.returncode == 0, self.proc.returncode
        self.log.close()


def maximum_error(a, b):
    assert len(a) == len(b)
    return max((abs(x - y) for x, y in zip(a, b)), default=0)


def run(args):
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    worker = Worker(args.binary, str(Path(args.output).with_suffix(".log")))
    result = {"model": str(Path(args.model).resolve()), "checks": {}, "started": time.time()}
    try:
        result["capabilities"] = worker.call("capabilities")
        loaded = worker.call("load", path=result["model"], context=8192 if args.bonsai else 512,
                             gpu_layers=args.gpu_layers, batch=512, microbatch=128)
        result["loaded"] = loaded
        width, depth = loaded["n_embd"], loaded["n_layer"]
        selected = sorted(set(max(1, min(depth - 1, round((depth - 1) * p)))
                              for p in [.25, .40, .55, .70, .85]))
        messages = [{"role": "user", "content": "Describe a quiet garden in one sentence." if args.bonsai else "Once upon a time"}]
        common = dict(messages=messages, raw=not args.bonsai, layers=selected, completion=" The garden is quiet and green.")
        first = worker.call("extract", **common, prefix_chunk=7)
        assert first["capture_position"] == len(first["token_ids"]) - 1
        assert first["rendered"] == first["prefix"] + common["completion"]
        assert set(c["layer"] for c in first["captures"]) == set(selected)
        assert all(len(c["values"]) == width for c in first["captures"])
        result["checks"]["capture_position_and_shape"] = True
        chunked = worker.call("extract", **common, prefix_chunk=1)
        default = worker.call("extract", **common, prefix_chunk=512)
        assert first["token_ids"] == chunked["token_ids"] == default["token_ids"]
        chunk_error = max(maximum_error(a["values"], b["values"])
                          for a, b in zip(first["captures"], chunked["captures"]))
        large_chunk_error = max(maximum_error(a["values"], b["values"])
                                for a, b in zip(first["captures"], default["captures"]))
        result["checks"]["prefix_chunk_max_abs_error"] = max(chunk_error, large_chunk_error)
        # Different Metal matmul/prefix shapes can have small rounding differences.
        scale = max(abs(x) for c in first["captures"] for x in c["values"])
        assert max(chunk_error, large_chunk_error) <= max(0.01, scale * 0.005)
        worker.call("extract", **{**common, "completion": " Nothing here resembles a garden."})
        again = worker.call("extract", **common, prefix_chunk=7)
        reset_error = max(maximum_error(a["values"], b["values"])
                          for a, b in zip(first["captures"], again["captures"]))
        result["checks"]["A_B_A_max_abs_error"] = reset_error
        assert reset_error <= 1e-5
        zero = [{"layer": layer, "values": [0.0] * width} for layer in selected]
        baseline = worker.call("probe", **common, include_logits=not args.bonsai)
        disabled = worker.call("probe", **common, include_logits=not args.bonsai,
                               controls={"revision": 1, "rows": zero})
        zero_error = max(maximum_error(a["values"], b["values"])
                         for a, b in zip(baseline["captures"], disabled["captures"]))
        result["checks"]["zero_disabled_capture_error"] = zero_error
        assert zero_error <= 1e-6
        if not args.bonsai:
            result["checks"]["zero_disabled_logits_error"] = maximum_error(baseline["logits"], disabled["logits"])
            assert result["checks"]["zero_disabled_logits_error"] <= 1e-6
        layer = selected[0]
        base_values = baseline["captures"][0]["values"]
        # Inject only on the separately decoded final token. Earlier prefix computation
        # remains unsteered, allowing a direct representation-level delta measurement.
        direction = [0.125 if i % 2 else -0.0625 for i in range(width)]
        positive = worker.call("probe", **common, controls={"revision": 2, "rows": [{"layer": layer, "values": direction}]})
        negative = worker.call("probe", **common, controls={"revision": 3, "rows": [{"layer": layer, "values": [-v for v in direction]}]})
        positive_delta = [x - y for x, y in zip(positive["captures"][0]["values"], base_values)]
        negative_delta = [x - y for x, y in zip(negative["captures"][0]["values"], base_values)]
        injection_error = max(maximum_error(positive_delta, direction), maximum_error(negative_delta, [-x for x in direction]))
        result["checks"]["signed_injection_max_abs_error"] = injection_error
        assert injection_error <= max(1e-5, scale * 1e-6), injection_error
        # Exercise leftover allocated device rows: upload two, then only one.
        worker.call("probe", **common, controls={"revision": 4, "rows": [
            {"layer": layer, "values": direction}, {"layer": selected[-1], "values": direction}]})
        after_remove = worker.call("probe", **common, controls={"revision": 5, "rows": [{"layer": layer, "values": direction}]})
        stale_error = max(maximum_error(a["values"], b["values"])
                          for a, b in zip(positive["captures"], after_remove["captures"]))
        result["checks"]["removed_rows_max_abs_error"] = stale_error
        assert stale_error <= 1e-5

        generation = dict(messages=messages, raw=not args.bonsai,
                          sampling={"temperature": 0, "seed": 42, "top_p": 1, "max_tokens": 64 if args.bonsai else 200})
        start = len(worker.events)
        plain = worker.call("generate", **generation)
        plain_tokens = [e["token"] for e in worker.events[start:] if e["event"] == "token"]
        start = len(worker.events)
        zero_generation = worker.call("generate", **generation, controls={"revision": 0, "rows": zero})
        zero_tokens = [e["token"] for e in worker.events[start:] if e["event"] == "token"]
        assert plain_tokens == zero_tokens and plain["output"] == zero_generation["output"]
        result["checks"]["zero_generation_identical"] = True
        start = len(worker.events)
        ident = worker.send("generate", **generation)
        update_id = switch_id = cancel_id = None
        while True:
            event = worker.recv()
            if event.get("id") == ident and event["event"] == "token":
                if update_id is None and event["index"] >= 1:
                    update_id = worker.send("controls", target=ident, revision=1,
                                            rows=[{"layer": layer, "values": direction}])
                    switch_id = worker.send("load", path=result["model"])
                if cancel_id is None and event["index"] >= 12:
                    cancel_id = worker.send("cancel", target=ident)
            if event.get("id") == ident and event["event"] in ("done", "error"):
                assert event["event"] == "done", event
                result["live_done"] = event
                break
        events = worker.events[start:]
        applied = next(e for e in events if e["event"] == "applied" and e["revision"] == 1)
        tokens = [e for e in events if e["event"] == "token"]
        assert all(e["token"] == plain_tokens[e["index"]] for e in tokens if e["index"] < applied["first_token_index"])
        assert all(e["revision"] == (0 if e["index"] < applied["first_token_index"] else 1) for e in tokens)
        assert any(e.get("id") == switch_id and e["event"] == "error" for e in events)
        assert result["live_done"]["cancelled"], "generation ended before cancellation test could run"
        result["checks"]["live_boundary_ack"] = applied
        result["checks"]["switch_while_busy_rejected"] = True
        result["checks"]["cancellation"] = True
        worker.call("unload")
        result["status"] = "passed"
    finally:
        result["finished"] = time.time()
        Path(args.output).write_text(json.dumps(result, indent=2))
        worker.close()
    print(json.dumps(result["checks"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--binary", default="engine/build/bin/torment-engine")
    parser.add_argument("--bonsai", action="store_true")
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--output", default="work/engine-conformance.json")
    run(parser.parse_args())
