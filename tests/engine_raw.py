#!/usr/bin/env python3
"""Real tiny-engine raw-mode regressions, without Codex or model downloads.

python3 tests/engine_raw.py --model work/models/stories260K.gguf
"""

import argparse
import json
from pathlib import Path
import time

from engine_conformance import Worker


def rejected(worker, op, expected, **fields):
    ident = worker.send(op, **fields)
    while True:
        event = worker.recv()
        if event.get("id") == ident and event["event"] in ("result", "done", "error"):
            assert event["event"] == "error", event
            assert expected in event["error"], event
            return event["error"]


def run(args):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"model": str(Path(args.model).resolve()), "started": time.time(), "checks": {}}
    worker = Worker(args.binary, str(output.with_suffix(".log")))
    try:
        loaded = worker.call("load", path=report["model"], context=1024,
                             gpu_layers=args.gpu_layers, batch=128, microbatch=32)
        report["loaded"] = loaded
        layer = max(1, min(loaded["n_layer"] - 1, loaded["n_layer"] // 2))
        completion = "The fox saw a flower."
        prompt = "  Once upon a time\n"
        single = [{"role": "user", "content": prompt}]
        common = {"raw": True, "layers": [layer], "completion": completion}
        capture = worker.call("extract", messages=single, **common)
        assert capture["prefix"] == prompt
        assert capture["rendered"] == prompt + completion
        assert capture["settings"]["raw_format"] == "single-user-verbatim-v1"
        assert capture["settings"]["add_generation_prompt"] is False
        assert capture["settings"]["enable_thinking"] is None
        assert capture["template"] == ""
        report["checks"]["single_user_verbatim"] = True

        messages = [
            {"role": "system", "content": "Write a short story.\nKeep it simple."},
            {"role": "user", "content": "Where did the fox go?"},
            {"role": "assistant", "content": "The fox went into a garden."},
            {"role": "user", "content": "What did it see?"},
        ]
        expected = "".join(f'{message["role"]}: {message["content"]}\n\n'
                           for message in messages) + "assistant: "
        capture = worker.call("extract", messages=messages, prefix_chunk=7, **common)
        assert capture["prefix"] == expected
        assert capture["rendered"] == expected + completion
        assert capture["capture_position"] == len(capture["token_ids"]) - 1
        assert capture["settings"]["raw_format"] == "role-labeled-dialogue-v1"
        assert capture["settings"]["add_generation_prompt"] is True
        assert capture["settings"]["enable_thinking"] is None
        assert capture["template"] == ""
        assert len(capture["captures"]) == 1
        assert len(capture["captures"][0]["values"]) == loaded["n_embd"]
        report["checks"]["multi_message_raw_capture"] = True

        start = len(worker.events)
        generated = worker.call("generate", messages=messages, raw=True,
                                sampling={"temperature": 0, "seed": 42, "top_p": 1, "max_tokens": 8})
        rendered = [event for event in worker.events[start:]
                    if event.get("id") == generated["id"] and event["event"] == "rendered"]
        assert len(rendered) == 1 and rendered[0]["rendered"] == expected
        assert rendered[0]["settings"]["raw_format"] == "role-labeled-dialogue-v1"
        assert rendered[0]["template"] == ""
        assert isinstance(generated["output"], str) and 0 <= generated["tokens"] <= 8
        assert generated["cancelled"] is False
        report["checks"]["multi_turn_raw_generation"] = True

        for role in ("developer", "tool", "root"):
            rejected(worker, "extract", "unsupported message role",
                     messages=[{"role": role, "content": "not a supported context role"}], **common)
        rejected(worker, "extract", "message requires text role and content",
                 messages=[{"role": "user", "content": None}], **common)
        report["checks"]["raw_roles_and_content_validated"] = True

        if not loaded.get("chat_template"):
            rejected(worker, "extract", "explicitly select raw completion mode",
                     messages=single, layers=[layer], completion=completion, raw=False)
            report["checks"]["no_implicit_raw_fallback"] = True

        for label, marker in (("non_eog_control", args.control_marker), ("end_marker", args.end_marker)):
            error = rejected(worker, "extract", "final token is a control/end marker",
                             messages=single, layers=[layer], completion=f"Literal marker {marker}", raw=True)
            report["checks"][f"final_{label}_rejected"] = {"marker": marker, "error": error}
        recovered = worker.call("extract", messages=messages, **common)
        assert recovered["token_ids"] == capture["token_ids"]
        assert recovered["capture_position"] == capture["capture_position"]
        report["checks"]["worker_recovers_after_rejected_capture"] = True
        worker.call("unload")
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        report["finished"] = time.time()
        output.write_text(json.dumps(report, indent=2))
        worker.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="work/models/stories260K.gguf")
    parser.add_argument("--binary", default="engine/build/bin/torment-engine")
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--control-marker", default="<s>")
    parser.add_argument("--end-marker", default="</s>")
    parser.add_argument("--output", default="work/engine-raw-conformance.json")
    run(parser.parse_args())
