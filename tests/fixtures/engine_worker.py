#!/usr/bin/env -S python3 -B
"""Pipe-ordering fixture only: this deliberately performs no inference.

Jobs stay active until a matching cancel arrives. Reports carry the observed
input order so a cancellation sent while idle cannot satisfy the race tests.
"""

import json
import sys


def emit(ident, event, **fields):
    print(json.dumps({"v": 1, "id": ident, "event": event, **fields}), flush=True)


observed = []
active = None
for line in sys.stdin:
    command = json.loads(line)
    ident, op = command["id"], command["op"]
    observed.append({key: command[key] for key in ("id", "op", "target") if key in command})
    if op in ("generate", "extract"):
        if active is not None:
            emit(ident, "error", error="fixture already has an active job")
            continue
        active = command
        emit(ident, "progress", stage="waiting for explicit cancellation")
    elif op == "cancel":
        if active is None or command.get("target") != active["id"]:
            emit(ident, "result", cancel_requested=False, observed=observed)
            continue
        finished, active = active, None
        if finished["terminal_order"] == "done_then_ack":
            emit(finished["id"], "done", cancelled=True, observed=observed)
            emit(ident, "result", cancel_requested=True, observed=observed)
        else:
            emit(ident, "result", cancel_requested=True, observed=observed)
            emit(finished["id"], "done", cancelled=True, observed=observed)
    elif op == "complete":
        emit(ident, "result", observed=observed)
    elif op == "fail":
        emit(ident, "error", error="intentional fixture failure")
    elif op == "inspect":
        emit(ident, "result", active=active, observed=observed)
    else:
        emit(ident, "error", error=f"unsupported fixture operation: {op}")
