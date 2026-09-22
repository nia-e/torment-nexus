#!/usr/bin/env python3
"""Remain in-flight after the rendered event to test application cleanup."""
import json
import sys
import time

command = json.loads(sys.stdin.readline())
assert command["op"] == "generate"
print(json.dumps({"v": 1, "id": command["id"], "event": "rendered", "runtime": {}}), flush=True)
while True:
    time.sleep(60)
