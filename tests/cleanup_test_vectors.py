#!/usr/bin/env python3
"""Explicit one-off cleanup of this development lab's known test copies.

Archives library entries only. Never deletes vectors, recipes, artifacts, or runs.
Requires the running app and verifies immutable records before/after the operation.
"""
import hashlib
import json
from pathlib import Path
import sqlite3

from app_e2e import Client


KEEP = [
    "5ead7e58-27ef-449c-a35e-e85adb171a17",
    "bfa64f93-c057-4ffb-a6f0-96ef0ea2c467",
    "c5f5f3ee-a574-4fc6-a810-a3237edcd67f",
]
DUPLICATES = {
    "2806a202-8ebc-4ea7-956c-9280612203a7": KEEP[0],
    "e580d67a-14ca-4ff4-a549-a11d00a2c046": KEEP[0],
    "1afa08a6-5124-4034-b4c3-6cadeed8fb26": KEEP[2],
}
EDIT_TEST = "b461baab-52cb-495a-8d4a-84397634b705"
RETAINED_KINDS = ("models", "recipes", "vectors", "presets", "jobs", "runs", "conversations")


def record_hashes(state):
    return {
        kind: {
            row["id"]: hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            for row in state[kind]
        }
        for kind in RETAINED_KINDS
    }


def main():
    root = Path("work/library-cleanup")
    root.mkdir(exist_ok=True)
    report_path = root / "report.json"
    assert not report_path.exists(), "cleanup report already exists; inspect rather than overwrite"
    client = Client("work/app-e2e.log")
    before = client.state()
    assert "vector_visibility" in before, "app needs reversible library archives"
    assert not any(job["status"] in ("queued", "running") for job in before["jobs"]), "app busy"
    vectors = {row["id"]: row for row in before["vectors"]}
    assert set(vectors) == set(KEEP + list(DUPLICATES) + [EDIT_TEST]), "lab changed; inspect before cleanup"
    for duplicate, original in DUPLICATES.items():
        assert vectors[duplicate]["manifest_hash"] == vectors[original]["manifest_hash"]
    prior_report = json.loads(Path("work/app-e2e-report.json").read_text())
    assert prior_report["browser_edited_vector_id"] == EDIT_TEST
    assert vectors[EDIT_TEST]["name"] == vectors[KEEP[1]]["name"]
    assert vectors[EDIT_TEST]["manifest_hash"] != vectors[KEEP[1]]["manifest_hash"]

    backup_path = root / "index-before.sqlite3"
    assert not backup_path.exists(), "preserve the existing backup"
    with sqlite3.connect("file:work/e2e-data/index.sqlite3?mode=ro", uri=True) as source:
        with sqlite3.connect(backup_path) as backup:
            source.backup(backup)
            assert backup.execute("PRAGMA quick_check").fetchone() == ("ok",)
    hashes = record_hashes(before)
    (root / "record-hashes-before.json").write_text(json.dumps(hashes, indent=2))
    archived = []
    for vector_id in [*DUPLICATES, EDIT_TEST]:
        result = client.action("set_vector_archived", vector_id=vector_id, archived=True)
        assert result["id"] == vector_id and result["archived"] is True
        archived.append({"id": vector_id, "name": vectors[vector_id]["name"],
                         "reason": "exact import-test copy" if vector_id in DUPLICATES else "browser-edit test version"})
    after = client.state()
    assert record_hashes(after) == hashes, "cleanup changed an original record"
    hidden = {row["id"] for row in after["vector_visibility"] if row["archived"]}
    assert hidden == set(DUPLICATES) | {EDIT_TEST}
    assert {row["id"] for row in after["vectors"] if row["id"] not in hidden} == set(KEEP)
    report = {"status": "passed", "kept": KEEP, "archived": archived,
              "retained_record_counts": {kind: len(rows) for kind, rows in hashes.items()},
              "all_existing_records_unchanged": True, "deleted_records": 0,
              "deleted_artifacts": 0, "database_backup": str(backup_path),
              "new_inference_or_cloud_jobs": 0}
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": report["status"], "visible_concepts": len(KEEP), "archived": len(archived)}, indent=2))


if __name__ == "__main__":
    main()
