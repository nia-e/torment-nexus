#!/usr/bin/env python3
"""Collect installed dependency notices while packaging; no runtime dependency."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

destination = Path(sys.argv[1])
metadata = json.loads(subprocess.check_output([
    "cargo", "metadata", "--format-version", "1", "--locked", "--offline",
    "--filter-platform", "aarch64-apple-darwin",
]))
index = []
for package in metadata["packages"]:
    if not package.get("source"):
        continue
    root = Path(package["manifest_path"]).parent
    ident = package["name"] + "-" + package["version"]
    paths = set()
    for pattern in ("LICENSE*", "LICENCE*", "COPYING*", "NOTICE*", "license*", "licence*"):
        paths.update(p for p in root.glob(pattern) if p.is_file())
    if package.get("license_file"):
        path = root / package["license_file"]
        if path.is_file():
            paths.add(path)
    for path in paths:
        target = destination / "rust" / ident / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    index.append({"package": ident, "license": package.get("license"),
                  "repository": package.get("repository"), "notices": sorted(p.name for p in paths)})
for name in ("react", "react-dom", "scheduler"):
    path = Path("frontend/node_modules") / name / "LICENSE"
    if not path.is_file():
        raise SystemExit(f"Missing frontend notice {path}; run npm ci in frontend before packaging")
    target = destination / "frontend" / name / "LICENSE"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
(destination / "rust-dependencies.json").write_text(json.dumps(index, indent=2) + "\n")
