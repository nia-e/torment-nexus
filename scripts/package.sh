#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
test "$(uname -sm)" = "Darwin arm64" || {
    echo 'This package target is macOS / Apple Silicon.' >&2
    exit 1
}
scripts/build-engine.sh
cargo build --release --locked
destination="dist/Torment-Nexus-macos-arm64"
mkdir -p "$destination/licenses"
cp target/release/torment-nexus engine/build/bin/torment-engine "$destination/"
cp vendor/llama.cpp/LICENSE "$destination/licenses/Prism-llama.cpp-LICENSE"
cp README.md "$destination/README.md"
cp -R docs "$destination/"
mkdir -p "$destination/benchmarks/pain"
cp benchmarks/pain/*.py benchmarks/pain/*.md benchmarks/pain/*.txt "$destination/benchmarks/pain/"
python3 scripts/collect-licenses.py "$destination/licenses"
cat > "$destination/torment" <<'LAUNCH'
#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$root/torment-nexus" --worker "$root/torment-engine" "$@"
LAUNCH
chmod +x "$destination/torment"
python3 - "$destination" <<'PY'
from pathlib import Path
import hashlib
import sys
root = Path(sys.argv[1])
files = sorted(p for p in root.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
with (root / 'SHA256SUMS').open('w') as output:
    for path in files:
        output.write(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}\n')
PY
tar -czf dist/Torment-Nexus-macos-arm64.tar.gz -C dist Torment-Nexus-macos-arm64
printf '\nPackage: %s\nLaunch: %s/torment\n' "dist/Torment-Nexus-macos-arm64.tar.gz" "$destination"
