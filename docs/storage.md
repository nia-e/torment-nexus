# Storage and acquisition

`Store::open(data_dir)` opens SQLite in WAL / FULL synchronous mode and takes an
OS file lock for the lifetime of the store and all its clones. A second process
using the same directory fails rather than interrupting the first one's jobs.
On reopen, queued/running jobs and runs become `interrupted`; completed stages,
partial output, and control events stay intact. Nothing automatically reruns.

Records use `put(kind, id, &value)`, `get(kind, id)`, and `list(kind)`. The value
must contain the same `id`. Recipe/vector records are immutable: an identical
write is idempotent, but changed contents require a new version ID. Logical
concept names are not unique. Other record kinds support snapshot updates.

## Artifact format and ordering

Artifacts live at `artifacts/<first-two-hash-characters>/<sha256>`. SHA-256 covers
every byte, including tensor shape headers. Writers install a synced temporary
file atomically without replacing an existing artifact; readers verify its hash
and size before parsing. A corrupt existing blob is an error, not silently fixed.

The store offers JSON, JSONL, raw-byte, and F32 helpers. F32 format v1 is:

| Bytes | Meaning |
| --- | --- |
| 0–7 | `TNF32\0\x01\0` |
| 8–11 | Little-endian u32 rank |
| next `8 * rank` | Little-endian u64 dimensions |
| remainder | Little-endian f32 values in row-major order |

Rank must be 1–8, dimensions positive, product at most 16,777,216, and every value
finite. Byte count must match shape exactly. All artifacts are capped at 128 MiB.

Write artifacts before records. `Store::put` additionally verifies referenced
`manifest_hash`, `direction_hash`, `unit_hash`, and `artifact_hash` blobs before
committing. Model binding, algorithm provenance, and semantic compatibility belong
to the typed vector/import layer, not to the untyped blob store.

## Model acquisition

`import_model(path, name)` keeps an absolute reference to the existing file and
streams a SHA-256 fingerprint; it does not duplicate the model. `verify_model`
rechecks that binding before load. Import validates GGUF magic/version, not the
entire GGUF structure; the pinned inference engine validates model compatibility.

`browse_hf(repo)` resolves `main` to an immutable commit and lists GGUF files with
their size and LFS SHA-256 where available. `download_model` resolves a requested
revision once and persists that resolution until the operation completes. Retry
uses the pinned descriptor and partial file. Transfers are serialized per request
and resolved content using OS file locks.

- HTTP 206 must name the exact requested offset and consistent total size.
- A server ignoring Range with HTTP 200 restarts the partial instead of appending.
- Truncation, network errors, rate limits, and cancellation retain partial data.
- Cancellation is polled while connecting, reading, and fingerprinting.
- Completed bytes are size/hash checked, GGUF checked, then atomically published.
- Failed complete downloads are retained under a `.rejected-*` filename; an explicit
  retry downloads fresh bytes. Existing published corruption fails visibly.
- Gated/private HF credentials are not imported. Use an authorized external HF
  client and import its local file. Codex credentials never enter this path.

The progress callback includes `downloaded`, optional `total`, immutable `revision`,
and `phase`. Preserve this resolved revision in job provenance. Model files remain
separate from application/package resources. Files over 512 GiB are rejected.

## Focused verification

Unit tests in these modules exercise corruption/non-finite/shape rejection,
immutable versions, restart retention, instance locking, import drift, exact HTTP
resumption, ignored ranges, truncation/retry, cancellation, and access/rate-limit
failures. HTTP tests bind loopback and therefore need an environment permitting
local sockets. They do not establish inference quality or native engine behavior.
