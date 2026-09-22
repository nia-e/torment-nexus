# End-to-end completion audit

21 September 2026 · macOS arm64 · local unsigned Torment Nexus 0.1.0.

This audit distinguishes real integration evidence from fixture tests and source
inspection. The original application scope is complete with the user's later
changes: default [extraction adapted from Tagliabue et al. (2026)](paper-extraction.md),
unrestricted finite coefficient signs and ranges, previews off by default, and
reversible test-library cleanup.

## Requirement coverage

| Requirement | Verification |
| --- | --- |
| One-command local Rust application, embedded browser UI, private worker | Packaged launcher tested from a relocated directory with system-only PATH; embedded assets and Codex discovery worked without Node/Rust. Loopback Host/Origin/session-token checks have regressions. |
| Pinned Prism / tiny GGUF / genuine Bonsai | Exact upstream revision and wrapper recorded in [STATUS](STATUS.md). Current packaged worker is byte-identical to the independently exercised worker. Metal conformance covers exact captures, supported layer mapping, signed injection, zero/stale-row clearing, chunk bounds and A→B→A state resets. |
| HF acquisition and local imports | Genuine 7,206,168,928-byte Bonsai download pinned to its immutable HF revision and SHA-256; tiny imported by reference. Transfer fixtures exercise interruption, cancellation, correct/ignored/mismatched resume ranges, authentication and rate limits. |
| Codex-login multi-agent factory | Three actual 96-pair factories contain 32 pairs each from Astra/Sol/Terra, independent agent threads, original/repaired outputs and complete provenance. Persisted isolation observations show no exposed tools for pinned CLI 0.155.1. |
| Durable validation/repair and visible cloud failures | Malformed structured-output, protocol/disconnect/rate-limit and cancellation regressions. Explicit signed-out/expired-auth fixtures prove no silent retry, durable failed attempts and explicit recovery without modifying the real login. |
| Editable immutable recipes and downstream invalidation | Real browser repair edit made a new version, preserved the original and five upstream stage hashes, reused cloud attempt IDs and completed 192 local captures. |
| Both extraction methods, calibrated vectors and diagnostics | Independent NumPy calculations reproduce completion directions/norms/scaling/pole reversal and control PCA/CV/all-pair fits. The adapted method is the new default; old recipes are not silently rewritten. Shape/non-finite/missing-hook errors fail explicitly. |
| Live N-axis controls and scratchpad/chat | Genuine two-axis Bonsai mixing, revision acknowledgements and complete clearing; deterministic engine checks establish future-token boundaries and busy-load/cancel handling. Latest ordinary smoke checks live revision at token 7, malformed frozen-membership snapshots, same-input baseline and two-turn chat. |
| UI workflows | Browser contracts plus real populated-app sessions cover create/edit, signed/expanded ranges, stream/stop/reset/baseline, save/reload, PCA/CV diagnostics and provenance, exports/imports, token rotation and archive/restore. |
| Durable storage and compatible checked bundles | Independently SHA-verified 921 unique blobs / 404,930,249 bytes across seven vector closures; datasets, bounded F32 shapes and finite values checked. Wrong-model/corrupt/hidden-reference bundles rejected. Database WAL and quick_check passed. |
| Restart/recovery | Actual SIGKILL during cloud review/repair retained completed stage hashes and partial output. Explicit offline local retry reused all stages with a deliberately unavailable Codex executable. Killed inference retained 22 durable tokens and was marked interrupted, never exactly resumable. |
| Clean library without broken provenance | Separate mutable archive metadata hides test copies without deleting or mutating vectors, recipes, runs, presets or artifact blobs. Historical axes remain resolvable even when their library cards are archived. |

## Evidence map

The workspace retains private/local evidence in `work/`; these reports and model
files are deliberately **not** shipped in the application archive.

- `packaged-{tiny,bonsai}-conformance.json`, `engine-raw-conformance.json`:
  real packaged engine checks; measured tolerances are in [STATUS](STATUS.md).
- `app-e2e-report.json`, `lab-numerical-conformance.json`: two original agent
  concepts, complete application loop and independent completion arithmetic.
- `final-package-smoke.json`: ordinary live mix, baseline, membership errors, chat.
- `recovery-proof.json`, `inference-recovery-proof.json`: actual interrupted jobs.
- `benchmarks/pain-matched-v2/`, `benchmarks/pain-paper-v3/`: Tagliabue et al. adaptation,
  independent arithmetic, exact benchmark records and retained behavioral stop.
- `real-paper-ui-report.json`, `paper-relocated-proof.json`: real browser and
  system-only-PATH package verification.
- `library-cleanup/report.json`, `library-cleanup/browser-report.json`:
  four archived test entries, unchanged historical records, actual restore/reload.
- `multilayer-real-report.json`, `multilayer-{rust-tests,frontend-unit,browser}.log`:
  subsequent independent per-layer controls, backward-compatible saved mixes and
  live events, real Bonsai clearing/acknowledgements, and conversation-first layout.
- `final-rust-tests.log`, `final-clippy.log`, browser/unit test reports:
  automated contracts and failure paths; not substitutes for the real runs above.

## Claims deliberately not made

This is a working local application, not a validated emotion or welfare meter.
The method is an explicit adaptation, not a literal reproduction of Tagliabue et al.'s
complete layer/perspective/injection protocol. Representation separation does
not prove causal steering quality. The bounded pain work stopped on repetition;
the fresh adapted-method zero baseline also repeated, so no further pain generations
were run for completion. See [the benchmark](pain-benchmark.md).

Only the pinned Bonsai and tiny GGUF were exercised. Other models require compatible
hooks/templates or explicit raw mode. Managed permission requirements still apply;
the Codex adapter fails closed on unverified CLI versions. Native/notarized packaging,
MLX, vision, remote serving, fine-tuning and speculative decoding remain out of scope.
