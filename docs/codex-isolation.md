# Codex dataset-worker boundary

This is a local integration with **Codex CLI 0.155.1**, verified on macOS/Apple
Silicon. Other versions fail closed until their tool contract is re-verified.
Authentication remains in Codex's normal home; Torment Nexus never reads
`auth.json`, copies tokens, sets a replacement `CODEX_HOME`, or edits global config.

## Why flags alone are not enough

The installed artifact was exercised, not merely compared with documentation:

- `mcp_servers={}` merges with existing configuration. It does **not** clear it.
  Startup first inspects configuration without creating a thread, then restarts
  with an inline table setting every discovered server's `enabled=false`.
  It verifies both configuration and per-thread runtime state.
- The installed model catalog can force `tool_mode=code_mode_only` and v2
  delegation despite disabled feature flags. With flag-only configuration, an
  actual outgoing request still exposed pure-V8 execution, patching, and
  collaboration. Treating `tools: []` as proof would also be wrong: this release
  carries tool definitions in `input[].type=additional_tools`.
- The worker therefore derives a private, content-addressed copy of the installed
  bundled catalog, setting `tool_mode`, `multi_agent_version`, and
  `apply_patch_tool_type` to null, `experimental_supported_tools` to an empty array,
  and `supports_search_tool` to false. Model IDs, weights/provider routing,
  context limits, and other inference metadata are unchanged. Actual account
  availability is discovered through ordinary `model/list`, intersected with the
  installed security-verified catalog before displaying or assigning models.
  Remotely advertised models absent from the installed catalog are not offered.
- `tools.experimental_request_user_input.enabled=false` removes the remaining
  question tool. Both outgoing tool locations were then **empty** in a loopback,
  no-auth fake-provider capture. There is no descendant-agent capability to inherit
  more permissive tools.

Flags additionally disable shell, apps, plugins, MCP dependencies, hooks,
memories, browser/computer use, image tools, goals, workspace dependencies, and
skill discovery/instructions. Web search, memory reads/writes, and notification
commands are explicitly disabled. Effective settings are checked before a turn.
Threads are independent and ephemeral. Their sandbox is read-only with tool
network access disabled. **Approval policy and reviewer are not overridden**;
conflicting managed requirements fail rather than being bypassed. Any unexpected
server-side tool/approval request is rejected, and unexpected tool items terminate
the worker as defense in depth, not as the primary sandbox.

Only concept-generation inputs are passed to this module. Local inference chat
transcripts have no route into it. Private catalog source/output checksums, the
exact Codex user-agent/version, schema, prompt, role/model, raw output, thread/turn
IDs, usage, and isolation projection are retained in stage artifacts. Raw Codex
configuration and diagnostic stderr are not persisted because they can contain
provider secrets.

## Stage durability

The factory persists a running attempt before invoking Codex, periodic partial
output checkpoints, every original answer (including malformed ones), and the
validated completed stage before advancing. Up to two schema/content repairs are
allowed. A transport, authentication, or rate-limit failure stops the stage rather
than silently buying another turn. Explicit retries reuse only completed stages
whose input hash still matches. An interrupted turn is not called exactly
resumable.

Explicit Retry refreshes discovery and replaces unavailable model assignments for
unfinished stages, recording the old roles and substitutions in job provenance.
Completed stages retain their assignments and outputs; a missing completed-stage
model requires an explicit edit/fork rather than silent regeneration.

Manual edits use `{ "edited": true, "output": ... }` checkpoints:

- `design`: invalidate writers, candidates, review, repair, and dataset.
- `writer_1`, `writer_2`, `writer_3`: preserve other writers, invalidate candidates,
  review, repair, dataset, and any prior dataset override.
- `dataset_input`: an edited pair array bypasses writers, then review/repair run.
- `review`: preserve candidates, replace acceptance/repair decisions, and
  invalidate repair/dataset.
- `repair`: preserve the review, replace its repaired pairs, invalidate the dataset.

The GUI can fork any completed checkpoint from the jobs inspector, including while
the original factory is still running. The original job and its outputs are unchanged.

Writer outputs, candidate data, reviewer decisions, repairs, and final pairs stay
separate. The steering module assigns the canonical scenario-family-grouped split.

## Verification

- `cargo test --lib codex::tests -- --skip live_discovery_and_schema`
- `cargo test --lib factory::tests`
- `cargo test --lib codex::tests::live_discovery_and_schema -- --ignored --nocapture`
  performed actual model discovery and a schema-constrained `{ "ok": true }`
  generation using the compiled Rust client and existing login.
- `cargo test --lib codex::tests::installed_catalog_has_no_tools_for_all_three_writer_models -- --ignored --nocapture`
  exercises the production client against the installed binary for Astra, Sol,
  and Terra. Only provider routing changes, to an unauthenticated loopback
  recorder; a pre-turn provider assertion prevents accidental remote generation.
  **All three produced empty legacy and `additional_tools` registries.**
- Deterministic process fixtures exercise 429 errors without automatic retry,
  malformed JSON, cancellation and partial-output retention, split JSON protocol
  lines across polling timeouts, two schema-repair attempts, and completed-stage
  reuse without a new generation call.
- Development investigation: `work/codex-probe/registry-constrained.py`,
  `work/codex-probe/tool-registry.json`, and
  `work/codex-probe/registry-constrained-output.txt`. These are ignored scratch
  evidence, not packaged runtime resources. The capture uses a loopback fake
  provider with authentication disabled and does not record HTTP headers.

The empty registry observation and real schema generation establish this specific
client/installed-artifact path. They do not prove arbitrary future Codex releases
safe. End-to-end Bonsai extraction, mixed controls, and application reload are
separate application integration checks.

Primary interface references: [App Server](https://learn.chatgpt.com/docs/app-server)
and [Configuration Reference](https://learn.chatgpt.com/docs/config-file/config-reference).
