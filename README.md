# Torment Nexus

A local activation-steering playground for Apple Silicon. Turn concepts into
model-specific vectors, mix them, and move sliders while a response streams.
An instrument panel, not an emotion meter.

## Launch

From the unpacked application folder or source checkout:

```sh
./torment
```

This opens the browser UI. The package needs no compiler or Node installation.
From source, the launcher builds on first use and needs Rust 1.97.1, CMake and
Apple's command-line developer tools. Models are downloaded separately; the
application is an unsigned local build, not a notarized macOS app.

Creating concepts requires **Codex CLI 0.155.1** and an existing login; other
versions are currently rejected. Existing vectors and local chat work offline.

## Your first steered response

1. Open **Manage models** and download the **Bonsai 2 PQ2_0** preset—the reference
   model. You can also browse Hugging Face or import a local GGUF by absolute path;
   other model architectures may not support extraction or steering.
2. Once the download finishes, select the model and click **Load into engine**.
3. Click **Create a concept**. Describe a contrast, such as “playful dry humor
   versus earnest literal delivery,” and submit. Codex agents design, write and
   review the dataset; local extraction then produces a vector. Follow progress
   in **Jobs & provenance**.
4. When the job completes, open **Vectors** and add the result to your mix. Each
   extracted layer has its own slider, initially at zero. Try **+5%** on the layer
   marked **✦**; it separated the dataset's two sides best, though that doesn't
   guarantee the best steering effect.
5. Enter a prompt and click **Run prompt**. Adjust sliders while it streams.
   Afterwards, **Compare baseline** reruns the same input and sampling settings
   without steering. **Save mix** preserves your slider settings.

New concepts use [extraction adapted from Tagliabue et al. (2026)](docs/paper-extraction.md).
Completion contrast is an advanced alternative. Automatic previews are off by default.

## Sliders and chat

- **Percentages describe perturbation size, not emotional intensity.** +10% adds
  a direction scaled to 10% of that layer's measured median unsteered activation
  magnitude. Negative subtracts it; zero disables it. The default range is ±20%, but
  numeric input accepts larger finite values—including 200%.
- **Layers are different depths inside the model.** Contributions sum at each
  layer without renormalization. Five layers at 20% are not one layer at 100%.
  Diagnostics warn about weak or extreme mixtures rather than forbidding them.
- **Add or remove vectors between responses; move sliders anytime.** Pending
  changes take effect when acknowledged. They affect future computation, not
  already-generated text or cached state. **Zero all** clears the controls.
- **Scratchpad** starts fresh. Switch to **Chat** to continue from its reply;
  **Duplicate** forks a new chat through the chosen reply, leaving the original
  unchanged. Chat re-encodes history each turn under the current mix. A baseline
  keeps that transcript; it cannot undo earlier steered replies.
- **Allow self-adjustment** lets the model change its current-mix sliders. It is
  off by default, revocable mid-response, and every change is recorded. Manual
  controls still work. [Tool behavior and external MCP setup](docs/steering.md#self-adjustment-and-mcp).
- **Unbounded output** removes the output-token cap, not the context limit.
  Generation ends naturally or with **Stop**. When context fills, older text is
  omitted from the model's view; the full output remains saved.
- **Library** and **Mix** collapse the sidebars; **Focus chat** hides both.

## Saved work and privacy

Data lives in `~/Library/Application Support/Torment Nexus`. To use another lab:

```sh
./torment --data-dir /path/to/lab
```

Keep using the same data directory. If your library looks empty, check both the
data directory and selected model: vectors belong to a specific model file.
Imported GGUFs stay at their original paths; don't move them without re-importing.

Vectors, recipes, conversations and saved mixes have **Rename** and **Delete**
controls. Delete removes entries from the UI, not their stored provenance; it is
not a disk wipe. **Archive** hides vectors; use **Show archived** to restore them.

Dataset and recipe edits create new versions and rerun only affected stages.
Recipes are shared across models; extraction reuses their examples to create
vectors for the selected model.
Interrupted jobs retain completed work; retry is explicit, never silently repeated
on startup. Interrupted responses are saved but cannot be resumed exactly.

Concept-generation inputs go to Codex; local chat transcripts are not sent
automatically. Codex retains control of account credentials. The UI is loopback-only;
keep its launch links and MCP connection tokens private.

## Development

The Rust server embeds the React UI. A separate C++ worker runs the pinned Prism
llama.cpp fork with Metal—not stock llama.cpp. Node is needed only to rebuild the UI.

```sh
(cd frontend && npm ci && npm test && npm run build)
scripts/build-engine.sh
cargo test --all-targets
(cd frontend && npm run test:browser)
scripts/package.sh
```

Packaging produces `dist/Torment-Nexus-macos-arm64.tar.gz`, without models or lab data.
See [recorded integration checks](docs/STATUS.md), [steering details](docs/steering.md),
[storage and provenance](docs/storage.md), [Codex isolation](docs/codex-isolation.md)
and [integration contracts](docs/contracts.md).
