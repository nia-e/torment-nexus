# Explicit raw-completion mode

Raw mode never invokes the model's chat template. It must be selected explicitly;
a model without a template otherwise produces an actionable error.

The engine accepts text messages with roles `system`, `user`, and `assistant`.
Rendering is deterministic and shared by extraction, previews, and generation:

- **One user message:** its content is the exact raw prompt. No trimming, role
  label, separator, or assistant prefix is added. Recorded as
  `settings.raw_format: "single-user-verbatim-v1"`.
- **Other supported message sequences:** each message becomes
  `role + ": " + content + "\n\n"`, followed by `"assistant: "`. Message contents
  remain unchanged. Recorded as `"role-labeled-dialogue-v1"`.

For example, a multi-turn context is rendered as:

```text
user: Where did the fox go?

assistant: The fox went into a garden.

user: What did it see?

assistant:
```

The final `assistant:` label includes one trailing space.

This is a plain dialogue convention, not a claim about a model's preferred chat
format. A raw chat's first single-user turn is verbatim; subsequent turns use
the labeled format. To control every byte yourself, provide the whole prefix as
one user message. The rendered text, token IDs, and format choice are retained
in provenance. Template-specific thinking/time settings are `null` in raw mode.

Extraction appends the supplied completion without closing markers, uses the
existing tokenizer and BOS policy, and captures the final token separately.
Recognized control tokens and end-of-generation tokens are rejected at that final
position: a structural marker is not assistant content. Tokenization still
parses recognized special-token spellings; this change does not silently alter
the tokenizer or strip supplied text.

Focused real-engine regression check (tiny GGUF, no Codex calls):

```sh
scripts/build-engine.sh
python3 tests/engine_raw.py --model work/models/stories260K.gguf
```
