import { describe, expect, it } from "vitest";
import {
  conversationHistory,
  conversationRuns,
  startupModelId,
} from "./workflows";
import { emptyState, type Run, type State } from "./types";
const run = (id: string, fields: Partial<Run> = {}): Run => ({
  id,
  model_id: "m",
  messages: [],
  axes: [],
  sampling: { seed: 42, temperature: 0.7, top_p: 0.95, max_tokens: 20 },
  output: "",
  status: "completed",
  conversation_id: "chat",
  created_at: 1234,
  ...fields,
});

describe("durable chat ordering", () => {
  it("shows a fork's shared prefix, even after deleting its parent, but not later parent turns", () => {
    const input = [
      run("later"),
      run("branch", { conversation_id: "fork" }),
      run("prefix", { deleted: true }),
      run("scratch", { conversation_id: undefined }),
    ];
    expect(
      conversationRuns(input, "fork", ["prefix"]).map((run) => run.id),
    ).toEqual(["prefix", "branch"]);
    expect(
      conversationRuns(input, "continued", ["scratch"]).map((run) => run.id),
    ).toEqual(["scratch"]);
  });
  it("retains tool-result turns with a fallback for historical runs", () => {
    const previous = run("tools", {
      messages: [{ role: "user", content: "Check the mix." }],
      reply_messages: [
        { role: "assistant", content: "<torment_tool>get_mix</torment_tool>" },
        { role: "user", content: "Tool result: zero." },
        { role: "assistant", content: "The mix is zero." },
      ],
    });
    expect(conversationHistory(previous, "visible prose")).toEqual([
      ...previous.messages,
      ...previous.reply_messages!,
    ]);
    expect(conversationHistory(run("old"), "Old reply")).toEqual([
      { role: "assistant", content: "Old reply" },
    ]);
  });
  it("preserves exact turn ordering when timestamps tie", () => {
    expect(
      conversationRuns([run("b"), run("a"), run("c")], "chat", [
        "a",
        "b",
        "c",
      ]).map((run) => run.id),
    ).toEqual(["a", "b", "c"]);
  });
  it("appends active turns not yet committed to the conversation and excludes baselines", () => {
    const input = [
      run("active", { status: "running" }),
      run("baseline", { baseline_of: "a" }),
      run("b"),
      run("other", { conversation_id: "other" }),
      run("a"),
    ];
    expect(
      conversationRuns(input, "chat", ["a", "b"]).map((run) => run.id),
    ).toEqual(["a", "b", "active"]);
  });
});

describe("cold-start model selection", () => {
  const state: State = {
    ...emptyState,
    models: [
      {
        id: "tiny",
        name: "Tiny test",
        path: "tiny.gguf",
        fingerprint: "tiny-fp",
      },
      {
        id: "bonsai",
        name: "Bonsai",
        path: "bonsai.gguf",
        fingerprint: "bonsai-fp",
      },
    ],
    vectors: [
      {
        id: "v",
        name: "Saved concept",
        model_id: "bonsai",
        model_fingerprint: "bonsai-fp",
        layers: [],
        selected_layer: 16,
      },
    ],
  };
  it("prefers the populated model over the most recently imported test model", () => {
    expect(startupModelId(state, "")).toBe("bonsai");
    expect(startupModelId(state, "missing-from-another-lab")).toBe("bonsai");
    expect(startupModelId(emptyState, "")).toBe("");
  });
  it("respects explicit selection, then loaded model, then recent use", () => {
    expect(startupModelId(state, "tiny")).toBe("tiny");
    expect(
      startupModelId(
        { ...state, engine: { status: "idle", model_id: "tiny" } },
        "",
      ),
    ).toBe("tiny");
    expect(
      startupModelId(
        {
          ...state,
          runs: [
            run("old", { model_id: "tiny", created_at: 1 }),
            run("latest", { model_id: "bonsai", created_at: 2 }),
          ],
        },
        "",
      ),
    ).toBe("bonsai");
  });
});
