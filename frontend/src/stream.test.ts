import { describe, expect, it } from "vitest";
import { TokenReconciler } from "./stream";

describe("atomic snapshot / token reconciliation", () => {
  it("does not duplicate a token already included in the next snapshot", () => {
    const stream = new TokenReconciler();
    expect(
      stream.snapshot({ id: "a", output: "Hello", output_token_count: 1 }),
    ).toBe("Hello");
    expect(stream.token("a", 1, " world")).toBe("Hello world");
    expect(
      stream.snapshot({
        id: "a",
        output: "Hello world",
        output_token_count: 2,
      }),
    ).toBe("Hello world");
    expect(stream.token("a", 1, " world")).toBe("Hello world");
    expect(stream.token("a", 2, "!")).toBe("Hello world!");
  });
  it("anchors resumed streams even when events arrive before the snapshot", () => {
    const stream = new TokenReconciler();
    expect(stream.token("a", 2, "!")).toBeUndefined();
    expect(
      stream.snapshot({
        id: "a",
        output: "Hello world",
        output_token_count: 2,
      }),
    ).toBe("Hello world!");
  });
  it("keeps newer events across stale response completion and recovers gaps", () => {
    const stream = new TokenReconciler();
    stream.snapshot({ id: "a", output: "A", output_token_count: 1 });
    expect(stream.token("a", 2, "C")).toBe("A");
    expect(stream.token("a", 1, "B")).toBe("ABC");
    expect(
      stream.snapshot({ id: "a", output: "A", output_token_count: 1 }),
    ).toBe("ABC");
    expect(
      stream.snapshot({ id: "a", output: "ABCDE", output_token_count: 5 }),
    ).toBe("ABCDE");
  });
  it("fresh runs stream from zero without needing a snapshot", () => {
    const stream = new TokenReconciler();
    expect(stream.token("a", 0, "A")).toBe("A");
    expect(stream.token("a", 1, "B")).toBe("AB");
    expect(
      stream.snapshot({ id: "a", output: "A", output_token_count: 1 }),
    ).toBe("AB");
  });
});
