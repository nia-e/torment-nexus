import { describe, expect, it } from "vitest";
import { checkpointRefs, completedOutput } from "./stages";

describe("immutable completed stage selection", () => {
  it("only exposes the seven editable base stages, including edited dataset inputs", () => {
    expect(
      checkpointRefs({
        design: "a",
        writer_1: "b",
        writer_1_attempt_0: "incomplete",
        candidates: "derived",
        dataset_input: "edited",
      }),
    ).toEqual({ design: "a", writer_1: "b", dataset: "edited" });
  });
  it("accepts completed and explicitly edited outputs, never partial or failed agent artifacts", () => {
    expect(
      completedOutput({ status: "completed", output: { pairs: [] } }),
    ).toEqual({ pairs: [] });
    expect(completedOutput({ edited: true, output: ["kept"] })).toEqual([
      "kept",
    ]);
    expect(completedOutput({ status: "edited", output: 0 })).toBe(0);
    expect(
      completedOutput({ status: "running", output: "partial" }),
    ).toBeUndefined();
    expect(
      completedOutput({ status: "failed", output: "invalid" }),
    ).toBeUndefined();
    expect(completedOutput({ status: "completed" })).toBeUndefined();
  });
});
