import { afterEach, describe, expect, it, vi } from "vitest";
import {
  expandLayerAxes,
  finitePercent,
  LiveControls,
  requestedAxes,
} from "./controls";
import type { Vector } from "./types";

const axes = [
  { vector_id: "a", layer: 3, percent: 0 },
  { vector_id: "b", layer: 7, percent: 0 },
];
afterEach(() => vi.useRealTimers());
describe("complete revisioned controls", () => {
  it("updates two layers of the same concept independently and restores the latest snapshot", async () => {
    vi.useFakeTimers();
    const same = [
      { vector_id: "a", layer: 3, percent: 0 },
      { vector_id: "a", layer: 7, percent: 0 },
    ];
    const send = vi.fn().mockResolvedValue({});
    const control = new LiveControls("run", same, 0, send, vi.fn(), vi.fn());
    control.update([
      { ...same[0], percent: 200 },
      { ...same[1], percent: -2 },
    ]);
    await vi.advanceTimersByTimeAsync(110);
    expect(send.mock.calls[0][0].coefficients).toEqual([
      { vector_id: "a", layer: 3, percent: 200 },
      { vector_id: "a", layer: 7, percent: -2 },
    ]);
    expect(
      requestedAxes(same, send.mock.calls[0][0].coefficients).map(
        (a) => a.percent,
      ),
    ).toEqual([200, -2]);
    expect(
      requestedAxes(same, [{ vector_id: "a", percent: 9 }]).map(
        (a) => a.percent,
      ),
    ).toEqual([0, 0]);
    expect(
      requestedAxes([same[1]], [{ vector_id: "a", percent: 9 }])[0].percent,
    ).toBe(9);
    control.dispose();
  });
  it("adds only zero-valued missing usable layers to an old saved mix", () => {
    const old = [{ vector_id: "a", layer: 7, percent: -8 }];
    const vectors = [
      {
        id: "a",
        layers: [{ layer: 3 }, { layer: 7 }, { layer: 9, usable: false }],
      },
    ] as Vector[];
    const expanded = expandLayerAxes(old, vectors);
    expect(expanded).toEqual([
      ...old,
      { vector_id: "a", layer: 3, percent: 0 },
    ]);
    expect(old).toEqual([{ vector_id: "a", layer: 7, percent: -8 }]);
    expect(expandLayerAxes(expanded, vectors)).toEqual(expanded);
  });
  it("coalesces movement but retains zero rows and monotone revisions", async () => {
    vi.useFakeTimers();
    const send = vi.fn().mockResolvedValue({});
    const pending = vi.fn();
    const control = new LiveControls("run", axes, 0, send, pending, vi.fn());
    control.update([{ ...axes[0], percent: 8 }, axes[1]]);
    control.update([{ ...axes[0], percent: -4 }, axes[1]]);
    control.update([{ ...axes[0], percent: 0 }, axes[1]]);
    expect(send).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(110);
    expect(send).toHaveBeenCalledExactlyOnceWith({
      run_id: "run",
      revision: 3,
      base_revision: 0,
      coefficients: [
        { vector_id: "a", layer: 3, percent: 0 },
        { vector_id: "b", layer: 7, percent: 0 },
      ],
    });
    expect(control.acknowledge(2)).toBe(false);
    expect(control.acknowledge(3)).toBe(true);
    control.dispose();
  });
  it("rejects shape changes and non-finite values for a frozen run", () => {
    const control = new LiveControls("run", axes, 0, vi.fn(), vi.fn(), vi.fn());
    expect(() => control.update([axes[0]])).toThrow(/frozen/);
    expect(() => control.update([{ ...axes[0], layer: 4 }, axes[1]])).toThrow(
      /frozen/,
    );
    expect(() =>
      control.update([{ ...axes[0], percent: Infinity }, axes[1]]),
    ).toThrow(/finite/);
    control.dispose();
  });
  it("adopts model changes without echoing them or replaying stale snapshots", async () => {
    vi.useFakeTimers();
    const send = vi.fn().mockResolvedValue({});
    const control = new LiveControls("run", axes, 0, send, vi.fn(), vi.fn());
    const changed = [{ ...axes[0], percent: -2 }, axes[1]];
    expect(control.acceptExternal(changed, 1)).toBe(true);
    control.update(changed);
    await vi.advanceTimersByTimeAsync(110);
    expect(send).not.toHaveBeenCalled();
    control.update([{ ...axes[0], percent: 200 }, axes[1]]);
    expect(control.acceptExternal(changed, 1)).toBe(false);
    await vi.advanceTimersByTimeAsync(110);
    expect(send.mock.calls[0][0]).toMatchObject({
      revision: 2,
      base_revision: 1,
    });
    control.dispose();
  });
  it("does not send delayed edits to a completed or cancelled run", async () => {
    vi.useFakeTimers();
    const send = vi.fn();
    const control = new LiveControls("run", axes, 4, send, vi.fn(), vi.fn());
    control.update([{ ...axes[0], percent: 12 }, axes[1]]);
    control.dispose();
    await vi.advanceTimersByTimeAsync(1000);
    expect(send).not.toHaveBeenCalled();
  });
  it("continues after the persisted highest requested revision on reload", async () => {
    vi.useFakeTimers();
    const send = vi.fn().mockResolvedValue({});
    const control = new LiveControls("run", axes, 17, send, vi.fn(), vi.fn());
    control.update([{ ...axes[0], percent: -300 }, axes[1]]);
    await vi.advanceTimersByTimeAsync(110);
    expect(send.mock.calls[0][0].revision).toBe(18);
    control.dispose();
  });
  it("never races HTTP delivery order even when the first request is slow", async () => {
    vi.useFakeTimers();
    let resolve!: () => void;
    const send = vi
      .fn()
      .mockImplementationOnce(
        () =>
          new Promise<void>((r) => {
            resolve = r;
          }),
      )
      .mockResolvedValue({});
    const control = new LiveControls("run", axes, 0, send, vi.fn(), vi.fn());
    control.update([{ ...axes[0], percent: 1 }, axes[1]]);
    await vi.advanceTimersByTimeAsync(110);
    control.update([{ ...axes[0], percent: 2 }, axes[1]]);
    await vi.advanceTimersByTimeAsync(110);
    expect(send).toHaveBeenCalledTimes(1);
    resolve();
    await vi.advanceTimersByTimeAsync(0);
    expect(send).toHaveBeenCalledTimes(2);
    expect(send.mock.calls.map((call) => call[0].revision)).toEqual([1, 2]);
    control.dispose();
  });
});
describe("finite numeric input", () => {
  it("accepts signed values without gatekeeping extremes, rejects absent/non-finite values", () => {
    expect(finitePercent("-4000")).toBe(-4000);
    expect(finitePercent("0")).toBe(0);
    expect(finitePercent("0.125")).toBe(0.125);
    expect(finitePercent("1e3")).toBe(1000);
    expect(finitePercent("")).toBeNull();
    expect(finitePercent(" ")).toBeNull();
    expect(finitePercent("Infinity")).toBeNull();
    expect(finitePercent("NaN")).toBeNull();
  });
});
