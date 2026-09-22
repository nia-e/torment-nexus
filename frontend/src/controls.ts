import type { Axis, ControlEvent, Vector } from "./types";
export type Coefficient = { vector_id: string; layer: number; percent: number };
export type ControlSnapshot = {
  run_id: string;
  revision: number;
  base_revision?: number;
  coefficients: Coefficient[];
};
export function finitePercent(value: string): number | null {
  if (!value.trim()) return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}
export function signature(axes: Axis[]): string {
  return JSON.stringify(
    axes.map(({ vector_id, layer, percent }) => [vector_id, layer, percent]),
  );
}
/** Upgrade saved single-layer selections by adding only zero-valued missing rows. */
export function expandLayerAxes(axes: Axis[], vectors: Vector[]): Axis[] {
  const result = [...axes];
  for (const id of new Set(axes.map((axis) => axis.vector_id))) {
    const vector = vectors.find((candidate) => candidate.id === id);
    for (const layer of vector?.layers ?? []) {
      if (
        layer.usable === false ||
        result.some(
          (axis) => axis.vector_id === id && axis.layer === layer.layer,
        )
      )
        continue;
      result.push({ vector_id: id, layer: layer.layer, percent: 0 });
    }
  }
  return result;
}
export function requestedAxes(
  axes: Axis[],
  coefficients: ControlEvent["coefficients"],
): Axis[] {
  return axes.map((axis) => {
    const coefficient = coefficients?.find(
      (candidate) =>
        candidate.vector_id === axis.vector_id &&
        (candidate.layer === axis.layer ||
          (candidate.layer === undefined &&
            axes.filter((member) => member.vector_id === axis.vector_id)
              .length === 1)),
    );
    return { ...axis, percent: coefficient?.percent ?? axis.percent };
  });
}
/** Run identity and layer membership are immutable; every request is a complete snapshot. */
export class LiveControls {
  private pending: ControlSnapshot | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private last: string;
  private revision: number;
  private latestRequested: number;
  private lastAcknowledged = 0;
  private disposed = false;
  private serverRevision: number;
  private externalRevision: number;
  private sending = false;
  readonly frozen: ReadonlyArray<{ vector_id: string; layer: number }>;
  constructor(
    readonly runId: string,
    axes: Axis[],
    initialRevision: number,
    private send: (snapshot: ControlSnapshot) => Promise<unknown>,
    private onPending: (revision: number) => void,
    private onError: (error: unknown) => void,
    private delayMs = 110,
  ) {
    this.frozen = axes.map(({ vector_id, layer }) =>
      Object.freeze({ vector_id, layer }),
    );
    this.last = signature(axes);
    this.revision = initialRevision;
    this.latestRequested = initialRevision;
    this.serverRevision = initialRevision;
    this.externalRevision = initialRevision;
  }
  update(axes: Axis[]): void {
    if (this.disposed) return;
    if (
      axes.length !== this.frozen.length ||
      axes.some(
        (axis, i) =>
          axis.vector_id !== this.frozen[i].vector_id ||
          axis.layer !== this.frozen[i].layer,
      )
    )
      throw new Error(
        "Vectors and layers are frozen until this response finishes.",
      );
    if (axes.some((axis) => !Number.isFinite(axis.percent)))
      throw new Error("Coefficients must be finite numbers.");
    const next = signature(axes);
    if (next === this.last) return;
    this.last = next;
    this.latestRequested = ++this.revision;
    this.pending = {
      run_id: this.runId,
      revision: this.revision,
      coefficients: axes.map(({ vector_id, layer, percent }) => ({
        vector_id,
        layer,
        percent,
      })),
    };
    this.onPending(this.revision);
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = setTimeout(() => {
      this.timer = null;
      void this.flush();
    }, this.delayMs);
  }
  private async flush(): Promise<void> {
    if (this.disposed || this.sending) return;
    this.sending = true;
    try {
      while (this.pending && !this.disposed && this.timer === null) {
        const pending = this.pending;
        this.pending = null;
        try {
          await this.send({ ...pending, base_revision: this.serverRevision });
          this.serverRevision = Math.max(this.serverRevision, pending.revision);
        } catch (error) {
          if (!this.disposed) this.onError(error);
        }
      }
    } finally {
      this.sending = false;
    }
  }
  acknowledge(revision: number): boolean {
    this.serverRevision = Math.max(this.serverRevision, revision);
    this.revision = Math.max(this.revision, revision);
    this.lastAcknowledged = Math.max(this.lastAcknowledged, revision);
    return this.lastAcknowledged >= this.latestRequested;
  }
  /** Accepted model/MCP edits are authoritative; never echo them as user writes. */
  acceptExternal(axes: Axis[], revision: number): boolean {
    if (revision < this.serverRevision || revision <= this.externalRevision)
      return false;
    this.externalRevision = revision;
    this.serverRevision = revision;
    this.revision = Math.max(this.revision, revision);
    this.latestRequested = revision;
    this.last = signature(axes);
    this.pending = null;
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
    return true;
  }
  dispose() {
    this.disposed = true;
    if (this.timer !== null) clearTimeout(this.timer);
    this.pending = null;
  }
}
