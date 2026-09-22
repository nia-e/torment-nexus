import type { Run } from "./types";

type Buffer = {
  output: string;
  count: number;
  pending: Map<number, string>;
  checkpoint: boolean;
};
/** Reconciles zero-based token events with atomic output/count snapshots, including out-of-order fetch completion. */
export class TokenReconciler {
  private runs = new Map<string, Buffer>();
  private buffer(id: string) {
    let buffer = this.runs.get(id);
    if (!buffer) {
      buffer = { output: "", count: 0, pending: new Map(), checkpoint: false };
      this.runs.set(id, buffer);
    }
    return buffer;
  }
  private drain(buffer: Buffer) {
    while (buffer.pending.has(buffer.count)) {
      buffer.output += buffer.pending.get(buffer.count)!;
      buffer.pending.delete(buffer.count++);
    }
  }
  token(id: string, index: number, text: string): string | undefined {
    if (!Number.isSafeInteger(index) || index < 0) return undefined;
    const buffer = this.buffer(id);
    if (index >= buffer.count) buffer.pending.set(index, text);
    this.drain(buffer);
    return buffer.checkpoint || buffer.count > 0 ? buffer.output : undefined;
  }
  snapshot(run: Pick<Run, "id" | "output" | "output_token_count">): string {
    const buffer = this.buffer(run.id);
    const count = run.output_token_count;
    if (count === undefined) {
      // Old servers cannot anchor partial token streams. Saved output remains authoritative.
      if (
        run.output.length >= buffer.output.length ||
        !buffer.output.startsWith(run.output)
      )
        return run.output;
      return buffer.output;
    }
    if (!Number.isSafeInteger(count) || count < 0) return run.output;
    if (count >= buffer.count) {
      buffer.output = run.output;
      buffer.count = count;
      buffer.checkpoint = true;
      for (const index of buffer.pending.keys())
        if (index < count) buffer.pending.delete(index);
    }
    buffer.checkpoint = true;
    this.drain(buffer);
    return buffer.output;
  }
}
