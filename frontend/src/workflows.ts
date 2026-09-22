import type { Message, Run, State } from "./types";

/** Replay actual tool exchanges, not just the visible pre-tool announcement. */
export function conversationHistory(run: Run, output: string): Message[] {
  return [
    ...run.messages,
    ...(run.reply_messages?.length
      ? run.reply_messages
      : [{ role: "assistant" as const, content: output }]),
  ];
}

/** A fresh origin has no browser workspace. Prefer actual use/library contents,
 * not import order (which can put a tiny test model ahead of the working model). */
export function startupModelId(state: State, savedId: string): string {
  const exists = (id: string | null | undefined) =>
    state.models.some((model) => model.id === id);
  if (exists(savedId)) return savedId;
  if (exists(state.engine.model_id)) return state.engine.model_id!;
  const recent = [...state.runs]
    .sort((a, b) => (b.created_at ?? 0) - (a.created_at ?? 0))
    .find((run) => !run.deleted && exists(run.model_id));
  if (recent) return recent.model_id;
  const archived = new Set(
    (state.vector_visibility ?? [])
      .filter((entry) => entry.archived)
      .map((entry) => entry.id),
  );
  const populated = state.models.find((model) =>
    state.vectors.some(
      (vector) =>
        !vector.deleted &&
        !archived.has(vector.id) &&
        (vector.model_id === model.id ||
          vector.model_fingerprint === model.fingerprint),
    ),
  );
  return populated?.id ?? state.models[0]?.id ?? "";
}

/** A fork shares only its recorded prefix, including runs from a deleted parent.
 * Whole-second timestamps cannot order rapid chat turns; persisted run IDs can. */
export function conversationRuns(
  runs: Run[],
  conversationId: string,
  runIds: string[] = [],
): Run[] {
  const order = new Map(runIds.map((id, index) => [id, index]));
  return runs
    .filter(
      (run) =>
        order.has(run.id) ||
        (!run.deleted &&
          run.conversation_id === conversationId &&
          !run.baseline_of),
    )
    .sort((a, b) => {
      const aIndex = order.get(a.id);
      const bIndex = order.get(b.id);
      if (aIndex !== undefined && bIndex !== undefined) return aIndex - bIndex;
      if (aIndex !== undefined) return -1;
      if (bIndex !== undefined) return 1;
      return (a.created_at ?? 0) - (b.created_at ?? 0);
    });
}
