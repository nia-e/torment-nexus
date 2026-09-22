import type { Json, Job, Recipe } from "./types";

export const stageNames = [
  "design",
  "writer_1",
  "writer_2",
  "writer_3",
  "review",
  "repair",
  "dataset",
] as const;
export type StageName = (typeof stageNames)[number];
export const stageLabels: Record<StageName, string> = {
  design: "Design / interpretation",
  writer_1: "Writer 1 / matched pairs",
  writer_2: "Writer 2 / matched pairs",
  writer_3: "Writer 3 / matched pairs",
  review: "Review / accepted revisions",
  repair: "Targeted repairs",
  dataset: "Final matched-pair dataset",
};
export function jsonObject(
  value: Json | undefined,
): { [key: string]: Json } | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value
    : undefined;
}
export function checkpointRefs(
  stages: Json | undefined,
): Partial<Record<StageName, string>> {
  const value = jsonObject(stages);
  const refs: Partial<Record<StageName, string>> = {};
  for (const stage of stageNames) {
    const hash =
      value?.[stage] ??
      (stage === "dataset" ? value?.dataset_input : undefined);
    if (typeof hash === "string" && hash) refs[stage] = hash;
  }
  return refs;
}
export function jobCheckpoints(job: Job): Json | undefined {
  return jsonObject(job.details)?.stages;
}
export function completedOutput(artifact: Json | undefined): Json | undefined {
  const value = jsonObject(artifact);
  if (
    !value ||
    (!["completed", "edited"].includes(String(value.status)) &&
      value.edited !== true)
  )
    return undefined;
  return Object.hasOwn(value, "output") ? value.output : undefined;
}
export function directRecipeStages(
  recipe?: Recipe,
): Partial<Record<StageName, Json>> {
  if (!recipe) return {};
  const values: Partial<Record<StageName, Json>> = {};
  for (const stage of ["design", "review", "dataset"] as const) {
    const value = recipe[stage];
    if (
      value !== undefined &&
      value !== null &&
      (!Array.isArray(value) || value.length > 0)
    )
      values[stage] = value;
  }
  return values;
}
