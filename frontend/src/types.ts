export type Json =
  null | boolean | number | string | Json[] | { [key: string]: Json };
export type Message = {
  role: "system" | "user" | "assistant";
  content: string;
};
export type Axis = { vector_id: string; layer: number; percent: number };
export type Sampling = {
  seed: number;
  temperature: number;
  top_p: number;
  max_tokens: number;
  unbounded?: boolean;
};
export type RecordBase = {
  id: string;
  created_at?: number;
  display_name?: string;
  deleted?: boolean;
};
export type Model = RecordBase & {
  name: string;
  path: string;
  fingerprint: string;
  source?: Json;
  size_bytes?: number;
};
export type Layer = {
  layer: number;
  usable?: boolean;
  auc?: number;
  residual_norm: number;
  width: number;
  direction_hash?: string;
  unit_hash?: string;
};
export type Vector = RecordBase & {
  extraction?: { method: "paper" | "completion"; readout_suffix: string };
  preview_mode?: "standard" | "negative_only" | "none";
  name: string;
  recipe_id?: string;
  model_id: string;
  model_fingerprint: string;
  selected_layer: number;
  layers: Layer[];
  warnings?: string[];
  manifest_hash?: string;
  previews?: Json;
};
export type Roles = { designer: string; writers: string[]; reviewer: string };
export type Recipe = RecordBase & {
  extraction?: { method: "paper" | "completion"; readout_suffix: string };
  raw?: boolean;
  concept: string;
  model_id: string;
  parent_id?: string;
  version: number;
  design?: Json;
  dataset?: Json;
  review?: Json;
  stages?: Json;
  roles?: Roles;
  warnings?: string[];
};
export type Job = RecordBase & {
  kind: string;
  status: string;
  stage?: string;
  progress?: number;
  error?: string;
  recipe_id?: string;
  model_id?: string;
  run_id?: string;
  details?: Json;
};
export type ControlEvent = {
  revision: number;
  source?: "user" | "model" | "mcp";
  first_token_index?: number;
  coefficients?: { vector_id: string; layer?: number; percent: number }[];
};
export type MixGeometry = {
  warnings: string[];
  layers: {
    layer: number;
    injected_norm: number;
    axis_count: number;
    percent_of_shared_calibration?: number | null;
    remaining_fraction_after_summation?: number | null;
  }[];
  cosines: {
    left_vector_id: string;
    right_vector_id: string;
    layer: number;
    cosine: number;
  }[];
  meaning?: string;
};
export type AppliedMixture = {
  revision: number;
  first_token_index: number;
  geometry: MixGeometry;
};
export type Run = RecordBase & {
  self_modification?: boolean;
  self_tools_available?: boolean;
  tool_calls?: {
    id: string;
    input: string;
    result: Json;
    after_token_index: number;
  }[];
  context_rollovers?: { first_token_index: number; dropped_tokens: number }[];
  model_id: string;
  messages: Message[];
  reply_messages?: Message[];
  axes: Axis[];
  sampling: Sampling;
  output: string;
  status: string;
  output_token_count?: number;
  mix_diagnostics?: AppliedMixture;
  raw?: boolean;
  requested_controls?: ControlEvent[];
  applied_controls?: ControlEvent[];
  conversation_id?: string;
  baseline_of?: string;
  manifest_hash?: string;
  error?: string;
};
export type Preset = RecordBase & {
  name: string;
  model_id: string;
  axes: Axis[];
};
export type Conversation = RecordBase & {
  title: string;
  messages?: Message[];
  run_ids?: string[];
  forked_from_run_id?: string;
  forked_from_conversation_id?: string;
};
export type CodexModel = {
  id: string;
  displayName?: string;
  display_name?: string;
  isDefault?: boolean;
  is_default?: boolean;
};
export type State = {
  mcp?: { url: string; token: string };
  models: Model[];
  recipes: Recipe[];
  vectors: Vector[];
  vector_visibility?: { id: string; archived: boolean }[];
  jobs: Job[];
  runs: Run[];
  presets: Preset[];
  conversations: Conversation[];
  engine: {
    status: string;
    model_id: string | null;
    capabilities?: Json;
    memory_bytes?: number;
    memory_kind?: "peak_worker_rss";
    error?: string;
  };
  codex: { models: CodexModel[]; error?: string | null; assignments?: Roles };
};
export type ServerEvent = {
  kind: string;
  run_id?: string;
  job_id?: string;
  data: Record<string, unknown>;
};
export type HfListing = {
  repo: string;
  revision: string;
  files: { name: string; size_bytes?: number }[];
};
export const emptyState: State = {
  models: [],
  recipes: [],
  vectors: [],
  jobs: [],
  runs: [],
  presets: [],
  conversations: [],
  engine: { status: "unloaded", model_id: null },
  codex: { models: [] },
};
export const running = (status: string) =>
  ["queued", "running", "loading", "generating", "extracting"].includes(status);
export function recordTime(a: RecordBase, b: RecordBase) {
  return (b.created_at ?? 0) - (a.created_at ?? 0);
}
