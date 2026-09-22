// Isolated browser-contract fixture. No inference, cloud requests, or production data.
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { resolve, extname } from "node:path";
import { fileURLToPath } from "node:url";
const dist = resolve(fileURLToPath(new URL("../dist/", import.meta.url)));
const clients = new Set();
const timers = new Set();
let state;
let artifacts = {};
let counter = 0;
const id = (prefix) => `${prefix}-${++counter}`;
const now = () => Date.now() / 1000;
function reset() {
  for (const timer of timers) clearInterval(timer);
  timers.clear();
  const layer = { layer: 4, auc: 0.91, residual_norm: 23.4, width: 16 };
  state = {
    models: [
      {
        id: "model-fixture",
        name: "Bonsai UI fixture",
        path: "/fixture/model.gguf",
        fingerprint: "fixture-fingerprint",
        size_bytes: 7200000000,
        created_at: now(),
      },
    ],
    vectors: ["Dry wit", "Measured certainty"].map((name, i) => ({
      id: `vector-${i}`,
      name,
      model_id: "model-fixture",
      model_fingerprint: "fixture-fingerprint",
      selected_layer: 4,
      layers: [layer, { ...layer, layer: 9, auc: 0.82 }],
      warnings: [],
      previews: [],
      manifest_hash: "fixture-manifest",
      created_at: now(),
    })),
    vector_visibility: [],
    recipes: [
      {
        id: "recipe-fixture",
        concept: "Dry wit",
        version: 1,
        model_id: "model-fixture",
        design: { interpretation: "Dry wit versus literal prose." },
        dataset: {
          pairs: [
            {
              id: "pair-1",
              family: "ordinary-things",
              messages: [{ role: "user", content: "Describe a chair." }],
              positive: "A committee for one, with excellent attendance.",
              negative: "A chair supports a seated person.",
              split: "train",
            },
          ],
        },
        review: { warnings: [] },
        stages: {
          design: "completed",
          dataset: "completed",
          review: "completed",
        },
        roles: {
          designer: "model-a",
          writers: ["model-a", "model-b", "model-c"],
          reviewer: "model-a",
        },
        warnings: [],
        created_at: now(),
      },
    ],
    jobs: [],
    runs: [],
    presets: [],
    conversations: [],
    engine: { status: "idle", model_id: "model-fixture", capabilities: {} },
    codex: {
      models: [
        { id: "model-a", displayName: "Model A", isDefault: true },
        { id: "model-b", displayName: "Model B" },
        { id: "model-c", displayName: "Model C" },
      ],
      error: null,
    },
  };
  artifacts = {
    "checkpoint-design": {
      status: "completed",
      output: state.recipes[0].design,
    },
    "checkpoint-writer-1": {
      status: "completed",
      output: { pairs: structuredClone(state.recipes[0].dataset.pairs) },
    },
    "checkpoint-writer-2": { status: "running", output: { pairs: [] } },
    "checkpoint-review": {
      status: "completed",
      output: state.recipes[0].review,
    },
    "checkpoint-dataset": {
      status: "completed",
      output: state.recipes[0].dataset,
    },
  };
  state.recipes[0].stages = {
    design: "checkpoint-design",
    writer_1: "checkpoint-writer-1",
    review: "checkpoint-review",
    dataset: "checkpoint-dataset",
  };
  state.jobs.push({
    id: "factory-in-flight",
    kind: "factory",
    status: "running",
    stage: "writer_2",
    created_at: now(),
    details: {
      stages: {
        design: "checkpoint-design",
        writer_1: "checkpoint-writer-1",
        writer_2: "checkpoint-writer-2",
      },
    },
  });
}
reset();
function emit(kind, data = {}, extra = {}) {
  for (const client of clients)
    client.write(`data: ${JSON.stringify({ kind, data, ...extra })}\n\n`);
}
function fixtureGeometry(run) {
  const coefficients = run.requested_controls.at(-1).coefficients;
  const axes = run.axes.map((axis) => ({
    ...axis,
    percent:
      coefficients.find(
        (c) =>
          c.vector_id === axis.vector_id &&
          (c.layer === axis.layer || c.layer === undefined),
      )?.percent ?? 0,
  }));
  const layers = [...new Set(axes.map((axis) => axis.layer))].map((layer) => {
    const contributions = axes
      .filter((axis) => axis.layer === layer)
      .map((axis) => (axis.percent / 100) * 23.4);
    let squaredNorm = contributions.reduce((sum, value) => sum + value ** 2, 0);
    for (let i = 0; i < contributions.length; i++)
      for (let j = i + 1; j < contributions.length; j++)
        squaredNorm += 2 * 0.35 * contributions[i] * contributions[j];
    const injected_norm = Math.sqrt(Math.max(0, squaredNorm));
    return {
      layer,
      axis_count: contributions.length,
      injected_norm,
      percent_of_shared_calibration: (injected_norm / 23.4) * 100,
    };
  });
  // Engine buffers also carry unselected zero rows; these must not swamp the UI.
  layers.unshift({ layer: 1, axis_count: 0, injected_norm: 0 });
  const cosines = [];
  for (let i = 0; i < axes.length; i++)
    for (let j = i + 1; j < axes.length; j++)
      if (axes[i].layer === axes[j].layer)
        cosines.push({
          left_vector_id: axes[i].vector_id,
          right_vector_id: axes[j].vector_id,
          layer: axes[i].layer,
          cosine: 0.35,
        });
  return {
    warnings: axes.some((axis) => Math.abs(axis.percent) > 20)
      ? ["Fixture advisory: large perturbation; generation remains allowed."]
      : [],
    layers,
    cosines,
  };
}
function finishConversation(run) {
  const chat = state.conversations.find(
    (chat) => chat.id === run.conversation_id,
  );
  if (!chat || !run.output) return;
  chat.messages = [
    ...run.messages,
    ...(run.reply_messages?.length
      ? run.reply_messages
      : [{ role: "assistant", content: run.output }]),
  ];
  if (!chat.run_ids.includes(run.id)) chat.run_ids.push(run.id);
}
function generation(body) {
  const runId = id("run");
  const jobId = id("job");
  const run = {
    ...body,
    id: runId,
    created_at: now(),
    status: "running",
    output: "",
    self_tools_available: !!body.self_modification,
    output_token_count: 0,
    requested_controls: [
      {
        revision: 0,
        coefficients: body.axes.map((axis) => ({
          vector_id: axis.vector_id,
          layer: axis.layer,
          percent: axis.percent,
        })),
      },
    ],
    applied_controls: [{ revision: 0, first_token_index: 0 }],
  };
  run.mix_diagnostics = {
    revision: 0,
    first_token_index: 0,
    geometry: fixtureGeometry(run),
  };
  const job = {
    id: jobId,
    run_id: runId,
    kind: "generation",
    status: "running",
    progress: 0,
    created_at: now(),
  };
  state.runs.push(run);
  state.jobs.push(job);
  state.engine.status = "running";
  emit("state");
  const tokens = (
    body.baseline_of
      ? "This is the unsteered comparison, with the same exact input. "
      : "A quiet room remembers every conversation, but has the decency not to repeat them. "
  )
    .repeat(5)
    .split(/(?<= )/);
  let revision = 0;
  const timer = setInterval(() => {
    if (run.status !== "running") {
      clearInterval(timer);
      timers.delete(timer);
      return;
    }
    const latest = run.requested_controls.at(-1).revision;
    if (latest !== revision) {
      revision = latest;
      const applied = { revision, first_token_index: run.output_token_count };
      run.applied_controls.push(applied);
      run.mix_diagnostics = { ...applied, geometry: fixtureGeometry(run) };
      emit("applied", applied, { run_id: runId });
    }
    const text = tokens[run.output_token_count];
    if (
      text === undefined ||
      (!body.sampling.unbounded &&
        run.output_token_count >= body.sampling.max_tokens)
    ) {
      run.status = "completed";
      job.status = "completed";
      state.engine.status = "idle";
      finishConversation(run);
      clearInterval(timer);
      timers.delete(timer);
      emit("completed", {}, { run_id: runId, job_id: jobId });
      return;
    }
    const index = run.output_token_count++;
    run.output += text;
    emit("token", { text, index, revision }, { run_id: runId });
  }, 55);
  timers.add(timer);
  return { run_id: runId, job_id: jobId };
}
function action(body) {
  switch (body.action) {
    case "set_self_modification": {
      const run = state.runs.find((run) => run.id === body.run_id);
      run.self_modification = body.enabled;
      emit("state");
      return { enabled: body.enabled };
    }
    case "fixture_model_controls": {
      const run = state.runs.find((run) => run.id === body.run_id);
      const controls = {
        revision: run.requested_controls.at(-1).revision + 1,
        coefficients: body.coefficients,
        source: "model",
      };
      run.requested_controls.push(controls);
      emit("controls_requested", controls, { run_id: run.id });
      return controls;
    }
    case "generate":
      return generation(body);
    case "controls": {
      const run = state.runs.find((run) => run.id === body.run_id);
      if (!run || run.status !== "running")
        throw new Error("Run is no longer active");
      run.requested_controls.push(body);
      return {};
    }
    case "cancel_run": {
      const run = state.runs.find((run) => run.id === body.run_id);
      run.status = "cancelled";
      finishConversation(run);
      state.jobs.find((job) => job.run_id === run.id).status = "cancelled";
      state.engine.status = "idle";
      emit("completed", {}, { run_id: run.id });
      return {};
    }
    case "rename_record":
    case "delete_record": {
      const record = state[body.kind].find((record) => record.id === body.id);
      if (!record) throw new Error("Record not found");
      if (body.action === "delete_record") {
        record.deleted = true;
        if (body.kind === "conversations")
          state.runs
            .filter((run) => run.conversation_id === body.id)
            .forEach((run) => (run.deleted = true));
      } else {
        record.display_name = body.name;
        if (body.kind === "vectors" || body.kind === "presets")
          record.name = body.name;
        if (body.kind === "conversations") record.title = body.name;
      }
      emit("state");
      return {};
    }
    case "new_conversation": {
      const record = {
        id: id("conversation"),
        title: body.title ?? "Untitled experiment",
        created_at: now(),
        messages: [],
        run_ids: [],
      };
      if (body.from_run_id) {
        const source = state.runs.find((run) => run.id === body.from_run_id);
        if (!source || ["queued", "running"].includes(source.status))
          throw new Error("Cannot fork this run");
        const parent = state.conversations.find(
          (chat) => chat.id === source.conversation_id,
        );
        const index = parent?.run_ids.indexOf(source.id) ?? -1;
        record.run_ids = [
          ...(index >= 0 ? parent.run_ids.slice(0, index) : []),
          source.id,
        ];
        record.messages = [
          ...source.messages,
          ...(source.reply_messages?.length
            ? source.reply_messages
            : [{ role: "assistant", content: source.output }]),
        ];
        record.forked_from_run_id = source.id;
      }
      state.conversations.push(record);
      return record;
    }
    case "save_mix": {
      const record = { ...body, id: id("mix"), created_at: now() };
      state.presets.push(record);
      return { id: record.id };
    }
    case "set_vector_archived": {
      if (!state.vectors.some((vector) => vector.id === body.vector_id))
        throw new Error("Vector not found");
      state.vector_visibility = state.vector_visibility.filter(
        (entry) => entry.id !== body.vector_id,
      );
      state.vector_visibility.push({
        id: body.vector_id,
        archived: body.archived,
      });
      emit("state");
      return {};
    }
    case "discover_codex":
      return state.codex;
    case "create_concept": {
      const job = {
        id: id("job"),
        kind: "factory",
        status: "running",
        stage: "design",
        progress: 0.1,
        created_at: now(),
      };
      state.jobs.push(job);
      const recipe = {
        ...state.recipes[0],
        id: id("recipe"),
        concept: body.concept,
        version: 1,
        roles: body.roles,
        created_at: now(),
      };
      state.recipes.push(recipe);
      job.recipe_id = recipe.id;
      const timer = setTimeout(() => {
        job.status = "completed";
        job.stage = "published";
        job.progress = 1;
        state.vectors.push({
          ...state.vectors[0],
          id: id("vector"),
          name: body.concept,
          recipe_id: recipe.id,
        });
        emit("completed", {}, { job_id: job.id });
      }, 450);
      timers.add(timer);
      emit("state");
      return { recipe_id: recipe.id, job_id: job.id };
    }
    case "edit_recipe": {
      const previous = state.recipes.find(
        (recipe) => recipe.id === body.recipe_id,
      );
      const next = {
        ...previous,
        id: id("recipe"),
        parent_id: previous.id,
        version: previous.version + 1,
        [body.stage]: body.value,
        created_at: now(),
      };
      const hash = id("checkpoint");
      artifacts[hash] = { edited: true, output: body.value };
      next.stages = { ...previous.stages, [body.stage]: hash };
      state.recipes.push(next);
      emit("state");
      return { id: next.id };
    }
    case "edit_job_stage": {
      const original = state.jobs.find((job) => job.id === body.job_id);
      const hash = id("checkpoint");
      artifacts[hash] = { edited: true, output: body.value };
      const recipe = {
        ...state.recipes[0],
        id: id("recipe"),
        concept: "Forked writer checkpoint",
        version: 1,
        source_job_id: original.id,
        stages: { ...original.details.stages, [body.stage]: hash },
        created_at: now(),
      };
      state.recipes.push(recipe);
      const job = {
        id: id("job"),
        kind: "factory",
        status: "queued",
        recipe_id: recipe.id,
        created_at: now(),
        details: { stages: recipe.stages },
      };
      state.jobs.push(job);
      emit("state");
      return { id: recipe.id, job_id: job.id };
    }
    case "artifact":
      return (
        artifacts[body.hash] ?? { algorithm: "fixture-only", hash: body.hash }
      );
    case "export_vector":
      return {
        manifest: state.vectors.find((vector) => vector.id === body.vector_id),
        tensors: [],
      };
    case "import_vector": {
      if (!body.bundle.manifest) throw new Error("Invalid vector bundle");
      const record = { ...body.bundle.manifest, id: id("vector") };
      state.vectors.push(record);
      return { id: record.id };
    }
    case "browse_hf":
      return {
        repo: body.repo,
        revision: "immutable-fixture-revision",
        files: [{ name: "fixture.gguf", size_bytes: 512 }],
      };
    case "download_model": {
      const job = {
        id: id("job"),
        kind: "download",
        status: "interrupted",
        stage: "download",
        error: "Fixture interruption: resumable partial file retained",
        created_at: now(),
      };
      state.jobs.push(job);
      return { job_id: job.id };
    }
    case "retry_job": {
      const job = state.jobs.find((job) => job.id === body.job_id);
      job.status = "completed";
      job.error = null;
      emit("state");
      return { job_id: job.id };
    }
    default:
      throw new Error(`Unsupported fixture action: ${body.action}`);
  }
}
createServer(async (req, res) => {
  if (req.url === "/__reset") {
    reset();
    res.end("reset");
    return;
  }
  if (req.url.startsWith("/api/")) {
    if (
      !["Bearer browser-fixture", "Bearer browser-fixture-rotated"].includes(
        req.headers.authorization,
      )
    ) {
      res.writeHead(401, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Launch token required" }));
      return;
    }
    if (req.url === "/api/events") {
      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-store",
        Connection: "keep-alive",
      });
      res.write(": ready\n\n");
      clients.add(res);
      req.on("close", () => clients.delete(res));
      return;
    }
    res.setHeader("Content-Type", "application/json");
    if (req.url === "/api/state") {
      res.end(JSON.stringify(state));
      return;
    }
    if (req.url === "/api/action") {
      try {
        let input = "";
        for await (const chunk of req) input += chunk;
        const result = action(JSON.parse(input));
        res.end(JSON.stringify(result));
      } catch (error) {
        res.writeHead(400);
        res.end(JSON.stringify({ error: error.message }));
      }
      return;
    }
  }
  try {
    const path = resolve(
      dist,
      `.${req.url === "/" ? "/index.html" : req.url.split("?")[0]}`,
    );
    if (!path.startsWith(`${dist}/`)) throw new Error("Invalid path");
    const body = await readFile(path);
    res.setHeader(
      "Content-Type",
      {
        ".html": "text/html",
        ".js": "text/javascript",
        ".css": "text/css",
        ".svg": "image/svg+xml",
      }[extname(path)] ?? "application/octet-stream",
    );
    res.end(body);
  } catch {
    res.writeHead(404);
    res.end("Not found");
  }
}).listen(47842, "127.0.0.1");
