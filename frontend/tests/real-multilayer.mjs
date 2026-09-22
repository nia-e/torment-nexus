// Opt-in real packaged-app check. Existing humor vector only; no cloud concepts.
import { chromium, expect } from "@playwright/test";
import { readFile, writeFile, access } from "node:fs/promises";
import { resolve } from "node:path";
const root = resolve(import.meta.dirname, "../..");
const reportPath = resolve(root, "work/multilayer-real-report.json");
try {
  await access(reportPath);
  throw new Error(
    "Report already exists: retain evidence, do not silently repeat inference.",
  );
} catch (error) {
  if (error.code !== "ENOENT") throw error;
}
const log = await readFile(resolve(root, "work/app-e2e.log"), "utf8");
const [, base, token] = log.match(
  /Open: (http:\/\/127\.0\.0\.1:\d+)\/#token=([a-f0-9]+)/,
);
async function api(path, body) {
  const response = await fetch(base + path, {
    method: body ? "POST" : "GET",
    headers: {
      Authorization: `Bearer ${token}`,
      Origin: base,
      "Content-Type": "application/json",
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  const value = await response.json();
  if (!response.ok) throw new Error(JSON.stringify(value));
  return value;
}
const before = await api("/api/state");
expect(before.jobs.some((j) => ["queued", "running"].includes(j.status))).toBe(
  false,
);
const vector = before.vectors.find(
  (v) => v.id === "5ead7e58-27ef-449c-a35e-e85adb171a17",
);
expect(vector.name.toLowerCase()).not.toContain("pain");
expect(vector.layers.map((l) => l.layer)).toEqual([16, 25, 35, 44, 54]);
const report = {
  vector_id: vector.id,
  model_id: vector.model_id,
  new_cloud_concepts: 0,
  checks: {},
  status: "in_progress",
};
const save = () => writeFile(reportPath, JSON.stringify(report, null, 2));
const browser = await chromium.launch();
let runId;
try {
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1000 },
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(`${base}/#token=${token}`);
  await expect(page.locator(".vector-card")).toHaveCount(3);
  await page
    .getByRole("button", { name: `Add ${vector.name} to mix`, exact: true })
    .click();
  const coefficient = (layer) =>
    page.getByRole("spinbutton", {
      name: `Coefficient for ${vector.name} at layer ${layer}`,
      exact: true,
    });
  await expect(page.locator(".axis-control .axis-slider")).toHaveCount(5);
  for (const layer of vector.layers)
    await expect(coefficient(layer.layer)).toHaveValue("0");
  await coefficient(16).fill("-1");
  await coefficient(25).fill("1");
  await page
    .getByRole("button", { name: "Sampling settings", exact: true })
    .click();
  await page
    .getByRole("spinbutton", { name: "Max tokens", exact: true })
    .fill("192");
  await page
    .getByRole("button", { name: "Sampling settings", exact: true })
    .click();
  await page
    .getByRole("textbox", { name: "Prompt", exact: true })
    .fill(
      "Describe a fictional library in five short paragraphs, focusing on everyday objects and layout.",
    );
  const starting = page.waitForResponse(
    (r) =>
      r.url().endsWith("/api/action") &&
      r.request().postDataJSON()?.action === "generate",
  );
  await page.getByRole("button", { name: "Run prompt", exact: true }).click();
  const result = await (await starting).json();
  runId = result.run_id;
  report.run_id = runId;
  report.job_id = result.job_id;
  await save();
  await expect
    .poll(
      async () =>
        (await api("/api/state")).runs.find((r) => r.id === runId)?.status,
    )
    .toBe("running");
  await coefficient(25).fill("-2");
  await expect(page.getByText(/Rev 1 · from token/)).toBeVisible();
  await expect(coefficient(16)).toHaveValue("-1");
  await page
    .getByRole("button", {
      name: `Zero ${vector.name} at layer 16`,
      exact: true,
    })
    .click();
  await expect(page.getByText(/Rev 2 · from token/)).toBeVisible();
  await expect(coefficient(25)).toHaveValue("-2");
  let state = await api("/api/state");
  let run = state.runs.find((r) => r.id === runId);
  expect(run.axes.length).toBe(5);
  const latest = run.requested_controls.at(-1);
  expect(latest.coefficients).toEqual(
    vector.layers.map((l) => ({
      vector_id: vector.id,
      layer: l.layer,
      percent: l.layer === 25 ? -2 : 0,
    })),
  );
  const geometry = run.mix_diagnostics.geometry.layers;
  expect(geometry.find((l) => l.layer === 16).injected_norm).toBe(0);
  const expectedNorm =
    vector.layers.find((l) => l.layer === 25).residual_norm * 0.02;
  expect(
    Math.abs(geometry.find((l) => l.layer === 25).injected_norm - expectedNorm),
  ).toBeLessThan(1e-5);
  expect(
    geometry.filter((l) => l.layer !== 25).every((l) => l.injected_norm === 0),
  ).toBe(true);
  report.checks.independent_live_layers_and_complete_clearing = true;
  report.applied_controls = run.applied_controls;
  if (run.status === "running")
    await page.getByRole("button", { name: "Stop", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Stop", exact: true }),
  ).toBeHidden();
  await page.reload();
  await expect(coefficient(16)).toHaveValue("0");
  await expect(coefficient(25)).toHaveValue("-2");
  report.checks.reload_retains_each_layer = true;
  await page.screenshot({
    path: resolve(root, "work/multilayer-real-ui.png"),
    fullPage: true,
  });
  const width = (await page.locator(".experiment-panel").boundingBox()).width;
  expect(width).toBeGreaterThan(1440 * 0.6);
  report.chat_width = width;
  await page.getByRole("button", { name: "Focus chat", exact: true }).click();
  await expect(page.locator(".mixer-panel")).toBeHidden();
  await page.screenshot({
    path: resolve(root, "work/multilayer-real-focus.png"),
    fullPage: true,
  });
  await page.getByRole("button", { name: "Show panels", exact: true }).click();
  await expect(coefficient(25)).toHaveValue("-2");
  // Leave this private browser workspace neutral; do not modify any saved preset.
  await page.getByRole("button", { name: "Zero all", exact: true }).click();
  state = await api("/api/state");
  run = state.runs.find((r) => r.id === runId);
  expect(state.vectors).toEqual(before.vectors);
  expect(state.recipes).toEqual(before.recipes);
  expect(state.presets).toEqual(before.presets);
  expect(state.vector_visibility).toEqual(before.vector_visibility);
  expect(errors).toEqual([]);
  report.checks.focus_preserves_mix = true;
  report.checks.library_artifacts_and_presets_unchanged = true;
  report.page_errors = errors;
  report.run_status = run.status;
  report.output_tokens = run.output_token_count;
  report.status = "passed";
  await save();
  console.log(JSON.stringify(report, null, 2));
} finally {
  if (runId) {
    const run = (await api("/api/state")).runs.find((r) => r.id === runId);
    if (["queued", "running"].includes(run.status))
      await api("/api/action", { action: "cancel_run", run_id: runId });
  }
  await browser.close();
}
