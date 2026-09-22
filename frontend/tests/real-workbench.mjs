// Opt-in real-lab integration; never included in the ordinary fixture suite.
// Runs local inference and edits a saved repair. Completed cloud stages are reused.
import { chromium, expect } from "@playwright/test";
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "../..");
const reportPath = resolve(root, "work/app-e2e-report.json");
const report = JSON.parse(await readFile(reportPath, "utf8"));
const log = await readFile(resolve(root, "work/app-e2e.log"), "utf8");
const [, base, token] = log.match(/Open: (http:\/\/127\.0\.0\.1:\d+)\/#token=([a-f0-9]+)/);
async function api(path, body) {
  const response = await fetch(base + path, {
    method: body ? "POST" : "GET",
    headers: { Authorization: `Bearer ${token}`, Origin: base, "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  const value = await response.json();
  if (!response.ok) throw new Error(JSON.stringify(value));
  return value;
}
const action = (action, fields = {}) => api("/api/action", { action, ...fields });
async function waitJob(id) {
  let previous;
  for (let attempt = 0; attempt < 1800; attempt++) {
    const state = await api("/api/state");
    const job = state.jobs.find((job) => job.id === id);
    if (job.stage !== previous) console.log(id.slice(0, 8), job.status, job.stage);
    previous = job.stage;
    if (job.status === "completed") return job;
    if (["failed", "interrupted", "cancelled"].includes(job.status)) throw new Error(JSON.stringify(job));
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error("real job timed out");
}

await waitJob((await action("load_model", { model_id: report.model_id })).job_id);
const before = await api("/api/state");
const originalJob = before.jobs.find((job) => job.id === report.concept_jobs[1]);
const originalRecipe = before.recipes.find((recipe) => recipe.id === originalJob.recipe_id);
const browser = await chromium.launch();
try {
  const page = await browser.newPage({ viewport: { width: 1500, height: 1020 } });
  const errors = [];
  page.on("pageerror", (error) => errors.push(String(error)));
  await page.goto(`${base}/#token=${token}`);
  await expect(page).toHaveURL(`${base}/`);
  await expect(page.getByText("Local inference · private session")).toBeVisible();
  await page.screenshot({ path: resolve(root, "work/real-ui-library.png"), fullPage: true });

  if (!report.browser_edited_job_id) {
    await page.getByRole("button", { name: /^Recipes/ }).click();
    await page.getByRole("button", { name: new RegExp(originalRecipe.concept + " v1") }).click();
    await page.getByRole("button", { name: "Edit stages", exact: true }).click();
    const picker = page.getByRole("combobox", { name: "Completed stage" });
    await expect(picker.locator('option[value="repair"]')).toHaveCount(1);
    await picker.selectOption("repair");
    const editor = page.getByRole("textbox", { name: "repair JSON editor" });
    const repaired = JSON.parse(await editor.inputValue());
    // An explicit, small content edit; no new pair, ID, scenario family or cloud writer.
    repaired.pairs[0].positive += " A faint scent of rain hangs in the air.";
    await editor.fill(JSON.stringify(repaired, null, 2));
    await page.screenshot({ path: resolve(root, "work/real-ui-edit.png"), fullPage: true });
    const responsePromise = page.waitForResponse((response) => {
      const body = response.request().postDataJSON();
      return response.url().endsWith("/api/action") && body?.action === "edit_recipe";
    });
    await page.getByRole("button", { name: "Save new version", exact: true }).click();
    const response = await responsePromise;
    expect(response.ok()).toBe(true);
    const result = await response.json();
    report.browser_edited_job_id = result.job_id;
    report.browser_edited_draft_id = result.id;
    await writeFile(reportPath, JSON.stringify(report, null, 2));
  }
  const editedJob = await waitJob(report.browser_edited_job_id);
  const after = await api("/api/state");
  const editedRecipe = after.recipes.find((recipe) => recipe.id === editedJob.recipe_id);
  expect(editedRecipe.dataset_hash).not.toBe(originalRecipe.dataset_hash);
  expect(after.recipes.find((recipe) => recipe.id === originalRecipe.id)).toEqual(originalRecipe);
  for (const stage of ["design", "writer_1", "writer_2", "writer_3", "review"])
    expect(editedJob.details.stages[stage]).toBe(originalJob.details.stages[stage]);
  expect(Object.keys(editedJob.details.stages).filter((stage) => stage.includes(".attempt_")).sort())
    .toEqual(Object.keys(originalJob.details.stages).filter((stage) => stage.includes(".attempt_")).sort());
  expect(after.vectors.some((vector) => vector.id === editedJob.vector_id)).toBe(true);
  report.browser_edited_vector_id = editedJob.vector_id;
  report.checks.real_browser_repair_edit_version_extraction = true;
  report.checks.real_browser_edit_reused_cloud_stages = true;
  await page.reload();
  await expect(page.getByText("Local inference · private session")).toBeVisible();
  await page.getByRole("button", { name: /^Recipes/ }).click();
  await expect(page.getByRole("button", { name: new RegExp(originalRecipe.concept + " v" + editedRecipe.version) })).toBeVisible();
  await page.screenshot({ path: resolve(root, "work/real-ui-reload.png"), fullPage: true });
  expect(errors).toEqual([]);
  report.checks.real_browser_no_page_errors = true;
  await writeFile(reportPath, JSON.stringify(report, null, 2));
  console.log("Real browser edit/re-extraction/reload passed; no new cloud generation attempt.");
} finally {
  await browser.close();
}
