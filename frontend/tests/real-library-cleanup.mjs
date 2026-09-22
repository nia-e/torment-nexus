// Opt-in verification against the populated local lab. No inference/cloud jobs.
import { chromium, expect } from "@playwright/test";
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "../..");
const log = await readFile(resolve(root, "work/app-e2e.log"), "utf8");
const [, base, token] = log.match(/Open: (http:\/\/127\.0\.0\.1:\d+)\/#token=([a-f0-9]+)/);
const browser = await chromium.launch();
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = [], actions = [];
  page.on("pageerror", (error) => errors.push(String(error)));
  page.on("request", (request) => {
    if (request.url().endsWith("/api/action")) actions.push(request.postDataJSON()?.action);
  });
  await page.goto(`${base}/#token=${token}`);
  const cards = page.locator(".vector-card");
  await expect(cards).toHaveCount(3);
  await expect(page.getByRole("checkbox", { name: "Show archived (4)" })).toBeVisible();
  await page.reload();
  await expect(cards).toHaveCount(3);
  await page.screenshot({ path: resolve(root, "work/library-cleanup/library-clean.png"), fullPage: true });
  await page.getByRole("checkbox", { name: "Show archived (4)" }).check();
  await expect(cards).toHaveCount(7);
  // Card order does not depend on visibility metadata. Restore one known test
  // copy, then archive it again; the user's final library remains clean.
  const index = await cards.evaluateAll((elements) => elements.findIndex((element) =>
    element.querySelector('button[aria-label^="Restore playful dry humor"]')));
  expect(index).toBeGreaterThanOrEqual(0);
  await cards.nth(index).getByRole("button", { name: /^Restore playful dry humor/ }).click();
  await expect(page.getByRole("checkbox", { name: "Show archived (3)" })).toBeVisible();
  await cards.nth(index).getByRole("button", { name: /^Archive playful dry humor/ }).click();
  await expect(page.getByRole("checkbox", { name: "Show archived (4)" })).toBeVisible();
  await page.getByRole("checkbox", { name: "Show archived (4)" }).uncheck();
  await expect(cards).toHaveCount(3);
  await page.reload();
  await expect(cards).toHaveCount(3);
  expect(errors).toEqual([]);
  expect(actions).toEqual(["set_vector_archived", "set_vector_archived"]);
  await writeFile(resolve(root, "work/library-cleanup/browser-report.json"), JSON.stringify({
    status: "passed", actual_backend: true, visible_vectors: 3, archived_vectors: 4,
    restored_and_rearchived: true, reload_persisted: true, page_errors: errors,
    new_inference_or_cloud_jobs: 0,
  }, null, 2));
  console.log("Real library cleanup UI passed: 3 visible, 4 archived, restore/reload, no inference.");
} finally {
  await browser.close();
}
