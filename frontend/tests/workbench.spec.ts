import { expect, test, type Page } from "@playwright/test";

test.beforeEach(async ({ page, request }) => {
  await request.get("/__reset");
  await page.goto("/#token=browser-fixture");
  await expect(
    page.getByText("Local inference · private session"),
  ).toBeVisible();
});

async function addDirections(page: Page) {
  await page
    .getByRole("button", { name: "Add Dry wit to mix", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Add Measured certainty to mix", exact: true })
    .click();
}
async function startPrompt(page: Page) {
  await page
    .getByRole("textbox", { name: "Prompt", exact: true })
    .fill("Describe a quiet room just before sunrise.");
  await page.getByRole("button", { name: "Run prompt", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Stop", exact: true }),
  ).toBeVisible();
}

test("library management and independent panels survive reload", async ({
  page,
}) => {
  await addDirections(page);
  const savedMix = await page.evaluate(async () => {
    const response = await fetch("/api/action", {
      method: "POST",
      headers: {
        Authorization: "Bearer browser-fixture",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        action: "save_mix",
        name: "Test mix",
        model_id: "model-fixture",
        axes: [],
      }),
    });
    return response.json();
  });
  await page.reload();
  await page
    .getByRole("combobox", { name: "Load saved mix" })
    .selectOption(savedMix.id);
  await page.getByRole("button", { name: "Chat", exact: true }).click();
  await page.getByRole("button", { name: "New chat", exact: true }).click();
  const rename = async (kind: string, previous: string, next: string) => {
    await page
      .getByRole("button", { name: `Rename ${kind} ${previous}`, exact: true })
      .click();
    await page
      .getByRole("textbox", { name: "New name", exact: true })
      .fill(next);
    await page.getByRole("button", { name: "Save name", exact: true }).click();
    await expect(page.getByRole("dialog")).toBeHidden();
    await expect(
      page.getByRole("button", { name: `Delete ${kind} ${next}`, exact: true }),
    ).toBeVisible();
  };
  const remove = async (kind: string, name: string) => {
    const button = page.getByRole("button", {
      name: `Delete ${kind} ${name}`,
      exact: true,
    });
    await button.click();
    await page
      .getByRole("dialog")
      .getByRole("button", { name: "Delete", exact: true })
      .click();
    await expect(page.getByRole("dialog")).toBeHidden();
    await expect(button).toHaveCount(0);
  };
  await rename("vector", "Dry wit", "Renamed wit");
  await rename("saved mix", "Test mix", "Renamed mix");
  const conversationTitle = await page
    .getByRole("combobox", { name: "Conversation", exact: true })
    .locator("option:checked")
    .innerText();
  await rename("conversation", conversationTitle, "Renamed conversation");
  await page.getByRole("button", { name: "Recipes", exact: false }).click();
  await rename("recipe", "Dry wit", "Renamed recipe");
  await page.reload();
  await expect(
    page.getByRole("combobox", { name: "Conversation", exact: true }),
  ).toContainText("Renamed conversation");
  await remove("vector", "Renamed wit");
  await page.getByRole("button", { name: "Recipes", exact: false }).click();
  await remove("recipe", "Renamed recipe");
  await page
    .getByRole("combobox", { name: "Load saved mix" })
    .selectOption(savedMix.id);
  await remove("saved mix", "Renamed mix");
  await remove("conversation", "Renamed conversation");
  await page
    .getByRole("button", { name: "Toggle library", exact: true })
    .click();
  await expect(page.locator(".library-panel")).toBeHidden();
  await expect(page.locator(".mixer-panel")).toBeVisible();
  await page.reload();
  await expect(page.locator(".library-panel")).toBeHidden();
  await expect(page.locator(".mixer-panel")).toBeVisible();
  await page.getByRole("button", { name: "Toggle mixer", exact: true }).click();
  await expect(page.locator(".mixer-panel")).toBeHidden();
  await page
    .getByRole("button", { name: "Toggle library", exact: true })
    .click();
  await expect(page.locator(".library-panel")).toBeVisible();
  await expect(page.locator(".mixer-panel")).toBeHidden();
  await page.screenshot({ path: "../work/manage-sidebar.png" });
});

test("self-adjustment is opt-in, mirrors model edits, and can be revoked live", async ({
  page,
}) => {
  const actions: Record<string, any>[] = [];
  page.on("request", (r) => {
    if (r.url().endsWith("/api/action")) actions.push(r.postDataJSON());
  });
  await addDirections(page);
  const self = page.getByRole("checkbox", {
    name: "Allow self-adjustment",
    exact: true,
  });
  await expect(self).not.toBeChecked();
  await self.check();
  await page
    .getByRole("checkbox", { name: "Unbounded output", exact: true })
    .check();
  await startPrompt(page);
  expect(actions.find((a) => a.action === "generate")).toMatchObject({
    self_modification: true,
    sampling: { unbounded: true },
  });
  await page.evaluate(async () => {
    const headers = {
      Authorization: "Bearer browser-fixture",
      "Content-Type": "application/json",
    };
    const s = await (await fetch("/api/state", { headers })).json();
    const run = s.runs.find((r: any) => r.status === "running");
    await fetch("/api/action", {
      method: "POST",
      headers,
      body: JSON.stringify({
        action: "fixture_model_controls",
        run_id: run.id,
        coefficients: run.axes.map((axis: any, i: number) => ({
          ...axis,
          percent: i === 0 ? -2 : 0,
        })),
      }),
    });
  });
  const slider = page.getByRole("spinbutton", {
    name: "Coefficient for Dry wit at layer 4",
    exact: true,
  });
  await expect(slider).toHaveValue("-2");
  expect(actions.filter((a) => a.action === "controls")).toHaveLength(0);
  await slider.fill("3");
  await expect
    .poll(() => actions.filter((a) => a.action === "controls").length)
    .toBe(1);
  expect(actions.find((a) => a.action === "controls")).toMatchObject({
    revision: 2,
    base_revision: 1,
  });
  await self.uncheck();
  await expect(self).not.toBeChecked();
  await page.getByRole("button", { name: "Stop", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Stop", exact: true }),
  ).toBeHidden();
  await page.reload();
  await expect(self).not.toBeChecked();
  await expect(
    page.getByRole("checkbox", { name: "Unbounded output", exact: true }),
  ).toBeChecked();
});

test("cold start selects the populated model rather than the first imported model", async ({
  page,
}) => {
  await page.route("**/api/state", async (route) => {
    const response = await route.fetch();
    const state = await response.json();
    state.models.unshift({
      id: "tiny-startup",
      name: "Tiny engine test",
      path: "/fixture/tiny.gguf",
      fingerprint: "tiny",
      created_at: Date.now(),
    });
    state.engine = { status: "unloaded", model_id: null };
    state.runs = [];
    await route.fulfill({ response, json: state });
  });
  await page.evaluate(() => localStorage.removeItem("torment-nexus.workspace"));
  await page.reload();
  await expect(page.locator(".model-selector")).toContainText(
    "Bonsai UI fixture",
  );
  await expect(page.locator(".vector-card")).toHaveCount(2);
  // Deliberate switching remains possible, but no longer resembles missing data.
  await page
    .getByRole("button", { name: "Manage models", exact: true })
    .click();
  const card = page
    .locator(".model-card")
    .filter({ hasText: "Tiny engine test" });
  await card.getByRole("button", { name: "Select model", exact: true }).click();
  await page.getByRole("button", { name: "Close dialog", exact: true }).click();
  await expect(
    page.getByText("No vectors for this model", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Choose another model", exact: true }),
  ).toBeVisible();
});

test("each concept exposes independent layer sliders in saved and live mixes", async ({
  page,
}) => {
  const actions: Record<string, any>[] = [];
  page.on("request", (r) => {
    if (r.url().endsWith("/api/action")) actions.push(r.postDataJSON());
  });
  await page
    .getByRole("button", { name: "Add Dry wit to mix", exact: true })
    .click();
  const layer4 = page.getByRole("spinbutton", {
    name: "Coefficient for Dry wit at layer 4",
    exact: true,
  });
  const layer9 = page.getByRole("spinbutton", {
    name: "Coefficient for Dry wit at layer 9",
    exact: true,
  });
  await expect(page.locator(".axis-control")).toHaveCount(1);
  await expect(page.locator(".axis-control .axis-slider")).toHaveCount(2);
  await expect(layer4).toHaveValue("0");
  await expect(layer9).toHaveValue("0");
  await layer4.fill("6");
  await layer9.fill("-9");
  await page.getByRole("button", { name: "Save mix", exact: true }).click();
  await page
    .getByRole("textbox", { name: "Mix name", exact: true })
    .fill("Two layers of wit");
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Save mix", exact: true })
    .click();
  await expect(page.getByRole("dialog")).toBeHidden();
  expect(actions.find((a) => a.action === "save_mix")?.axes).toEqual([
    { vector_id: "vector-0", layer: 4, percent: 6 },
    { vector_id: "vector-0", layer: 9, percent: -9 },
  ]);
  await startPrompt(page);
  await layer9.fill("-3");
  await expect(page.getByText(/Rev 1 · from token/)).toBeVisible();
  expect(
    actions.filter((a) => a.action === "controls").at(-1)?.coefficients,
  ).toEqual([
    { vector_id: "vector-0", layer: 4, percent: 6 },
    { vector_id: "vector-0", layer: 9, percent: -3 },
  ]);
  await page
    .getByRole("button", { name: "Zero Dry wit at layer 4", exact: true })
    .click();
  await expect(page.getByText(/Rev 2 · from token/)).toBeVisible();
  await expect(layer9).toHaveValue("-3");
  await page.reload();
  await expect(layer4).toHaveValue("0");
  await expect(layer9).toHaveValue("-3");
  await page.getByRole("button", { name: "Stop", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Stop", exact: true }),
  ).toBeHidden();
  await page
    .getByRole("combobox", { name: "Load saved mix" })
    .selectOption({ label: "Two layers of wit" });
  await expect(layer4).toHaveValue("6");
  await expect(layer9).toHaveValue("-9");
});

test("desktop layout fits the viewport with the jobs drawer open or closed", async ({
  page,
}) => {
  await expect(page.locator("footer")).toHaveCount(0);
  await expect(page.getByText(/Chat stays on this machine/)).toHaveCount(0);
  await expect(page.locator(".model-footprint")).not.toContainText(
    "last sample",
  );
  for (const [width, height] of [
    [1440, 980],
    [1280, 720],
    [1024, 600],
    [1100, 480],
  ]) {
    await page.setViewportSize({ width, height });
    const fits = async () => {
      const size = await page.evaluate(() => ({
        height: document.documentElement.scrollHeight,
        width: document.documentElement.scrollWidth,
        viewportHeight: innerHeight,
        viewportWidth: innerWidth,
      }));
      expect(size.height).toBeLessThanOrEqual(size.viewportHeight);
      expect(size.width).toBeLessThanOrEqual(size.viewportWidth);
    };
    await fits();
    await page.getByRole("button", { name: /Jobs & provenance/ }).click();
    await fits();
    await page.getByRole("button", { name: "Focus chat", exact: true }).click();
    await fits();
    await page
      .getByRole("button", { name: "Show panels", exact: true })
      .click();
    await page.getByRole("button", { name: /Jobs & provenance/ }).click();
  }
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.screenshot({ path: "../work/viewport-fit.png", fullPage: true });
});

test("old failures stay visible until dismissed instead of haunting the attention count", async ({
  page,
}) => {
  let dismissed = false;
  await page.route("**/api/state", async (route) => {
    const response = await route.fetch();
    const state = await response.json();
    state.jobs = [
      {
        id: "old-failure",
        kind: "generate",
        status: "interrupted",
        created_at: 1,
        error: "Old interrupted response",
        attention_dismissed: dismissed,
      },
      ...Array.from({ length: 45 }, (_, i) => ({
        id: `recent-${i}`,
        kind: "generate",
        status: "completed",
        created_at: i + 2,
      })),
    ];
    await route.fulfill({ json: state });
  });
  await page.route("**/api/action", async (route) => {
    const body = route.request().postDataJSON();
    if (body.action !== "dismiss_job_attention") return route.continue();
    expect(body.job_id).toBe("old-failure");
    dismissed = true;
    await route.fulfill({ json: { ok: true } });
  });
  await page.reload();
  const toggle = page.getByRole("button", { name: /Jobs & provenance/ });
  await expect(toggle).toContainText("1 needs attention");
  await toggle.click();
  await expect(
    page.getByText("Old interrupted response", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Dismiss", exact: true }).click();
  await expect(toggle).not.toContainText("needs attention");
  await page.reload();
  await expect(toggle).not.toContainText("needs attention");
});

test("chat gets most of the viewport and focus mode preserves the mixer", async ({
  page,
}) => {
  await addDirections(page);
  const bounds = await page.locator(".experiment-panel").boundingBox();
  expect(bounds!.width).toBeGreaterThan(1440 * 0.6);
  const transcript = await page.locator(".transcript").boundingBox();
  expect(transcript!.height).toBeGreaterThan(500);
  const before = await page.evaluate(() =>
    localStorage.getItem("torment-nexus.workspace"),
  );
  await page.getByRole("button", { name: "Focus chat", exact: true }).click();
  await expect(page.locator(".library-panel")).toBeHidden();
  await expect(page.locator(".mixer-panel")).toBeHidden();
  expect(
    (await page.locator(".experiment-panel").boundingBox())!.width,
  ).toBeGreaterThan(1380);
  await page.getByRole("button", { name: "Show panels", exact: true }).click();
  await expect(page.locator(".mixer-panel")).toBeVisible();
  expect(
    await page.evaluate(() => localStorage.getItem("torment-nexus.workspace")),
  ).toBe(before);
  await page.screenshot({
    path: "../work/multilayer-desktop-fixture.png",
    fullPage: true,
  });
});

test("archive persists without changing live axes, saved mixes, or historical labels; restore reverses it", async ({
  page,
}) => {
  const actions: Record<string, any>[] = [];
  page.on("request", (request) => {
    if (request.url().endsWith("/api/action"))
      actions.push(request.postDataJSON());
  });
  await addDirections(page);
  const coefficient = page.getByRole("spinbutton", {
    name: "Coefficient for Dry wit at layer 4",
  });
  await coefficient.fill("5");
  await page.getByRole("button", { name: "Save mix", exact: true }).click();
  await page
    .getByRole("textbox", { name: "Mix name", exact: true })
    .fill("Keep archived references");
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Save mix", exact: true })
    .click();
  await expect(page.getByRole("dialog")).toBeHidden();
  await startPrompt(page);
  await page
    .getByRole("button", { name: "Archive Dry wit", exact: true })
    .click();
  await expect(page.locator(".vector-card")).toHaveCount(1);
  await expect(page.locator(".library-heading")).toHaveText("Your directions1");
  await expect(coefficient).toHaveValue("5");
  await expect(page.locator(".run-axis-tags")).toContainText("Dry wit");
  expect(actions.filter((action) => action.action === "controls")).toEqual([]);
  expect(
    actions.find((action) => action.action === "set_vector_archived"),
  ).toMatchObject({ vector_id: "vector-0", archived: true });
  await page.getByRole("button", { name: "Stop", exact: true }).click();
  await page.reload();
  await expect(page.locator(".vector-card")).toHaveCount(1);
  await expect(coefficient).toHaveValue("5");
  await expect(page.locator(".run-axis-tags")).toContainText("Dry wit");
  await page
    .getByRole("button", { name: "Remove Dry wit from mixer", exact: true })
    .click();
  await expect(coefficient).toHaveCount(0);
  await page
    .getByRole("combobox", { name: "Load saved mix", exact: true })
    .selectOption({ label: "Keep archived references" });
  await expect(coefficient).toHaveValue("5");
  await expect(page.locator(".vector-card")).toHaveCount(1);
  const toggle = page.getByRole("checkbox", { name: "Show archived (1)" });
  await toggle.check();
  await expect(page.locator(".vector-card")).toHaveCount(2);
  await expect(page.locator(".library-heading")).toHaveText("Your directions2");
  await page
    .getByRole("button", { name: "Inspect Dry wit", exact: true })
    .click();
  await expect(page.getByRole("dialog")).toContainText("fixture-fi");
  await page.getByRole("button", { name: "Close dialog", exact: true }).click();
  await page
    .getByRole("button", { name: "Restore Dry wit", exact: true })
    .click();
  await page.getByRole("checkbox", { name: "Show archived (0)" }).click();
  await expect(
    page.getByRole("checkbox", { name: /Show archived/ }),
  ).toHaveCount(0);
  await page.reload();
  await expect(page.locator(".vector-card")).toHaveCount(2);
  await expect(
    page.getByRole("button", { name: "Archive Dry wit" }),
  ).toBeVisible();
  await expect(coefficient).toHaveValue("5");
});

test("older state snapshots without vector visibility leave the library intact", async ({
  page,
}) => {
  await page.route("**/api/state", async (route) => {
    const response = await route.fetch();
    const state = await response.json();
    delete state.vector_visibility;
    await route.fulfill({ response, json: state });
  });
  await page.reload();
  await expect(page.locator(".vector-card")).toHaveCount(2);
  await expect(
    page.getByRole("checkbox", { name: /Show archived/ }),
  ).toHaveCount(0);
  await addDirections(page);
  await expect(
    page.getByRole("spinbutton", {
      name: "Coefficient for Dry wit at layer 4",
    }),
  ).toHaveValue("0");
});

test("paper is the new default and automatic previews start disabled", async ({
  page,
}) => {
  const actions: Record<string, any>[] = [];
  page.on("request", (r) => {
    if (r.url().endsWith("/api/action")) actions.push(r.postDataJSON());
  });
  await page
    .getByRole("button", { name: "Create a concept", exact: true })
    .click();
  await expect(
    page.getByRole("combobox", { name: "Extraction method", exact: true }),
  ).toHaveValue("paper");
  await expect(
    page.getByRole("textbox", { name: "Fixed readout suffix" }),
  ).toHaveValue("I feel:");
  await expect(
    page.getByRole("checkbox", { name: /Non-positive coefficients only/ }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("combobox", { name: "Automatic previews", exact: true }),
  ).toHaveValue("none");
  await expect(
    page
      .getByRole("combobox", { name: "Automatic previews", exact: true })
      .getByRole("option", { name: /Standard/ }),
  ).toHaveCount(1);
  await page
    .getByRole("textbox", { name: "Concept", exact: true })
    .fill("A restrained fictional contrast");
  await page
    .getByRole("button", { name: "Create concept", exact: true })
    .click();
  await expect
    .poll(() => actions.find((a) => a.action === "create_concept"))
    .toMatchObject({
      extraction: { method: "paper", readout_suffix: "I feel:" },
      preview_mode: "none",
    });
  expect(actions.find((a) => a.action === "create_concept")).not.toHaveProperty(
    "coefficient_policy",
  );
});

test("standard signed previews remain an explicit option", async ({ page }) => {
  const actions: Record<string, any>[] = [];
  page.on("request", (r) => {
    if (r.url().endsWith("/api/action")) actions.push(r.postDataJSON());
  });
  await page
    .getByRole("button", { name: "Create a concept", exact: true })
    .click();
  await page
    .getByRole("combobox", { name: "Automatic previews", exact: true })
    .selectOption("standard");
  await page
    .getByRole("textbox", { name: "Concept", exact: true })
    .fill("Dry wit rather than earnest literalness");
  await page
    .getByRole("button", { name: "Create concept", exact: true })
    .click();
  await expect
    .poll(() => actions.find((a) => a.action === "create_concept"))
    .toMatchObject({ preview_mode: "standard" });
});

test("200 percent is not clamped; slider expands beyond 160 and live negatives survive", async ({
  page,
}) => {
  const actions: Record<string, any>[] = [];
  page.on("request", (request) => {
    if (request.url().endsWith("/api/action"))
      actions.push(request.postDataJSON());
  });
  await addDirections(page);
  const slider = page.getByRole("slider", {
    name: "Steering slider for Dry wit at layer 4",
  });
  const number = page.getByRole("spinbutton", {
    name: "Coefficient for Dry wit at layer 4",
  });
  await expect(slider).toHaveAttribute("max", "20");
  for (const extent of [40, 80, 160, 320]) {
    await page
      .getByRole("button", {
        name: "Expand range for Dry wit at layer 4",
        exact: true,
      })
      .click();
    await expect(slider).toHaveAttribute("max", String(extent));
  }
  await number.fill("200");
  await startPrompt(page);
  expect(
    actions.find((action) => action.action === "generate")?.axes[0].percent,
  ).toBe(200);
  await number.fill("-200");
  await expect
    .poll(
      () =>
        actions.filter((action) => action.action === "controls").at(-1)
          ?.coefficients[0].percent,
    )
    .toBe(-200);
  await expect(number).toHaveValue("-200");
  await page.getByRole("button", { name: "Stop", exact: true }).click();
  await page.reload();
  await expect(number).toHaveValue("-200");
  await expect(slider).toHaveAttribute("max", "200");
});

test("legacy sign-policy metadata never locks coefficients or claims enforcement", async ({
  page,
}) => {
  const actions: Record<string, any>[] = [];
  page.on("request", (r) => {
    if (r.url().endsWith("/api/action")) actions.push(r.postDataJSON());
  });
  await page.route("**/api/state", async (route) => {
    const response = await route.fetch();
    const state = await response.json();
    Object.assign(state.vectors[0], {
      coefficient_policy: "non_positive",
      preview_mode: "none",
      extraction: { method: "paper", readout_suffix: "I feel:" },
      warnings: ["Fixture diagnostic remains available on demand."],
    });
    await route.fulfill({ response, json: state });
  });
  await page.reload();
  await page
    .getByRole("button", { name: "Add Dry wit to mix", exact: true })
    .click();
  const slider = page.getByRole("slider", {
    name: "Steering slider for Dry wit at layer 4",
  });
  const number = page.getByRole("spinbutton", {
    name: "Coefficient for Dry wit at layer 4",
  });
  await expect(slider).toHaveAttribute("max", "20");
  await expect(number).not.toHaveAttribute("max");
  await expect(page.getByText("≤ 0 only", { exact: true })).toHaveCount(0);
  await number.fill("200");
  await number.press("Tab");
  await expect(number).toHaveValue("200");
  await expect(slider).toHaveAttribute("max", "200");
  await startPrompt(page);
  expect(actions.find((a) => a.action === "generate")?.axes[0].percent).toBe(
    200,
  );
  await number.fill("-200");
  await expect
    .poll(
      () =>
        actions.filter((a) => a.action === "controls").at(-1)?.coefficients[0]
          .percent,
    )
    .toBe(-200);
  await expect(number).toHaveValue("-200");
  await expect(slider).toHaveAttribute("min", "-200");
  await page.getByRole("button", { name: "Stop", exact: true }).click();
  await page
    .getByRole("button", { name: "Inspect Dry wit", exact: true })
    .click();
  await expect(
    page.getByText(/The selection score is not an independent test/),
  ).toBeVisible();
  await expect(page.getByText(/enforced by the server/)).toHaveCount(0);
  const diagnostic = page.getByText(
    "Fixture diagnostic remains available on demand.",
    { exact: true },
  );
  await expect(diagnostic).toBeHidden();
  await page.getByText("Diagnostics (1)", { exact: true }).click();
  await expect(diagnostic).toBeVisible();
  await page.getByRole("button", { name: "Previews", exact: true }).click();
  await expect(
    page.getByText(/Automatic previews were disabled/),
  ).toBeVisible();
  await expect(page.getByText("No completed previews yet")).toHaveCount(0);
});

test("shared recipes extract and edit for the selected model while preserving their method", async ({
  page,
}) => {
  const requests: Record<string, unknown>[] = [];
  await page.route("**/api/state", async (route) => {
    const response = await route.fetch();
    const state = await response.json();
    state.models.push({
      ...state.models[0],
      id: "second-model",
      name: "Second model",
      fingerprint: "second-fingerprint",
    });
    await route.fulfill({ json: state });
  });
  await page.route("**/api/action", async (route) => {
    const body = route.request().postDataJSON();
    if (!["extract_recipe", "edit_recipe"].includes(body.action))
      return route.continue();
    requests.push(body);
    return route.fulfill({
      json: { id: "recipe-fixture", job_id: "explicit-extraction" },
    });
  });
  await page.reload();
  await page.locator(".model-selector").click();
  await page.getByRole("button", { name: "Select model", exact: true }).click();
  await page.getByRole("button", { name: "Close dialog", exact: true }).click();
  await page.getByRole("button", { name: /^Recipes/ }).click();
  await expect(
    page.getByText("Shared across all models.", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: /Dry wit v1/ }).click();
  await expect(
    page.getByRole("dialog").getByText("Second model", { exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Extract for selected model", exact: true })
    .click();
  await expect.poll(() => requests.length).toBe(1);
  expect(requests[0]).toEqual({
    action: "extract_recipe",
    recipe_id: "recipe-fixture",
    model_id: "second-model",
  });
  await page
    .getByText("Re-extract with different settings", { exact: true })
    .click();
  const method = page.getByRole("combobox", {
    name: "Extraction method for new version",
  });
  await expect(method).toHaveValue("completion");
  await method.selectOption("paper");
  await page
    .getByRole("textbox", { name: "Readout suffix for new version" })
    .fill("This seems:");
  await page
    .getByRole("button", { name: "Extract with these settings", exact: true })
    .click();
  await expect.poll(() => requests.length).toBe(2);
  expect(requests[1]).toMatchObject({
    model_id: "second-model",
    recipe_id: "recipe-fixture",
    extraction: { method: "paper", readout_suffix: "This seems:" },
  });
  await page.getByRole("button", { name: "Edit stages", exact: true }).click();
  const editor = page.getByRole("textbox", { name: "dataset JSON editor" });
  const dataset = JSON.parse(await editor.inputValue());
  dataset.pairs[0].positive = "A quieter kind of wit.";
  await editor.fill(JSON.stringify(dataset));
  await page
    .getByRole("button", { name: "Save new version", exact: true })
    .click();
  await expect.poll(() => requests.length).toBe(3);
  expect(requests[2]).toMatchObject({
    action: "edit_recipe",
    model_id: "second-model",
    recipe_id: "recipe-fixture",
  });
  await page.getByRole("button", { name: "Close dialog", exact: true }).click();
  await page.reload();
  await page.getByRole("button", { name: /^Recipes/ }).click();
  await expect(page.getByRole("button", { name: /Dry wit v1/ })).toBeVisible();
  await page.locator(".model-selector").click();
  await page.getByRole("button", { name: "Select model", exact: true }).click();
  await page.getByRole("button", { name: "Close dialog", exact: true }).click();
  await expect(page.getByRole("button", { name: /Dry wit v1/ })).toBeVisible();
});

test("token is scrubbed; factory roles, create, immutable dataset editing and reload", async ({
  page,
}) => {
  await expect(page).toHaveURL("http://127.0.0.1:47842/");
  await page
    .getByRole("button", { name: "Create a concept", exact: true })
    .click();
  await page
    .getByRole("textbox", { name: "Concept", exact: true })
    .fill("Quiet curiosity");
  await page.getByRole("button", { name: "Edit role assignments" }).click();
  await expect(
    page.getByRole("combobox", { name: "Writer 1", exact: true }),
  ).toHaveValue("model-a");
  await expect(
    page.getByRole("combobox", { name: "Writer 2", exact: true }),
  ).toHaveValue("model-b");
  await expect(
    page.getByRole("combobox", { name: "Writer 3", exact: true }),
  ).toHaveValue("model-c");
  await page
    .getByRole("button", { name: "Create concept", exact: true })
    .click();
  await page.getByRole("button", { name: /Quiet curiosity v1/ }).click();
  await page.getByRole("button", { name: "Edit stages", exact: true }).click();
  const editor = page.getByRole("textbox", { name: "dataset JSON editor" });
  const json = JSON.parse(await editor.inputValue());
  json.pairs[0].positive = "A quiet invitation to examine the ordinary.";
  await editor.fill(JSON.stringify(json));
  await page
    .getByRole("button", { name: "Save new version", exact: true })
    .click();
  await expect(
    page.getByRole("dialog").getByText("v2", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Close dialog", exact: true }).click();
  await page.reload();
  await page.getByRole("button", { name: /^Recipes/ }).click();
  await expect(
    page.getByRole("button", { name: /Quiet curiosity v1/ }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /Quiet curiosity v2/ }),
  ).toBeVisible();
});

test("mix streams, freezes membership, acknowledges signed changes, zeroes and compares", async ({
  page,
}) => {
  const controls: {
    coefficients: { vector_id: string; layer: number; percent: number }[];
    revision: number;
  }[] = [];
  page.on("request", (request) => {
    if (request.url().endsWith("/api/action")) {
      const body = request.postDataJSON();
      if (body?.action === "controls") controls.push(body);
    }
  });
  await addDirections(page);
  await page
    .getByRole("spinbutton", { name: "Coefficient for Dry wit at layer 4" })
    .fill("8");
  await startPrompt(page);
  await expect(
    page.getByRole("button", {
      name: "Remove Dry wit from mixer",
      exact: true,
    }),
  ).toBeDisabled();
  await expect(
    page.getByRole("button", { name: "Remove Dry wit from mix", exact: true }),
  ).toBeDisabled();
  await page
    .getByRole("spinbutton", { name: "Coefficient for Dry wit at layer 4" })
    .fill("-7");
  await expect(page.getByText(/Rev 1 · from token/)).toBeVisible();
  expect(controls.at(-1)?.coefficients).toEqual([
    { vector_id: "vector-0", layer: 4, percent: -7 },
    { vector_id: "vector-0", layer: 9, percent: 0 },
    { vector_id: "vector-1", layer: 4, percent: 0 },
    { vector_id: "vector-1", layer: 9, percent: 0 },
  ]);
  await page.getByRole("button", { name: "Zero all", exact: true }).click();
  await expect(page.getByText(/Rev 2 · from token/)).toBeVisible();
  expect(
    controls
      .at(-1)
      ?.coefficients.every((coefficient) => coefficient.percent === 0),
  ).toBe(true);
  await page.getByRole("button", { name: "Stop", exact: true }).click();
  await expect(
    page.getByText("cancelled", { exact: true }).first(),
  ).toBeVisible();
  await page
    .getByRole("spinbutton", { name: "Coefficient for Dry wit at layer 4" })
    .fill("12");
  await page
    .getByRole("button", { name: "Compare baseline", exact: true })
    .click();
  await expect(
    page.getByText("ZERO / BASELINE", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Stop", exact: true }),
  ).toBeHidden();
  await expect(
    page.getByRole("spinbutton", {
      name: "Coefficient for Dry wit at layer 4",
    }),
  ).toHaveValue("12");
  await expect(
    page.getByText(/This is the unsteered comparison/),
  ).toBeVisible();
});

test("save mix, continue scratchpad in chat and fork at an earlier reply", async ({
  page,
}) => {
  const actions: Record<string, any>[] = [];
  page.on("request", (r) => {
    if (r.url().endsWith("/api/action")) actions.push(r.postDataJSON());
  });
  await addDirections(page);
  await page
    .getByRole("spinbutton", { name: "Coefficient for Dry wit at layer 4" })
    .fill("5");
  await page.getByRole("button", { name: "Save mix", exact: true }).click();
  await page
    .getByRole("textbox", { name: "Mix name", exact: true })
    .fill("A very dry room");
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Save mix", exact: true })
    .click();
  await page.reload();
  await expect(
    page.getByRole("spinbutton", {
      name: "Coefficient for Dry wit at layer 4",
    }),
  ).toHaveValue("5");
  await page.getByRole("button", { name: "Zero all", exact: true }).click();
  await page
    .getByRole("combobox", { name: "Load saved mix", exact: true })
    .selectOption({ label: "A very dry room" });
  await expect(
    page.getByRole("spinbutton", {
      name: "Coefficient for Dry wit at layer 4",
    }),
  ).toHaveValue("5");
  await startPrompt(page);
  await expect(page.locator(".run-card .response-text")).toContainText("quiet");
  await page.getByRole("button", { name: "Stop", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Stop", exact: true }),
  ).toBeHidden();
  const firstOutput = await page
    .locator(".run-card .response-text")
    .innerText();
  await page.getByRole("button", { name: "Chat", exact: true }).click();
  const conversation = page.getByRole("combobox", {
    name: "Conversation",
    exact: true,
  });
  await expect(conversation).not.toHaveValue("");
  const originalChat = await conversation.inputValue();
  await expect(page.locator(".run-card .response-text")).toHaveText(
    firstOutput,
  );
  await expect(
    page.getByRole("textbox", { name: "Prompt", exact: true }),
  ).toBeEmpty();
  await page.reload();
  await expect(page.locator(".run-card .response-text")).toHaveText(
    firstOutput,
  );
  // Toggling the modes without a new scratchpad response resumes the same chat.
  await page.getByRole("button", { name: "Scratchpad", exact: true }).click();
  await page.getByRole("button", { name: "Chat", exact: true }).click();
  await expect(conversation).toHaveValue(originalChat);
  await page
    .getByRole("textbox", { name: "Prompt", exact: true })
    .fill("Now describe the window.");
  await page.getByRole("button", { name: "Run prompt", exact: true }).click();
  await expect(page.locator(".run-card .response-text").last()).toContainText(
    "quiet",
  );
  await page.getByRole("button", { name: "Stop", exact: true }).click();
  expect(
    actions.filter((a) => a.action === "generate").at(-1)?.messages,
  ).toEqual([
    { role: "user", content: "Describe a quiet room just before sunrise." },
    { role: "assistant", content: expect.stringContaining("quiet") },
    { role: "user", content: "Now describe the window." },
  ]);
  await expect(
    page.getByRole("button", { name: "Duplicate", exact: true }),
  ).toHaveCount(2);
  await page
    .getByRole("button", { name: "Duplicate", exact: true })
    .first()
    .click();
  await expect(conversation).not.toHaveValue(originalChat);
  const forkChat = await conversation.inputValue();
  await expect(page.locator(".run-card")).toHaveCount(1);
  await expect(
    page.getByRole("textbox", { name: "Prompt", exact: true }),
  ).toBeEmpty();
  await page.reload();
  await expect(conversation).toHaveValue(forkChat);
  await expect(page.locator(".run-card")).toHaveCount(1);
  await page
    .getByRole("textbox", { name: "Prompt", exact: true })
    .fill("Describe the ceiling instead.");
  await page.getByRole("button", { name: "Run prompt", exact: true }).click();
  await expect(page.locator(".run-card .response-text").last()).toContainText(
    "quiet",
  );
  await page.getByRole("button", { name: "Stop", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Stop", exact: true }),
  ).toBeHidden();
  expect(
    actions.filter((a) => a.action === "generate").at(-1)?.messages,
  ).toEqual([
    { role: "user", content: "Describe a quiet room just before sunrise." },
    { role: "assistant", content: expect.stringContaining("quiet") },
    { role: "user", content: "Describe the ceiling instead." },
  ]);
  await conversation.selectOption(originalChat);
  await expect(page.locator(".run-card")).toHaveCount(2);
  await expect(page.locator(".run-card").last()).toContainText(
    "Now describe the window.",
  );
  await page.getByRole("button", { name: "New chat", exact: true }).click();
  await expect(page.locator(".run-card")).toHaveCount(0);
  await page.getByRole("button", { name: /^History/ }).click();
  await expect(
    page.getByRole("heading", { name: "Every experiment, kept." }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Now describe the window." }),
  ).toBeVisible();
});

test("applied geometry stays revision-tagged while controls are pending and survives reload", async ({
  page,
}) => {
  await addDirections(page);
  await startPrompt(page);
  const geometry = page.locator(".run-card .mixture-diagnostics");
  await geometry.locator("summary").click();
  await expect(geometry.locator("summary")).toHaveText(
    "Applied mixture geometry · revision 0",
  );
  await expect(geometry.getByRole("row")).toHaveCount(3);
  await expect(geometry.getByText("+0.350", { exact: true })).toHaveCount(2);

  let releaseControls!: () => void;
  const held = new Promise<void>((resolve) => {
    releaseControls = resolve;
  });
  await page.route("**/api/action", async (route) => {
    if (route.request().postDataJSON()?.action === "controls") await held;
    await route.continue();
  });
  await page
    .getByRole("spinbutton", { name: "Coefficient for Dry wit at layer 4" })
    .fill("25");
  await expect(page.getByText(/Revision 1 pending/)).toBeVisible();
  await expect(geometry.locator("summary")).toHaveText(
    "Applied mixture geometry · revision 0",
  );
  releaseControls();
  await expect(geometry.locator("summary")).toHaveText(
    "Applied mixture geometry · revision 1",
  );
  await expect(
    geometry.getByText(/Applied revision 1 · first affected output token \d+/),
  ).toBeVisible();
  await expect(
    geometry.getByText(
      "Fixture advisory: large perturbation; generation remains allowed.",
      { exact: true },
    ),
  ).toBeVisible();
  await expect(
    geometry.getByRole("cell", { name: "5.850", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Stop", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Stop", exact: true }),
  ).toBeHidden();
  await page.reload();
  await expect(geometry.locator("summary")).toHaveText(
    "Applied mixture geometry · revision 1",
  );
});

test("download interruption stays visible and can be explicitly retried", async ({
  page,
}) => {
  await page
    .getByRole("button", { name: "Manage models", exact: true })
    .click();
  await page.getByRole("button", { name: "Hugging Face", exact: true }).click();
  await page.getByRole("button", { name: "Browse files", exact: true }).click();
  await expect(page.getByText("immutable-fixture-revision")).toBeVisible();
  await page
    .getByRole("button", { name: "Download selected file", exact: true })
    .click();
  await page.getByRole("button", { name: "Close dialog", exact: true }).click();
  await page.getByRole("button", { name: /^Jobs & provenance/ }).click();
  await expect(
    page.getByText("Fixture interruption: resumable partial file retained"),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Resume / retry", exact: true })
    .click();
  await expect(
    page.getByText("Fixture interruption: resumable partial file retained"),
  ).toBeHidden();
});

test("small viewport retains library, prompt, mixer and bounded layout", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await addDirections(page);
  await expect(
    page.getByRole("textbox", { name: "Prompt", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("spinbutton", {
      name: "Coefficient for Dry wit at layer 4",
    }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: "../work/frontend-mobile.png",
    fullPage: true,
  });
});

test("long multiline concepts keep cards and sticky inspector headings bounded", async ({
  page,
}) => {
  const longName =
    "Pain, in the broad present-suffering sense: physical hurt and social hurt.\n\n" +
    "Keep the entire clarified concept in provenance, not in a viewport-sized title. ".repeat(
      28,
    ) +
    "\n\nThis final clarification must remain available.";
  await page.route("**/api/state", async (route) => {
    const response = await route.fetch();
    const state = await response.json();
    state.vectors[0].name = longName;
    state.recipes[0].concept = longName;
    await route.fulfill({ response, json: state });
  });
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 844 });
    await page.reload();
    const card = page.locator(".vector-card").first();
    await expect(card.locator("strong")).toHaveAttribute("title", longName);
    expect((await card.boundingBox())!.height).toBeLessThan(150);
    const add = page.getByRole("button", { name: /^Add Pain,/ });
    if (await add.count()) await add.click();
    const heading = page.locator(".axis-heading h3").first();
    await expect(heading).toHaveAttribute("title", longName);
    expect((await heading.boundingBox())!.height).toBeLessThan(40);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    ).toBe(true);
    await page.getByRole("button", { name: /^Inspect Pain,/ }).click();
    const dialog = page.getByRole("dialog");
    await expect(dialog.locator("h2")).toHaveAttribute("title", longName);
    expect(
      (await dialog.locator(".modal-header").boundingBox())!.height,
    ).toBeLessThan(165);
    await expect(
      dialog.getByRole("button", { name: "Overview", exact: true }),
    ).toBeInViewport();
    await dialog
      .getByRole("button", { name: "Record JSON", exact: true })
      .click();
    const record = JSON.parse(
      await dialog.locator(".json-preview").innerText(),
    );
    expect(record.name).toBe(longName);
    const close = dialog.getByRole("button", {
      name: "Close dialog",
      exact: true,
    });
    await expect(close).toBeInViewport();
    await close.click();
    await expect(dialog).toHaveCount(0);
  }
});

test("fresh launch fragments rotate a same-tab session and leave no token in its URL", async ({
  page,
}) => {
  await page.goto("/#token=browser-fixture-rotated");
  await expect(page).toHaveURL("http://127.0.0.1:47842/");
  await expect(
    page.getByText("Local inference · private session"),
  ).toBeVisible();
  expect(
    await page.evaluate(() =>
      sessionStorage.getItem("torment-nexus.launch-token"),
    ),
  ).toBe("browser-fixture-rotated");
});

test("forks only completed in-flight stages and edits writer versions without touching original job", async ({
  page,
  request,
}) => {
  await page.getByRole("button", { name: /^Jobs & provenance/ }).click();
  await page
    .getByRole("button", { name: "Fork completed stage", exact: true })
    .click();
  const dialog = page.getByRole("dialog");
  const picker = dialog.getByRole("combobox", { name: "Completed stage" });
  await expect(picker.locator("option")).toHaveCount(2);
  expect(await picker.locator("option").allTextContents()).toEqual([
    "Design / interpretation",
    "Writer 1 / matched pairs",
  ]);
  await picker.selectOption("writer_1");
  const editor = dialog.getByRole("textbox", { name: "writer_1 JSON editor" });
  const value = JSON.parse(await editor.inputValue());
  value.pairs[0].positive = "A browser-edited writer completion.";
  await editor.fill(JSON.stringify(value));
  await dialog
    .getByRole("button", { name: "Save fork & resume", exact: true })
    .click();
  await expect(
    page.getByRole("heading", {
      name: "Forked writer checkpoint",
      exact: true,
    }),
  ).toBeVisible();
  const state = await (
    await request.get("/api/state", {
      headers: { Authorization: "Bearer browser-fixture" },
    })
  ).json();
  const original = state.jobs.find(
    (job: { id: string }) => job.id === "factory-in-flight",
  );
  expect(original.status).toBe("running");
  expect(original.details.stages.writer_1).toBe("checkpoint-writer-1");
  await page.getByRole("button", { name: "Edit stages", exact: true }).click();
  await page
    .getByRole("combobox", { name: "Completed stage" })
    .selectOption("writer_1");
  const nextEditor = page.getByRole("textbox", {
    name: "writer_1 JSON editor",
  });
  await expect(nextEditor).toHaveValue(/A browser-edited writer completion/);
  value.pairs[0].positive = "Second immutable writer revision.";
  await nextEditor.fill(JSON.stringify(value));
  await page
    .getByRole("button", { name: "Save new version", exact: true })
    .click();
  await expect(
    page.getByRole("dialog").getByText("v2", { exact: true }),
  ).toBeVisible();
});
