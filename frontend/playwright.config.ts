import { defineConfig } from "@playwright/test";
import { resolve } from "node:path";

process.env.PLAYWRIGHT_BROWSERS_PATH ??= resolve("../work/playwright");
export default defineConfig({
  testDir: "./tests",
  testMatch: "workbench.spec.ts",
  fullyParallel: false,
  workers: 1,
  timeout: 25000,
  expect: { timeout: 7000 },
  reporter: [
    ["list"],
    ["html", { outputFolder: "../work/browser-report", open: "never" }],
  ],
  outputDir: "../work/browser-results",
  use: {
    baseURL: "http://127.0.0.1:47842",
    viewport: { width: 1440, height: 980 },
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  webServer: {
    command: "node tests/fixture-server.mjs",
    url: "http://127.0.0.1:47842",
    reuseExistingServer: false,
  },
});
