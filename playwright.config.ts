import { defineConfig, devices } from '@playwright/test'

const PORT = 4173

/**
 * E2E smoke + accessibility against the production build (`vite build` + `vite preview`).
 * `npm run test:e2e` builds first; `npm run screenshots` refreshes docs/screenshots/.
 * Every request that leaves localhost is aborted and fails the test (see e2e/fixtures.ts).
 */
export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['github'], ['list']] : 'list',
  timeout: 60_000,
  expect: { timeout: 15_000 },
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: 'retain-on-failure',
    // Headless Chromium has no GPU: WebGL falls back to SwiftShader, which recent Chrome only allows when asked.
    launchOptions: { args: ['--enable-unsafe-swiftshader'] },
  },
  projects: [
    { name: 'chromium', testIgnore: /screenshots\.spec\.ts/, use: { ...devices['Desktop Chrome'] } },
    {
      name: 'screenshots',
      testMatch: /screenshots\.spec\.ts/,
      use: { ...devices['Desktop Chrome'], viewport: { width: 1280, height: 800 }, colorScheme: 'dark' },
    },
  ],
  webServer: {
    command: `npx vite preview --port ${PORT} --strictPort`,
    url: `http://localhost:${PORT}`,
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
})
