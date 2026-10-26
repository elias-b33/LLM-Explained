import type { Page } from '@playwright/test'
import { expect, test } from './fixtures'

/** the scene's canvas exists and holds a live WebGL2 context (SwiftShader in headless CI) */
async function expectWebGLCanvas(page: Page) {
  const canvas = page.locator('canvas').first()
  await expect(canvas).toBeAttached({ timeout: 30_000 })
  // getContext returns the canvas's existing context, or null if it was created as something else
  await expect.poll(() => canvas.evaluate((c: HTMLCanvasElement) => !!c.getContext('webgl2')), { timeout: 15_000 }).toBe(true)
}

test.describe('every route renders', () => {
  test('landing (with the 3D hero)', async ({ page }) => {
    await page.goto('/')
    await expect(page.getByRole('heading', { level: 1 })).toBeVisible()
    await expect(page.getByRole('link', { name: /workshop/i }).first()).toBeVisible()
    await expectWebGLCanvas(page)
  })

  for (const slug of ['tokenization', 'attention', 'sampling']) {
    test(`module: ${slug}`, async ({ page }) => {
      await page.goto(`/module/${slug}`)
      await expect(page).toHaveURL(new RegExp(`/module/${slug}$`))
      await expect(page.getByRole('heading', { level: 1 })).toBeVisible()
      // the lazy chapter body has rendered (not just the shell)
      await expect(page.locator('main h2').first()).toBeVisible()
    })
  }

  test('unknown paths redirect home', async ({ page }) => {
    await page.goto('/no/such/page')
    await expect(page).toHaveURL(/\/$/)
  })

  test('workshop and ask render their headings', async ({ page }) => {
    await page.goto('/workshop')
    await expect(page.getByRole('heading', { level: 1 })).toContainText(/model/i)
    await page.goto('/ask')
    await expect(page.getByRole('heading', { level: 1 })).toBeVisible()
  })

  test('galaxy: a WebGL canvas', async ({ page }) => {
    await page.goto('/galaxy')
    await expectWebGLCanvas(page)
    await expect(page).toHaveTitle(/Word Universes/)
  })

  test('museum: a WebGL canvas', async ({ page }) => {
    // reduced motion would show the calm-version prompt first; the default does not
    await page.goto('/museum/orientation')
    await expectWebGLCanvas(page)
    // no WebGL would have redirected to the reading version
    await expect(page).toHaveURL(/\/museum\/orientation$/)
  })
})
