import type { Page } from '@playwright/test'
import { expect, test } from './fixtures'

/**
 * README screenshots (docs/screenshots/*.jpg). Not part of `npm run test:e2e`;
 * refresh them with `npm run screenshots`.
 */
const OUT = 'docs/screenshots'

async function shot(page: Page, name: string) {
  // let entrance animations finish
  await page.waitForFunction(() =>
    document.getAnimations().every((a) => a.playState !== 'running' || a.effect?.getTiming().iterations === Infinity),
  )
  await page.waitForTimeout(600)
  await page.screenshot({ path: `${OUT}/${name}.jpg`, type: 'jpeg', quality: 78 })
}

test('landing', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('canvas').first()).toBeAttached()
  await page.waitForTimeout(2500) // the hero network fades in
  await shot(page, 'landing')
})

test('chapter with a live demo', async ({ page }) => {
  await page.goto('/module/attention')
  const demo = page.locator('span[title$="%"]').first()
  await demo.evaluate((el) => {
    el.scrollIntoView({ block: 'center', behavior: 'instant' })
    window.scrollBy({ top: -120, behavior: 'instant' })
  })
  await shot(page, 'chapter-attention')
})

test('workshop training', async ({ page }) => {
  await page.goto('/workshop')
  const train = page.getByRole('button', { name: 'Train', exact: true })
  await expect(train).toBeEnabled({ timeout: 30_000 })
  await train.click()
  await page.waitForTimeout(15_000)
  // stats, loss curve and the evolving samples in one frame, still training
  await page.getByText(/^Engine: /).evaluate((el) => {
    el.scrollIntoView({ block: 'start', behavior: 'instant' })
    window.scrollBy({ top: -260, behavior: 'instant' })
  })
  await page.waitForTimeout(400)
  await page.screenshot({ path: `${OUT}/workshop.jpg`, type: 'jpeg', quality: 78 })
  await page.getByRole('button', { name: 'Pause' }).click()
})

test('ask the text', async ({ page }) => {
  await page.goto('/ask')
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible()
  await page.waitForTimeout(1000)
  await shot(page, 'ask')
})

test('galaxy', async ({ page }) => {
  await page.goto('/galaxy')
  await expect(page.locator('canvas').first()).toBeAttached({ timeout: 30_000 })
  await page.waitForTimeout(6000) // camera fly-in on a software renderer
  await page.screenshot({ path: `${OUT}/galaxy.jpg`, type: 'jpeg', quality: 78 })
})
