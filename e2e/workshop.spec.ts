import type { Page } from '@playwright/test'
import { expect, test } from './fixtures'

/** the value cell of a Workshop stat tile ("Step", "Train loss", ...) */
const stat = (page: Page, label: string) =>
  page.getByText(label, { exact: true }).locator('xpath=following-sibling::div[1]')

const num = async (page: Page, label: string) => Number((await stat(page, label).innerText()).replace(/[^\d.]/g, '') || '0')

test.describe('Workshop', () => {
  test('trains Tiny on the CPU from scratch, then pauses', async ({ page }) => {
    await page.goto('/workshop')
    // defaults: Macbeth, Tiny, CPU, from scratch
    await expect(page.getByRole('radio', { name: /Tiny/ })).toHaveAttribute('aria-checked', 'true')
    const train = page.getByRole('button', { name: 'Train', exact: true })
    await expect(train).toBeEnabled({ timeout: 30_000 })
    await expect(stat(page, 'Step')).toHaveText('0')

    await train.click()
    // steps advance and a training loss appears
    await expect.poll(() => num(page, 'Step'), { timeout: 30_000 }).toBeGreaterThan(5)
    await expect(stat(page, 'Train loss')).toHaveText(/^\d+\.\d{3}$/)

    await page.getByRole('button', { name: 'Pause' }).click()
    await expect(page.getByRole('button', { name: 'Resume' })).toBeVisible()
    // once paused, the counter stops (after the steps already in flight have been reported)
    await page.waitForTimeout(1000)
    const stopped = await num(page, 'Step')
    await page.waitForTimeout(1500)
    expect(await num(page, 'Step')).toBe(stopped)
  })

  test('GPU self-test passes when this browser has a hardware WebGPU adapter', async ({ page }) => {
    await page.goto('/workshop')
    const hasAdapter = await page.evaluate(async () => {
      type Adapter = { info?: { isFallbackAdapter?: boolean } }
      const gpu = (navigator as unknown as { gpu?: { requestAdapter(): Promise<Adapter | null> } }).gpu
      if (!gpu) return false
      const a = await gpu.requestAdapter().catch(() => null)
      return !!a && !a.info?.isFallbackAdapter
    })
    test.skip(!hasAdapter, 'no hardware WebGPU adapter in this browser (expected in headless CI)')
    const run = page.getByRole('button', { name: /Run GPU self-test/ })
    // the page shows the button only for a hardware adapter; a software one is skipped too
    test.skip(!(await run.isVisible({ timeout: 15_000 }).catch(() => false)), 'the page classed the adapter as software')
    await run.click()
    await expect(page.getByText(/^(PASS|FAIL|Could not run)/)).toBeVisible({ timeout: 60_000 })
    await expect(page.getByText(/^PASS · /)).toBeVisible()
  })
})
