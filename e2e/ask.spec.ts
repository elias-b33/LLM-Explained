import { expect, test } from './fixtures'

test.describe('Ask the text', () => {
  test.beforeEach(async ({ page }) => {
    // the CI browser has no GPU; make the no-WebGPU path deterministic on any machine
    await page.addInitScript(() => {
      Object.defineProperty(Navigator.prototype, 'gpu', { get: () => undefined, configurable: true })
    })
  })

  test('without WebGPU: explains why, offers search only, downloads nothing', async ({ page, external }) => {
    const modelHosts: string[] = []
    page.on('request', (r) => {
      if (/huggingface|hf\.co|jsdelivr|githubusercontent|mlc\.ai/.test(r.url())) modelHosts.push(r.url())
    })
    await page.goto('/ask')
    await expect(page).toHaveTitle(/Ask the text/)
    await expect(page.getByRole('alert').filter({ hasText: 'The chat models need WebGPU' })).toBeVisible()
    await expect(page.getByText('This browser has no WebGPU.', { exact: false })).toBeVisible()
    // nothing is fetched just by visiting: no model, no tokenizer, no wasm from a CDN
    await page.waitForTimeout(2000)
    expect(modelHosts).toEqual([])
    expect(external).toEqual([])
  })
})
