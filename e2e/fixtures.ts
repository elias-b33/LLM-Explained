import { test as base, expect } from '@playwright/test'

const LOCAL = /^(localhost|127\.0\.0\.1|\[::1\])$/

/**
 * Two automatic guards on every test, checked when it ends:
 *  - pageErrors: no uncaught exceptions in the page;
 *  - external: no network traffic beyond the local preview server. The app is
 *    self-hosted, and the Ask page must never fetch a model (Hugging Face, the
 *    WebLLM CDN) on its own, so any external request is aborted and fails the test.
 */
export const test = base.extend<{ pageErrors: string[]; external: string[] }>({
  pageErrors: [
    async ({ page }, use) => {
      const errors: string[] = []
      page.on('pageerror', (e) => errors.push(`${e.name}: ${e.message}`))
      await use(errors)
      expect(errors, 'uncaught page errors').toEqual([])
    },
    { auto: true },
  ],
  external: [
    async ({ context }, use) => {
      const hits: string[] = []
      await context.route(
        (url) => (url.protocol === 'http:' || url.protocol === 'https:') && !LOCAL.test(url.hostname),
        (route) => {
          hits.push(route.request().url())
          return route.abort('blockedbyclient')
        },
      )
      await use(hits)
      expect(hits, 'requests that left localhost').toEqual([])
    },
    { auto: true },
  ],
})

export { expect }
