import AxeBuilder from '@axe-core/playwright'
import { expect, test } from './fixtures'

/**
 * axe-core (WCAG 2.1 A/AA rules) on the main 2D pages, in both themes. Serious and
 * critical violations fail; moderate/minor ones are attached to the report.
 *
 * The gallery look layers low-opacity texture and glass-reflection pseudo-elements
 * over the page. axe cannot compute contrast through those (it reports "incomplete"
 * for nearly every text node), so they are switched off while measuring and text is
 * checked against the solid theme colours underneath.
 */
const DECOR_OFF = `
  *::before, *::after { content: none !important; }
  .gallery-backdrop, .limestone-wall, .vitrine { background-image: none !important; }
`

const PAGES: { path: string; ready?: string }[] = [
  { path: '/' },
  { path: '/module/attention' },
  { path: '/workshop', ready: 'Train' },
  { path: '/ask' },
]

for (const theme of ['dark', 'light'] as const) {
  for (const { path, ready } of PAGES) {
    test(`axe (${theme}): ${path}`, async ({ page }, info) => {
      await page.addInitScript((t) => {
        localStorage.setItem('llm-theme', JSON.stringify(t))
        // Ask: the no-WebGPU path, as in CI
        Object.defineProperty(Navigator.prototype, 'gpu', { get: () => undefined, configurable: true })
      }, theme)
      await page.emulateMedia({ reducedMotion: 'reduce' }) // no entrance animations mid-measurement
      await page.goto(path)
      await expect(page.getByRole('heading', { level: 1 })).toBeVisible()
      if (ready) await expect(page.getByRole('button', { name: ready, exact: true })).toBeEnabled({ timeout: 30_000 })
      // let fade-ins finish (a half-faded heading would be measured at its in-between colour)
      await page.waitForFunction(() =>
        document.getAnimations().every((a) => a.playState !== 'running' || a.effect?.getTiming().iterations === Infinity),
      )
      await page.waitForTimeout(800)
      await page.addStyleTag({ content: DECOR_OFF })

      const results = await new AxeBuilder({ page })
        .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
        .exclude('canvas') // WebGL previews: no DOM text inside
        .analyze()

      const fmt = (v: (typeof results.violations)[number]) =>
        `${v.impact} ${v.id}: ${v.help} (${v.nodes.length}x)\n` +
        v.nodes
          .slice(0, 5)
          .map((n) => `    ${n.target.join(' ')} :: ${(n.any[0] ?? n.all[0] ?? n.none[0])?.message ?? ''}`)
          .join('\n')
      const blocking = results.violations.filter((v) => v.impact === 'serious' || v.impact === 'critical')
      const other = results.violations.filter((v) => !blocking.includes(v))
      if (other.length) await info.attach('axe-moderate.txt', { body: other.map(fmt).join('\n'), contentType: 'text/plain' })
      expect(blocking.map(fmt), `serious/critical axe violations on ${path} (${theme})`).toEqual([])
    })
  }
}
