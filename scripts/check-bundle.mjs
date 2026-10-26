#!/usr/bin/env node
/**
 * Bundle size report + budget for the production build (run after `vite build`):
 *
 *   node scripts/check-bundle.mjs          # fails (exit 1) if over budget
 *
 * "Initial" = what index.html loads before any route code: the entry script, its
 * modulepreloads and the stylesheets. Sizes are gzip (zlib default level), kB = 1000 bytes
 * (vite build's own report differs by about 1%).
 * Budgets sit ~10% above the measured baseline; raise them deliberately, in review.
 */
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { gzipSync } from 'node:zlib'

const BUDGET_KB = {
  initialJs: 160, // measured 144.6 kB gzip (react-dom, react-router, motion, app shell)
  initialCss: 40, // measured 36.1 kB gzip (Tailwind + KaTeX)
}

const dist = new URL('../dist/', import.meta.url)
let html
try {
  html = readFileSync(new URL('index.html', dist), 'utf8')
} catch {
  console.error('dist/index.html not found: run `npm run build` first')
  process.exit(1)
}

const attr = (tag, name) => tag.match(new RegExp(`${name}="([^"]+)"`))?.[1]
const tags = html.match(/<(script|link)\b[^>]*>/g) ?? []
const initialJs = new Set()
const initialCss = new Set()
for (const t of tags) {
  if (t.startsWith('<script') && attr(t, 'type') === 'module' && attr(t, 'src')) initialJs.add(attr(t, 'src'))
  if (t.startsWith('<link') && attr(t, 'rel') === 'modulepreload') initialJs.add(attr(t, 'href'))
  if (t.startsWith('<link') && attr(t, 'rel') === 'stylesheet') initialCss.add(attr(t, 'href'))
}

const file = (p) => new URL(p.replace(/^\//, ''), dist)
const gz = (p) => gzipSync(readFileSync(p)).length
const kb = (b) => (b / 1000).toFixed(1)
const sum = (set) => [...set].reduce((s, p) => s + gz(file(p)), 0)

const jsGz = sum(initialJs)
const cssGz = sum(initialCss)

// every other emitted JS chunk / data asset, largest first (informational)
const assetsDir = new URL('assets/', dist)
const lazy = readdirSync(assetsDir)
  .filter((f) => /\.(js|bin)$/.test(f) && !initialJs.has(`/assets/${f}`))
  .map((f) => {
    const p = join(assetsDir.pathname.replace(/^\/([A-Za-z]:)/, '$1'), f)
    return { f, raw: statSync(p).size, gz: f.endsWith('.bin') ? statSync(p).size : gz(p) }
  })
  .sort((a, b) => b.gz - a.gz)

console.log('Initial load (index.html)')
for (const p of initialJs) console.log(`  js   ${kb(gz(file(p))).padStart(7)} kB gz  ${p}`)
for (const p of initialCss) console.log(`  css  ${kb(gz(file(p))).padStart(7)} kB gz  ${p}`)
console.log(`  total JS ${kb(jsGz)} kB gz (budget ${BUDGET_KB.initialJs}), CSS ${kb(cssGz)} kB gz (budget ${BUDGET_KB.initialCss})`)
console.log('\nLargest lazy chunks (loaded on demand; .bin is already compressed)')
for (const c of lazy.slice(0, 10)) console.log(`  ${kb(c.gz).padStart(8)} kB gz  ${kb(c.raw).padStart(8)} kB raw  ${c.f}`)

const over = []
if (jsGz / 1000 > BUDGET_KB.initialJs) over.push(`initial JS ${kb(jsGz)} kB > ${BUDGET_KB.initialJs} kB`)
if (cssGz / 1000 > BUDGET_KB.initialCss) over.push(`initial CSS ${kb(cssGz)} kB > ${BUDGET_KB.initialCss} kB`)
if (over.length) {
  console.error(`\nBundle budget exceeded: ${over.join('; ')}`)
  process.exit(1)
}
console.log('\nBundle budget OK')
