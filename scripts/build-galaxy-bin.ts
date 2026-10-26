/**
 * src/data/galaxy.json -> src/data/galaxy.bin (gzip of the binary format in
 * src/three/galaxy/galaxyBin.ts). Run after build_galaxy.py / build_universes.py:
 *
 *   node scripts/build-galaxy-bin.ts
 *
 * (Node 22.18+ / 24 runs TypeScript directly.) The unit test src/three/galaxy/galaxyBin.test.ts
 * fails if the committed .bin no longer decodes to exactly the JSON.
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { gzipSync } from 'node:zlib'
import { decodeGalaxy, encodeGalaxy } from '../src/three/galaxy/galaxyBin.ts'
import type { GalaxyData } from '../src/three/galaxy/types.ts'

const src = new URL('../src/data/galaxy.json', import.meta.url)
const out = new URL('../src/data/galaxy.bin', import.meta.url)
const json = readFileSync(src, 'utf8')
const data = JSON.parse(json) as GalaxyData
const raw = encodeGalaxy(data)
if (JSON.stringify(decodeGalaxy(raw)) !== JSON.stringify(data)) throw new Error('round trip failed')
// no mtime and an "unknown" OS byte in the gzip header: the same JSON always gives the same bytes
const gz = gzipSync(raw, { level: 9 })
gz.writeUInt32LE(0, 4)
gz[9] = 0xff
writeFileSync(out, gz)
const kb = (b: number) => `${(b / 1024).toFixed(0)} kB`
console.log(`galaxy.json ${kb(Buffer.byteLength(json))} -> ${data.words.length} words, binary ${kb(raw.length)}, gzip ${kb(gz.length)} -> ${out.pathname}`)
