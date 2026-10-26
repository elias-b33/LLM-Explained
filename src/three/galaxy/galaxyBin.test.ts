import { readFileSync } from 'node:fs'
import { gzipSync } from 'node:zlib'
import { describe, expect, it } from 'vitest'
import { decodeGalaxy, decodeGalaxyFile, encodeGalaxy } from './galaxyBin'
import type { GalaxyData } from './types'

const read = (rel: string) => readFileSync(new URL(rel, import.meta.url))
const json = JSON.parse(read('../../data/galaxy.json').toString('utf8')) as GalaxyData

const tiny: GalaxyData = {
  words: [
    { w: 'king', p: [1.25, -0.5, 30.01], c: 0, nn: [1, 2] },
    { w: 'drottning', p: [-50.24, 0, 0.01], c: 1, nn: [0, 2] },
    { w: 'ملك', p: [0, 2.5, -3.75], c: 2, nn: [0, 1] },
  ],
  clusters: [
    { name: 'English', color: '#8b7bff', center: [-34, 0, 0] },
    { name: 'Swedish', color: '#2bd3e0' },
    { name: 'Arabic', color: '#f0b429' },
  ],
  analogies: [{ a: 'king', b: 'man', c: 'woman', d: 'queen' }],
}

describe('galaxy binary format', () => {
  it('round-trips exactly (UTF-8 words, 2-decimal positions, neighbours, clusters)', () => {
    expect(decodeGalaxy(encodeGalaxy(tiny))).toEqual(tiny)
  })

  it('refuses data it cannot store losslessly', () => {
    const bad = (patch: Partial<GalaxyData['words'][number]>) => ({ ...tiny, words: [{ ...tiny.words[0], ...patch }, ...tiny.words.slice(1)] })
    expect(() => encodeGalaxy(bad({ p: [1.234, 0, 0] }))).toThrow(/representable/)
    expect(() => encodeGalaxy(bad({ p: [400, 0, 0] }))).toThrow(/representable/)
    expect(() => encodeGalaxy(bad({ nn: [1] }))).toThrow(/neighbours/)
    expect(() => encodeGalaxy(bad({ nn: [1, 9] }))).toThrow(/range/)
    expect(() => encodeGalaxy(bad({ w: 'a\nb' }))).toThrow(/newline/)
  })

  it('rejects other files, such as an HTML page or a truncated download', () => {
    expect(() => decodeGalaxy(new TextEncoder().encode('<!doctype html><html></html>'))).toThrow(/not a galaxy/)
    const b = encodeGalaxy(tiny)
    expect(() => decodeGalaxy(b.subarray(0, b.length - 1))).toThrow(/bytes, expected/)
  })

  it('accepts the gzip file and an already-decompressed one', async () => {
    const raw = encodeGalaxy(tiny)
    expect(await decodeGalaxyFile(new Uint8Array(gzipSync(raw)))).toEqual(tiny)
    expect(await decodeGalaxyFile(raw)).toEqual(tiny)
  })

  it('the committed src/data/galaxy.bin decodes to exactly src/data/galaxy.json', async () => {
    const bin = new Uint8Array(read('../../data/galaxy.bin'))
    const decoded = await decodeGalaxyFile(bin)
    expect(decoded.words.length).toBe(45_000)
    // if this fails after editing the JSON, regenerate: node scripts/build-galaxy-bin.ts
    // (compared through JSON, which also writes the JSON's few -0 coordinates as 0)
    expect(decoded).toEqual(JSON.parse(JSON.stringify(json)))
    expect(bin.length).toBeLessThan(1_000_000)
  })
})
