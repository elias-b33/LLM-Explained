/**
 * Compact binary form of src/data/galaxy.json (45k words), so the galaxy route does not
 * ship and parse a 3.6 MB JavaScript object literal. Written by scripts/build-galaxy-bin.ts
 * and stored gzip-compressed as src/data/galaxy.bin; lossless apart from the sign of zero (a
 * unit test decodes it back to exactly the JSON).
 *
 *   bytes 0..7   magic "LLMXGAL1"
 *   u32 x4 (LE)  word count n, neighbours per word k, meta JSON length, words length
 *   meta         UTF-8 JSON { clusters, analogies }
 *   words        UTF-8, joined by "\n"
 *   cluster      u8 [n]
 *   positions    i16 [n*3] = round(p * 100) (the JSON has exactly 2 decimals; |p| < 327)
 *   neighbours   u16 [n*k]
 * Both 16-bit arrays are stored as byte planes (all low bytes, then all high bytes), which
 * gzip compresses noticeably better than interleaved little-endian.
 *
 * No DOM or Node APIs: the same code runs in the browser, in Vitest and in the build script.
 */
import type { GalaxyData } from './types'

export const GALAXY_MAGIC = 'LLMXGAL1'
const SCALE = 100

const enc = new TextEncoder()

function planes16(v: Int16Array | Uint16Array): Uint8Array {
  const n = v.length
  const out = new Uint8Array(n * 2)
  for (let i = 0; i < n; i++) {
    const x = v[i] & 0xffff
    out[i] = x & 0xff
    out[n + i] = x >>> 8
  }
  return out
}

function unplanes16(b: Uint8Array, n: number): Uint16Array {
  const out = new Uint16Array(n)
  for (let i = 0; i < n; i++) out[i] = b[i] | (b[n + i] << 8)
  return out
}

/** GalaxyData -> uncompressed binary (throws if the data does not fit the format losslessly) */
export function encodeGalaxy(d: GalaxyData): Uint8Array {
  const n = d.words.length
  const k = d.words[0]?.nn.length ?? 0
  const pos = new Int16Array(n * 3)
  const nn = new Uint16Array(n * k)
  const cl = new Uint8Array(n)
  d.words.forEach((w, i) => {
    if (w.w.includes('\n')) throw new Error(`word ${i} contains a newline`)
    if (w.nn.length !== k) throw new Error(`word ${i} has ${w.nn.length} neighbours, expected ${k}`)
    if (w.c < 0 || w.c > 255) throw new Error(`word ${i} has cluster ${w.c}`)
    cl[i] = w.c
    for (let a = 0; a < 3; a++) {
      const q = Math.round(w.p[a] * SCALE)
      if (q / SCALE !== w.p[a] || q < -32768 || q > 32767) throw new Error(`word ${i} position ${w.p[a]} is not representable`)
      pos[i * 3 + a] = q
    }
    w.nn.forEach((j, b) => {
      if (j < 0 || j >= n || !Number.isInteger(j)) throw new Error(`word ${i} neighbour ${j} out of range`)
      nn[i * k + b] = j
    })
  })
  const meta = enc.encode(JSON.stringify({ clusters: d.clusters, analogies: d.analogies }))
  const words = enc.encode(d.words.map((w) => w.w).join('\n'))
  const parts = [meta, words, cl, planes16(pos), planes16(nn)]
  const out = new Uint8Array(24 + parts.reduce((s, p) => s + p.length, 0))
  for (let i = 0; i < 8; i++) out[i] = GALAXY_MAGIC.charCodeAt(i)
  const dv = new DataView(out.buffer)
  dv.setUint32(8, n, true)
  dv.setUint32(12, k, true)
  dv.setUint32(16, meta.length, true)
  dv.setUint32(20, words.length, true)
  let o = 24
  for (const p of parts) {
    out.set(p, o)
    o += p.length
  }
  return out
}

export function hasGalaxyMagic(b: Uint8Array): boolean {
  if (b.length < 8) return false
  for (let i = 0; i < 8; i++) if (b[i] !== GALAXY_MAGIC.charCodeAt(i)) return false
  return true
}

/** uncompressed binary -> GalaxyData (the same shape the JSON had) */
export function decodeGalaxy(b: Uint8Array): GalaxyData {
  if (!hasGalaxyMagic(b)) throw new Error('not a galaxy data file')
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength)
  const n = dv.getUint32(8, true)
  const k = dv.getUint32(12, true)
  const metaLen = dv.getUint32(16, true)
  const wordsLen = dv.getUint32(20, true)
  const expected = 24 + metaLen + wordsLen + n + n * 3 * 2 + n * k * 2
  if (b.length !== expected) throw new Error(`galaxy data has ${b.length} bytes, expected ${expected}`)
  const dec = new TextDecoder()
  let o = 24
  const meta = JSON.parse(dec.decode(b.subarray(o, (o += metaLen)))) as Pick<GalaxyData, 'clusters' | 'analogies'>
  const names = dec.decode(b.subarray(o, (o += wordsLen))).split('\n')
  if (names.length !== n) throw new Error('galaxy word list is damaged')
  const cl = b.subarray(o, (o += n))
  const pos = new Int16Array(unplanes16(b.subarray(o, (o += n * 6)), n * 3).buffer)
  const nn = unplanes16(b.subarray(o, o + n * k * 2), n * k)
  const words: GalaxyData['words'] = new Array(n)
  for (let i = 0; i < n; i++) {
    const p = i * 3
    words[i] = {
      w: names[i],
      p: [pos[p] / SCALE, pos[p + 1] / SCALE, pos[p + 2] / SCALE],
      c: cl[i],
      nn: Array.from(nn.subarray(i * k, (i + 1) * k)),
    }
  }
  return { words, clusters: meta.clusters, analogies: meta.analogies }
}

const GZIP = (b: Uint8Array) => b.length > 2 && b[0] === 0x1f && b[1] === 0x8b

/**
 * Bytes as fetched -> GalaxyData. The file is gzip-compressed on disk (so it is small even
 * when the host does not compress octet-streams); if a server or proxy already decoded it,
 * the raw format is accepted too.
 */
export async function decodeGalaxyFile(bytes: Uint8Array): Promise<GalaxyData> {
  if (!GZIP(bytes)) return decodeGalaxy(bytes)
  const stream = new Blob([bytes as Uint8Array<ArrayBuffer>]).stream().pipeThrough(new DecompressionStream('gzip'))
  return decodeGalaxy(new Uint8Array(await new Response(stream).arrayBuffer()))
}
