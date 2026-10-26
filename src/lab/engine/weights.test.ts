import { readFileSync } from 'node:fs'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { GPT, type GPTConfig } from './gpt'
import { encode, byteVocab } from './data'
import {
  NotWeightsError,
  WEIGHTS_MAGIC,
  WEIGHTS_VERSION,
  hasWeightsMagic,
  modelFromWeights,
  parseWeights,
  readWeightsHeader,
  type WeightsHeader,
} from './weights'
import { fetchWeights } from '../pretrained'
import { getSize, paramCount } from '../sizes'

const CFG: GPTConfig = { vocabSize: 256, ctx: 8, nLayer: 2, nHead: 2, dModel: 8 }

/** float32 -> IEEE half bits (round to nearest even), enough for test data */
function toHalf(f: number): number {
  const b = new DataView(new ArrayBuffer(4))
  b.setFloat32(0, f)
  const x = b.getUint32(0)
  const sign = (x >>> 16) & 0x8000
  const exp = ((x >>> 23) & 0xff) - 127 + 15
  let mant = x & 0x7fffff
  if (exp <= 0) return sign // flush tiny values to zero
  if (exp >= 31) return sign | 0x7c00
  const round = mant & 0x1fff
  mant >>>= 13
  let h = sign | (exp << 10) | mant
  if (round > 0x1000 || (round === 0x1000 && (mant & 1))) h++
  return h
}

/** The writer side of the format (mirrors scripts/pretrain/export.py). */
function writeWeights(m: GPT, dtype: 'f16' | 'f32', patch: Partial<WeightsHeader> = {}): ArrayBuffer {
  const header: WeightsHeader = {
    format: 'llm-explainer-gpt',
    size: 'test',
    config: m.cfg,
    vocab: { type: 'bytes', size: m.cfg.vocabSize },
    dtype,
    layout: m.layout().map((t) => ({ name: t.name, shape: [t.size] })),
    numParams: m.numParams,
    training: { finalValLoss: 1.23 },
    ...patch,
  }
  let json = new TextEncoder().encode(JSON.stringify(header))
  const pad = (16 - ((16 + json.length) % 16)) % 16
  if (pad) {
    const padded = new Uint8Array(json.length + pad).fill(0x20)
    padded.set(json)
    json = padded
  }
  const width = dtype === 'f16' ? 2 : 4
  const buf = new ArrayBuffer(16 + json.length + m.numParams * width)
  const bytes = new Uint8Array(buf)
  for (let i = 0; i < 8; i++) bytes[i] = WEIGHTS_MAGIC.charCodeAt(i)
  const dv = new DataView(buf)
  dv.setUint32(8, WEIGHTS_VERSION, true)
  dv.setUint32(12, json.length, true)
  bytes.set(json, 16)
  const start = 16 + json.length
  for (let i = 0; i < m.numParams; i++) {
    if (dtype === 'f32') dv.setFloat32(start + i * 4, m.params[i], true)
    else dv.setUint16(start + i * 2, toHalf(m.params[i]), true)
  }
  return buf
}

describe('weights loader', () => {
  it('round-trips a tiny model exactly in f32, and the reloaded model gives the same logits', () => {
    const m = new GPT(CFG, 21)
    const w = parseWeights(writeWeights(m, 'f32'))
    expect(w.header.config).toEqual(CFG)
    expect(Array.from(w.params)).toEqual(Array.from(m.params))
    const tokens = Array.from(encode(byteVocab(), 'hej då!'))
    const r = modelFromWeights(w)
    m.forward(tokens, null, 1, tokens.length)
    const a = Array.from(m.logitsRow(1, tokens.length, tokens.length - 1))
    r.forward(tokens, null, 1, tokens.length)
    const b = Array.from(r.logitsRow(1, tokens.length, tokens.length - 1))
    expect(b).toEqual(a)
  })

  it('round-trips f16 within half precision', () => {
    const m = new GPT(CFG, 22)
    const w = parseWeights(writeWeights(m, 'f16'))
    // half precision: ~11 significant bits; this test writer flushes |x| < 2^-14 to zero
    let worst = 0
    for (let i = 0; i < m.numParams; i++) {
      const err = Math.abs(w.params[i] - m.params[i])
      worst = Math.max(worst, err / (Math.abs(m.params[i]) * 2 ** -11 + 2 ** -14))
    }
    expect(worst).toBeLessThanOrEqual(1)
    expect(readWeightsHeader(writeWeights(m, 'f16')).training?.finalValLoss).toBe(1.23)
  })

  it('rejects a wrong magic, a truncated file and a mismatched layout', () => {
    const m = new GPT(CFG, 23)
    const good = writeWeights(m, 'f32')

    const bad = good.slice(0)
    new Uint8Array(bad)[0] = 0x58
    expect(() => parseWeights(bad)).toThrow(NotWeightsError)
    expect(() => readWeightsHeader(bad)).toThrow(NotWeightsError)

    expect(() => parseWeights(good.slice(0, good.byteLength - 10))).toThrow(/truncated/)
    expect(() => parseWeights(good.slice(0, 40))).toThrow(/truncated/)

    const otherCfg = writeWeights(m, 'f32', { config: { ...CFG, dModel: 16 } })
    expect(() => parseWeights(otherCfg)).toThrow(/layout|engine|vocab|numParams/)

    const v2 = good.slice(0)
    new DataView(v2).setUint32(8, 2, true)
    expect(() => parseWeights(v2)).toThrow(/version/)
  })

  it('treats an HTML page as not-weights', () => {
    const html = new TextEncoder().encode('<!doctype html><html><body>index</body></html>')
    expect(hasWeightsMagic(html)).toBe(false)
    expect(() => parseWeights(html.buffer as ArrayBuffer)).toThrow(NotWeightsError)
  })
})

describe('fetchWeights ("coming soon" path)', () => {
  afterEach(() => vi.unstubAllGlobals())

  const stub = (res: Response) => vi.stubGlobal('fetch', vi.fn(async () => res))

  it('an SPA rewrite answering with index.html counts as missing', async () => {
    stub(new Response('<!doctype html><html></html>', { headers: { 'content-type': 'text/html; charset=utf-8' } }))
    expect(await fetchWeights('large', () => {})).toEqual({ status: 'missing' })
  })

  it('HTML without a content type is caught by the magic check', async () => {
    stub(new Response('<!doctype html><html><body>not here</body></html>', { headers: { 'content-type': 'application/octet-stream' } }))
    expect(await fetchWeights('large', () => {})).toEqual({ status: 'missing' })
  })

  it('a 404 counts as missing', async () => {
    stub(new Response('nope', { status: 404 }))
    expect(await fetchWeights('huge', () => {})).toEqual({ status: 'missing' })
  })

  it('a real file streams to ready with progress', async () => {
    const buf = writeWeights(new GPT(CFG, 24), 'f16')
    stub(new Response(buf, { headers: { 'content-type': 'application/octet-stream', 'content-length': String(buf.byteLength) } }))
    const seen: number[] = []
    const r = await fetchWeights('large', (loaded) => seen.push(loaded))
    expect(r.status).toBe('ready')
    if (r.status === 'ready') {
      expect(r.bytes).toBe(buf.byteLength)
      expect(r.header.config).toEqual(CFG)
    }
    expect(seen.at(-1)).toBe(buf.byteLength)
  })

  it('a network failure is an error, not missing', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => Promise.reject(new TypeError('offline'))))
    const r = await fetchWeights('large', () => {})
    expect(r.status).toBe('error')
  })
})

describe('the shipped pretrained Large checkpoint', () => {
  // public/models/large.bin is committed (Vercel serves it); this reads it from disk, no network
  const file = readFileSync(new URL('../../../public/models/large.bin', import.meta.url))
  const buf = file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength) as ArrayBuffer

  it('parses, matches the Large size and its recorded validation loss', () => {
    const w = parseWeights(buf)
    const large = getSize('large')
    expect(w.header.config).toMatchObject({ nLayer: large.nLayer, dModel: large.dModel, nHead: large.nHead, ctx: large.ctx, vocabSize: 256 })
    expect(w.header.numParams).toBe(paramCount(large, 256))
    expect(w.header.training?.finalValLoss).toBeCloseTo(1.29, 2)
  })

  it('predicts English text far better than chance in the TS engine', () => {
    const m = modelFromWeights(parseWeights(buf))
    const text = 'It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness.'
    const ids = Array.from(encode(byteVocab(), text))
    const loss = m.forward(ids.slice(0, -1), ids.slice(1), 1, ids.length - 1)
    // chance is ln(256) = 5.55 nats per byte; the pretrained model should be well under 2.5
    expect(loss).toBeLessThan(2.5)
  })
})
