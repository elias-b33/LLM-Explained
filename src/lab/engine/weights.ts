/**
 * Pretrained weight files (.bin) for the Workshop GPT. Written by
 * scripts/pretrain/export.py; the format is documented in scripts/pretrain/README.md.
 *
 *   bytes 0..7    magic "LLMXWGT\0"
 *   bytes 8..11   format version, u32 little-endian (currently 1)
 *   bytes 12..15  header length H in bytes, u32 LE (H is padded so the data starts 16-byte aligned)
 *   bytes 16..    H bytes of UTF-8 JSON (config, vocab, dtype, layout, training metadata)
 *   then          every parameter as little-endian f16 (or f32 for parity tests),
 *                 in exactly the engine's storage order (GPT.layout()).
 *
 * No dependencies on the DOM, so the same loader runs in the worker and in Node (parity test).
 */
import { GPT, type GPTConfig } from './gpt.ts'

export const WEIGHTS_MAGIC = 'LLMXWGT\0'
export const WEIGHTS_VERSION = 1

export interface WeightsHeader {
  format: 'llm-explainer-gpt'
  /** Workshop size id these weights are for (large / huge / xl), or a free label for tests */
  size: string
  config: GPTConfig
  vocab: { type: 'bytes' | 'chars'; size: number; chars?: string[] }
  dtype: 'f16' | 'f32'
  /** parameter tensors in storage order */
  layout: { name: string; shape: number[] }[]
  numParams: number
  training?: {
    dataset?: string
    tokensSeen?: number
    steps?: number
    finalValLoss?: number
    finalTrainLoss?: number
    date?: string
    [k: string]: unknown
  }
}

export interface LoadedWeights {
  header: WeightsHeader
  params: Float32Array
}

/** Not a weights file (for example a 404 page, or index.html from an SPA rewrite). */
export class NotWeightsError extends Error {}

/** Cheap check of the first bytes, before anything is parsed. */
export function hasWeightsMagic(head: Uint8Array): boolean {
  if (head.length < 8) return false
  for (let i = 0; i < 8; i++) if (head[i] !== WEIGHTS_MAGIC.charCodeAt(i)) return false
  return true
}

let halfTable: Float32Array | null = null
/** All 65536 fp16 bit patterns as float32 (256 kB, built once). */
function halfToFloatTable(): Float32Array {
  if (halfTable) return halfTable
  const t = new Float32Array(65536)
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1
    const e = (h >> 10) & 0x1f
    const m = h & 0x3ff
    if (e === 0) t[h] = s * m * 2 ** -24
    else if (e === 31) t[h] = m ? NaN : s * Infinity
    else t[h] = s * (1 + m / 1024) * 2 ** (e - 15)
  }
  halfTable = t
  return t
}

/** Parse and validate a weights file; throws NotWeightsError if the magic header is missing. */
export function parseWeights(buf: ArrayBuffer): LoadedWeights {
  const bytes = new Uint8Array(buf)
  if (!hasWeightsMagic(bytes)) throw new NotWeightsError('not a Workshop weights file')
  const dv = new DataView(buf)
  const version = dv.getUint32(8, true)
  if (version !== WEIGHTS_VERSION) throw new Error(`unsupported weights version ${version}`)
  const hlen = dv.getUint32(12, true)
  if (16 + hlen > buf.byteLength) throw new Error('weights file is truncated (header)')
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(16, 16 + hlen))) as WeightsHeader
  if (header.format !== 'llm-explainer-gpt') throw new Error(`unknown weights format ${String(header.format)}`)
  const cfg = header.config
  if (header.vocab.size !== cfg.vocabSize) throw new Error('vocab size does not match the config')

  // the layout must be exactly the engine's parameter order
  const expected = expectedLayout(cfg)
  if (header.layout.length !== expected.length) throw new Error('parameter layout does not match the engine')
  let n = 0
  header.layout.forEach((t, i) => {
    const size = t.shape.reduce((a, b) => a * b, 1)
    if (t.name !== expected[i].name || size !== expected[i].size)
      throw new Error(`parameter ${i} is ${t.name} [${t.shape}] but the engine expects ${expected[i].name} (${expected[i].size})`)
    n += size
  })
  if (n !== header.numParams) throw new Error('numParams does not match the layout')

  const width = header.dtype === 'f16' ? 2 : header.dtype === 'f32' ? 4 : 0
  if (!width) throw new Error(`unsupported dtype ${String(header.dtype)}`)
  const start = 16 + hlen
  if (start % 4 !== 0) throw new Error('weights data is not aligned')
  if (buf.byteLength !== start + n * width)
    throw new Error(`weights file has ${buf.byteLength} bytes, expected ${start + n * width} (truncated download?)`)

  let params: Float32Array
  if (width === 4) {
    params = new Float32Array(buf.slice(start, start + n * 4))
  } else {
    const h = new Uint16Array(buf, start, n)
    const t = halfToFloatTable()
    params = new Float32Array(n)
    for (let i = 0; i < n; i++) params[i] = t[h[i]]
  }
  for (let i = 0; i < n; i++) if (!Number.isFinite(params[i])) throw new Error(`non-finite weight at index ${i}`)
  return { header, params }
}

/** The engine's parameter order for a config (names and element counts), taken from the GPT itself. */
export function expectedLayout(cfg: GPTConfig): { name: string; size: number }[] {
  return layoutCache.get(key(cfg)) ?? computeLayout(cfg)
}

const layoutCache = new Map<string, { name: string; size: number }[]>()
const key = (c: GPTConfig) => `${c.vocabSize}/${c.ctx}/${c.nLayer}/${c.nHead}/${c.dModel}`

function computeLayout(cfg: GPTConfig): { name: string; size: number }[] {
  // Mirrors the GPT constructor without allocating its (possibly 100 MB) buffers;
  // checked against GPT.layout() on a small config below.
  const { vocabSize: V, ctx, nLayer: L, dModel: d } = cfg
  const out: { name: string; size: number }[] = [
    { name: 'tok', size: V * d },
    { name: 'pos', size: ctx * d },
  ]
  for (let l = 0; l < L; l++) {
    out.push(
      { name: `${l}.ln1g`, size: d },
      { name: `${l}.ln1b`, size: d },
      { name: `${l}.wqkv`, size: d * 3 * d },
      { name: `${l}.bqkv`, size: 3 * d },
      { name: `${l}.wo`, size: d * d },
      { name: `${l}.bo`, size: d },
      { name: `${l}.ln2g`, size: d },
      { name: `${l}.ln2b`, size: d },
      { name: `${l}.wfc`, size: d * 4 * d },
      { name: `${l}.bfc`, size: 4 * d },
      { name: `${l}.wproj`, size: 4 * d * d },
      { name: `${l}.bproj`, size: d },
    )
  }
  out.push({ name: 'lnfg', size: d }, { name: 'lnfb', size: d }, { name: 'wout', size: d * V }, { name: 'bout', size: V })
  layoutCache.set(key(cfg), out)
  return out
}

// guard against the mirror above drifting from the real engine
{
  const probe: GPTConfig = { vocabSize: 5, ctx: 3, nLayer: 2, nHead: 2, dModel: 4 }
  const real = new GPT(probe, 1).layout()
  const mine = computeLayout(probe)
  if (real.length !== mine.length || real.some((r, i) => r.name !== mine[i].name || r.size !== mine[i].size))
    throw new Error('weights.ts layout is out of sync with gpt.ts')
}

/** Build a CPU model from loaded weights. */
export function modelFromWeights(w: LoadedWeights, seed = 1337): GPT {
  const m = new GPT(w.header.config, seed)
  m.params.set(w.params)
  return m
}

/** Read just the JSON header (cheap: no tensor conversion). Throws NotWeightsError without the magic. */
export function readWeightsHeader(buf: ArrayBuffer): WeightsHeader {
  const bytes = new Uint8Array(buf)
  if (!hasWeightsMagic(bytes)) throw new NotWeightsError('not a Workshop weights file')
  const hlen = new DataView(buf).getUint32(12, true)
  return JSON.parse(new TextDecoder().decode(bytes.subarray(16, 16 + hlen))) as WeightsHeader
}
