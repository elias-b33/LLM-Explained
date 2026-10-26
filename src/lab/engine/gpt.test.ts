import { describe, expect, it } from 'vitest'
import { Decoder, GPT, sampleFrom, topK, type AdamW, type GPTConfig } from './gpt'
import { Rng } from './rng'

const TINY: GPTConfig = { vocabSize: 7, ctx: 6, nLayer: 2, nHead: 2, dModel: 8 }

/** a fixed batch: B rows of T tokens, targets = the next token */
function batch(rng: Rng, B: number, T: number, V: number) {
  const idx: number[] = []
  const targets: number[] = []
  for (let n = 0; n < B * T; n++) {
    idx.push(rng.int(V))
    targets.push(rng.int(V))
  }
  return { idx, targets }
}

describe('CPU GPT engine', () => {
  it('backward matches central finite differences on every parameter (Float64)', () => {
    const m = new GPT(TINY, 7, Float64Array)
    // perturb the ones/zeros-initialised tensors so LayerNorm gains and biases get non-trivial gradients
    const prng = new Rng(99)
    for (let i = 0; i < m.numParams; i++) m.params[i] += (prng.next() - 0.5) * 0.2
    const B = 2
    const T = 5
    const { idx, targets } = batch(new Rng(3), B, T, TINY.vocabSize)

    m.zeroGrad()
    m.forward(idx, targets, B, T)
    m.backward()
    const analytic = Float64Array.from(m.grads)

    const h = 1e-5
    const numeric = new Float64Array(m.numParams)
    for (let i = 0; i < m.numParams; i++) {
      const w = m.params[i]
      m.params[i] = w + h
      const lp = m.forward(idx, targets, B, T)
      m.params[i] = w - h
      const lm = m.forward(idx, targets, B, T)
      m.params[i] = w
      numeric[i] = (lp - lm) / (2 * h)
    }

    let diff = 0
    let na = 0
    let nn = 0
    let maxAbs = 0
    for (let i = 0; i < m.numParams; i++) {
      diff += (analytic[i] - numeric[i]) ** 2
      na += analytic[i] ** 2
      nn += numeric[i] ** 2
      maxAbs = Math.max(maxAbs, Math.abs(analytic[i] - numeric[i]))
    }
    const relErr = Math.sqrt(diff) / (Math.sqrt(na) + Math.sqrt(nn))
    // measured ~1e-9 to 1e-8 with h = 1e-5; anything near 1e-4 would mean a wrong kernel
    expect(relErr).toBeLessThan(1e-6)
    expect(maxAbs).toBeLessThan(1e-6)
    expect(Math.sqrt(na)).toBeGreaterThan(1e-3) // the check is not vacuous
  })

  it('is deterministic for a fixed seed and differs across seeds', () => {
    const { idx, targets } = batch(new Rng(5), 2, 6, TINY.vocabSize)
    const a = new GPT(TINY, 42)
    const b = new GPT(TINY, 42)
    const c = new GPT(TINY, 43)
    expect(Array.from(a.params)).toEqual(Array.from(b.params))
    const la = a.forward(idx, targets, 2, 6)
    const lb = b.forward(idx, targets, 2, 6)
    const lc = c.forward(idx, targets, 2, 6)
    expect(la).toBe(lb)
    expect(lc).not.toBe(la)
    // an untrained model is close to uniform: loss ~ ln(V)
    expect(la).toBeGreaterThan(Math.log(TINY.vocabSize) - 0.1)
    expect(la).toBeLessThan(Math.log(TINY.vocabSize) + 0.1)
  })

  it('the KV-cached decoder reproduces the full forward pass logits', () => {
    const m = new GPT(TINY, 11)
    const tokens = [1, 4, 2, 6, 0, 3]
    m.forward(tokens, null, 1, tokens.length)
    const full = Array.from(m.logitsRow(1, tokens.length, tokens.length - 1))
    const dec = new Decoder(m)
    dec.prime(tokens)
    const cached = Array.from(dec.logits)
    for (let i = 0; i < full.length; i++) expect(cached[i]).toBeCloseTo(full[i], 5)
    const p = dec.probs(1)
    expect(p.reduce((s, x) => s + x, 0)).toBeCloseTo(1, 6)
  })

  it('loss goes down over a few AdamW steps on a learnable pattern', () => {
    const cfg: GPTConfig = { vocabSize: 5, ctx: 8, nLayer: 1, nHead: 2, dModel: 16 }
    const m = new GPT(cfg, 1)
    const opt: AdamW = { lr: 1e-2, beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0.01 }
    // a periodic sequence 0 1 2 3 4 0 1 2 ...: the next token is fully predictable
    const B = 4
    const T = 8
    const idx: number[] = []
    const targets: number[] = []
    for (let b = 0; b < B; b++)
      for (let t = 0; t < T; t++) {
        idx.push((b + t) % 5)
        targets.push((b + t + 1) % 5)
      }
    const losses: number[] = []
    for (let s = 0; s < 40; s++) {
      m.zeroGrad()
      losses.push(m.forward(idx, targets, B, T))
      m.backward()
      m.clipGrads(1)
      m.step(opt)
    }
    expect(losses.every(Number.isFinite)).toBe(true)
    expect(losses[losses.length - 1]).toBeLessThan(losses[0] * 0.5)
  })

  it('rejects bad shapes', () => {
    expect(() => new GPT({ ...TINY, nHead: 3 })).toThrow(/divisible/)
    const m = new GPT(TINY)
    expect(() => m.forward(new Array(TINY.ctx + 1).fill(0), null, 1, TINY.ctx + 1)).toThrow(/context/)
    expect(() => new GPT(TINY).backward()).toThrow(/forward/)
  })
})

describe('sampling helpers', () => {
  it('topK returns the largest entries in order', () => {
    expect(topK([0.1, 0.5, 0.05, 0.35], 2)).toEqual([1, 3])
  })

  it('sampleFrom follows the distribution', () => {
    const rng = new Rng(123)
    const counts = [0, 0, 0]
    for (let i = 0; i < 20000; i++) counts[sampleFrom([0.2, 0.5, 0.3], rng)]++
    expect(counts[0] / 20000).toBeCloseTo(0.2, 1)
    expect(counts[1] / 20000).toBeCloseTo(0.5, 1)
    expect(counts[2] / 20000).toBeCloseTo(0.3, 1)
  })

  it('Rng is reproducible and normal() has unit variance', () => {
    const a = new Rng(9)
    const b = new Rng(9)
    expect([a.next(), a.next(), a.int(10)]).toEqual([b.next(), b.next(), b.int(10)])
    const r = new Rng(1)
    let s = 0
    let s2 = 0
    const n = 20000
    for (let i = 0; i < n; i++) {
      const x = r.normal()
      s += x
      s2 += x * x
    }
    expect(s / n).toBeCloseTo(0, 1)
    expect(s2 / n).toBeCloseTo(1, 1)
  })
})
