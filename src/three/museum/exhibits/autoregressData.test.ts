import { afterEach, describe, expect, it, vi } from 'vitest'
import { FORCE_END, MAX_STEPS, PROMPTS, TEMPERATURES, distFor, isEnd, sample, shape } from './autoregressData'

describe('autoregression machine data', () => {
  afterEach(() => vi.restoreAllMocks())

  it('every table has positive weights and every prompt can reach an end', () => {
    for (const p of PROMPTS) {
      for (const d of [p.start, ...Object.values(p.next)]) {
        expect(d.length).toBeGreaterThan(0)
        for (const [tok, w] of d) {
          expect(tok.length).toBeGreaterThan(0)
          expect(w).toBeGreaterThan(0)
        }
      }
      // a greedy walk ends within MAX_STEPS
      let last: string | null = null
      let steps = 0
      for (; steps < MAX_STEPS; steps++) {
        const d = distFor(p, last, steps)
        last = d.reduce((a, b) => (b[1] > a[1] ? b : a))[0]
        if (isEnd(last)) break
      }
      expect(isEnd(last!)).toBe(true)
    }
  })

  it('forces an ending at the step limit', () => {
    expect(distFor(PROMPTS[0], ' mat', MAX_STEPS - 1)).toBe(FORCE_END)
    expect(distFor(PROMPTS[0], null, 0)).toBe(PROMPTS[0].start)
  })

  it('temperature reshapes softmax(log p / T): T=1 is the plain weights, low T sharpens, high T flattens', () => {
    const d = PROMPTS[0].start
    const total = d.reduce((a, [, w]) => a + w, 0)
    const plain = shape(d, 1)
    plain.forEach((p, i) => expect(p).toBeCloseTo(d[i][1] / total, 10))
    const tops = TEMPERATURES.map((T) => shape(d, T)[0])
    for (let i = 1; i < tops.length; i++) expect(tops[i]).toBeLessThan(tops[i - 1])
    for (const T of TEMPERATURES) expect(shape(d, T).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
  })

  it('sample walks the cumulative distribution', () => {
    const p = [0.2, 0.5, 0.3]
    vi.spyOn(Math, 'random').mockReturnValue(0.1)
    expect(sample(p)).toBe(0)
    vi.spyOn(Math, 'random').mockReturnValue(0.69)
    expect(sample(p)).toBe(1)
    vi.spyOn(Math, 'random').mockReturnValue(0.999999)
    expect(sample(p)).toBe(2)
  })
})
