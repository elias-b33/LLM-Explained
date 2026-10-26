import { describe, expect, it } from 'vitest'
import { GPT } from './engine/gpt'
import { expectedLayout } from './engine/weights'
import { FINETUNE, SCRATCH_WARMUP, SIZES, estStepsPerSec, fmtBytes, fmtSpeed, getSize, gpuMemory, lrAt, paramCount } from './sizes'

const cfgOf = (id: Parameters<typeof getSize>[0], V: number) => {
  const s = getSize(id)
  return { vocabSize: V, ctx: s.ctx, nLayer: s.nLayer, nHead: s.nHead, dModel: s.dModel }
}

describe('model sizes', () => {
  it('every size is a valid transformer shape, ordered small to large', () => {
    const ids = SIZES.map((s) => s.id)
    expect(new Set(ids).size).toBe(ids.length)
    for (const s of SIZES) {
      expect(s.dModel % s.nHead).toBe(0)
      expect(s.batch).toBeGreaterThan(0)
      expect(s.lr).toBeGreaterThan(0)
      if (s.gpuOnly) expect(s.gpuRecommended).toBe(true)
    }
    const counts = SIZES.map((s) => paramCount(s, 256))
    expect([...counts].sort((a, b) => a - b)).toEqual(counts)
  })

  it('paramCount matches the real engine (small sizes) and the loader layout (all sizes)', () => {
    for (const id of ['tiny', 'small', 'medium'] as const) {
      expect(new GPT(cfgOf(id, 65)).numParams).toBe(paramCount(getSize(id), 65))
    }
    for (const s of SIZES) {
      const total = expectedLayout(cfgOf(s.id, 256)).reduce((a, t) => a + t.size, 0)
      expect(total).toBe(paramCount(s, 256))
    }
    // the advertised sizes on the page
    expect(paramCount(getSize('large'), 256)).toBeCloseTo(1.9e6, -5)
    expect(paramCount(getSize('xl'), 256)).toBeCloseTo(25.6e6, -5)
  })

  it('learning-rate schedule: warmup, then the base rate (1/50 when fine-tuning)', () => {
    const s = getSize('large')
    expect(lrAt(s, 'scratch', 0)).toBeCloseTo(s.lr / SCRATCH_WARMUP)
    expect(lrAt(s, 'scratch', SCRATCH_WARMUP - 1)).toBeCloseTo(s.lr)
    expect(lrAt(s, 'scratch', 10_000)).toBeCloseTo(s.lr)
    expect(lrAt(s, 'pretrained', 10_000)).toBeCloseTo(s.lr * FINETUNE.lrScale)
    expect(lrAt(s, 'pretrained', 0)).toBeCloseTo((s.lr * FINETUNE.lrScale) / FINETUNE.warmupSteps)
  })

  it('GPU memory grows with size and the largest buffer is part of the total', () => {
    let prev = 0
    for (const s of SIZES) {
      const m = gpuMemory(s, 256)
      expect(m.total).toBeGreaterThan(prev)
      expect(m.largestBuffer).toBeLessThan(m.total)
      prev = m.total
    }
    expect(estStepsPerSec(getSize('tiny'), 65, 'gpu')).toBeGreaterThan(estStepsPerSec(getSize('tiny'), 65, 'cpu'))
  })

  it('getSize falls back to Tiny and formatters read well', () => {
    expect(getSize('nope' as never).id).toBe('tiny')
    expect(fmtBytes(512)).toBe('1 kB')
    expect(fmtBytes(3.8e6)).toBe('4 MB')
    expect(fmtBytes(2.5e9)).toBe('2.5 GB')
    expect(fmtSpeed(250)).toBe('~250 steps/s')
    expect(fmtSpeed(2)).toBe('~2 steps/s')
    expect(fmtSpeed(0.5)).toBe('~2 s per step')
    expect(fmtSpeed(1 / 180)).toBe('~3 min per step')
  })
})
