export type SizeId = 'tiny' | 'small' | 'medium'

export interface ModelSize {
  id: SizeId
  label: string
  nLayer: number
  dModel: number
  nHead: number
  ctx: number
  batch: number
  lr: number
  blurb: string
}

export const SIZES: ModelSize[] = [
  { id: 'tiny', label: 'Tiny', nLayer: 1, dModel: 32, nHead: 2, ctx: 32, batch: 32, lr: 5e-3, blurb: 'Learns fast. Words within a minute.' },
  { id: 'small', label: 'Small', nLayer: 2, dModel: 64, nHead: 4, ctx: 64, batch: 12, lr: 4e-3, blurb: 'Slower, but remembers more.' },
  { id: 'medium', label: 'Medium', nLayer: 3, dModel: 96, nHead: 4, ctx: 64, batch: 8, lr: 3e-3, blurb: 'Patient visitors only.' },
]

export function getSize(id: SizeId): ModelSize {
  return SIZES.find((s) => s.id === id) ?? SIZES[0]
}

/** Exact parameter count of the GPT in engine/gpt.ts for a given vocabulary size. */
export function paramCount(s: ModelSize, vocabSize: number): number {
  const d = s.dModel
  const perLayer = 2 * d + (d * 3 * d + 3 * d) + (d * d + d) + 2 * d + (d * 4 * d + 4 * d) + (4 * d * d + d)
  return vocabSize * d + s.ctx * d + s.nLayer * perLayer + 2 * d + d * vocabSize + vocabSize
}

/** Frontier LLMs are commonly estimated at around a trillion parameters (exact sizes are not published). */
export const FRONTIER_PARAMS = 1e12

export function frontierRatio(params: number): string {
  const r = FRONTIER_PARAMS / params
  if (r >= 1e6) return `${Math.round(r / 1e6)} million`
  return `${Math.round(r / 1e3)} thousand`
}
