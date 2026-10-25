/** Numerically-stable softmax with optional temperature. */
export function softmax(logits: number[], temperature = 1): number[] {
  const t = Math.max(temperature, 1e-6)
  const scaled = logits.map((l) => l / t)
  const max = Math.max(...scaled)
  const exps = scaled.map((l) => Math.exp(l - max))
  const sum = exps.reduce((a, b) => a + b, 0) || 1
  return exps.map((e) => e / sum)
}

/**
 * Apply top-k and top-p (nucleus) filtering to a probability distribution.
 * Returns the renormalized distribution (excluded tokens become 0) plus a mask
 * of which tokens were cut. Pure and unit-testable.
 */
export function filterTopKTopP(
  probs: number[],
  { topK = 0, topP = 1 }: { topK?: number; topP?: number },
): { probs: number[]; excluded: boolean[] } {
  const n = probs.length
  const order = [...probs.keys()].sort((a, b) => probs[b] - probs[a])
  const k = topK > 0 ? Math.min(topK, n) : n

  // Nucleus: smallest prefix whose cumulative probability reaches topP.
  let cum = 0
  let pCount = n
  for (let r = 0; r < n; r++) {
    cum += probs[order[r]]
    if (cum >= topP) {
      pCount = r + 1
      break
    }
  }

  const keep = Math.max(1, Math.min(k, pCount))
  const excluded = new Array<boolean>(n).fill(false)
  for (let r = keep; r < n; r++) excluded[order[r]] = true

  const keptSum = order.slice(0, keep).reduce((s, i) => s + probs[i], 0) || 1
  const out = probs.map((p, i) => (excluded[i] ? 0 : p / keptSum))
  return { probs: out, excluded }
}

/** Draw an index from a probability distribution (probabilities should sum to 1). */
export function sampleIndex(probs: number[], rand: number): number {
  let cum = 0
  for (let i = 0; i < probs.length; i++) {
    cum += probs[i]
    if (rand < cum) return i
  }
  return probs.length - 1
}
