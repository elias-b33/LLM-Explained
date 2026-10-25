/** Small vector helpers for the embedding and attention demos. */
export const dot = (a: number[], b: number[]): number =>
  a.reduce((s, x, i) => s + x * b[i], 0)

export const norm = (a: number[]): number => Math.sqrt(dot(a, a))

export const cosine = (a: number[], b: number[]): number => dot(a, b) / (norm(a) * norm(b) || 1)

export const add = (a: number[], b: number[]): number[] => a.map((x, i) => x + b[i])

export const sub = (a: number[], b: number[]): number[] => a.map((x, i) => x - b[i])
