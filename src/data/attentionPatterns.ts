export const ATTENTION_SENTENCE = ['The', 'tired', 'cat', 'sat', 'because', 'it', 'was', 'sleepy']

type Rule = (i: number) => Record<number, number>

/** Build a row-normalized attention matrix from a per-query emphasis rule. */
function buildHead(rule: Rule): number[][] {
  const N = ATTENTION_SENTENCE.length
  return Array.from({ length: N }, (_, i) => {
    const row = new Array<number>(N).fill(0.03)
    const emph = rule(i)
    for (const key in emph) row[+key] += emph[+key]
    const sum = row.reduce((a, b) => a + b, 0)
    return row.map((x) => x / sum)
  })
}

export interface AttnHead {
  name: string
  desc: string
  weights: number[][]
}

/** Curated, illustrative attention heads — they show the kinds of links attention learns. */
export const ATTENTION_HEADS: AttnHead[] = [
  {
    name: 'Coreference',
    desc: '“it” and the words around it reach back to the noun “cat”.',
    weights: buildHead((i) =>
      i === 5 ? { 2: 1.3, 5: 0.2 } : i === 6 || i === 7 ? { 2: 0.6, [i]: 0.5 } : { [i]: 0.7, [Math.max(0, i - 1)]: 0.2 },
    ),
  },
  {
    name: 'Previous token',
    desc: 'Each token leans mostly on the token immediately before it.',
    weights: buildHead((i) => ({ [Math.max(0, i - 1)]: 1.1, [i]: 0.2 })),
  },
  {
    name: 'Adjective → noun',
    desc: 'The adjectives “tired” and “sleepy” attach to the noun “cat”.',
    weights: buildHead((i) => (i === 1 || i === 7 ? { 2: 1.3 } : { [i]: 0.85 })),
  },
]
