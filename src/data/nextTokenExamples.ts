export interface NextTokenExample {
  prompt: string
  /** top candidate next tokens with curated probabilities (descending) */
  tokens: { token: string; prob: number }[]
}

/**
 * Curated next-token distributions for the orientation demo. These are
 * illustrative — hand-set to feel realistic — not outputs of a live model.
 */
export const NEXT_TOKEN_EXAMPLES: NextTokenExample[] = [
  {
    prompt: 'The capital of France is',
    tokens: [
      { token: ' Paris', prob: 0.86 },
      { token: ' the', prob: 0.04 },
      { token: ' located', prob: 0.025 },
      { token: ' a', prob: 0.018 },
      { token: ' home', prob: 0.012 },
      { token: ' now', prob: 0.008 },
    ],
  },
  {
    prompt: 'Once upon a',
    tokens: [
      { token: ' time', prob: 0.94 },
      { token: ' midnight', prob: 0.012 },
      { token: ' star', prob: 0.008 },
      { token: ' dream', prob: 0.006 },
      { token: ' hill', prob: 0.004 },
    ],
  },
  {
    prompt: '2 + 2 =',
    tokens: [
      { token: ' 4', prob: 0.79 },
      { token: ' four', prob: 0.07 },
      { token: ' 5', prob: 0.03 },
      { token: ' the', prob: 0.02 },
      { token: ' ?', prob: 0.015 },
    ],
  },
  {
    prompt: 'The opposite of hot is',
    tokens: [
      { token: ' cold', prob: 0.88 },
      { token: ' warm', prob: 0.03 },
      { token: ' cool', prob: 0.025 },
      { token: ' not', prob: 0.012 },
      { token: ' freezing', prob: 0.01 },
    ],
  },
]
