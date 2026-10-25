/** Character-level tokenizer: every distinct character is one token. */
export interface Vocab {
  chars: string[]
  index: Map<string, number>
  /** id used for characters that are not in the vocabulary */
  unk: number
}

/** Characters beyond this many distinct ones are folded into a single "unknown" token. */
const MAX_VOCAB = 120
const UNK_CHAR = '¿' // ¿

export function buildVocab(text: string): Vocab {
  const counts = new Map<string, number>()
  for (const ch of text) counts.set(ch, (counts.get(ch) ?? 0) + 1)
  let kept = [...counts.entries()]
  let needUnk = false
  if (kept.length > MAX_VOCAB) {
    kept.sort((a, b) => b[1] - a[1])
    kept = kept.slice(0, MAX_VOCAB - 1)
    needUnk = true
  }
  const chars = kept.map((e) => e[0]).sort()
  if (needUnk && !chars.includes(UNK_CHAR)) chars.push(UNK_CHAR)
  const index = new Map(chars.map((c, i) => [c, i]))
  const unk = index.get(UNK_CHAR) ?? index.get(' ') ?? 0
  return { chars, index, unk }
}

export function encode(v: Vocab, text: string): Uint16Array {
  const out: number[] = []
  for (const ch of text) out.push(v.index.get(ch) ?? v.unk)
  return Uint16Array.from(out)
}

export function decode(v: Vocab, ids: ArrayLike<number>): string {
  let s = ''
  for (let i = 0; i < ids.length; i++) s += v.chars[ids[i]] ?? ''
  return s
}
