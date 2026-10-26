/**
 * Tokenizers for the Workshop. Two kinds:
 *  - chars: every distinct character of the training text is one token (from-scratch mode).
 *  - bytes: a fixed vocabulary of the 256 possible UTF-8 bytes (pretrained mode), so any text
 *    can be encoded. ASCII is one byte per character; å, ä, ö are two bytes each.
 */
export interface Vocab {
  kind: 'chars' | 'bytes'
  /** display label of each token id (for bytes: the character for ASCII, else a hex placeholder) */
  chars: string[]
  /** chars mode: character -> id (bytes mode: single-byte characters only) */
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
  return { kind: 'chars', chars, index, unk }
}

/* ------------------------------------------------------------------ bytes */

const hex = (b: number) => b.toString(16).toUpperCase().padStart(2, '0')

/**
 * Readable label for one byte on its own: printable ASCII is itself, everything
 * else (control bytes, and the pieces of multi-byte characters) is a hex
 * placeholder such as ‹C3›.
 */
export function byteLabel(b: number): string {
  if (b === 10) return '\n'
  if (b === 9) return '\t'
  if (b >= 32 && b < 127) return String.fromCharCode(b)
  return `‹${hex(b)}›`
}

/** True for labels produced by byteLabel() for a non-printable byte. */
export const isBytePlaceholder = (label: string) => label.length === 4 && label[0] === '‹' && label[3] === '›'

let byteVocabCache: Vocab | null = null
export function byteVocab(): Vocab {
  if (!byteVocabCache) {
    const chars = Array.from({ length: 256 }, (_, b) => byteLabel(b))
    const index = new Map<string, number>()
    for (let b = 0; b < 128; b++) index.set(String.fromCharCode(b), b)
    byteVocabCache = { kind: 'bytes', chars, index, unk: 63 /* '?' */ }
  }
  return byteVocabCache
}

const utf8 = new TextEncoder()

export function encode(v: Vocab, text: string): Uint16Array {
  if (v.kind === 'bytes') return Uint16Array.from(utf8.encode(text))
  const out: number[] = []
  for (const ch of text) out.push(v.index.get(ch) ?? v.unk)
  return Uint16Array.from(out)
}

/**
 * Number of bytes at the end of `bytes` that form an incomplete UTF-8 sequence
 * (a lead byte still waiting for its continuation bytes), 0 if it ends cleanly.
 */
export function incompleteTail(bytes: ArrayLike<number>): number {
  const n = bytes.length
  for (let k = 1; k <= Math.min(3, n); k++) {
    const b = bytes[n - k]
    if ((b & 0xc0) === 0x80) continue // continuation byte: keep looking for the lead
    const need = (b & 0xe0) === 0xc0 ? 2 : (b & 0xf0) === 0xe0 ? 3 : (b & 0xf8) === 0xf0 ? 4 : 1
    return need > k ? k : 0
  }
  return 0
}

/**
 * Token ids back to text. In bytes mode, invalid sequences become U+FFFD; with
 * dropIncomplete, a multi-byte character cut off at the very end is left out
 * instead (used for fixed-length samples).
 */
export function decode(v: Vocab, ids: ArrayLike<number>, dropIncomplete = false): string {
  if (v.kind === 'bytes') {
    let bytes = Uint8Array.from(ids as ArrayLike<number>)
    if (dropIncomplete) bytes = bytes.subarray(0, bytes.length - incompleteTail(bytes))
    return new TextDecoder('utf-8').decode(bytes)
  }
  let s = ''
  for (let i = 0; i < ids.length; i++) s += v.chars[ids[i]] ?? ''
  return s
}

/**
 * Streaming decoder: push one token at a time, get back the text it completes
 * ('' while a multi-byte character is still unfinished). For chars mode it is
 * just a lookup.
 */
export class TokenStream {
  private readonly v: Vocab
  private dec: TextDecoder | null
  /** bytes of the character currently being assembled */
  pending: number[] = []

  constructor(v: Vocab) {
    this.v = v
    this.dec = v.kind === 'bytes' ? new TextDecoder('utf-8') : null
  }

  push(id: number): string {
    if (!this.dec) return this.v.chars[id] ?? ''
    const out = this.dec.decode(new Uint8Array([id]), { stream: true })
    // track the unfinished character so candidate labels can preview it
    if (id < 0x80) this.pending = []
    else if (id >= 0xc0) this.pending = [id]
    else if (this.pending.length) this.pending.push(id)
    if (this.pending.length && incompleteTail(this.pending) === 0) this.pending = []
    return out
  }

  /** Whatever is left at the end (an unfinished character becomes U+FFFD). */
  flush(): string {
    if (!this.dec) return ''
    this.pending = []
    return this.dec.decode()
  }

  /**
   * Label for a candidate next token, shown in the probability bars. In bytes
   * mode, if the candidate would finish the character being assembled (or is a
   * whole character by itself) it shows that character; otherwise a hex
   * placeholder. Returns [label, isPlaceholder].
   */
  candidate(id: number): [string, boolean] {
    if (!this.dec) return [this.v.chars[id] ?? '', false]
    const seq = [...this.pending, id]
    if (incompleteTail(seq) === 0) {
      const s = new TextDecoder('utf-8', { fatal: false }).decode(Uint8Array.from(seq))
      if (s && !s.includes('�')) return [s, false]
    }
    const label = byteLabel(id)
    return [label, isBytePlaceholder(label)]
  }
}

/**
 * Per-token labels for a byte sequence (attention heatmap axes): the lead byte
 * of a multi-byte character carries the whole character, its continuation
 * bytes show as '◦'.
 */
export function tokenLabels(v: Vocab, ids: ArrayLike<number>): string[] {
  if (v.kind === 'chars') return Array.from(ids as ArrayLike<number>, (t) => v.chars[t])
  const out: string[] = []
  for (let i = 0; i < ids.length; i++) {
    const b = ids[i]
    const need = (b & 0xe0) === 0xc0 ? 2 : (b & 0xf0) === 0xe0 ? 3 : (b & 0xf8) === 0xf0 ? 4 : 1
    if (need > 1 && i + need <= ids.length) {
      const seq = Array.from({ length: need }, (_, k) => ids[i + k])
      if (seq.slice(1).every((c) => (c & 0xc0) === 0x80)) {
        out.push(new TextDecoder().decode(Uint8Array.from(seq)))
        for (let k = 1; k < need; k++) out.push('◦')
        i += need - 1
        continue
      }
    }
    out.push((b & 0xc0) === 0x80 ? '◦' : byteLabel(b))
  }
  return out
}
