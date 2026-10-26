/**
 * A tiny BM25 keyword index, used next to the embeddings (hybrid retrieval).
 * The embedding model alone often misses the obvious passage in old verse (e5 matched
 * "What do the witches predict?" to random Witch lines, not "All hail, Macbeth! … king
 * hereafter"), while plain word overlap finds it. The rankings are merged with
 * reciprocal rank fusion (retrieve.ts). Words are lower-cased, stop words dropped and crudely stemmed by
 * cutting them to five letters ("witches" and "witch" both become "witch").
 */

const STOP = new Set(
  (
    'the and for are but not you all any can her was one our out his has had him how who why what when where which with that this ' +
    'from they them their there then than does did doe will would shall should into upon your yours its ' +
    'och det att som han hon den inte med för har var vad vem vilken vilka vilket varför hur när till sig sin sitt sina ' +
    'hans hennes ett men eller jag mig dig oss ska skulle kan kunde heter gör gjorde finns mot från över under efter'
  ).split(' '),
)

export function terms(text: string): string[] {
  const out: string[] = []
  for (const w of text.toLowerCase().match(/\p{L}+/gu) ?? []) {
    if (w.length < 3 || STOP.has(w)) continue
    out.push(w.length > 5 ? w.slice(0, 5) : w)
  }
  return out
}

export interface KeywordIndex {
  score(query: string): Float32Array
}

export function buildKeywordIndex(docs: string[], k1 = 1.2, b = 0.75): KeywordIndex {
  const tfs = docs.map((d) => {
    const m = new Map<string, number>()
    for (const t of terms(d)) m.set(t, (m.get(t) ?? 0) + 1)
    return m
  })
  const lens = tfs.map((m) => [...m.values()].reduce((a, c) => a + c, 0))
  const avg = lens.reduce((a, c) => a + c, 0) / Math.max(1, lens.length)
  const df = new Map<string, number>()
  for (const m of tfs) for (const t of m.keys()) df.set(t, (df.get(t) ?? 0) + 1)
  const N = docs.length
  return {
    score(query: string) {
      const out = new Float32Array(N)
      const qs = [...new Set(terms(query))]
      for (const q of qs) {
        const n = df.get(q)
        if (!n) continue
        const idf = Math.log(1 + (N - n + 0.5) / (n + 0.5))
        for (let i = 0; i < N; i++) {
          const f = tfs[i].get(q)
          if (!f) continue
          out[i] += (idf * f * (k1 + 1)) / (f + k1 * (1 - b + (b * lens[i]) / avg))
        }
      }
      return out
    },
  }
}
