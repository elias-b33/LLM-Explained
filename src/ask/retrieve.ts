/**
 * Retrieval pipeline (pure; the worker and the offline eval harness both use it):
 *   1. score every passage against each query text (the question, plus an optional
 *      hypothetical answer) with embeddings and with BM25 keywords,
 *   2. merge all rankings with reciprocal rank fusion,
 *   3. optionally re-sort the top candidates with a cross-encoder,
 *   4. hand the top K to the model, each widened with its neighbouring passages
 *      from the same scene or chapter.
 */
import { chunkWithHeader, type Chunk } from './chunk'
import type { KeywordIndex } from './keyword'

export interface PassageIndex {
  chunks: Chunk[]
  vectors: Float32Array
  dim: number
  keywords: KeywordIndex
}

export interface Pick {
  chunk: number
  cos: number
  keyword: number
  rerank: number | null
  /** passage text with neighbours merged in */
  text: string
  /** chunk ids included (pick + neighbours) */
  covers: number[]
}

export interface RetrieveOptions {
  topK: number
  neighbours: 0 | 1 | 2
  candidates?: number
  rerank?: (docs: string[]) => Promise<number[]>
}

function ranks(s: ArrayLike<number>): Int32Array {
  const idx = Array.from({ length: s.length }, (_, i) => i).sort((a, b) => s[b] - s[a])
  const r = new Int32Array(s.length)
  idx.forEach((i, pos) => (r[i] = pos))
  return r
}

/** reciprocal rank fusion; a sparse list only credits documents it actually matched */
export function rrf(lists: { scores: ArrayLike<number>; sparse: boolean }[], k = 60): Float32Array {
  const n = lists[0]?.scores.length ?? 0
  const out = new Float32Array(n)
  for (const l of lists) {
    const r = ranks(l.scores)
    for (let i = 0; i < n; i++) if (!l.sparse || l.scores[i] > 0) out[i] += 1 / (k + r[i])
  }
  return out
}

export function cosines(ix: PassageIndex, q: Float32Array): Float32Array {
  const out = new Float32Array(ix.chunks.length)
  for (let i = 0; i < ix.chunks.length; i++) {
    let s = 0
    const o = i * ix.dim
    for (let d = 0; d < ix.dim; d++) s += q[d] * ix.vectors[o + d]
    out[i] = s
  }
  return out
}

/** append b to a, dropping the overlap the chunker put at the start of b */
function joinOverlap(a: string, b: string): string {
  const max = Math.min(200, a.length, b.length)
  for (let len = max; len >= 20; len--) if (a.endsWith(b.slice(0, len))) return a + b.slice(len)
  return `${a}\n${b}`
}

export async function retrieve(
  ix: PassageIndex,
  queries: { text: string; vec: Float32Array }[],
  o: RetrieveOptions,
): Promise<Pick[]> {
  const lists: { scores: ArrayLike<number>; sparse: boolean }[] = []
  const cos0 = cosines(ix, queries[0].vec)
  const kw0 = ix.keywords.score(queries[0].text)
  for (const [k, q] of queries.entries()) {
    lists.push({ scores: k === 0 ? cos0 : cosines(ix, q.vec), sparse: false })
    lists.push({ scores: k === 0 ? kw0 : ix.keywords.score(q.text), sparse: true })
  }
  const fused = rrf(lists)
  let order = Array.from({ length: ix.chunks.length }, (_, i) => i).sort((a, b) => fused[b] - fused[a])
  let rr: Map<number, number> | null = null
  if (o.rerank) {
    const cand = order.slice(0, o.candidates ?? 20)
    const s = await o.rerank(cand.map((i) => chunkWithHeader(ix.chunks[i])))
    rr = new Map(cand.map((i, k) => [i, s[k]]))
    order = [...cand].sort((a, b) => rr!.get(b)! - rr!.get(a)!)
  }

  const used = new Set<number>()
  const picks: Pick[] = []
  for (const i of order) {
    if (picks.length >= o.topK) break
    if (used.has(i)) continue
    const c = ix.chunks[i]
    const same = (j: number) => j >= 0 && j < ix.chunks.length && ix.chunks[j].section === c.section && !used.has(j)
    const covers = [i]
    let text = c.text
    if (o.neighbours >= 2 && same(i - 1)) {
      text = joinOverlap(ix.chunks[i - 1].text, text)
      covers.unshift(i - 1)
    }
    if (o.neighbours >= 1 && same(i + 1)) {
      text = joinOverlap(text, ix.chunks[i + 1].text)
      covers.push(i + 1)
    }
    covers.forEach((j) => used.add(j))
    picks.push({ chunk: i, cos: cos0[i], keyword: kw0[i], rerank: rr?.get(i) ?? null, text, covers })
  }
  return picks
}
