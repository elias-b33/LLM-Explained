import { describe, expect, it, vi } from 'vitest'
import type { Chunk } from './chunk'
import { buildKeywordIndex, terms } from './keyword'
import { cosines, retrieve, rrf, type PassageIndex } from './retrieve'

describe('BM25 keyword index', () => {
  it('normalises terms: lower case, stop words out, crude 5-letter stems', () => {
    expect(terms('What do the Witches predict for Macbeth?')).toEqual(['witch', 'predi', 'macbe'])
    expect(terms('Vad heter hans bror?')).toEqual(['bror'])
  })

  it('ranks the passage that shares rare words first', () => {
    const docs = [
      'The king rode out across the heath at dawn.',
      'All hail, Macbeth, that shalt be king hereafter! said the witches.',
      'The porter knocks and knocks and grumbles at the gate.',
      'A king, a castle, and a feast in the great hall.',
    ]
    const ix = buildKeywordIndex(docs)
    const s = ix.score('What do the witches predict? Macbeth will be king hereafter.')
    const best = Array.from(s).indexOf(Math.max(...s))
    expect(best).toBe(1)
    // a doc with no query term scores exactly 0 (sparse)
    expect(s[2]).toBe(0)
    // a common word ("king", in three docs) is worth less than a rare one ("porter", in one)
    const king = ix.score('king')
    const porter = ix.score('porter')
    expect(Math.max(...porter)).toBeGreaterThan(Math.max(...king))
    expect(Array.from(ix.score('zebra'))).toEqual([0, 0, 0, 0])
  })

  it('term frequency saturates and long documents are penalised', () => {
    const ix = buildKeywordIndex(['ghost', 'ghost ghost ghost ghost ghost ghost', 'ghost ' + 'filler words here '.repeat(20)])
    const [one, many, long] = ix.score('ghost')
    expect(many).toBeGreaterThan(one)
    expect(many).toBeLessThan(one * 3) // k1 saturation, not linear
    expect(long).toBeLessThan(one)
  })
})

/** a hand-made index: unit vectors in 4-d, so cosine = dot product */
function index(): PassageIndex {
  const chunks: Chunk[] = [
    { id: 0, header: 'Act 1, Scene 1', text: 'Thunder and lightning. Enter three Witches.', section: 0 },
    { id: 1, header: 'Act 1, Scene 1', text: 'Fair is foul, and foul is fair: hover through the fog and filthy air.', section: 0 },
    { id: 2, header: 'Act 1, Scene 3', text: 'All hail, Macbeth, that shalt be king hereafter!', section: 1 },
    { id: 3, header: 'Act 1, Scene 3', text: 'Banquo asks the witches to speak to him too.', section: 1 },
    { id: 4, header: 'Act 2, Scene 1', text: 'Is this a dagger which I see before me?', section: 2 },
  ]
  const vecs = [
    [1, 0, 0, 0],
    [0.8, 0.6, 0, 0],
    [0, 1, 0, 0],
    [0, 0.6, 0.8, 0],
    [0, 0, 0, 1],
  ]
  return {
    chunks,
    vectors: Float32Array.from(vecs.flat()),
    dim: 4,
    keywords: buildKeywordIndex(chunks.map((c) => c.text)),
  }
}

describe('hybrid retrieval', () => {
  it('cosines are dot products against every passage', () => {
    const c = cosines(index(), Float32Array.from([0, 1, 0, 0]))
    expect(Array.from(c).map((x) => Number(x.toFixed(3)))).toEqual([0, 0.6, 1, 0.6, 0])
  })

  it('reciprocal rank fusion: dense lists credit everyone, sparse lists only their matches', () => {
    const fused = rrf(
      [
        { scores: [3, 2, 1], sparse: false },
        { scores: [0, 0, 5], sparse: true },
      ],
      60,
    )
    expect(fused[0]).toBeCloseTo(1 / 60)
    expect(fused[1]).toBeCloseTo(1 / 61)
    expect(fused[2]).toBeCloseTo(1 / 62 + 1 / 60)
    expect(fused[2]).toBeGreaterThan(fused[0])
  })

  it('fuses embedding and keyword rankings and widens picks with neighbours from the same section', async () => {
    const ix = index()
    // the embedding points at passage 2; the keywords agree ("hail", "king")
    const picks = await retrieve(ix, [{ text: 'hail king', vec: Float32Array.from([0, 1, 0, 0]) }], { topK: 2, neighbours: 1 })
    expect(picks[0].chunk).toBe(2)
    // the next passage (3) is in the same scene, so it is merged in and not picked again
    expect(picks[0].covers).toEqual([2, 3])
    expect(picks[0].text).toContain('All hail')
    expect(picks[0].text).toContain('Banquo')
    expect(picks[0].cos).toBeCloseTo(1)
    expect(picks[0].keyword).toBeGreaterThan(0)
    expect(picks[0].rerank).toBeNull()
    expect(picks.flatMap((p) => p.covers)).toHaveLength(new Set(picks.flatMap((p) => p.covers)).size)
  })

  it('never borrows a neighbour from another section, and neighbours: 2 adds the previous passage too', async () => {
    const ix = index()
    const last = await retrieve(ix, [{ text: 'dagger', vec: Float32Array.from([0, 0, 0, 1]) }], { topK: 1, neighbours: 2 })
    expect(last[0].covers).toEqual([4]) // 3 is another scene
    const mid = await retrieve(ix, [{ text: 'foul fair', vec: Float32Array.from([0.8, 0.6, 0, 0]) }], { topK: 1, neighbours: 2 })
    expect(mid[0].chunk).toBe(1)
    expect(mid[0].covers).toEqual([0, 1])
  })

  it('drops the chunker overlap when joining neighbours', async () => {
    const overlap = 'hover through the fog and filthy air.'
    const ix = index()
    ix.chunks[1] = { ...ix.chunks[1], text: `${overlap} Fair is foul, and foul is fair.` }
    ix.chunks[0] = { ...ix.chunks[0], text: `Thunder. Enter three Witches. ${overlap}` }
    const [p] = await retrieve(ix, [{ text: 'thunder', vec: Float32Array.from([1, 0, 0, 0]) }], { topK: 1, neighbours: 1 })
    expect(p.covers).toEqual([0, 1])
    expect(p.text.split(overlap)).toHaveLength(2) // the shared tail appears once
  })

  it('an optional reranker re-sorts the fused candidates', async () => {
    const ix = index()
    const rerank = vi.fn(async (docs: string[]) => docs.map((d) => (d.includes('dagger') ? 10 : 0)))
    const picks = await retrieve(ix, [{ text: 'hail king', vec: Float32Array.from([0, 1, 0, 0]) }], {
      topK: 1,
      neighbours: 0,
      candidates: 5,
      rerank,
    })
    expect(rerank).toHaveBeenCalledOnce()
    expect(rerank.mock.calls[0][0][0]).toMatch(/^Act 1, Scene 3\n/) // the reranker sees header + passage
    expect(picks[0].chunk).toBe(4)
    expect(picks[0].rerank).toBe(10)
  })

  it('a second query (e.g. a hypothetical answer) adds its own rankings', async () => {
    const ix = index()
    const one = await retrieve(ix, [{ text: '', vec: Float32Array.from([1, 0, 0, 0]) }], { topK: 1, neighbours: 0 })
    expect(one[0].chunk).toBe(0)
    const two = await retrieve(
      ix,
      [
        { text: '', vec: Float32Array.from([1, 0, 0, 0]) },
        { text: 'dagger before me', vec: Float32Array.from([0, 0, 0, 1]) },
      ],
      { topK: 1, neighbours: 0 },
    )
    expect(two[0].chunk).toBe(4)
  })
})
