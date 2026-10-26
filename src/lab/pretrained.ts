/**
 * Lazy download of pretrained weights from public/models/<size>.bin, with
 * progress. A missing file is a normal state ("coming soon"), not an error:
 * a 404, an HTML page (Vercel's SPA rewrite and Vite's dev fallback both answer
 * unknown paths with index.html) or anything else without the magic header all
 * count as missing.
 */
import { hasWeightsMagic, NotWeightsError, readWeightsHeader, type WeightsHeader } from './engine/weights.ts'
import type { SizeId } from './sizes'

export type PretrainedState =
  | { status: 'loading'; loaded: number; total: number | null }
  | { status: 'ready'; buf: ArrayBuffer; header: WeightsHeader; bytes: number }
  | { status: 'missing' }
  | { status: 'error'; message: string }

export const weightsUrl = (id: SizeId) => `${import.meta.env.BASE_URL}models/${id}.bin`

export async function fetchWeights(
  id: SizeId,
  onProgress: (loaded: number, total: number | null) => void,
  signal?: AbortSignal,
): Promise<PretrainedState> {
  let res: Response
  try {
    res = await fetch(weightsUrl(id), { signal })
  } catch (err) {
    if (signal?.aborted) throw err
    return { status: 'error', message: `could not reach the server (${String(err)})` }
  }
  if (!res.ok || !res.body) return { status: 'missing' }
  if ((res.headers.get('content-type') ?? '').includes('text/html')) {
    void res.body.cancel()
    return { status: 'missing' }
  }
  const len = Number(res.headers.get('content-length'))
  const total = Number.isFinite(len) && len > 0 ? len : null
  const reader = res.body.getReader()
  const chunks: Uint8Array[] = []
  let loaded = 0
  let checked = false
  onProgress(0, total)
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    loaded += value.length
    if (!checked && loaded >= 8) {
      checked = true
      const head = new Uint8Array(8)
      let k = 0
      for (const c of chunks) for (let i = 0; i < c.length && k < 8; i++) head[k++] = c[i]
      if (!hasWeightsMagic(head)) {
        void reader.cancel()
        return { status: 'missing' }
      }
    }
    onProgress(loaded, total)
  }
  if (!checked) return { status: 'missing' }
  const out = new Uint8Array(loaded)
  let o = 0
  for (const c of chunks) {
    out.set(c, o)
    o += c.length
  }
  try {
    const header = readWeightsHeader(out.buffer)
    return { status: 'ready', buf: out.buffer, header, bytes: loaded }
  } catch (err) {
    if (err instanceof NotWeightsError) return { status: 'missing' }
    return { status: 'error', message: `the weights file is damaged (${String(err)})` }
  }
}
