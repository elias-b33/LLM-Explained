import type { Backend, Dtype, RetrievalConfig } from './models'

export interface Passage {
  /** citation number shown to the model and the reader, 1-based */
  n: number
  chunkId: number
  /** cosine similarity of the embeddings */
  score: number
  /** BM25 keyword score (0 = no shared words) */
  keyword: number
  /** cross-encoder relevance (only when the reranker is on) */
  rerank: number | null
  /** where it sits, e.g. "Act 1, Scene 7 · MACBETH" */
  header: string
  /** the passage, plus its neighbours when neighbour expansion added them */
  text: string
}

export interface GpuProbe {
  /** ok = hardware WebGPU; software = fallback adapter only; none = no WebGPU */
  status: 'ok' | 'software' | 'none'
  f16: boolean
  adapter: string
  reason: string
}

export type ErrorKind = 'oom' | 'lost' | 'network' | 'quota' | 'other'

/* page -> worker */
export type ToAsk =
  /** llm null = retrieval only (no WebGPU: the chat model can't run) */
  | {
      type: 'load'
      llm: string | null
      noThink: boolean
      backend: Backend
      embedDtype: Dtype
      keep: boolean
      retrieval: RetrievalConfig
    }
  | { type: 'index'; textId: string; text: string; title: string }
  | { type: 'ask'; id: number; question: string }
  | { type: 'stop' }
  | { type: 'dispose' }

/* worker -> page */
export type FromAsk =
  | { type: 'file'; group: 'embed' | 'rerank'; file: string; loaded: number; total: number; done: boolean; cached: boolean }
  | { type: 'llmProgress'; progress: number; text: string }
  | { type: 'loaded'; ms: number; backend: Backend; llm: string | null; vramMB: number | null; storedCleared: number }
  | { type: 'indexProgress'; done: number; total: number }
  | { type: 'indexed'; textId: string; chunks: number; ms: number }
  | { type: 'retrieved'; id: number; passages: Passage[]; lang: 'sv' | 'en'; hyde: string | null; retrievalMs: number }
  | { type: 'token'; id: number; text: string }
  | {
      type: 'answerDone'
      id: number
      promptTokens: number
      tokens: number
      firstTokenMs: number
      decodeTps: number
      stopped: boolean
      workerFetchesSinceLoad: number
    }
  | { type: 'disposed' }
  | { type: 'error'; where: 'load' | 'index' | 'ask' | 'gpu'; message: string; kind: ErrorKind }
