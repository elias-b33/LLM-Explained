/**
 * Model choices for "Ask the text".
 *
 * Chat models run on WebLLM (@mlc-ai/web-llm 0.2.86, WebGPU only). IDs and `vramMB` are copied
 * from that version's `prebuiltAppConfig` (vram_required_MB); the worker re-checks the ID
 * against the config at load time. Download sizes are the sum of the files in each
 * mlc-ai/<id> Hugging Face repo (weights shards + tokenizer + config, from the HF API) plus the
 * model's WebGPU kernel library (.wasm, Content-Length from its GitHub URL).
 *
 * Embeddings stay on Transformers.js (multilingual-e5-small), which retrieved well.
 */

export type Dtype = 'fp16' | 'fp32' | 'q8'

export interface LlmSpec {
  /** WebLLM model_id */
  id: string
  label: string
  name: string
  params: string
  bytes: number
  vramMB: number
  licence: string
  licenceUrl: string
  /** Qwen3: switch thinking mode off */
  noThink?: boolean
  note: string
}

export const LLMS: LlmSpec[] = [
  {
    id: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
    label: 'Fast · 0.5B',
    name: 'Qwen2.5-0.5B-Instruct',
    params: '0.5 billion parameters',
    bytes: 289_700_000 + 5_552_384,
    vramMB: 944.62,
    licence: 'Apache-2.0',
    licenceUrl: 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/LICENSE',
    note: 'Tiny and quick. Often misreads the excerpts or answers something else; good for seeing how small models fail.',
  },
  {
    id: 'Qwen2.5-1.5B-Instruct-q4f16_1-MLC',
    label: 'Standard · 1.5B',
    name: 'Qwen2.5-1.5B-Instruct',
    params: '1.5 billion parameters',
    bytes: 880_300_000 + 6_031_744,
    vramMB: 1629.75,
    licence: 'Apache-2.0',
    licenceUrl: 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/LICENSE',
    note: 'The default: light enough for most graphics cards. Gets simple questions right, but often paraphrases loosely or gives up on old verse.',
  },
  {
    id: 'Qwen3-4B-q4f16_1-MLC',
    label: 'XXL · 4B',
    name: 'Qwen3-4B',
    params: '4 billion parameters',
    bytes: 2_279_200_000 + 6_667_148,
    vramMB: 3431.59,
    licence: 'Apache-2.0',
    licenceUrl: 'https://huggingface.co/Qwen/Qwen3-4B/blob/main/LICENSE',
    noThink: true,
    note: 'Best in our tests: quotes the text and more often admits when the excerpts don’t say. Big download; wants a good graphics card.',
  },
]

/** XXL candidates that were evaluated (dev only, via ?llm=<id>); see the report in eval.ts */
export const CANDIDATES: LlmSpec[] = [
  {
    id: 'Qwen2.5-3B-Instruct-q4f16_1-MLC',
    label: 'Candidate · Qwen2.5 3B',
    name: 'Qwen2.5-3B-Instruct',
    params: '3 billion parameters',
    bytes: 1_748_000_000 + 6_243_281,
    vramMB: 2504.76,
    licence: 'Qwen Research License (non-commercial)',
    licenceUrl: 'https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE',
    note: '',
  },
  {
    id: 'Llama-3.2-3B-Instruct-q4f16_1-MLC',
    label: 'Candidate · Llama 3.2 3B',
    name: 'Llama-3.2-3B-Instruct',
    params: '3.2 billion parameters',
    bytes: 1_816_800_000 + 6_746_418,
    vramMB: 2263.69,
    licence: 'Llama 3.2 Community License',
    licenceUrl: 'https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/blob/main/LICENSE.txt',
    note: '',
  },
  {
    id: 'Qwen3-1.7B-q4f16_1-MLC',
    label: 'Candidate · Qwen3 1.7B',
    name: 'Qwen3-1.7B',
    params: '1.7 billion parameters',
    bytes: 984_200_000 + 6_386_879,
    vramMB: 2036.66,
    licence: 'Apache-2.0',
    licenceUrl: 'https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/LICENSE',
    noThink: true,
    note: '',
  },
]

export const DEFAULT_LLM = 'Qwen2.5-1.5B-Instruct-q4f16_1-MLC'

export function findLlm(id: string): LlmSpec | undefined {
  return [...LLMS, ...CANDIDATES].find((l) => l.id === id)
}

/* ------------------------------------------------------------------ retrieval */

export interface EncoderSpec {
  name: string
  repo: string
  licence: string
  /** bytes per dtype: ONNX weights + tokenizer/config files */
  bytes: Partial<Record<Dtype, number>>
}

/** e5 side files: tokenizer.json, tokenizer_config.json, config.json */
const E5S_SIDE = 17_082_730 + 443 + 658
const E5B_SIDE = 17_082_660 + 418 + 686

export const EMBEDDERS: Record<'small' | 'base', EncoderSpec> = {
  small: {
    name: 'multilingual-e5-small',
    repo: 'Xenova/multilingual-e5-small',
    licence: 'MIT',
    bytes: { fp16: 235_336_732 + E5S_SIDE, fp32: 470_268_533 + E5S_SIDE, q8: 118_308_185 + E5S_SIDE },
  },
  base: {
    name: 'multilingual-e5-base',
    repo: 'Xenova/multilingual-e5-base',
    licence: 'MIT',
    bytes: { fp16: 555_234_024 + E5B_SIDE, q8: 278_647_662 + E5B_SIDE },
  },
}

/** cross-encoder reranker (optional, see RETRIEVAL) */
export const RERANKER: EncoderSpec = {
  name: 'bge-reranker-v2-m3',
  repo: 'onnx-community/bge-reranker-v2-m3-ONNX',
  licence: 'Apache-2.0',
  bytes: { q8: 570_727_094 + 17_082_900 + 1_203 + 848, fp16: 1_136_209_678 + 17_082_900 + 1_203 + 848 },
}

export interface RetrievalConfig {
  embedder: keyof typeof EMBEDDERS
  /** let the chat model write a short hypothetical answer and search with it too (HyDE) */
  hyde: boolean
  rerank: boolean
  /** passages handed to the chat model */
  topK: number
  /** also include the passage before / after each pick (same scene or chapter) */
  neighbours: 0 | 1 | 2
}

/**
 * Chosen after the round-3 experiments (eval.ts, 8 questions + 4 hold-out):
 *  - embedder: multilingual-e5-small. e5-base ranked better offline (Node, q8) but not in the
 *    browser: fp16 on WebGPU retrieved worse, and q8 on WebGPU took ~150 s to index a book.
 *  - no reranker: bge-reranker-v2-m3 (571 MB, q8) added nothing at top 6 offline.
 *  - HyDE only for XXL: the 4B model's guesses pull in the right scenes; the 1.5B model's were
 *    noise ("filminställningsspel") and made answers worse.
 */
export function retrievalFor(llmId: string | null): RetrievalConfig {
  const xxl = Boolean(llmId?.startsWith('Qwen3-4B'))
  return { embedder: 'small', hyde: xxl, rerank: false, topK: xxl ? 6 : 4, neighbours: xxl ? 2 : 1 }
}

export const EMBEDDER = EMBEDDERS.small

export type Backend = 'webgpu' | 'wasm'

export interface Plan {
  /** where the embedder runs; the chat model needs WebGPU */
  backend: Backend
  embedDtype: Dtype
}

export function makePlan(backend: Backend, f16: boolean): Plan {
  if (backend === 'wasm') return { backend, embedDtype: 'q8' }
  return { backend, embedDtype: f16 ? 'fp16' : 'fp32' }
}

export function embedBytes(plan: Plan, cfg: RetrievalConfig = retrievalFor(null)): number {
  const e = EMBEDDERS[cfg.embedder]
  return (e.bytes[plan.embedDtype] ?? e.bytes.q8 ?? 0) + (cfg.rerank ? (RERANKER.bytes.q8 ?? 0) : 0)
}

export const fmtMB = (b: number) =>
  b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : b >= 1e6 ? `${Math.round(b / 1e6)} MB` : `${Math.max(1, Math.round(b / 1e3))} kB`

export const fmtVram = (mb: number) => (mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${Math.round(mb)} MB`)
