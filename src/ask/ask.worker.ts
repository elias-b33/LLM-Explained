/**
 * "Ask the text" worker: owns the chat model (WebLLM), the embedding model and optional
 * reranker (Transformers.js), the passage index and generation, so the page never blocks.
 * Everything lives in this worker's memory; terminating it frees the lot.
 *
 * Both libraries are imported only here, so they end up in this worker's chunk and are
 * fetched only on /ask, when the visitor presses Download.
 */
import { CreateMLCEngine, prebuiltAppConfig, type AppConfig, type MLCEngine } from '@mlc-ai/web-llm'
import {
  AutoModelForSequenceClassification,
  AutoTokenizer,
  env,
  pipeline,
  type FeatureExtractionPipeline,
  type PreTrainedModel,
  type PreTrainedTokenizer,
  type Tensor,
} from '@huggingface/transformers'
import { chunkText, chunkWithHeader, type Chunk } from './chunk'
import { buildKeywordIndex } from './keyword'
import { opfsCache } from './opfsCache'
import { EMBEDDERS, RERANKER, type RetrievalConfig } from './models'
import { retrieve, type PassageIndex } from './retrieve'
import { clearModelStorage } from './wipe'
import type { ErrorKind, FromAsk, Passage, ToAsk } from './protocol'

interface WorkerScope {
  postMessage(msg: unknown, transfer?: Transferable[]): void
  onmessage: ((e: MessageEvent) => void) | null
  fetch: typeof fetch
}
const scope = self as unknown as WorkerScope
const post = (m: FromAsk) => scope.postMessage(m)

const EMBED_BATCH = 16
const MAX_NEW_TOKENS = 160
const TEMPERATURE = 0.15
/** WebLLM's prebuilt Qwen models are compiled for a 4096-token window; set it explicitly */
const CONTEXT_WINDOW = 4096

/* ------------------------------------------------------------ fetch watching */

let keepCache = false
/** Transformers.js files that actually came over the network during this load (vs. OPFS) */
const fetchedFiles = new Set<string>()
/** fetch() calls the worker made after loading finished (privacy check; should stay 0) */
let fetchesAfterLoad = 0
let loadedFlag = false

// Transformers.js v3 calls the global `fetch(url)` and has no fetch hook, so it is wrapped:
// unless the visitor opted in to keeping the model, its files are fetched with
// cache: 'no-store' and never land in the HTTP cache. WebLLM's own downloads go through
// Cache Storage (it always caches); see wipe.ts for how those are removed.
const nativeFetch = scope.fetch.bind(self)
scope.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
  const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
  if (loadedFlag) fetchesAfterLoad++
  const m = /huggingface\.co\/(?:Xenova|onnx-community)\/[^/]+\/resolve\/[^/]+\/(.+)$/.exec(url)
  if (m) fetchedFiles.add(decodeURIComponent(m[1]))
  return nativeFetch(input, keepCache ? init : { ...init, cache: 'no-store' })
}) as typeof fetch

env.allowLocalModels = false
env.allowRemoteModels = true
env.useBrowserCache = false
env.useFSCache = false
env.useCustomCache = false
env.backends.onnx.logLevel = 'error'
// ONNX Runtime's wasm runtime is bundled by Vite and served from this site (no CDN).
if (env.backends.onnx.wasm) {
  ;(env.backends.onnx.wasm as { wasmPaths?: unknown }).wasmPaths = undefined
}

/* --------------------------------------------------------------------- state */

let engine: MLCEngine | null = null
let embedder: FeatureExtractionPipeline | null = null
let rerankTok: PreTrainedTokenizer | null = null
let rerankModel: PreTrainedModel | null = null
let cfg: RetrievalConfig | null = null
let noThink = false
let chunks: Chunk[] = []
let index: PassageIndex | null = null
let indexedId = ''
let bookTitle = ''
let generating = false
let stopRequested = false

function classify(err: unknown): { message: string; kind: ErrorKind } {
  const message = err instanceof Error ? `${err.name !== 'Error' ? err.name + ': ' : ''}${err.message}` : String(err)
  if (/quota/i.test(message)) return { message, kind: 'quota' }
  if (/^\d+$/.test(message.trim()))
    return { message: `ONNX Runtime error ${message} (usually means it ran out of memory)`, kind: 'oom' }
  // check device loss first: WebLLM's DeviceLostError message also mentions OOM
  if (/device.*lost|lost.*device|GPUDevice|device is destroyed|Device was lost/i.test(message)) return { message, kind: 'lost' }
  if (/out of memory|oom|Aborted\(\)|bad_alloc|allocat|array buffer|memory access out of bounds|RangeError|Maximum.*size/i.test(message))
    return { message, kind: 'oom' }
  if (/fetch|network|Could not locate|Failed to load|NetworkError|load failed/i.test(message)) return { message, kind: 'network' }
  return { message, kind: 'other' }
}

function fileProgress(group: 'embed' | 'rerank') {
  return (p: { status: string; file?: string; loaded?: number; total?: number }) => {
    if (!p.file) return
    if (p.status === 'progress') {
      post({ type: 'file', group, file: p.file, loaded: p.loaded ?? 0, total: p.total ?? 0, done: false, cached: false })
    } else if (p.status === 'done') {
      post({ type: 'file', group, file: p.file, loaded: -1, total: -1, done: true, cached: !fetchedFiles.has(p.file) })
    }
  }
}

let dim = 0
async function embed(texts: string[]): Promise<Float32Array> {
  const out = (await embedder!(texts, { pooling: 'mean', normalize: true })) as Tensor
  dim = out.dims[out.dims.length - 1]
  const copy = new Float32Array(out.data as Float32Array)
  out.dispose?.()
  return copy
}

async function rerankScores(question: string, docs: string[]): Promise<number[]> {
  const out: number[] = []
  for (let i = 0; i < docs.length; i += 5) {
    const batch = docs.slice(i, i + 5)
    const inputs = rerankTok!(new Array(batch.length).fill(question), {
      text_pair: batch,
      padding: true,
      truncation: true,
      max_length: 384,
    })
    const { logits } = (await rerankModel!(inputs)) as { logits: Tensor }
    out.push(...Array.from(logits.data as Float32Array))
  }
  return out
}

/* ------------------------------------------------------------------ prompting */

const SV_WORDS =
  /\b(vad|vem|vilka|vilken|vilket|varför|hur|när|var|är|och|det|den|som|han|hon|inte|att|för|med|om|till|sin|sitt|hans|hennes|heter|gör|finns)\b/gi
const EN_WORDS = /\b(what|who|whom|why|how|when|where|which|is|are|does|did|the|and|of|to|his|her|in|do)\b/gi

/** the question's language: å/ä/ö, or more Swedish than English function words */
export function questionLang(q: string): 'sv' | 'en' {
  if (/[åäöÅÄÖ]/.test(q)) return 'sv'
  const sv = (q.match(SV_WORDS) ?? []).length
  const en = (q.match(EN_WORDS) ?? []).length
  return sv > en ? 'sv' : 'en'
}

const NOT_FOUND = { sv: 'Det står inte i texten.', en: 'The text doesn’t say.' }

function excerptText(passages: Passage[]): string {
  return passages.map((p) => `[${p.n}]${p.header ? ` (${p.header})` : ''}\n${p.text}`).join('\n\n')
}

function buildMessages(question: string, passages: Passage[], lang: 'sv' | 'en') {
  const ex = excerptText(passages)
  if (lang === 'sv') {
    return [
      {
        role: 'system' as const,
        content:
          'Du svarar på frågor om en bok och får bara använda de numrerade utdragen som användaren ger dig. ' +
          'Svara på svenska med 1–3 meningar. Citera texten ordagrant inom citattecken där det går. ' +
          'Avsluta varje mening med numret på utdraget den bygger på, till exempel [1]. ' +
          'Läs noga vem som är vem: förväxla inte personer som nämns i samma mening. ' +
          `Bara om inget av utdragen innehåller svaret: svara exakt ${NOT_FOUND.sv} och inget annat.`,
      },
      {
        role: 'user' as const,
        content: `Utdrag:\n\n${ex}\n\nFråga: ${question}\n\nSvara på svenska, med 1–3 meningar, bara utifrån utdragen, och ange källa som [1].`,
      },
    ]
  }
  return [
    {
      role: 'system' as const,
      content:
        'You answer questions about a book using only the numbered excerpts the user gives you. ' +
        'Answer in English in 1–3 sentences. Quote the text directly, in quotation marks, where possible. ' +
        'End each sentence with the number of the excerpt it is based on, like [1]. ' +
        'Check who says what: in a play, the speaker’s name stands on the line before their words. ' +
        `Only if none of the excerpts contains the answer: reply exactly ${NOT_FOUND.en} and nothing else.`,
    },
    {
      role: 'user' as const,
      content: `Excerpts:\n\n${ex}\n\nQuestion: ${question}\n\nAnswer in English, in 1–3 sentences, using only the excerpts, and cite them like [1].`,
    },
  ]
}

/** HyDE: a short hypothetical passage that would answer the question, used only for searching */
async function hypothetical(question: string, lang: 'sv' | 'en'): Promise<string> {
  if (!engine) return ''
  await engine.resetChat()
  const book = bookTitle ? (lang === 'sv' ? ` ur ${bookTitle}` : ` from ${bookTitle}`) : ''
  const r = await engine.chat.completions.create({
    messages: [
      {
        role: 'system',
        content:
          lang === 'sv'
            ? `Skriv ett kort stycke (2 meningar)${book} som skulle besvara frågan, med de namn och ord som boken troligen använder. Ingen inledning.`
            : `Write a short passage (2 sentences)${book} that would answer the question, using the names and words the text itself would likely use. No preamble.`,
      },
      { role: 'user', content: question },
    ],
    temperature: 0,
    max_tokens: 70,
    ...(noThink ? { extra_body: { enable_thinking: false } } : {}),
  })
  return (r.choices[0]?.message?.content ?? '').replace(/<think>[\s\S]*?<\/think>/g, '').trim()
}

/* ------------------------------------------------------------------ handlers */

async function load(m: Extract<ToAsk, { type: 'load' }>) {
  const t0 = performance.now()
  keepCache = m.keep
  noThink = m.noThink
  cfg = m.retrieval
  // opt-in persistence for the Transformers.js models goes to OPFS (see opfsCache.ts)
  env.useCustomCache = m.keep
  env.customCache = m.keep ? opfsCache : null
  fetchedFiles.clear()
  // without WebGPU the encoders run on the CPU; proxy keeps this worker responsive
  if (m.backend === 'wasm' && env.backends.onnx.wasm) env.backends.onnx.wasm.proxy = true

  // (cast: the full pipeline() overload union is too large for tsc)
  const makePipeline = pipeline as unknown as (
    task: 'feature-extraction',
    model: string,
    opts: Record<string, unknown>,
  ) => Promise<FeatureExtractionPipeline>
  embedder = await makePipeline('feature-extraction', EMBEDDERS[cfg.embedder].repo, {
    session_options: { logSeverityLevel: 3 },
    dtype: m.embedDtype,
    device: m.backend,
    progress_callback: fileProgress('embed'),
  })
  if (cfg.rerank) {
    rerankTok = await AutoTokenizer.from_pretrained(RERANKER.repo, { progress_callback: fileProgress('rerank') })
    rerankModel = await AutoModelForSequenceClassification.from_pretrained(RERANKER.repo, {
      session_options: { logSeverityLevel: 3 },
      dtype: 'q8',
      device: m.backend,
      progress_callback: fileProgress('rerank'),
    })
  }

  let vramMB: number | null = null
  let storedCleared = 0
  if (m.llm) {
    const record = prebuiltAppConfig.model_list.find((r) => r.model_id === m.llm)
    if (!record) throw new Error(`Model ${m.llm} is not in WebLLM's prebuilt list.`)
    vramMB = record.vram_required_MB ?? null
    const appConfig: AppConfig = { ...prebuiltAppConfig, cacheBackend: 'cache' }
    engine = await CreateMLCEngine(
      m.llm,
      { appConfig, initProgressCallback: (r) => post({ type: 'llmProgress', progress: r.progress, text: r.text }) },
      { context_window_size: CONTEXT_WINDOW },
    )
    // Ephemeral by default: WebLLM has to save the files in Cache Storage while loading,
    // but once the weights are on the GPU they are no longer needed, so delete them now.
    if (!m.keep) storedCleared = await clearModelStorage()
  }
  loadedFlag = true
  fetchesAfterLoad = 0
  post({ type: 'loaded', ms: performance.now() - t0, backend: m.backend, llm: m.llm, vramMB, storedCleared })
}

async function buildIndex(textId: string, text: string, title: string) {
  if (!embedder) throw new Error('Model not loaded')
  const t0 = performance.now()
  const cs = chunkText(text)
  chunks = []
  index = null
  indexedId = ''
  const texts = cs.map(chunkWithHeader)
  let buf: Float32Array | null = null
  for (let i = 0; i < cs.length; i += EMBED_BATCH) {
    const v = await embed(texts.slice(i, i + EMBED_BATCH).map((t) => `passage: ${t}`))
    if (!buf) buf = new Float32Array(cs.length * dim)
    buf.set(v, i * dim)
    post({ type: 'indexProgress', done: Math.min(cs.length, i + EMBED_BATCH), total: cs.length })
  }
  chunks = cs
  index = { chunks: cs, vectors: buf ?? new Float32Array(), dim, keywords: buildKeywordIndex(texts) }
  indexedId = textId
  bookTitle = title
  post({ type: 'indexed', textId, chunks: cs.length, ms: performance.now() - t0 })
}

async function ask(id: number, question: string) {
  if (!embedder || !index || !cfg) throw new Error('Load the model and index a text first.')
  const tStart = performance.now()
  const lang = questionLang(question)
  generating = true
  try {
    const hyde = cfg.hyde && engine ? await hypothetical(question, lang) : ''
    const queries = [{ text: question, vec: await embed([`query: ${question}`]) }]
    if (hyde) queries.push({ text: hyde, vec: await embed([`query: ${hyde}`]) })
    const picks = await retrieve(index, queries, {
      topK: cfg.topK,
      neighbours: cfg.neighbours,
      candidates: 20,
      rerank: cfg.rerank && rerankModel ? (docs) => rerankScores(question, docs) : undefined,
    })
    const passages: Passage[] = picks.map((p, k) => ({
      n: k + 1,
      chunkId: chunks[p.chunk].id,
      score: p.cos,
      keyword: p.keyword,
      rerank: p.rerank,
      header: chunks[p.chunk].header,
      text: p.text,
    }))
    post({ type: 'retrieved', id, passages, lang, hyde: hyde || null, retrievalMs: performance.now() - tStart })

    const done = (o: Partial<Extract<FromAsk, { type: 'answerDone' }>>) =>
      post({
        type: 'answerDone',
        id,
        promptTokens: 0,
        tokens: 0,
        firstTokenMs: 0,
        decodeTps: 0,
        stopped: false,
        workerFetchesSinceLoad: fetchesAfterLoad,
        ...o,
      })
    if (!engine) {
      // retrieval only (no WebGPU)
      done({})
      return
    }
    if (stopRequested) {
      done({ stopped: true })
      return
    }

    let stopped = false
    let tFirst = 0
    let usage: { prompt_tokens: number; completion_tokens: number; extra?: { decode_tokens_per_s?: number } } | null = null
    // every question starts from a clean conversation
    await engine.resetChat()
    const stream = await engine.chat.completions.create({
      messages: buildMessages(question, passages, lang),
      stream: true,
      stream_options: { include_usage: true },
      temperature: TEMPERATURE,
      top_p: 0.9,
      max_tokens: MAX_NEW_TOKENS,
      ...(noThink ? { extra_body: { enable_thinking: false } } : {}),
    })
    // With thinking off, WebLLM prepends an empty "<think></think>" block to Qwen3's reply; hold
    // back the start of the stream until that block (if any) has passed.
    let head = ''
    let headDone = !noThink
    for await (const chunk of stream) {
      let delta = chunk.choices[0]?.delta?.content
      if (delta && !headDone) {
        head += delta
        const t = head.trimStart()
        if (t.startsWith('<think>') || '<think>'.startsWith(t)) {
          const end = t.indexOf('</think>')
          if (end < 0) continue
          delta = t.slice(end + '</think>'.length).trimStart()
        } else delta = head
        headDone = true
      }
      if (delta) {
        if (!tFirst) tFirst = performance.now()
        post({ type: 'token', id, text: delta })
      }
      if (chunk.choices[0]?.finish_reason === 'abort') stopped = true
      if (chunk.usage) usage = chunk.usage as typeof usage
    }
    done({
      promptTokens: usage?.prompt_tokens ?? 0,
      tokens: usage?.completion_tokens ?? 0,
      firstTokenMs: tFirst ? tFirst - tStart : 0,
      decodeTps: usage?.extra?.decode_tokens_per_s ?? 0,
      stopped: stopped || stopRequested,
    })
  } finally {
    generating = false
    stopRequested = false
  }
}

async function dispose() {
  try {
    if (generating) await engine?.interruptGenerate()
    await engine?.unload()
  } catch {
    /* already gone */
  }
  for (const m of [embedder, rerankModel]) {
    try {
      await m?.dispose()
    } catch {
      /* already gone */
    }
  }
  engine = null
  embedder = null
  rerankModel = null
  rerankTok = null
  chunks = []
  index = null
  indexedId = ''
  post({ type: 'disposed' })
}

scope.onmessage = (e: MessageEvent<ToAsk>) => {
  const m = e.data
  switch (m.type) {
    case 'load':
      load(m).catch((err) => post({ type: 'error', where: 'load', ...classify(err) }))
      break
    case 'index':
      if (m.textId === indexedId && index) {
        post({ type: 'indexed', textId: m.textId, chunks: chunks.length, ms: 0 })
        break
      }
      buildIndex(m.textId, m.text, m.title).catch((err) => post({ type: 'error', where: 'index', ...classify(err) }))
      break
    case 'ask':
      ask(m.id, m.question).catch((err) => post({ type: 'error', where: 'ask', ...classify(err) }))
      break
    case 'stop':
      if (generating) {
        stopRequested = true
        void engine?.interruptGenerate()
      }
      break
    case 'dispose':
      void dispose()
      break
  }
}
