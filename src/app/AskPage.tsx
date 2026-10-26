import { useEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from 'react'
import { motion } from 'motion/react'
import { Button } from '@/primitives/Button'
import { DemoFrame } from '@/primitives/DemoFrame'
import { DeepDive } from '@/primitives/DeepDive'
import { cn } from '@/lib/cn'
import { CORPORA, CUSTOM_MAX } from '@/lab/corpora'
import {
  DEFAULT_LLM,
  EMBEDDERS,
  LLMS,
  RERANKER,
  embedBytes,
  findLlm,
  fmtMB,
  fmtVram,
  makePlan,
  retrievalFor,
  type Dtype,
  type LlmSpec,
  type RetrievalConfig,
} from '@/ask/models'
import { EXAMPLES } from '@/ask/examples'
import type { GpuProbe } from '@/ask/protocol'
import { cachedModelFiles, clearModelCache, probe, readKeep, writeKeep } from '@/ask/storage'
import { useAsk, type AskState, type FileProgress } from '@/ask/useAsk'
import { TurnView } from '@/ask/AnswerView'
import { EVAL, scoreTurn, type EvalRow } from '@/ask/eval'

type TextId = 'shakespeare' | 'strindberg' | 'grimm' | 'custom'
const CUSTOM_MIN_ASK = 300

const devParams = import.meta.env.DEV && typeof location !== 'undefined' ? new URLSearchParams(location.search) : null
/** dev-only: ?nogpu previews the no-WebGPU path; ?llm=<WebLLM id> loads an evaluated candidate */
const forceNoGpu = Boolean(devParams?.has('nogpu'))
const devLlm = devParams?.get('llm') ? findLlm(devParams.get('llm')!) : undefined
/** dev-only: ?ret=emb:small,hyde:0,rerank:1,topk:6,nb:2 overrides the retrieval settings */
function devRetrieval(base: RetrievalConfig): RetrievalConfig {
  const raw = devParams?.get('ret')
  if (!raw) return base
  const o = Object.fromEntries(raw.split(',').map((kv) => kv.split(':')))
  return {
    embedder: o.emb === 'small' || o.emb === 'base' ? o.emb : base.embedder,
    hyde: o.hyde !== undefined ? o.hyde === '1' : base.hyde,
    rerank: o.rerank !== undefined ? o.rerank === '1' : base.rerank,
    topK: o.topk ? Number(o.topk) : base.topK,
    neighbours: o.nb !== undefined ? (Number(o.nb) as 0 | 1 | 2) : base.neighbours,
  }
}

function Step({ n, title, hint, children }: { n: string; title: string; hint: string; children: ReactNode }) {
  return (
    <section className="min-w-0 space-y-3">
      <div>
        <p className="exhibit-label">Step {n}</p>
        <h2 className="mt-0.5 font-display text-[1.7rem] leading-tight text-fg">{title}</h2>
        <p className="mt-1 text-sm text-fg-muted">{hint}</p>
      </div>
      {children}
    </section>
  )
}

function Choice({
  selected,
  onClick,
  children,
  disabled,
}: {
  selected: boolean
  onClick: () => void
  children: ReactNode
  disabled?: boolean
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={selected}
      disabled={disabled}
      onClick={onClick}
      className={cn(
        'relative flex h-full w-full min-w-0 flex-col items-start border p-3.5 text-left transition-colors enabled:cursor-pointer disabled:cursor-not-allowed disabled:opacity-50',
        selected
          ? 'border-accent bg-surface shadow-[0_0_0_3px_color-mix(in_srgb,var(--accent)_14%,transparent)]'
          : 'border-border bg-surface/60 enabled:hover:border-brass enabled:hover:bg-surface',
      )}
    >
      {children}
    </button>
  )
}

function Bar({ value, max, label, right }: { value: number; max: number; label: ReactNode; right: ReactNode }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0
  return (
    <div className="min-w-0">
      <div className="flex items-baseline justify-between gap-3 text-xs">
        <span className="min-w-0 truncate font-mono text-fg-muted">{label}</span>
        <span className="shrink-0 font-mono text-fg-subtle tabular-nums">{right}</span>
      </div>
      <div className="mt-1 h-2 overflow-hidden rounded-full border border-border bg-surface-2">
        <div className="h-full bg-brass transition-[width] duration-200" style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

function FileBars({ files, title }: { files: FileProgress[]; title: string }) {
  if (!files.length) return null
  return (
    <div className="min-w-0 space-y-2">
      <p className="text-[10px] font-semibold tracking-[0.16em] text-fg-subtle uppercase">{title}</p>
      {files.map((f) => (
        <Bar
          key={f.file}
          label={f.file}
          value={f.done ? 1 : f.loaded}
          max={f.done ? 1 : f.total}
          right={
            f.done
              ? f.cached
                ? `${f.total > 0 ? fmtMB(f.total) + ' · ' : ''}from cache`
                : `${f.total > 0 ? fmtMB(f.total) + ' · ' : ''}done`
              : f.total > 0
                ? `${fmtMB(f.loaded)} / ${fmtMB(f.total)}`
                : fmtMB(f.loaded)
          }
        />
      ))}
    </div>
  )
}

const ERROR_HELP: Record<string, string> = {
  oom: 'The browser ran out of memory. Close other tabs, or try a smaller model.',
  lost: 'The graphics card reset (WebGPU “device lost”), usually from running out of video memory. Remove the model and try a smaller one.',
  network: 'A download failed. Check your connection and try again.',
  quota:
    'The browser would not let this site store the model files (storage quota exceeded). WebLLM has to store them while loading. Free up disk space, or try a smaller model.',
  other: 'Try removing the model and downloading it again. If it keeps failing, try a smaller model.',
}

/** Offline eval (src/ask/eval.ts): 8 tuning + 4 held-out questions on the Swedish and English texts; ½ = partly right. */
const MEASURED = [
  { model: 'XXL · 4B', right: 5.5 },
  { model: 'Standard · 1.5B', right: 3.5 },
]

function MeasuredResults() {
  return (
    <div className="mx-auto mt-5 max-w-md border border-border bg-surface px-5 py-4 text-left text-sm">
      <p className="exhibit-label">Measured on 12 test questions</p>
      <ul className="mt-3 space-y-2">
        {MEASURED.map(({ model, right }) => (
          <li key={model}>
            <Bar value={right} max={12} label={model} right={`${right} / 12 correct`} />
          </li>
        ))}
      </ul>
      <p className="mt-3 text-xs leading-relaxed text-fg-muted">
        The search finds the right passage for 7 of 8 questions. Most mistakes happen after that: the model misreads what it
        was given. A bigger model reads better, which is why the XXL size exists.
      </p>
    </div>
  )
}

export function AskPage() {
  const asker = useAsk()
  const { state } = asker
  const [textId, setTextId] = useState<TextId>('shakespeare')
  const [draft, setDraft] = useState('')
  const [customText, setCustomText] = useState<{ id: string; text: string } | null>(null)
  const [llmId, setLlmId] = useState<string>(devLlm?.id ?? DEFAULT_LLM)
  const [keep, setKeep] = useState(readKeep)
  const [cachedCount, setCachedCount] = useState(0)
  const [gpu, setGpu] = useState<GpuProbe | null>(null)
  const [question, setQuestion] = useState('')
  const [notice, setNotice] = useState<string | null>(null)
  const [storageWarn, setStorageWarn] = useState<string | null>(null)
  const texts = useRef(new Map<string, string>())
  const customSeq = useRef(0)
  const turnsEnd = useRef<HTMLDivElement>(null)

  useEffect(() => {
    document.title = 'Ask the text · LLM Explainer'
  }, [])

  // Ephemeral by default: on arrival, delete anything a previous visit left behind ...
  useEffect(() => {
    void (async () => {
      if (!readKeep()) await clearModelCache()
      setCachedCount(await cachedModelFiles())
    })()
    void probe(forceNoGpu).then(setGpu)
  }, [])

  // ... and, best effort, when leaving (the browser may not let this finish)
  const { kill } = asker
  useEffect(() => {
    const onHide = () => {
      if (readKeep()) return
      kill()
      void clearModelCache()
    }
    window.addEventListener('pagehide', onHide)
    return () => window.removeEventListener('pagehide', onHide)
  }, [kill])

  const hasGpu = gpu?.status === 'ok'
  const canRunLlm = hasGpu && gpu.f16
  const plan = useMemo(() => {
    const p = makePlan(hasGpu ? 'webgpu' : 'wasm', gpu?.f16 ?? false)
    const edt = devParams?.get('edt') // dev-only: embedder dtype override
    return edt === 'q8' || edt === 'fp16' ? { ...p, embedDtype: edt as Dtype } : p
  }, [gpu, hasGpu])
  const choices: LlmSpec[] = devLlm && !LLMS.includes(devLlm) ? [...LLMS, devLlm] : LLMS
  const spec = findLlm(llmId)!
  const retrieval = devRetrieval(retrievalFor(canRunLlm ? spec.id : null))
  const searchBytes = embedBytes(plan, retrieval)
  const totalBytes = (canRunLlm ? spec.bytes : 0) + searchBytes
  const embedSpec = EMBEDDERS[retrieval.embedder]

  const loaded = state.phase === 'ready' || state.phase === 'indexing' || state.phase === 'answering'
  const busy = state.phase === 'loading' || state.phase === 'removing'
  const loadedSpec = state.llm ? findLlm(state.llm) : undefined
  const retrievalOnly = loaded && !state.llm

  const currentTextKey = textId === 'custom' ? (customText?.id ?? null) : textId

  // once the models are in memory, (re)index whichever text is picked
  const { index } = asker
  useEffect(() => {
    if (state.phase !== 'ready' || !state.loadMs || !currentTextKey || state.index.textId === currentTextKey) return
    let cancelled = false
    void (async () => {
      let text: string | undefined
      if (textId === 'custom') text = customText?.text
      else {
        text = texts.current.get(textId)
        if (!text) {
          text = await CORPORA.find((c) => c.id === textId)!.load()
          texts.current.set(textId, text)
        }
      }
      const title = textId === 'custom' ? '' : `${CORPORA.find((c) => c.id === textId)!.title} (${CORPORA.find((c) => c.id === textId)!.author})`
      if (!cancelled && text) index(currentTextKey, text, title)
    })()
    return () => {
      cancelled = true
    }
  }, [state.phase, state.loadMs, state.index.textId, currentTextKey, textId, customText, index])

  useEffect(() => {
    if (state.phase === 'answering') turnsEnd.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [state.turns.length, state.phase])

  // dev-only: window.__askEval() runs the 8-question eval set with the loaded model
  const stateRef = useRef<AskState>(state)
  useEffect(() => {
    stateRef.current = state
  })
  const { ask } = asker
  useEffect(() => {
    if (!devParams) return
    const w = window as unknown as { __askEval?: () => Promise<EvalRow[]>; __evalRows?: EvalRow[] }
    w.__askEval = async () => {
      const rows: EvalRow[] = []
      w.__evalRows = rows
      const model = stateRef.current.llm ?? 'retrieval-only'
      for (const item of EVAL) {
        setTextId(item.text)
        const t0 = performance.now()
        while (
          !(stateRef.current.phase === 'ready' && stateRef.current.index.textId === item.text) &&
          performance.now() - t0 < 120_000
        )
          await new Promise((r) => setTimeout(r, 200))
        const turn = await ask(item.question)
        rows.push(scoreTurn(model, item, turn))
      }
      console.table(rows.map((r) => ({ q: r.question, langOk: r.langOk, cites: r.cites, hits: r.keywordHits.join(','), tps: r.decodeTps })))
      return rows
    }
    return () => {
      delete w.__askEval
    }
  }, [ask])

  const onKeep = async (v: boolean) => {
    setKeep(v)
    writeKeep(v)
    if (!v) {
      const n = await clearModelCache()
      setNotice(n ? `Deleted ${n} stored model files from this browser.` : 'Nothing was stored.')
    } else {
      setNotice(loaded ? 'Takes effect the next time a model downloads.' : null)
    }
    setCachedCount(await cachedModelFiles())
  }

  const onDownload = async () => {
    setNotice(null)
    setStorageWarn(null)
    asker.clearError()
    // WebLLM must store the files while loading. If the whole quota is too small it can never
    // work, so say so up front. (Chrome reports deleted Cache Storage as "used" for a while,
    // so a low *free* estimate is not treated as fatal; a real quota error is caught later.)
    if (canRunLlm) {
      try {
        const est = await navigator.storage.estimate()
        if (est.quota !== undefined && est.quota < spec.bytes * 1.05) {
          setStorageWarn(
            `This browser lets the site store at most ${fmtMB(est.quota)}, but ${spec.name} needs ${fmtMB(spec.bytes)} while it loads. ` +
              'Browsers allow a share of the free disk: free up space, or pick a smaller model.',
          )
          return
        }
      } catch {
        /* no estimate: just try */
      }
    }
    asker.load(canRunLlm ? spec : null, plan, keep, retrieval)
  }

  const onRemove = async () => {
    const n = await asker.remove()
    setCachedCount(await cachedModelFiles())
    setNotice(`Model removed: unloaded from the GPU, worker terminated${n ? `, ${n} stored files deleted` : ''}.`)
  }

  const onAsk = (e?: FormEvent, q = question) => {
    e?.preventDefault()
    const t = q.trim()
    if (!t || state.phase !== 'ready' || state.index.textId !== currentTextKey) return
    void asker.ask(t)
    setQuestion('')
  }

  const embedFiles = Object.values(state.files)
  const indexed = state.index.textId !== null && state.index.textId === currentTextKey
  const canAsk = state.phase === 'ready' && indexed
  const lastTurn = state.turns[state.turns.length - 1]
  const examples = textId === 'custom' ? [] : EXAMPLES[textId]
  const corpus = CORPORA.find((c) => c.id === textId)

  return (
    <article className="mx-auto max-w-5xl px-4 py-10 sm:px-8 sm:py-14" data-model-worker={asker.workerAlive ? 'running' : 'none'}>
      <motion.header
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="mb-10 text-center"
      >
        <p className="exhibit-label">The Annex</p>
        <h1 className="mt-4 font-display text-[2.6rem] leading-[1.05] text-fg sm:text-6xl">
          Ask the <span className="holo-text">text</span>
        </h1>
        <div className="ornament mx-auto mt-5 w-48" aria-hidden="true">
          <span />
        </div>
        <p className="mx-auto mt-5 max-w-2xl font-display text-xl leading-snug text-fg-muted italic sm:text-[1.35rem]">
          Download a small, real chat model into this tab and ask questions about a book. It looks up the right passages first, then
          answers from them.
        </p>
        <p className="mx-auto mt-5 flex max-w-2xl flex-wrap items-center justify-center gap-2 text-sm text-fg-muted">
          <span className="tag text-success">Private</span>
          Everything runs on your computer. Your text and questions never leave it.
        </p>
        <p className="mt-2 flex items-center justify-center gap-2 text-sm text-fg-muted">
          <span className="tag text-warning">Experiment</span>
          Small models, often wrong. Here is how often, measured.
        </p>
        <MeasuredResults />
      </motion.header>

      <div className="grid grid-cols-1 gap-10">
        <Step n="I" title="Pick a text" hint="The model will only be allowed to answer from passages of this text.">
          <div role="radiogroup" aria-label="Text" className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
            {CORPORA.map((c) => (
              <Choice
                key={c.id}
                selected={textId === c.id}
                onClick={() => setTextId(c.id)}
                disabled={state.phase === 'indexing' || state.phase === 'answering'}
              >
                <span className="font-display text-lg leading-tight text-fg">{c.title}</span>
                <span className="text-xs text-fg-muted">{c.author}</span>
                <span className="tag mt-2 text-brass-ink">{c.lang}</span>
              </Choice>
            ))}
            <Choice
              selected={textId === 'custom'}
              onClick={() => setTextId('custom')}
              disabled={state.phase === 'indexing' || state.phase === 'answering'}
            >
              <span className="font-display text-lg leading-tight text-fg">Paste your own</span>
              <span className="text-xs text-fg-muted">An article, notes, a chapter… up to 1 MB</span>
              <span className="tag mt-2 text-accent">Your text</span>
            </Choice>
          </div>
          {textId === 'custom' && (
            <div className="space-y-2 border border-border bg-surface/70 p-3">
              <label htmlFor="ask-custom" className="text-sm text-fg-muted">
                Paste at least {CUSTOM_MIN_ASK} characters. It stays in this tab.
              </label>
              <textarea
                id="ask-custom"
                value={draft}
                onChange={(e) => setDraft(e.target.value.slice(0, CUSTOM_MAX))}
                rows={6}
                className="w-full resize-y border border-border-strong bg-surface p-2 font-mono text-xs text-fg outline-none focus:border-accent"
                placeholder="Paste a text here…"
              />
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className={cn('font-mono text-xs', draft.length >= CUSTOM_MIN_ASK ? 'text-success' : 'text-fg-subtle')}>
                  {draft.length.toLocaleString('en-US')} characters{draft.length >= CUSTOM_MAX && ' (capped at 1 MB)'}
                </span>
                <Button
                  variant="secondary"
                  disabled={draft.length < CUSTOM_MIN_ASK || state.phase === 'indexing' || state.phase === 'answering'}
                  onClick={() => setCustomText({ id: `custom-${++customSeq.current}`, text: draft })}
                >
                  {customText ? 'Use updated text' : 'Use this text'}
                </Button>
              </div>
            </div>
          )}
        </Step>

        <Step n="II" title="Pick a model" hint="Real instruction-tuned chat models, shrunk to 4-bit weights and compiled for your graphics card.">
          <GpuBanner gpu={gpu} />
          <div role="radiogroup" aria-label="Model" className="grid grid-cols-1 gap-2.5 md:grid-cols-3">
            {choices.map((l) => (
              <Choice key={l.id} selected={llmId === l.id} onClick={() => setLlmId(l.id)} disabled={loaded || busy || !canRunLlm}>
                <span className="flex w-full flex-wrap items-baseline justify-between gap-2">
                  <span className="font-display text-xl leading-tight text-fg">{l.label}</span>
                  {l.id === DEFAULT_LLM && <span className="tag text-brass-ink">Default</span>}
                  {l.label.startsWith('XXL') && <span className="tag text-accent">Good GPU</span>}
                </span>
                <span className="text-xs text-fg-muted">
                  {l.name} · {l.params}
                </span>
                <dl className="mt-2.5 grid w-full grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs">
                  <dt className="text-fg-subtle">Download</dt>
                  <dd className="font-mono text-fg">{fmtMB(l.bytes)}</dd>
                  <dt className="text-fg-subtle">Video memory</dt>
                  <dd className="font-mono text-fg">{fmtVram(l.vramMB)}</dd>
                  <dt className="text-fg-subtle">Licence</dt>
                  <dd className="text-fg">{l.licence}</dd>
                </dl>
                <span className="mt-2.5 text-xs leading-relaxed text-fg-muted">{l.note}</span>
              </Choice>
            ))}
          </div>
          <p className="font-mono text-[11px] text-fg-subtle">
            Chat model: WebLLM, q4f16 (4-bit weights, 16-bit maths) on WebGPU · search: {embedSpec.name} ({plan.embedDtype}
            {retrieval.rerank && ` + ${RERANKER.name} reranker`}, {fmtMB(searchBytes)}) on {plan.backend === 'webgpu' ? 'WebGPU' : 'the CPU'}
            {canRunLlm && ` · ${retrieval.topK} passages${retrieval.neighbours ? ' with neighbours' : ''}${retrieval.hyde ? ', HyDE query' : ''}`} · video memory figures are WebLLM’s own
            estimates
          </p>

          <div className="space-y-3 border border-border bg-surface/70 p-4">
            <label className="flex cursor-pointer items-start gap-2.5 text-sm text-fg">
              <input type="checkbox" className="mt-1" checked={keep} onChange={(e) => void onKeep(e.target.checked)} />
              <span>
                Keep the model for next time
                <span className="block text-xs text-fg-muted">
                  Off: removed when you leave, and double-checked every time you come back. (WebLLM has to save the files in browser
                  storage while it loads; with this off they are deleted as soon as the model is on your graphics card.) On: they
                  stay in this site’s browser storage, so the next visit skips the download. Turning it off deletes them.
                  {cachedCount > 0 && ` ${cachedCount} model files are stored right now.`}
                </span>
              </span>
            </label>
            {!loaded && (
              <div className="flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  className="btn-museum disabled:cursor-not-allowed disabled:opacity-50"
                  disabled={busy || !gpu}
                  onClick={() => void onDownload()}
                >
                  {state.phase === 'loading'
                    ? 'Downloading…'
                    : canRunLlm
                      ? `Download model (≈ ${fmtMB(totalBytes)})`
                      : `Download the search model only (≈ ${fmtMB(totalBytes)})`}
                </button>
                <span className="text-xs text-fg-subtle">Nothing is downloaded until you press this.</span>
              </div>
            )}
            <p className="text-xs text-fg-muted">
              <b className="text-warning">Memory:</b> loading needs roughly the download size in RAM for a moment, plus the video
              memory shown above. Close heavy tabs first on a laptop.
            </p>
          </div>

          {(state.phase === 'loading' || loaded) && (state.llmProgress || embedFiles.length > 0) && (
            <div className="grid gap-4 border border-border bg-surface/70 p-4 md:grid-cols-2">
              <FileBars files={embedFiles.filter((f) => f.group === 'embed')} title={`Embeddings · ${embedSpec.repo}`} />
              <FileBars files={embedFiles.filter((f) => f.group === 'rerank')} title={`Reranker · ${RERANKER.repo}`} />
              {state.llmProgress && (
                <div className="min-w-0 space-y-2">
                  <p className="text-[10px] font-semibold tracking-[0.16em] text-fg-subtle uppercase">
                    Chat model · mlc-ai/{state.llm}
                  </p>
                  <Bar
                    label={state.llmProgress.text}
                    value={state.llmProgress.progress}
                    max={1}
                    right={`${Math.round(state.llmProgress.progress * 100)}%`}
                  />
                </div>
              )}
              {loaded && (
                <p className="font-mono text-[11px] text-fg-subtle md:col-span-2">
                  Ready in {(state.loadMs / 1000).toFixed(1)} s
                  {state.llm ? ` · chat model on WebGPU (WebLLM estimates ${fmtVram(state.vramMB ?? 0)} of video memory)` : ' · search only'}
                  {state.llm && !keep && ' · downloaded files already deleted from browser storage'}.
                </p>
              )}
            </div>
          )}

          {state.error && (
            <div role="alert" className="border p-3 text-sm" style={{ borderColor: 'var(--danger)', color: 'var(--danger)' }}>
              <p className="font-semibold">Something went wrong{state.error.where === 'load' ? ' while loading the model' : ''}.</p>
              <p className="mt-1 text-fg">{ERROR_HELP[state.error.kind]}</p>
              <p className="mt-1 font-mono text-[11px] break-words text-fg-subtle">{state.error.message}</p>
            </div>
          )}
          {storageWarn && (
            <div role="alert" className="border p-3 text-sm" style={{ borderColor: 'var(--danger)' }}>
              <p className="font-semibold" style={{ color: 'var(--danger)' }}>
                Not enough browser storage.
              </p>
              <p className="mt-1 text-fg">{storageWarn}</p>
            </div>
          )}
          {notice && <p className="text-xs text-fg-muted">{notice}</p>}
        </Step>
      </div>

      <div className="mt-12">
        <p className="exhibit-label">Step III</p>
        <h2 className="mt-0.5 mb-3 font-display text-[1.7rem] leading-tight text-fg">Ask</h2>
        <DemoFrame
          title="The reading room"
          status="real"
          caption="Your question is compared with every passage by meaning and by shared words, and the closest passages, with their neighbours, are handed to the model together with the question."
        >
          {!loaded && state.phase !== 'loading' && <p className="text-sm text-fg-muted">Download a model above to start asking.</p>}
          {state.phase === 'loading' && <p className="text-sm text-fg-muted">Downloading the models…</p>}
          {loaded && (
            <div className="space-y-4">
              {retrievalOnly && (
                <p className="text-sm text-fg-muted">
                  Search only: without WebGPU there is no chat model, but you can still see which passages answer your question best.
                </p>
              )}
              {!indexed || state.phase === 'indexing' ? (
                <Bar
                  label={`Reading ${textId === 'custom' ? 'your text' : corpus?.title} into passages and embedding them…`}
                  value={state.index.done}
                  max={state.index.total}
                  right={state.index.total ? `${state.index.done} / ${state.index.total}` : '…'}
                />
              ) : (
                <p className="font-mono text-[11px] text-fg-subtle">
                  {state.index.chunks} passages of ~350–450 characters, embedded
                  {state.index.ms > 0 && ` in ${(state.index.ms / 1000).toFixed(1)} s`}.
                </p>
              )}
              {textId === 'custom' && !customText && <p className="text-sm text-fg-muted">Paste a text in Step I first.</p>}

              {examples.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {examples.map((q) => (
                    <button
                      key={q}
                      type="button"
                      disabled={!canAsk}
                      onClick={() => onAsk(undefined, q)}
                      className="cursor-pointer rounded-sm border border-border bg-surface px-2.5 py-1 text-left text-xs text-fg-muted transition-colors enabled:hover:border-brass enabled:hover:text-fg disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              )}

              <form onSubmit={onAsk} className="flex flex-col gap-2 sm:flex-row">
                <label htmlFor="ask-q" className="sr-only">
                  Your question
                </label>
                <input
                  id="ask-q"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value.slice(0, 500))}
                  placeholder={textId === 'strindberg' ? 'Ställ en fråga om texten…' : 'Ask a question about the text…'}
                  className="min-w-0 flex-1 border border-border-strong bg-surface px-3 py-2 text-sm text-fg outline-none focus:border-accent"
                  autoComplete="off"
                />
                {state.phase === 'answering' ? (
                  <Button variant="secondary" type="button" onClick={asker.stop}>
                    ■ Stop
                  </Button>
                ) : (
                  <Button variant="primary" type="submit" disabled={!canAsk || !question.trim()}>
                    {retrievalOnly ? 'Find passages' : 'Ask'}
                  </Button>
                )}
              </form>

              <div className="space-y-3">
                {[...state.turns].reverse().map((t) => (
                  <TurnView
                    key={t.id}
                    turn={t}
                    retrievalOnly={retrievalOnly}
                    answering={state.phase === 'answering' && t === lastTurn}
                    onStop={asker.stop}
                  />
                ))}
                <div ref={turnsEnd} />
              </div>
            </div>
          )}

          <div className="mt-5 flex flex-wrap items-center gap-3 border-t border-border pt-4">
            <button
              type="button"
              className="btn-outline disabled:cursor-not-allowed disabled:opacity-50"
              onClick={() => void onRemove()}
              disabled={state.phase === 'idle' && !state.error && cachedCount === 0}
            >
              {state.phase === 'removing' ? 'Removing…' : 'Remove model now'}
            </button>
            <span className="text-xs text-fg-subtle">
              Unloads the model from the GPU, stops the worker and deletes any stored files.
              {loadedSpec && ` Loaded: ${loadedSpec.name}.`}
            </span>
          </div>
        </DemoFrame>
      </div>

      <div className="mt-10 space-y-3">
        <DeepDive title="How does this work?">
          <p>
            This is <b>retrieval-augmented generation</b> (RAG). First the text is cut into passages of about 400 characters,
            following its own structure: every passage is labelled with its act and scene, or its chapter, and in a play with who
            is speaking. An <b>embedding model</b> turns each passage into a list of 384 numbers that captures its meaning, so
            passages about similar things end up close together. That happens once, right after the download.
          </p>
          <p>
            When you ask something (with the XXL model), the chat model first writes a short guess at the answer. Your question,
            and that guess, are compared with every passage, by meaning (cosine similarity of the numbers) and by shared words (BM25), and the
            rankings are merged. The best passages, each with its neighbours, are pasted into the prompt, numbered, with strict
            instructions: answer in one to three sentences, quote the text, cite the passage, and say “The text doesn’t say” if the
            answer isn’t there. The guess is only used for searching, never shown as the answer. The model never sees the whole
            book, only these passages, which is why it can miss things that are spread across the text.
          </p>
        </DeepDive>
        <DeepDive title="Why can this model answer when the Workshop’s can’t?">
          <p>
            The Workshop’s model learns from one book for a few minutes, so it only picks up spelling and rhythm. The Qwen models
            here were <b>pretrained</b> on 18 to 36 trillion tokens of web pages, books and code, by predicting the next token over
            and over on thousands of GPUs. That is where they learned grammar, facts and how arguments hang together, in dozens of
            languages.
          </p>
          <p>
            Then they were <b>instruction-tuned</b>: trained further on examples of questions with good answers, plus feedback on
            which answers people preferred. That turns a text-continuer into an assistant that follows instructions like “cite the
            excerpts”.
          </p>
          <p>
            <b>Scale</b> matters too: half a billion to four billion parameters, against at most ten million in the Workshop. Even
            so, these are tiny next to frontier models, which is why they still misread and make things up.
          </p>
        </DeepDive>
      </div>

      <aside className="mt-12 border-t border-border pt-6 text-xs leading-relaxed text-fg-subtle">
        <p className="exhibit-label mb-2">Credits</p>
        <ul className="list-disc space-y-1 pl-5">
          {LLMS.map((l) => (
            <li key={l.id}>
              <b>{l.name}</b> by the Qwen team, Alibaba Cloud,{' '}
              <a className="underline decoration-brass underline-offset-2 hover:text-fg" href={l.licenceUrl} target="_blank" rel="noreferrer">
                {l.licence}
              </a>
              . Compiled for the browser by MLC (
              <a
                className="underline decoration-brass underline-offset-2 hover:text-fg"
                href={`https://huggingface.co/mlc-ai/${l.id}`}
                target="_blank"
                rel="noreferrer"
              >
                mlc-ai/{l.id}
              </a>
              ).
            </li>
          ))}
          <li>
            <a className="underline decoration-brass underline-offset-2 hover:text-fg" href="https://github.com/mlc-ai/web-llm" target="_blank" rel="noreferrer">
              WebLLM
            </a>{' '}
            (@mlc-ai/web-llm 0.2.86) by the MLC team, Apache-2.0. Model files come from Hugging Face, the compiled GPU kernels from
            MLC’s GitHub.
          </li>
          <li>
            <b>multilingual-e5-small</b> by Wang et al. (Microsoft, intfloat/multilingual-e5-small), MIT licence; ONNX conversion{' '}
            <a className="underline decoration-brass underline-offset-2 hover:text-fg" href="https://huggingface.co/Xenova/multilingual-e5-small" target="_blank" rel="noreferrer">
              Xenova/multilingual-e5-small
            </a>
            , run with{' '}
            <a className="underline decoration-brass underline-offset-2 hover:text-fg" href="https://github.com/huggingface/transformers.js" target="_blank" rel="noreferrer">
              Transformers.js
            </a>{' '}
            v3 (Apache-2.0) and ONNX Runtime Web (MIT), whose WebAssembly file is served by this site.
          </li>
          <li>Texts are public domain, from Project Gutenberg (see the Workshop’s credits).</li>
        </ul>
      </aside>
    </article>
  )
}

function GpuBanner({ gpu }: { gpu: GpuProbe | null }) {
  if (!gpu) return <p className="text-xs text-fg-subtle">Checking for a graphics card…</p>
  if (gpu.status === 'ok' && gpu.f16) {
    return (
      <p className="flex flex-wrap items-center gap-2 text-xs text-fg-muted">
        <span className="tag text-success">WebGPU</span>
        Runs on your graphics card{gpu.adapter && ` (${gpu.adapter})`}.
      </p>
    )
  }
  const reason =
    gpu.status === 'ok'
      ? 'Your graphics card does not offer 16-bit shaders (shader-f16), which these compiled models need.'
      : gpu.reason
  return (
    <div role="alert" className="border p-3 text-sm" style={{ borderColor: 'var(--warning)' }}>
      <p className="font-semibold text-warning">The chat models need WebGPU, which isn’t available here.</p>
      <p className="mt-1 text-xs text-fg-muted">
        {reason} There is no CPU fallback for the chat model: it would take minutes per answer. You can still download the small
        search model ({EMBEDDERS.small.name}) and see which passages match your question. For answers, use Chrome or Edge on a computer
        with a graphics card.
      </p>
    </div>
  )
}
