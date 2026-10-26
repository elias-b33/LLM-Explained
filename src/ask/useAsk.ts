import { useCallback, useEffect, useRef, useState } from 'react'
import type { Backend, LlmSpec, Plan, RetrievalConfig } from './models'
import type { ErrorKind, FromAsk, Passage, ToAsk } from './protocol'
import { clearModelCache } from './storage'

export interface FileProgress {
  group: 'embed' | 'rerank'
  file: string
  loaded: number
  total: number
  done: boolean
  cached: boolean
}

export interface Turn {
  id: number
  question: string
  answer: string
  passages: Passage[]
  lang: 'sv' | 'en' | null
  /** the hypothetical answer used for searching (HyDE), if any */
  hyde: string | null
  retrievalMs: number
  promptTokens: number
  tokens: number
  firstTokenMs: number
  decodeTps: number
  done: boolean
  stopped: boolean
  error: string | null
  workerFetches: number | null
  /** false when only retrieval ran (no WebGPU) */
  generated: boolean
}

export type Phase = 'idle' | 'loading' | 'indexing' | 'ready' | 'answering' | 'removing'

export interface AskError {
  message: string
  kind: ErrorKind
  where: string
}

export interface AskState {
  phase: Phase
  llm: string | null
  backend: Backend | null
  files: Record<string, FileProgress>
  llmProgress: { progress: number; text: string } | null
  loadMs: number
  vramMB: number | null
  storedCleared: number
  index: { done: number; total: number; textId: string | null; chunks: number; ms: number }
  turns: Turn[]
  error: AskError | null
}

const initial: AskState = {
  phase: 'idle',
  llm: null,
  backend: null,
  files: {},
  llmProgress: null,
  loadMs: 0,
  vramMB: null,
  storedCleared: 0,
  index: { done: 0, total: 0, textId: null, chunks: 0, ms: 0 },
  turns: [],
  error: null,
}

/** Owns the model worker. It exists only between "Download" and "Remove model now". */
export function useAsk() {
  const [state, setState] = useState<AskState>(initial)
  const worker = useRef<Worker | null>(null)
  const [workerAlive, setWorkerAlive] = useState(false)
  const nextId = useRef(1)
  const disposedWaiter = useRef<(() => void) | null>(null)
  /** resolvers for ask() promises, keyed by turn id */
  const waiters = useRef(new Map<number, (t: Turn) => void>())
  const turnsRef = useRef<Turn[]>([])

  const send = (m: ToAsk) => worker.current?.postMessage(m)

  /** resolve an ask() promise; React may apply the state update later, so wait a frame or two */
  const settle = (id: number, patch: Partial<Turn>) => {
    const w = waiters.current.get(id)
    if (!w) return
    waiters.current.delete(id)
    setTimeout(() => {
      const t = turnsRef.current.find((x) => x.id === id)
      if (t) w({ ...t, ...patch, done: true })
    }, 50)
  }

  const updateTurns = (s: AskState, f: (t: Turn) => Turn, id: number): AskState => {
    const turns = s.turns.map((t) => (t.id === id ? f(t) : t))
    turnsRef.current = turns
    return { ...s, turns }
  }

  const onMessage = useCallback((e: MessageEvent<FromAsk>) => {
    const m = e.data
    switch (m.type) {
      case 'file':
        setState((s) => {
          const key = `${m.group}:${m.file}`
          const prev = s.files[key]
          const next: FileProgress = m.done
            ? { ...(prev ?? { group: m.group, file: m.file, loaded: 0, total: 0 }), done: true, cached: m.cached }
            : { group: m.group, file: m.file, loaded: m.loaded, total: m.total, done: false, cached: false }
          return { ...s, files: { ...s.files, [key]: next } }
        })
        break
      case 'llmProgress':
        setState((s) => ({ ...s, llmProgress: { progress: m.progress, text: m.text } }))
        break
      case 'loaded':
        setState((s) => ({
          ...s,
          phase: 'ready',
          loadMs: Math.max(1, m.ms),
          backend: m.backend,
          vramMB: m.vramMB,
          storedCleared: m.storedCleared,
        }))
        break
      case 'indexProgress':
        setState((s) => ({ ...s, phase: 'indexing', index: { ...s.index, done: m.done, total: m.total } }))
        break
      case 'indexed':
        setState((s) => ({
          ...s,
          phase: s.loadMs > 0 ? 'ready' : s.phase,
          index: { done: m.chunks, total: m.chunks, textId: m.textId, chunks: m.chunks, ms: m.ms || s.index.ms },
        }))
        break
      case 'retrieved':
        setState((s) => updateTurns(s, (t) => ({ ...t, passages: m.passages, lang: m.lang, hyde: m.hyde, retrievalMs: m.retrievalMs }), m.id))
        break
      case 'token':
        setState((s) => updateTurns(s, (t) => ({ ...t, answer: t.answer + m.text }), m.id))
        break
      case 'answerDone':
        setState((s) => ({
          ...updateTurns(
            s,
            (t) => ({
              ...t,
              done: true,
              stopped: m.stopped,
              promptTokens: m.promptTokens,
              tokens: m.tokens,
              firstTokenMs: m.firstTokenMs,
              decodeTps: m.decodeTps,
              workerFetches: m.workerFetchesSinceLoad,
            }),
            m.id,
          ),
          phase: 'ready',
        }))
        settle(m.id, {
          stopped: m.stopped,
          promptTokens: m.promptTokens,
          tokens: m.tokens,
          firstTokenMs: m.firstTokenMs,
          decodeTps: m.decodeTps,
          workerFetches: m.workerFetchesSinceLoad,
        })
        break
      case 'disposed':
        disposedWaiter.current?.()
        break
      case 'error':
        if (m.where === 'load') {
          worker.current?.terminate()
          worker.current = null
          setWorkerAlive(false)
        }
        setState((s) => {
          const err = { message: m.message, kind: m.kind, where: m.where }
          if (m.where === 'ask') {
            const last = s.turns[s.turns.length - 1]
            if (last && !last.done) settle(last.id, { error: m.message })
            return {
              ...updateTurns(s, (t) => ({ ...t, done: true, error: m.message }), last?.id ?? -1),
              phase: 'ready',
              error: m.kind === 'other' ? null : err,
            }
          }
          // an index error before the models finished loading must not look like "ready"
          if (m.where === 'index') return { ...s, phase: s.loadMs > 0 ? 'ready' : s.phase, error: err }
          return { ...s, phase: m.where === 'load' ? 'idle' : s.phase, error: err }
        })
        break
    }
  }, [])

  const kill = useCallback(() => {
    worker.current?.terminate()
    worker.current = null
    setWorkerAlive(false)
  }, [])

  useEffect(() => kill, [kill])

  const load = useCallback(
    (llm: LlmSpec | null, plan: Plan, keep: boolean, retrieval: RetrievalConfig) => {
      kill()
      const w = new Worker(new URL('./ask.worker.ts', import.meta.url), { type: 'module', name: 'ask-the-text' })
      w.onmessage = onMessage
      w.onerror = (ev) => {
        ev.preventDefault()
        setState((s) => ({
          ...s,
          phase: 'idle',
          error: { message: ev.message || 'The model worker crashed (possibly out of memory).', kind: 'oom', where: 'load' },
        }))
      }
      worker.current = w
      setWorkerAlive(true)
      turnsRef.current = []
      setState({ ...initial, phase: 'loading', llm: llm?.id ?? null, backend: plan.backend })
      send({
        type: 'load',
        llm: llm?.id ?? null,
        noThink: Boolean(llm?.noThink),
        backend: plan.backend,
        embedDtype: plan.embedDtype,
        keep,
        retrieval,
      })
    },
    [kill, onMessage],
  )

  const index = useCallback((textId: string, text: string, title = '') => {
    setState((s) => ({ ...s, phase: 'indexing', error: null, index: { ...s.index, done: 0, total: 0, textId: null } }))
    send({ type: 'index', textId, text, title })
  }, [])

  /** ask a question; resolves with the finished turn */
  const ask = useCallback((question: string) => {
    const id = nextId.current++
    const turn: Turn = {
      id,
      question,
      answer: '',
      passages: [],
      lang: null,
      hyde: null,
      retrievalMs: 0,
      promptTokens: 0,
      tokens: 0,
      firstTokenMs: 0,
      decodeTps: 0,
      done: false,
      stopped: false,
      error: null,
      workerFetches: null,
      generated: true,
    }
    const p = new Promise<Turn>((res) => waiters.current.set(id, res))
    setState((s) => {
      const turns = [...s.turns, turn]
      turnsRef.current = turns
      return { ...s, phase: 'answering', error: null, turns }
    })
    send({ type: 'ask', id, question })
    return p
  }, [])

  const stop = useCallback(() => send({ type: 'stop' }), [])

  /** unload the model, terminate the worker, drop references and delete stored files */
  const remove = useCallback(async () => {
    setState((s) => ({ ...s, phase: 'removing' }))
    if (worker.current) {
      await new Promise<void>((resolve) => {
        const t = setTimeout(resolve, 4000)
        disposedWaiter.current = () => {
          clearTimeout(t)
          resolve()
        }
        send({ type: 'dispose' })
      })
      disposedWaiter.current = null
    }
    kill()
    const removed = await clearModelCache()
    turnsRef.current = []
    setState(initial)
    return removed
  }, [kill])

  const clearError = useCallback(() => setState((s) => ({ ...s, error: null })), [])

  return { state, load, index, ask, stop, remove, kill, clearError, workerAlive }
}
