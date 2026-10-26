import { useCallback, useEffect, useRef, useState } from 'react'
import type { FromTrainer, GenMsg, GpuInfo, InsightMsg, ReadyMsg, SampleMsg, SelfTestResult, ToHelper, ToTrainer } from './protocol'
import type { Device, SizeId, StartFrom } from './sizes'

export interface TrainerView {
  phase: 'booting' | 'loading' | 'ready' | 'training' | 'paused'
  info: ReadyMsg | null
  step: number
  charsPerSec: number
  trainMs: number
  backend: string
  sampleEvery: number
  /** per-step training loss, index = step */
  losses: number[]
  val: [number, number][]
  samples: SampleMsg[]
  /** pretrained mode: the sample written before any fine-tuning (kept for comparison) */
  before: SampleMsg | null
  insight: InsightMsg | null
  error: string | null
  /** WebGPU availability as seen from the training worker (null while probing) */
  gpu: GpuInfo | null
  selfTest: SelfTestResult | null
  selfTestRunning: boolean
  /** bumps on every commit so memoised children can depend on it */
  version: number
}

interface GenHandlers {
  onChar: (m: GenMsg) => void
  /** tail: leftover text of an unfinished character at the very end */
  onDone: (tail?: string) => void
}

const MAX_SAMPLES = 40

function spawnWorker(): Worker {
  return new Worker(new URL('./trainer.worker.ts', import.meta.url), { type: 'module', name: 'workshop-trainer' })
}

/**
 * Owns the training worker (plus data-parallel helper workers) for the page's
 * lifetime. Worker messages land in a mutable store; React state is committed
 * at most once per animation frame, so a hidden tab never piles up renders.
 */
export function useTrainer() {
  const workerRef = useRef<Worker | null>(null)
  const helpersRef = useRef<Worker[]>([])
  const store = useRef<TrainerView>({
    phase: 'booting',
    info: null,
    step: 0,
    charsPerSec: 0,
    trainMs: 0,
    backend: 'CPU · TypeScript engine',
    sampleEvery: 0,
    losses: [],
    val: [],
    samples: [],
    before: null,
    insight: null,
    error: null,
    gpu: null,
    selfTest: null,
    selfTestRunning: false,
    version: 0,
  })
  const [view, setView] = useState<TrainerView>(store.current)
  const rafRef = useRef(0)
  const genRef = useRef<{ id: number; h: GenHandlers } | null>(null)
  const genId = useRef(0)
  const selfTestWaiters = useRef<((r: SelfTestResult) => void)[]>([])

  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const commit = useCallback(() => {
    if (rafRef.current) return
    const flush = () => {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = 0
      if (timerRef.current) clearTimeout(timerRef.current)
      timerRef.current = null
      const s = store.current
      s.version++
      setView({ ...s, losses: s.losses, val: s.val, samples: s.samples })
    }
    rafRef.current = requestAnimationFrame(flush)
    // rAF is paused in hidden tabs; fall back to a slow timer so state still catches up
    timerRef.current = setTimeout(flush, 500)
  }, [])

  useEffect(() => {
    const w = spawnWorker()
    workerRef.current = w
    // data-parallel helpers: use up to half the cores, max 4 threads in total
    const cores = navigator.hardwareConcurrency || 2
    const threads = Math.max(1, Math.min(4, Math.floor(cores / 2)))
    const ports: MessagePort[] = []
    for (let i = 0; i < threads - 1; i++) {
      try {
        const hw = spawnWorker()
        const ch = new MessageChannel()
        const msg: ToHelper = { type: 'helperInit', port: ch.port2 }
        hw.postMessage(msg, [ch.port2])
        helpersRef.current.push(hw)
        ports.push(ch.port1)
      } catch {
        break
      }
    }
    const connect: ToTrainer = { type: 'connect', helpers: ports }
    w.postMessage(connect, ports)

    w.onmessage = (e: MessageEvent<FromTrainer>) => {
      const m = e.data
      const s = store.current
      switch (m.type) {
        case 'ready':
          s.info = m
          s.phase = 'ready'
          s.step = 0
          s.trainMs = 0
          s.charsPerSec = 0
          s.losses = []
          s.val = []
          s.samples = []
          s.before = null
          s.insight = null
          s.error = null
          break
        case 'progress': {
          if (m.fromStep === s.losses.length) {
            for (const l of m.losses) s.losses.push(l)
          } else if (m.fromStep < s.losses.length) {
            s.losses.length = m.fromStep
            for (const l of m.losses) s.losses.push(l)
          }
          for (const v of m.val) s.val.push(v)
          s.step = m.step
          s.charsPerSec = m.charsPerSec
          s.trainMs = m.trainMs
          s.backend = m.backend
          s.sampleEvery = m.sampleEvery
          s.phase = m.running ? 'training' : m.step > 0 ? 'paused' : 'ready'
          // new array identities so memoised consumers notice
          s.losses = s.losses.slice()
          s.val = s.val.slice()
          break
        }
        case 'sample': {
          const next = [...s.samples, m]
          // keep the very first (random) sample as the "before" picture
          if (next.length > MAX_SAMPLES) next.splice(1, 1)
          s.samples = next
          if (m.before) s.before = m
          break
        }
        case 'gen':
          if (genRef.current?.id === m.id) genRef.current.h.onChar(m)
          return
        case 'genDone':
          if (genRef.current?.id === m.id) {
            const h = genRef.current.h
            genRef.current = null
            h.onDone(m.tail)
          }
          return
        case 'insight':
          s.insight = m
          break
        case 'error':
          s.error = m.message
          break
        case 'gpuInfo':
          s.gpu = m.info
          // dev-only: ?nogpu previews the fallback UI on a machine that has WebGPU
          if (import.meta.env.DEV && new URLSearchParams(location.search).has('nogpu'))
            s.gpu = { ...m.info, status: 'none', reason: 'WebGPU is present, but no GPU adapter was offered.' }
          break
        case 'selfTest': {
          s.selfTest = m.result
          s.selfTestRunning = false
          const ws = selfTestWaiters.current
          selfTestWaiters.current = []
          for (const w of ws) w(m.result)
          break
        }
      }
      commit()
    }
    w.onerror = (e) => {
      store.current.error = e.message || 'The training worker crashed.'
      commit()
    }
    const helpers = helpersRef.current
    return () => {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = 0
      if (timerRef.current) clearTimeout(timerRef.current)
      w.terminate()
      for (const h of helpers) h.terminate()
      helpersRef.current = []
      workerRef.current = null
    }
  }, [commit])

  const send = useCallback((m: ToTrainer) => workerRef.current?.postMessage(m), [])

  const init = useCallback(
    (text: string, size: SizeId, prompt: string, device: Device, start: StartFrom = 'scratch', weights?: ArrayBuffer) => {
      store.current.phase = 'loading'
      store.current.error = null
      commit()
      // the page keeps its copy of the weights (structured clone, not transfer) for the next re-init
      send({ type: 'init', text, size, prompt, seed: 1337, device, start, weights })
    },
    [send, commit],
  )

  const start = useCallback(() => {
    store.current.phase = 'training'
    commit()
    send({ type: 'start' })
  }, [send, commit])

  const pause = useCallback(() => {
    store.current.phase = 'paused'
    commit()
    send({ type: 'pause' })
  }, [send, commit])

  const reset = useCallback(() => {
    store.current.phase = 'loading'
    commit()
    send({ type: 'reset' })
  }, [send, commit])

  const generate = useCallback(
    (prompt: string, temperature: number, length: number, h: GenHandlers) => {
      const prev = genRef.current
      if (prev) prev.h.onDone()
      const id = ++genId.current
      genRef.current = { id, h }
      send({ type: 'generate', id, prompt, temperature, length })
    },
    [send],
  )

  const stopGenerate = useCallback(() => send({ type: 'stopGenerate' }), [send])
  const requestInsight = useCallback((prompt: string) => send({ type: 'insight', prompt }), [send])

  /** Compare the GPU engine against the CPU engine; resolves with the per-tensor results. */
  const runSelfTest = useCallback((weights?: ArrayBuffer): Promise<SelfTestResult> => {
    store.current.selfTestRunning = true
    commit()
    return new Promise((resolve) => {
      selfTestWaiters.current.push(resolve)
      send({ type: 'gpuSelfTest', weights })
    })
  }, [send, commit])

  return { view, runSelfTest, init, start, pause, reset, generate, stopGenerate, requestInsight }
}
