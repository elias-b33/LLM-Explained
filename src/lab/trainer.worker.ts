/**
 * Training worker. One file, two roles:
 *  - trainer: owns the model, the data and the optimizer; talks to the page.
 *  - helper:  computes gradients for a slice of each batch (data parallelism),
 *             connected to the trainer through a MessagePort.
 * The model lives only in this tab's memory and vanishes when it closes.
 */
import { Decoder, GPT, sampleFrom, topK, type GPTConfig } from './engine/gpt.ts'
import { Rng } from './engine/rng.ts'
import { buildVocab, decode, encode, type Vocab } from './engine/data.ts'
import { getSize, type ModelSize } from './sizes'
import type { FromHelper, FromTrainer, ToHelper, ToTrainer } from './protocol'

interface WorkerScope {
  postMessage(msg: unknown, transfer?: Transferable[]): void
  onmessage: ((e: MessageEvent) => void) | null
}
const scope = self as unknown as WorkerScope
const post = (m: FromTrainer) => scope.postMessage(m)

const SAMPLE_CHARS = 200
const SAMPLE_TEMP = 0.8
const FLUSH_MS = 150

/* =================================================================== helper */

function runHelper(port: MessagePort) {
  let model: GPT | null = null
  port.onmessage = (e: MessageEvent<ToHelper>) => {
    const m = e.data
    try {
      if (m.type === 'cfg') {
        model = new GPT(m.cfg, 1)
      } else if (m.type === 'job' && model) {
        model.params.set(m.params)
        model.zeroGrad()
        const loss = model.forward(m.x, m.y, m.B, m.T)
        model.backward()
        const grads = (model.grads as Float32Array).slice()
        const out: FromHelper = { type: 'grads', seq: m.seq, grads, loss }
        port.postMessage(out, [grads.buffer])
      }
    } catch (err) {
      const out: FromHelper = { type: 'helperError', message: String(err) }
      port.postMessage(out)
    }
  }
}

/* ================================================================== trainer */

let helpers: MessagePort[] = []
const pending = new Map<number, (r: FromHelper) => void>()
let seqCounter = 0
let helpersBroken = false

function connectHelpers(ports: MessagePort[]) {
  helpers = ports
  for (const p of ports) {
    p.onmessage = (e: MessageEvent<FromHelper>) => {
      const m = e.data
      if (m.type === 'helperError') {
        helpersBroken = true
        for (const r of pending.values()) r(m)
        pending.clear()
        return
      }
      const r = pending.get(m.seq)
      if (r) {
        pending.delete(m.seq)
        r(m)
      }
    }
  }
}

function callHelper(port: MessagePort, job: Omit<Extract<ToHelper, { type: 'job' }>, 'type' | 'seq'>): Promise<FromHelper> {
  const seq = ++seqCounter
  return new Promise((resolve) => {
    pending.set(seq, resolve)
    const msg: ToHelper = { type: 'job', seq, ...job }
    port.postMessage(msg, [job.params.buffer, job.x.buffer, job.y.buffer])
  })
}

// ---- state
let epoch = 0
let vocab: Vocab | null = null
let ids: Uint16Array = new Uint16Array(0)
let split = 0
let size: ModelSize = getSize('tiny')
let model: GPT | null = null
let promptIds: number[] = []
let promptText = ''
let seed = 1337
let rng = new Rng(1)
let running = false
let looping = false
let step = 0
let trainMs = 0
/** wall-clock time spent running (includes evals, samples, yielding) */
let wallMs = 0
let lossBuf: number[] = []
let lossFrom = 0
/** absolute step index of lossBuf[0] */
let lossBase = 0
let valBuf: [number, number][] = []
let lastFlush = 0
let rate: { t: number; chars: number }[] = []
let totalChars = 0
let sampleEvery = 0
let lastSampleAt = 0
let chunkStart = 0
const SAMPLE_EVERY_MS = 5000
let lastEvalAt = 0
let evalCost = 0
let lastLoss = NaN
let mode: 'probe' | 'single' | 'parallel' = 'single'
let probe = { single: 0, singleN: 0, parallel: 0, parallelN: 0 }
let x = new Uint16Array(0)
let y = new Uint16Array(0)
let gen: { id: number; dec: Decoder; left: number; temp: number; rng: Rng } | null = null
let genTimer: ReturnType<typeof setTimeout> | null = null

const threadsTotal = () => (helpersBroken ? 1 : helpers.length + 1)
const backendLabel = () => {
  if (mode === 'probe') return `CPU · TypeScript engine · ${threadsTotal()} threads available, picks 1 or ${threadsTotal()} on the first steps`
  const n = mode === 'parallel' ? threadsTotal() : 1
  return `CPU · TypeScript engine · ${n} thread${n === 1 ? '' : 's'}`
}

function buildModel() {
  epoch++
  running = false
  stopGen()
  const v = vocab!
  const cfg: GPTConfig = { vocabSize: v.chars.length, ctx: size.ctx, nLayer: size.nLayer, nHead: size.nHead, dModel: size.dModel }
  model = new GPT(cfg, seed)
  for (const h of helpers) {
    const msg: ToHelper = { type: 'cfg', cfg }
    h.postMessage(msg)
  }
  rng = new Rng(seed + 1)
  step = 0
  trainMs = 0
  wallMs = 0
  lossBuf = []
  lossFrom = 0
  lossBase = 0
  valBuf = []
  rate = []
  totalChars = 0
  sampleEvery = 0
  lastSampleAt = 0
  lastEvalAt = 0
  evalCost = 0
  lastLoss = NaN
  mode = helpers.length > 0 && !helpersBroken ? 'probe' : 'single'
  probe = { single: 0, singleN: 0, parallel: 0, parallelN: 0 }
  x = new Uint16Array(size.batch * size.ctx)
  y = new Uint16Array(size.batch * size.ctx)

  post({
    type: 'ready',
    vocab: v.chars,
    params: model.numParams,
    trainChars: split,
    valChars: ids.length - split,
    prompt: promptText,
    threadsAvailable: threadsTotal(),
  })
  const v0 = evalVal()
  if (!Number.isNaN(v0)) valBuf.push([0, v0])
  emitSample(Number.isNaN(v0) ? Math.log(v.chars.length) : v0)
  flush(true)
}

function fillBatch() {
  const T = size.ctx
  const B = size.batch
  const hi = split - T - 1
  for (let b = 0; b < B; b++) {
    const o = rng.int(Math.max(1, hi))
    for (let t = 0; t < T; t++) {
      x[b * T + t] = ids[o + t]
      y[b * T + t] = ids[o + t + 1]
    }
  }
}

/** Mean loss on a fixed set of windows from the held-out 10% (so the curve isn't noisy). */
function evalVal(): number {
  const m = model!
  const T = size.ctx
  const B = size.batch
  const span = ids.length - split - T - 1
  if (span < 1) return NaN
  const nb = size.id === 'tiny' ? 4 : 2
  const vx = new Uint16Array(B * T)
  const vy = new Uint16Array(B * T)
  let tot = 0
  for (let k = 0; k < nb; k++) {
    for (let b = 0; b < B; b++) {
      const o = split + Math.floor(((k * B + b) / (nb * B)) * span)
      for (let t = 0; t < T; t++) {
        vx[b * T + t] = ids[o + t]
        vy[b * T + t] = ids[o + t + 1]
      }
    }
    tot += m.forward(vx, vy, B, T)
  }
  return tot / nb
}

function shares(B: number, k: number): number[] {
  const out = new Array<number>(k).fill(Math.floor(B / k))
  for (let i = 0; i < B % k; i++) out[i]++
  return out
}

function singleStep(m: GPT): number {
  m.zeroGrad()
  const loss = m.forward(x, y, size.batch, size.ctx)
  m.backward()
  return loss
}

async function parallelStep(m: GPT): Promise<number> {
  const B = size.batch
  const T = size.ctx
  const sh = shares(B, helpers.length + 1)
  let row = sh[0]
  const jobs: Promise<{ r: FromHelper; n: number }>[] = []
  helpers.forEach((port, j) => {
    const n = sh[j + 1]
    if (n === 0) return
    const job = {
      params: (m.params as Float32Array).slice(),
      x: x.slice(row * T, (row + n) * T),
      y: y.slice(row * T, (row + n) * T),
      B: n,
      T,
    }
    row += n
    jobs.push(callHelper(port, job).then((r) => ({ r, n })))
  })
  m.zeroGrad()
  const own = m.forward(x.subarray(0, sh[0] * T), y.subarray(0, sh[0] * T), sh[0], T)
  m.backward()
  const results = await Promise.all(jobs)
  const g = m.grads
  const w0 = sh[0] / B
  for (let i = 0; i < g.length; i++) g[i] *= w0
  let loss = own * w0
  for (const { r, n } of results) {
    if (r.type !== 'grads') throw new Error('helper failed')
    const w = n / B
    const hg = r.grads
    for (let i = 0; i < g.length; i++) g[i] += w * hg[i]
    loss += w * r.loss
  }
  return loss
}

async function trainStep(myEpoch: number) {
  const m = model!
  fillBatch()
  const t0 = performance.now()
  let useParallel = mode === 'parallel'
  if (mode === 'probe') useParallel = step >= 5
  let loss: number
  if (useParallel && !helpersBroken) {
    try {
      loss = await parallelStep(m)
    } catch {
      helpersBroken = true
      mode = 'single'
      loss = singleStep(m)
    }
  } else {
    loss = singleStep(m)
  }
  if (myEpoch !== epoch) return
  m.clipGrads(1)
  const warm = Math.min(1, (step + 1) / 30)
  m.step({ lr: size.lr * warm, beta1: 0.9, beta2: 0.99, eps: 1e-8, weightDecay: 0.01 })
  const dt = performance.now() - t0
  trainMs += dt
  // the benchmark: steps 2-4 single-threaded vs 6-8 data-parallel, keep the faster
  if (mode === 'probe') {
    if (step >= 2 && step <= 4) {
      probe.single += dt
      probe.singleN++
    } else if (step >= 6 && step <= 8) {
      probe.parallel += dt
      probe.parallelN++
    }
    if (step === 8) {
      const s = probe.single / Math.max(1, probe.singleN)
      const p = probe.parallel / Math.max(1, probe.parallelN)
      mode = !helpersBroken && p < s * 0.9 ? 'parallel' : 'single'
    }
  }
  step++
  lastLoss = loss
  lossBuf.push(loss)
  totalChars += size.batch * size.ctx
  const now = performance.now()
  rate.push({ t: now, chars: totalChars })
  while (rate.length > 2 && now - rate[0].t > 3000) rate.shift()

  // sample on a round step count, roughly every 5 s of training (re-estimated as speed changes)
  const runNow = wallMs + (now - chunkStart)
  if (step >= 10 && runNow - lastSampleAt >= SAMPLE_EVERY_MS * 0.8) {
    const sps = (step / Math.max(1, runNow)) * 1000
    sampleEvery = niceStep(sps * (SAMPLE_EVERY_MS / 1000))
    if (step % sampleEvery === 0 || runNow - lastSampleAt >= SAMPLE_EVERY_MS * 1.6) {
      lastSampleAt = runNow
      emitSample(currentLossEstimate())
    }
  }
  if (trainMs - lastEvalAt >= Math.max(1000, evalCost * 10)) {
    const e0 = performance.now()
    const v = evalVal()
    evalCost = performance.now() - e0
    lastEvalAt = trainMs
    if (!Number.isNaN(v)) valBuf.push([step, v])
  }
}

function currentLossEstimate(): number {
  const n = Math.min(lossBuf.length, 20)
  if (n === 0) return lastLoss
  let s = 0
  for (let i = lossBuf.length - n; i < lossBuf.length; i++) s += lossBuf[i]
  return s / n
}

function niceStep(v: number): number {
  if (v <= 5) return 5
  const p = Math.pow(10, Math.floor(Math.log10(v)))
  const f = v / p
  const nice = f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10
  return nice * p
}

function emitSample(loss: number) {
  if (!model || !vocab) return
  const dec = new Decoder(model)
  dec.prime(promptIds)
  const r = new Rng(4242) // same dice every time: differences come from the model, not luck
  const out: number[] = []
  for (let i = 0; i < SAMPLE_CHARS; i++) {
    const t = sampleFrom(dec.probs(SAMPLE_TEMP), r)
    out.push(t)
    dec.push(t)
  }
  post({ type: 'sample', step, loss, prompt: promptText, text: decode(vocab, out) })
}

function flush(force = false) {
  const now = performance.now()
  if (!force && now - lastFlush < FLUSH_MS) return
  lastFlush = now
  let cps = 0
  if (running && rate.length >= 2) {
    const a = rate[0]
    const b = rate[rate.length - 1]
    if (b.t > a.t) cps = ((b.chars - a.chars) / (b.t - a.t)) * 1000
  }
  post({
    type: 'progress',
    step,
    fromStep: lossBase + lossFrom,
    losses: lossBuf.slice(lossFrom),
    val: valBuf,
    charsPerSec: cps,
    trainMs: wallMs,
    running,
    backend: backendLabel(),
    sampleEvery,
  })
  lossFrom = lossBuf.length
  valBuf = []
  // the page keeps the full history; drop ours once sent
  if (lossBuf.length > 4096) {
    const keep = 64
    lossBase += lossBuf.length - keep
    lossBuf = lossBuf.slice(-keep)
    lossFrom = lossBuf.length
  }
}

const channel = new MessageChannel()
let yieldResolve: (() => void) | null = null
channel.port1.onmessage = () => {
  const r = yieldResolve
  yieldResolve = null
  r?.()
}
/** Yield to the event loop (so pause/generate messages are handled) without timer clamping. */
function yieldNow(): Promise<void> {
  return new Promise((r) => {
    yieldResolve = r
    channel.port2.postMessage(0)
  })
}

async function loop() {
  if (looping) return
  looping = true
  const myEpoch = epoch
  try {
    while (running && myEpoch === epoch) {
      const t0 = performance.now()
      chunkStart = t0
      while (running && myEpoch === epoch && performance.now() - t0 < 40) {
        await trainStep(myEpoch)
      }
      if (myEpoch !== epoch) break
      flush()
      await yieldNow()
      if (myEpoch === epoch) wallMs += performance.now() - t0
    }
  } catch (err) {
    running = false
    post({ type: 'error', message: String(err) })
  } finally {
    looping = false
    if (myEpoch === epoch) flush(true)
  }
}

/* ---- generation (play with your model) */

function stopGen() {
  if (genTimer) clearTimeout(genTimer)
  genTimer = null
  gen = null
}

function startGen(id: number, prompt: string, temperature: number, length: number) {
  if (!model || !vocab) return
  stopGen()
  const dec = new Decoder(model)
  let p = Array.from(encode(vocab, prompt))
  if (p.length === 0) p = [vocab.index.get('\n') ?? vocab.index.get(' ') ?? 0]
  dec.prime(p)
  gen = { id, dec, left: length, temp: temperature, rng: new Rng((Math.random() * 2 ** 31) | 0) }
  genTick()
}

function genTick() {
  genTimer = null
  const g = gen
  if (!g || !vocab) return
  const n = running ? 3 : 1
  for (let k = 0; k < n && g.left > 0; k++) {
    const probs = g.dec.probs(g.temp)
    const top = topK(probs, 5).map((i) => [vocab!.chars[i], probs[i]] as [string, number])
    const tok = sampleFrom(probs, g.rng)
    post({ type: 'gen', id: g.id, ch: vocab.chars[tok], top })
    g.left--
    if (g.left > 0) g.dec.push(tok)
  }
  if (g.left <= 0) {
    post({ type: 'genDone', id: g.id })
    gen = null
    return
  }
  genTimer = setTimeout(genTick, 16)
}

/* ---- insight: what did it learn? */

function pca2(rows: number[][]): [number, number][] {
  const n = rows.length
  const d = rows[0].length
  const mean = new Array<number>(d).fill(0)
  for (const r of rows) for (let i = 0; i < d; i++) mean[i] += r[i] / n
  const X = rows.map((r) => r.map((v, i) => v - mean[i]))
  const cov = Array.from({ length: d }, () => new Array<number>(d).fill(0))
  for (const r of X) for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) cov[i][j] += r[i] * r[j]
  const vecs: number[][] = []
  const pr = new Rng(99)
  for (let c = 0; c < 2; c++) {
    let v = Array.from({ length: d }, () => pr.normal())
    for (let it = 0; it < 100; it++) {
      const w = new Array<number>(d).fill(0)
      for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) w[i] += cov[i][j] * v[j]
      for (const u of vecs) {
        const dot = w.reduce((s, wi, i) => s + wi * u[i], 0)
        for (let i = 0; i < d; i++) w[i] -= dot * u[i]
      }
      const norm = Math.hypot(...w) || 1
      v = w.map((wi) => wi / norm)
    }
    vecs.push(v)
  }
  return X.map((r) => [r.reduce((s, ri, i) => s + ri * vecs[0][i], 0), r.reduce((s, ri, i) => s + ri * vecs[1][i], 0)])
}

function insight(prompt: string) {
  if (!model || !vocab) return
  const proj = pca2(model.embeddings())
  const points = vocab.chars.map((ch, i) => ({ ch, x: proj[i][0], y: proj[i][1] }))
  let attn: { tokens: string[]; heads: number[][] } | null = null
  const p = Array.from(encode(vocab, prompt)).slice(-Math.min(24, size.ctx))
  if (p.length >= 2) {
    const T = p.length
    model.forward(p, null, 1, T)
    const layers = model.attention(1, T)
    attn = { tokens: p.map((t) => vocab!.chars[t]), heads: layers[layers.length - 1].map((h) => Array.from(h)) }
  }
  post({ type: 'insight', points, attn })
}

/* ---- message pump */

scope.onmessage = (e: MessageEvent<ToTrainer | ToHelper>) => {
  const m = e.data
  try {
    switch (m.type) {
      case 'helperInit':
        runHelper(m.port)
        return
      case 'connect':
        connectHelpers(m.helpers)
        return
      case 'init': {
        size = getSize(m.size)
        seed = m.seed
        vocab = buildVocab(m.text)
        ids = encode(vocab, m.text)
        split = Math.floor(ids.length * 0.9)
        promptText = Array.from(m.prompt)
          .filter((c) => vocab!.index.has(c))
          .join('')
        promptIds = Array.from(encode(vocab, promptText))
        if (promptIds.length === 0) {
          promptIds = [ids[0]]
          promptText = vocab.chars[ids[0]]
        }
        buildModel()
        return
      }
      case 'reset':
        if (vocab) buildModel()
        return
      case 'start':
        if (!model || running) return
        running = true
        flush(true)
        void loop()
        return
      case 'pause':
        running = false
        flush(true)
        return
      case 'generate':
        startGen(m.id, m.prompt, m.temperature, m.length)
        return
      case 'stopGenerate':
        if (gen) post({ type: 'genDone', id: gen.id })
        stopGen()
        return
      case 'insight':
        insight(m.prompt)
        return
    }
  } catch (err) {
    post({ type: 'error', message: String(err) })
  }
}
