/// <reference types="@webgpu/types" />
/**
 * GPU vs CPU self-test. Builds the same model (same seed) on both engines,
 * feeds both the same batch and compares the forward loss, every parameter
 * gradient, and the parameters after a few clipped AdamW steps. The CPU side
 * runs in float64 as the reference; the GPU computes in float32.
 */
import { GPT, type GPTConfig } from '../gpt.ts'
import { Rng } from '../rng.ts'
import { GpuGPT } from './gpuGpt.ts'

export interface SelfTestRow {
  tensor: string
  size: number
  /** max |gpu - cpu| */
  maxAbs: number
  /** max |cpu| over the tensor */
  refMax: number
  /** maxAbs / refMax */
  rel: number
  ok: boolean
}

export interface SelfTestCase {
  label: string
  lossCpu: number
  lossGpu: number
  lossRel: number
  grads: SelfTestRow[]
  /** after `steps` AdamW steps; rel here is max |Δ| in units of the learning rate */
  params: SelfTestRow[]
  stepLosses: { cpu: number; gpu: number }[]
  maxGradRel: number
  maxParamDrift: number
  ok: boolean
}

export interface SelfTestResult {
  ok: boolean
  cases: SelfTestCase[]
  ms: number
  summary: string
  error?: string
}

const LOSS_TOL = 1e-4
const GRAD_TOL = 1e-3
/**
 * After a few steps, every weight must agree to within this fraction of one
 * learning-rate-sized move. Not tighter: AdamW divides each gradient by its own
 * running magnitude, so for the few weights whose gradient is ~1e-6 of the
 * tensor's largest, fp32 rounding (a sizeable share of such a tiny gradient)
 * becomes a few percent of a step. Typical drift is 1e-3 x lr or less.
 */
const PARAM_TOL = 0.1
const STEPS = 3
const OPT = { lr: 1e-2, beta1: 0.9, beta2: 0.99, eps: 1e-8, weightDecay: 0.01 }

interface Case {
  label: string
  cfg: GPTConfig
  B: number
}

const CASES: Case[] = [
  // odd sizes: exercises every tile edge and the head split
  { label: 'odd shapes (V29 d24 h3 L2 T7 B3)', cfg: { vocabSize: 29, ctx: 7, nLayer: 2, nHead: 3, dModel: 24 }, B: 3 },
  // the Tiny model as trained on the page (smaller batch)
  { label: 'Tiny (V65 d32 h2 L1 T32 B4)', cfg: { vocabSize: 65, ctx: 32, nLayer: 1, nHead: 2, dModel: 32 }, B: 4 },
  // multi-tile matmuls with a width that is not a multiple of the 64-wide tile
  { label: 'wide (V70 d96 h4 L2 T40 B3)', cfg: { vocabSize: 70, ctx: 40, nLayer: 2, nHead: 4, dModel: 96 }, B: 3 },
]

/** Named tensors of a GPT as [name, offset, length], in parameter order. */
function tensors(m: GPT): [string, number, number][] {
  const base = m.params.byteOffset
  const out: [string, number, number][] = []
  const add = (name: string, v: Float32Array | Float64Array) => out.push([name, (v.byteOffset - base) / v.BYTES_PER_ELEMENT, v.length])
  add('tok', m.p.tok)
  add('pos', m.p.pos)
  m.p.layers.forEach((l, i) => {
    add(`${i}.ln1g`, l.ln1g)
    add(`${i}.ln1b`, l.ln1b)
    add(`${i}.wqkv`, l.wqkv)
    add(`${i}.bqkv`, l.bqkv)
    add(`${i}.wo`, l.wo)
    add(`${i}.bo`, l.bo)
    add(`${i}.ln2g`, l.ln2g)
    add(`${i}.ln2b`, l.ln2b)
    add(`${i}.wfc`, l.wfc)
    add(`${i}.bfc`, l.bfc)
    add(`${i}.wproj`, l.wproj)
    add(`${i}.bproj`, l.bproj)
  })
  add('lnfg', m.p.lnfg)
  add('lnfb', m.p.lnfb)
  add('wout', m.p.wout)
  add('bout', m.p.bout)
  return out
}

function compare(name: string, gpu: Float32Array, cpu: ArrayLike<number>, off: number, len: number, skip?: (i: number) => boolean) {
  let maxAbs = 0
  let refMax = 0
  for (let i = 0; i < len; i++) {
    if (skip?.(i)) continue
    const r = cpu[off + i]
    const d = Math.abs(gpu[off + i] - r)
    if (d > maxAbs || Number.isNaN(d)) maxAbs = Number.isNaN(d) ? Infinity : d
    if (Math.abs(r) > refMax) refMax = Math.abs(r)
  }
  return { tensor: name, size: len, maxAbs, refMax }
}

function batch(rng: Rng, V: number, n: number) {
  const x = new Uint16Array(n)
  const y = new Uint16Array(n)
  for (let i = 0; i < n; i++) {
    x[i] = rng.int(V)
    y[i] = rng.int(V)
  }
  return { x, y }
}

async function runCase(device: GPUDevice, c: Case): Promise<SelfTestCase> {
  const { cfg, B } = c
  const T = cfg.ctx
  const d = cfg.dModel
  const cpu = new GPT(cfg, 7, Float64Array)
  // both engines start from the identical fp32 weights
  cpu.params.set(Float32Array.from(cpu.params))
  const gpu = await GpuGPT.create(device, cpu, B, T)
  try {
    const rng = new Rng(2024)
    const { x, y } = batch(rng, cfg.vocabSize, B * T)

    cpu.zeroGrad()
    const lossCpu = cpu.forward(x, y, B, T)
    cpu.backward()
    const lossGpu = await gpu.forwardBackward(x, y)
    const gg = await gpu.readGrads()
    const names = tensors(cpu)
    const grads: SelfTestRow[] = names.map(([name, off, len]) => {
      const r = compare(name, gg, cpu.grads, off, len)
      const rel = r.refMax > 0 ? r.maxAbs / r.refMax : r.maxAbs
      return { ...r, rel, ok: rel <= GRAD_TOL }
    })

    // a few full training steps (clip + AdamW) on fresh batches
    const stepLosses: { cpu: number; gpu: number }[] = []
    // restart both from the same state: the forwardBackward above left grads but no update
    for (let s = 0; s < STEPS; s++) {
      const b = batch(rng, cfg.vocabSize, B * T)
      cpu.zeroGrad()
      const lc = cpu.forward(b.x, b.y, B, T)
      cpu.backward()
      cpu.clipGrads(1)
      cpu.step(OPT)
      const lg = await gpu.trainStep(b.x, b.y, OPT, 1)
      stepLosses.push({ cpu: lc, gpu: lg })
    }
    const pg = await gpu.readParams()
    const params: SelfTestRow[] = names.map(([name, off, len]) => {
      // The key bias gets an exactly-zero true gradient (softmax ignores a constant
      // added to every score), so both engines hold pure rounding noise there and
      // Adam's normalisation turns that noise into full-size random steps. Skip it.
      const skip = name.endsWith('.bqkv') ? (i: number) => i >= d && i < 2 * d : undefined
      const r = compare(name + (skip ? ' (q,v part)' : ''), pg, cpu.params, off, len, skip)
      const drift = r.maxAbs / OPT.lr
      return { ...r, rel: drift, ok: drift <= PARAM_TOL }
    })
    const lossRel = Math.abs(lossGpu - lossCpu) / Math.abs(lossCpu)
    const stepOk = stepLosses.every((l) => Math.abs(l.gpu - l.cpu) / Math.abs(l.cpu) <= LOSS_TOL)
    const maxGradRel = Math.max(...grads.map((g) => g.rel))
    const maxParamDrift = Math.max(...params.map((p) => p.rel))
    return {
      label: c.label,
      lossCpu,
      lossGpu,
      lossRel,
      grads,
      params,
      stepLosses,
      maxGradRel,
      maxParamDrift,
      ok: lossRel <= LOSS_TOL && stepOk && grads.every((g) => g.ok) && params.every((p) => p.ok),
    }
  } finally {
    gpu.destroy()
  }
}

export async function runSelfTest(device: GPUDevice): Promise<SelfTestResult> {
  const t0 = performance.now()
  const cases: SelfTestCase[] = []
  for (const c of CASES) cases.push(await runCase(device, c))
  const ok = cases.every((c) => c.ok)
  const maxLoss = Math.max(...cases.map((c) => c.lossRel), ...cases.flatMap((c) => c.stepLosses.map((l) => Math.abs(l.gpu - l.cpu) / Math.abs(l.cpu))))
  const maxGrad = Math.max(...cases.map((c) => c.maxGradRel))
  const maxDrift = Math.max(...cases.map((c) => c.maxParamDrift))
  const summary =
    `${ok ? 'PASS' : 'FAIL'} · ${cases.length} configs · loss rel err ≤ ${maxLoss.toExponential(1)} (tol ${LOSS_TOL.toExponential(0)})` +
    ` · grad rel err ≤ ${maxGrad.toExponential(1)} (tol ${GRAD_TOL.toExponential(0)})` +
    ` · params after ${STEPS} AdamW steps within ${maxDrift.toExponential(1)}×lr (tol ${PARAM_TOL}×lr)`
  return { ok, cases, ms: performance.now() - t0, summary }
}
