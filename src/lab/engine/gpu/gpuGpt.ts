/// <reference types="@webgpu/types" />
/**
 * The WebGPU training engine: the same GPT as ../gpt.ts, with every forward
 * and backward kernel, the gradient clipping and the AdamW update running as
 * WGSL compute shaders. Parameters, gradients and optimizer state live on the
 * GPU for the whole run; the CPU only uploads token ids and reads back the loss
 * (and, every few seconds, a snapshot of the weights for sampling).
 *
 * Memory layout is identical to the CPU engine: one flat fp32 buffer of
 * parameters (same order, same offsets), so weights can be copied straight
 * into a CPU GPT for text generation and the self-test can compare tensors.
 */
import type { AdamW, GPT, GPTConfig } from '../gpt.ts'
import { KERNELS, type KernelName } from './kernels.ts'

type Pipes = Record<KernelName, GPUComputePipeline>

const pipeCache = new WeakMap<GPUDevice, Promise<Pipes>>()

/** Compile every kernel once per device. */
export function getPipelines(device: GPUDevice): Promise<Pipes> {
  let p = pipeCache.get(device)
  if (!p) {
    p = (async () => {
      const names = Object.keys(KERNELS) as KernelName[]
      const pipes = await Promise.all(
        names.map((name) =>
          device.createComputePipelineAsync({
            label: name,
            layout: 'auto',
            compute: { module: device.createShaderModule({ label: name, code: KERNELS[name] }), entryPoint: 'main' },
          }),
        ),
      )
      return Object.fromEntries(names.map((n, i) => [n, pipes[i]])) as Pipes
    })()
    pipeCache.set(device, p)
    p.catch(() => pipeCache.delete(device))
  }
  return p
}

/** A uniform field: plain numbers are u32, {f} are f32. */
type UField = number | { f: number }
const f = (v: number): UField => ({ f: v })

interface Step {
  k: KernelName
  uni: UField[] | GPUBuffer
  bufs: GPUBuffer[]
  x: number
  y: number
}

interface Disp {
  pipe: GPUComputePipeline
  bg: GPUBindGroup
  x: number
  y: number
}

const SLOT = 256 // minUniformBufferOffsetAlignment upper bound
const UNI_SIZE = 64

/** Workgroup grid for a 256-wide elementwise kernel over n items. */
function grid(n: number): [number, number] {
  const wg = Math.max(1, Math.ceil(n / 256))
  const x = Math.min(wg, 65535)
  return [x, Math.ceil(wg / x)]
}

interface LayerOff {
  ln1g: number
  ln1b: number
  wqkv: number
  bqkv: number
  wo: number
  bo: number
  ln2g: number
  ln2b: number
  wfc: number
  bfc: number
  wproj: number
  bproj: number
}

interface LayerActs {
  ln1: GPUBuffer
  ln1m: GPUBuffer
  ln1r: GPUBuffer
  qkv: GPUBuffer
  att: GPUBuffer
  atty: GPUBuffer
  res1: GPUBuffer
  ln2: GPUBuffer
  ln2m: GPUBuffer
  ln2r: GPUBuffer
  fch: GPUBuffer
  fcg: GPUBuffer
  res2: GPUBuffer
}

export class GpuGPT {
  readonly device: GPUDevice
  readonly cfg: GPTConfig
  readonly numParams: number
  /** training batch shape the activation buffers are sized for */
  readonly B: number
  readonly T: number
  /** fixed padded length for attention snapshots ("Under the glass") */
  readonly inferT: number
  /** bytes of GPU memory this model allocated */
  bytes = 0

  private readonly pipes: Pipes
  private readonly buffers: GPUBuffer[] = []
  private readonly off: { tok: number; pos: number; lnfg: number; lnfb: number; wout: number; bout: number; layers: LayerOff[] }
  private params!: GPUBuffer
  private grads!: GPUBuffer
  private mom!: GPUBuffer
  private vel!: GPUBuffer
  private mask!: GPUBuffer
  private partial!: GPUBuffer
  private normSq!: GPUBuffer
  private adamU!: GPUBuffer
  private lossOut!: GPUBuffer
  private idx!: GPUBuffer
  private tgt!: GPUBuffer
  private dummy!: GPUBuffer
  private x0!: GPUBuffer
  private layers: LayerActs[] = []
  private lnf!: GPUBuffer
  private lnfm!: GPUBuffer
  private lnfr!: GPUBuffer
  private logits!: GPUBuffer
  private dlogits!: GPUBuffer
  private rowLoss!: GPUBuffer
  private s!: {
    dres: GPUBuffer
    dln: GPUBuffer
    dqkv: GPUBuffer
    datty: GPUBuffer
    dfcg: GPUBuffer
    dfch: GPUBuffer
    dlnf: GPUBuffer
    datt: GPUBuffer
  }
  private progs = new Map<string, Disp[]>()
  private t = 0
  private destroyed = false

  private constructor(device: GPUDevice, pipes: Pipes, model: GPT, B: number, T: number) {
    this.device = device
    this.pipes = pipes
    this.cfg = model.cfg
    this.numParams = model.numParams
    this.B = B
    this.T = T
    this.inferT = Math.min(24, model.cfg.ctx)
    const base = model.params
    const o = (v: ArrayBufferView & { BYTES_PER_ELEMENT: number }) => (v.byteOffset - base.byteOffset) / v.BYTES_PER_ELEMENT
    const p = model.p
    this.off = {
      tok: o(p.tok),
      pos: o(p.pos),
      lnfg: o(p.lnfg),
      lnfb: o(p.lnfb),
      wout: o(p.wout),
      bout: o(p.bout),
      layers: p.layers.map((l) => ({
        ln1g: o(l.ln1g),
        ln1b: o(l.ln1b),
        wqkv: o(l.wqkv),
        bqkv: o(l.bqkv),
        wo: o(l.wo),
        bo: o(l.bo),
        ln2g: o(l.ln2g),
        ln2b: o(l.ln2b),
        wfc: o(l.wfc),
        bfc: o(l.bfc),
        wproj: o(l.wproj),
        bproj: o(l.bproj),
      })),
    }
  }

  /**
   * Upload `model`'s current weights to the GPU and allocate everything needed
   * to train on batches of B sequences of length T.
   */
  static async create(device: GPUDevice, model: GPT, B: number, T: number): Promise<GpuGPT> {
    if (B * T > 65535) throw new Error('batch too large for one dispatch row')
    const pipes = await getPipelines(device)
    const g = new GpuGPT(device, pipes, model, B, T)
    device.pushErrorScope('out-of-memory')
    device.pushErrorScope('validation')
    let allocErr: unknown = null
    try {
      g.allocate(model)
    } catch (e) {
      allocErr = e
    }
    const val = await device.popErrorScope()
    const oom = await device.popErrorScope()
    if (allocErr || oom || val) {
      g.destroy()
      if (oom) throw new Error(`Not enough GPU memory (${oom.message})`)
      throw new Error(`GPU setup failed: ${val ? val.message : String(allocErr)}`)
    }
    // build the training program eagerly so shader/bind-group errors surface now
    device.pushErrorScope('validation')
    g.program('train')
    g.program('eval')
    const err = await device.popErrorScope()
    if (err) {
      g.destroy()
      throw new Error(`GPU setup failed: ${err.message}`)
    }
    return g
  }

  private buf(floats: number, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST): GPUBuffer {
    const size = Math.max(16, Math.ceil((floats * 4) / 16) * 16)
    const b = this.device.createBuffer({ size, usage })
    this.buffers.push(b)
    this.bytes += size
    return b
  }

  private allocate(model: GPT) {
    const { dModel: d, nHead: H, vocabSize: V, nLayer: L } = this.cfg
    const { B, T } = this
    const N = B * T
    const P = this.numParams
    this.params = this.buf(P)
    this.grads = this.buf(P)
    this.mom = this.buf(P)
    this.vel = this.buf(P)
    this.mask = this.buf(Math.ceil(P / 32))
    this.partial = this.buf(256)
    this.normSq = this.buf(4)
    this.adamU = this.buf(UNI_SIZE / 4, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
    this.lossOut = this.buf(8)
    this.idx = this.buf(N)
    this.tgt = this.buf(N)
    this.dummy = this.buf(4)
    this.x0 = this.buf(N * d)
    const att = B * H * T * T
    for (let l = 0; l < L; l++) {
      this.layers.push({
        ln1: this.buf(N * d),
        ln1m: this.buf(N),
        ln1r: this.buf(N),
        qkv: this.buf(N * 3 * d),
        att: this.buf(att),
        atty: this.buf(N * d),
        res1: this.buf(N * d),
        ln2: this.buf(N * d),
        ln2m: this.buf(N),
        ln2r: this.buf(N),
        fch: this.buf(N * 4 * d),
        fcg: this.buf(N * 4 * d),
        res2: this.buf(N * d),
      })
    }
    this.lnf = this.buf(N * d)
    this.lnfm = this.buf(N)
    this.lnfr = this.buf(N)
    this.logits = this.buf(N * V)
    this.dlogits = this.buf(N * V)
    this.rowLoss = this.buf(N)
    this.s = {
      dres: this.buf(N * d),
      dln: this.buf(N * d),
      dqkv: this.buf(N * 3 * d),
      datty: this.buf(N * d),
      dfcg: this.buf(N * 4 * d),
      dfch: this.buf(N * 4 * d),
      dlnf: this.buf(N * d),
      datt: this.buf(att),
    }

    // weights (fp32) and the weight-decay bitmask: matrices and embeddings decay, gains and biases don't
    const w = model.params instanceof Float32Array ? model.params : Float32Array.from(model.params)
    this.device.queue.writeBuffer(this.params, 0, w.buffer, w.byteOffset, P * 4)
    const mask = new Uint32Array(Math.ceil(P / 32))
    const setRange = (start: number, len: number) => {
      for (let i = start; i < start + len; i++) mask[i >>> 5] |= 1 << (i & 31)
    }
    const p = model.p
    const o = this.off
    setRange(o.tok, p.tok.length)
    setRange(o.pos, p.pos.length)
    p.layers.forEach((lp, l) => {
      const lo = o.layers[l]
      setRange(lo.wqkv, lp.wqkv.length)
      setRange(lo.wo, lp.wo.length)
      setRange(lo.wfc, lp.wfc.length)
      setRange(lo.wproj, lp.wproj.length)
    })
    setRange(o.wout, p.wout.length)
    this.device.queue.writeBuffer(this.mask, 0, mask)
    // grads/moments start at zero (fresh buffers are zero-initialised by WebGPU)
  }

  /* ------------------------------------------------------------ programs */

  private mm(
    st: Step[],
    kind: 'mmNN' | 'mmNT' | 'mmTN',
    M: number,
    Nc: number,
    K: number,
    A: GPUBuffer,
    aOff: number,
    Bb: GPUBuffer,
    bOff: number,
    C: GPUBuffer,
    cOff: number,
    bias: number | null,
    accumulate: boolean,
  ) {
    const flags = (bias !== null ? 1 : 0) | (accumulate ? 2 : 0)
    st.push({
      k: kind,
      uni: [M, Nc, K, aOff, bOff, cOff, bias ?? 0, flags],
      bufs: [A, Bb, C, bias !== null ? this.params : this.dummy],
      x: Math.ceil(Nc / 64),
      y: Math.ceil(M / 64),
    })
  }

  private forwardSteps(st: Step[], B: number, T: number) {
    const { dModel: d, nHead: H, vocabSize: V } = this.cfg
    const N = B * T
    const o = this.off
    const P = this.params
    st.push({ k: 'embed', uni: [N, T, d, o.tok, o.pos], bufs: [P, this.idx, this.x0], ...xy(grid(N * d)) })
    let x = this.x0
    this.layers.forEach((la, l) => {
      const lo = o.layers[l]
      st.push({ k: 'layernorm', uni: [N, d, lo.ln1g, lo.ln1b], bufs: [x, P, la.ln1, la.ln1m, la.ln1r], x: N, y: 1 })
      this.mm(st, 'mmNN', N, 3 * d, d, la.ln1, 0, P, lo.wqkv, la.qkv, 0, lo.bqkv, false)
      st.push({ k: 'attnFwd', uni: [B, T, d, H, f(1 / Math.sqrt(d / H))], bufs: [la.qkv, la.att, la.atty], x: Math.ceil((B * H * T) / 64), y: 1 })
      this.mm(st, 'mmNN', N, d, d, la.atty, 0, P, lo.wo, la.res1, 0, lo.bo, false)
      st.push({ k: 'add', uni: [N * d], bufs: [x, la.res1], ...xy(grid(N * d)) })
      st.push({ k: 'layernorm', uni: [N, d, lo.ln2g, lo.ln2b], bufs: [la.res1, P, la.ln2, la.ln2m, la.ln2r], x: N, y: 1 })
      this.mm(st, 'mmNN', N, 4 * d, d, la.ln2, 0, P, lo.wfc, la.fch, 0, lo.bfc, false)
      st.push({ k: 'geluFwd', uni: [N * 4 * d], bufs: [la.fch, la.fcg], ...xy(grid(N * 4 * d)) })
      this.mm(st, 'mmNN', N, d, 4 * d, la.fcg, 0, P, lo.wproj, la.res2, 0, lo.bproj, false)
      st.push({ k: 'add', uni: [N * d], bufs: [la.res1, la.res2], ...xy(grid(N * d)) })
      x = la.res2
    })
    st.push({ k: 'layernorm', uni: [N, d, o.lnfg, o.lnfb], bufs: [x, P, this.lnf, this.lnfm, this.lnfr], x: N, y: 1 })
    this.mm(st, 'mmNN', N, V, d, this.lnf, 0, P, o.wout, this.logits, 0, o.bout, false)
  }

  private lossSteps(st: Step[], N: number, slot: number) {
    const V = this.cfg.vocabSize
    st.push({ k: 'xent', uni: [N, V], bufs: [this.logits, this.tgt, this.dlogits, this.rowLoss], x: Math.ceil(N / 64), y: 1 })
    st.push({ k: 'meanReduce', uni: [N, slot], bufs: [this.rowLoss, this.lossOut], x: 1, y: 1 })
  }

  private backwardSteps(st: Step[], B: number, T: number) {
    const { dModel: d, nHead: H, vocabSize: V, nLayer: L } = this.cfg
    const N = B * T
    const o = this.off
    const P = this.params
    const G = this.grads
    const s = this.s
    const colsum = (D: GPUBuffer, C: number, off: number) =>
      st.push({ k: 'colsum', uni: [N, C, off, 0], bufs: [D, G], x: Math.ceil(C / 16), y: 1 })
    const lnBack = (dout: GPUBuffer, inp: GPUBuffer, gOff: number, bOff: number, m: GPUBuffer, r: GPUBuffer, acc: boolean) => {
      st.push({ k: 'layernormBackInput', uni: [N, d, gOff, acc ? 1 : 0], bufs: [dout, inp, P, m, r, s.dres], x: N, y: 1 })
      st.push({ k: 'layernormBackParams', uni: [N, d, gOff, bOff], bufs: [dout, inp, m, r, G], x: Math.ceil(d / 16), y: 1 })
    }

    // LM head: logits = lnf @ wout + bout
    this.mm(st, 'mmNT', N, d, V, this.dlogits, 0, P, o.wout, s.dlnf, 0, null, false)
    this.mm(st, 'mmTN', d, V, N, this.lnf, 0, this.dlogits, 0, G, o.wout, null, true)
    colsum(this.dlogits, V, o.bout)
    const xLast = L > 0 ? this.layers[L - 1].res2 : this.x0
    lnBack(s.dlnf, xLast, o.lnfg, o.lnfb, this.lnfm, this.lnfr, false)

    for (let l = L - 1; l >= 0; l--) {
      const la = this.layers[l]
      const lo = o.layers[l]
      const xin = l === 0 ? this.x0 : this.layers[l - 1].res2
      // MLP branch
      this.mm(st, 'mmNT', N, 4 * d, d, s.dres, 0, P, lo.wproj, s.dfcg, 0, null, false)
      this.mm(st, 'mmTN', 4 * d, d, N, la.fcg, 0, s.dres, 0, G, lo.wproj, null, true)
      colsum(s.dres, d, lo.bproj)
      st.push({ k: 'geluBack', uni: [N * 4 * d], bufs: [la.fch, s.dfcg, s.dfch], ...xy(grid(N * 4 * d)) })
      this.mm(st, 'mmNT', N, d, 4 * d, s.dfch, 0, P, lo.wfc, s.dln, 0, null, false)
      this.mm(st, 'mmTN', d, 4 * d, N, la.ln2, 0, s.dfch, 0, G, lo.wfc, null, true)
      colsum(s.dfch, 4 * d, lo.bfc)
      lnBack(s.dln, la.res1, lo.ln2g, lo.ln2b, la.ln2m, la.ln2r, true)
      // attention branch
      this.mm(st, 'mmNT', N, d, d, s.dres, 0, P, lo.wo, s.datty, 0, null, false)
      this.mm(st, 'mmTN', d, d, N, la.atty, 0, s.dres, 0, G, lo.wo, null, true)
      colsum(s.dres, d, lo.bo)
      const au = [B, T, d, H, f(1 / Math.sqrt(d / H))]
      const rows = Math.ceil((B * H * T) / 64)
      st.push({ k: 'attnBack1', uni: au, bufs: [la.qkv, la.att, s.datty, s.datt, s.dqkv], x: rows, y: 1 })
      st.push({ k: 'attnBack2', uni: au, bufs: [la.qkv, la.att, s.datt, s.datty, s.dqkv], x: rows, y: 1 })
      this.mm(st, 'mmNT', N, d, 3 * d, s.dqkv, 0, P, lo.wqkv, s.dln, 0, null, false)
      this.mm(st, 'mmTN', d, 3 * d, N, la.ln1, 0, s.dqkv, 0, G, lo.wqkv, null, true)
      colsum(s.dqkv, 3 * d, lo.bqkv)
      lnBack(s.dln, xin, lo.ln1g, lo.ln1b, la.ln1m, la.ln1r, true)
    }
    st.push({ k: 'embedBackTok', uni: [N, d, V, o.tok], bufs: [s.dres, this.idx, G], ...xy(grid(V * d)) })
    st.push({ k: 'embedBackPos', uni: [B, T, d, o.pos], bufs: [s.dres, G], ...xy(grid(T * d)) })
  }

  private optSteps(st: Step[]) {
    const n = this.numParams
    st.push({ k: 'normPartial', uni: [n], bufs: [this.grads, this.partial], x: 256, y: 1 })
    st.push({ k: 'normFinal', uni: [256], bufs: [this.partial, this.normSq], x: 1, y: 1 })
    st.push({
      k: 'adamw',
      uni: this.adamU,
      bufs: [this.params, this.grads, this.mom, this.vel, this.mask, this.normSq],
      ...xy(grid(n)),
    })
  }

  private program(name: 'train' | 'eval' | 'fb' | 'opt' | 'infer'): Disp[] {
    const cached = this.progs.get(name)
    if (cached) return cached
    const st: Step[] = []
    const { B, T } = this
    if (name === 'infer') this.forwardSteps(st, 1, this.inferT)
    if (name === 'train' || name === 'eval' || name === 'fb') {
      this.forwardSteps(st, B, T)
      this.lossSteps(st, B * T, name === 'eval' ? 1 : 0)
    }
    if (name === 'train' || name === 'fb') this.backwardSteps(st, B, T)
    if (name === 'train' || name === 'opt') this.optSteps(st)
    const disps = this.compile(st)
    this.progs.set(name, disps)
    return disps
  }

  private compile(st: Step[]): Disp[] {
    const nUni = st.filter((s) => !(s.uni instanceof GPUBuffer)).length
    const ubuf = this.buf((Math.max(1, nUni) * SLOT) / 4, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
    const data = new ArrayBuffer(Math.max(1, nUni) * SLOT)
    const dv = new DataView(data)
    let slot = 0
    const out: Disp[] = []
    for (const s of st) {
      let uniEntry: GPUBindGroupEntry
      if (s.uni instanceof GPUBuffer) {
        uniEntry = { binding: 0, resource: { buffer: s.uni } }
      } else {
        const base = slot * SLOT
        s.uni.forEach((v, i) => {
          if (typeof v === 'number') dv.setUint32(base + i * 4, v, true)
          else dv.setFloat32(base + i * 4, v.f, true)
        })
        uniEntry = { binding: 0, resource: { buffer: ubuf, offset: base, size: UNI_SIZE } }
        slot++
      }
      const pipe = this.pipes[s.k]
      const bg = this.device.createBindGroup({
        label: s.k,
        layout: pipe.getBindGroupLayout(0),
        entries: [uniEntry, ...s.bufs.map((b, i) => ({ binding: i + 1, resource: { buffer: b } }))],
      })
      out.push({ pipe, bg, x: s.x, y: s.y })
    }
    this.device.queue.writeBuffer(ubuf, 0, data)
    return out
  }

  private encode(enc: GPUCommandEncoder, ...progs: Disp[][]) {
    const pass = enc.beginComputePass()
    for (const prog of progs) {
      for (const d of prog) {
        pass.setPipeline(d.pipe)
        pass.setBindGroup(0, d.bg)
        pass.dispatchWorkgroups(d.x, d.y)
      }
    }
    pass.end()
  }

  /* ---------------------------------------------------------------- API */

  private writeTokens(x: ArrayLike<number>, y: ArrayLike<number> | null) {
    const q = this.device.queue
    q.writeBuffer(this.idx, 0, Uint32Array.from(x as ArrayLike<number>))
    if (y) q.writeBuffer(this.tgt, 0, Uint32Array.from(y as ArrayLike<number>))
  }

  private readLoss(enc: GPUCommandEncoder, slot: number): () => Promise<number> {
    const st = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    enc.copyBufferToBuffer(this.lossOut, slot * 4, st, 0, 4)
    return async () => {
      await st.mapAsync(GPUMapMode.READ)
      const v = new Float32Array(st.getMappedRange())[0]
      st.unmap()
      st.destroy()
      return v
    }
  }

  private writeAdam(o: AdamW, maxNorm: number) {
    this.t++
    const ab = new ArrayBuffer(UNI_SIZE)
    const dv = new DataView(ab)
    dv.setUint32(0, this.numParams, true)
    const fl = [o.lr, o.beta1, o.beta2, o.eps, o.weightDecay, 1 - Math.pow(o.beta1, this.t), 1 - Math.pow(o.beta2, this.t), maxNorm]
    fl.forEach((v, i) => dv.setFloat32(4 + i * 4, v, true))
    this.device.queue.writeBuffer(this.adamU, 0, ab)
  }

  /**
   * One full training step on the GPU (forward, backward, clip, AdamW), all in
   * a single compute pass. Returns as soon as the work is queued; the promise
   * resolves with the loss when the GPU has finished it.
   */
  trainStep(x: ArrayLike<number>, y: ArrayLike<number>, o: AdamW, maxNorm = 1): Promise<number> {
    this.writeTokens(x, y)
    this.writeAdam(o, maxNorm)
    const enc = this.device.createCommandEncoder()
    enc.clearBuffer(this.grads)
    this.encode(enc, this.program('train'))
    const done = this.readLoss(enc, 0)
    this.device.queue.submit([enc.finish()])
    return done()
  }

  /** Forward-only mean loss of one batch (validation). */
  evalLoss(x: ArrayLike<number>, y: ArrayLike<number>): Promise<number> {
    this.writeTokens(x, y)
    const enc = this.device.createCommandEncoder()
    this.encode(enc, this.program('eval'))
    const done = this.readLoss(enc, 1)
    this.device.queue.submit([enc.finish()])
    return done()
  }

  /** Forward + backward without an optimizer step (self-test): returns the loss; grads stay on the GPU. */
  forwardBackward(x: ArrayLike<number>, y: ArrayLike<number>): Promise<number> {
    this.writeTokens(x, y)
    const enc = this.device.createCommandEncoder()
    enc.clearBuffer(this.grads)
    this.encode(enc, this.program('fb'))
    const done = this.readLoss(enc, 0)
    this.device.queue.submit([enc.finish()])
    return done()
  }

  /** Clip + AdamW on the gradients currently on the GPU (self-test). */
  optimizerStep(o: AdamW, maxNorm = 1) {
    this.writeAdam(o, maxNorm)
    const enc = this.device.createCommandEncoder()
    this.encode(enc, this.program('opt'))
    this.device.queue.submit([enc.finish()])
  }

  private async read(src: GPUBuffer, offsetFloats: number, count: number): Promise<Float32Array> {
    const bytes = Math.max(4, count * 4)
    const st = this.device.createBuffer({ size: Math.ceil(bytes / 4) * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    const enc = this.device.createCommandEncoder()
    enc.copyBufferToBuffer(src, offsetFloats * 4, st, 0, bytes)
    this.device.queue.submit([enc.finish()])
    await st.mapAsync(GPUMapMode.READ)
    const out = new Float32Array(st.getMappedRange().slice(0, count * 4))
    st.unmap()
    st.destroy()
    return out
  }

  /** Snapshot of all weights (same layout as GPT.params). */
  readParams(): Promise<Float32Array> {
    return this.read(this.params, 0, this.numParams)
  }

  /** Logits of the last forward pass over the training batch shape (B*T rows of V). */
  readLogits(): Promise<Float32Array> {
    return this.read(this.logits, 0, this.B * this.T * this.cfg.vocabSize)
  }

  readGrads(): Promise<Float32Array> {
    return this.read(this.grads, 0, this.numParams)
  }

  /** Last-layer attention for a prompt of up to inferT tokens: heads x (len*len), like GPT.attention(). */
  async attention(tokens: number[]): Promise<number[][]> {
    const Ti = this.inferT
    const len = Math.min(tokens.length, Ti)
    const toks = tokens.slice(-len)
    // pad at the end: causal attention means padding never changes the rows we keep
    const padded = new Uint32Array(Ti)
    for (let i = 0; i < Ti; i++) padded[i] = toks[Math.min(i, len - 1)]
    this.device.queue.writeBuffer(this.idx, 0, padded)
    const enc = this.device.createCommandEncoder()
    this.encode(enc, this.program('infer'))
    this.device.queue.submit([enc.finish()])
    const H = this.cfg.nHead
    const last = this.layers[this.layers.length - 1]
    if (!last) return []
    const a = await this.read(last.att, 0, H * Ti * Ti)
    const heads: number[][] = []
    for (let h = 0; h < H; h++) {
      const out = new Array<number>(len * len)
      for (let t = 0; t < len; t++) for (let t2 = 0; t2 < len; t2++) out[t * len + t2] = a[(h * Ti + t) * Ti + t2]
      heads.push(out)
    }
    return heads
  }

  /** Wait until all submitted work has finished. */
  idle(): Promise<void> {
    return this.device.queue.onSubmittedWorkDone()
  }

  destroy() {
    if (this.destroyed) return
    this.destroyed = true
    for (const b of this.buffers) b.destroy()
    this.buffers.length = 0
    this.progs.clear()
  }
}

function xy([x, y]: [number, number]) {
  return { x, y }
}
