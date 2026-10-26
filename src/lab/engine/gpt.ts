/**
 * A tiny GPT (decoder-only transformer) with a hand-written forward and
 * backward pass on flat typed arrays. No dependencies, no autograd graph:
 * each layer has an explicit forward and backward kernel, in the spirit of
 * Karpathy's llm.c. Verified against finite differences (see gradcheck).
 *
 *   x = tokEmb[idx] + posEmb[t]
 *   per block:  x = x + Attn(LN1(x));  x = x + MLP(LN2(x))
 *   logits = LNf(x) @ Wout + bout
 */
import { Rng } from './rng.ts'

export type Arr = Float32Array | Float64Array
type ArrCtor = Float32ArrayConstructor | Float64ArrayConstructor

export interface GPTConfig {
  vocabSize: number
  ctx: number
  nLayer: number
  nHead: number
  dModel: number
}

interface LayerParams {
  ln1g: Arr
  ln1b: Arr
  wqkv: Arr
  bqkv: Arr
  wo: Arr
  bo: Arr
  ln2g: Arr
  ln2b: Arr
  wfc: Arr
  bfc: Arr
  wproj: Arr
  bproj: Arr
}

interface Params {
  tok: Arr
  pos: Arr
  layers: LayerParams[]
  lnfg: Arr
  lnfb: Arr
  wout: Arr
  bout: Arr
}

interface Spec {
  key: string
  size: number
  decay: boolean
  init: 'normal' | 'zeros' | 'ones'
  std: number
  offset: number
}

interface LayerActs {
  ln1: Arr
  ln1m: Arr
  ln1r: Arr
  qkv: Arr
  att: Arr
  atty: Arr
  res1: Arr
  ln2: Arr
  ln2m: Arr
  ln2r: Arr
  fch: Arr
  fcg: Arr
  res2: Arr
}

interface Acts {
  B: number
  T: number
  x0: Arr
  layers: LayerActs[]
  lnf: Arr
  lnfm: Arr
  lnfr: Arr
  /** logits, overwritten in place by softmax probabilities */
  probs: Arr
  // gradient scratch (allocated on first backward)
  g?: {
    dres: Arr
    dln: Arr
    dqkv: Arr
    datty: Arr
    dfcg: Arr
    dfch: Arr
    dlnf: Arr
    dlogits: Arr
    scratch: Arr
  }
}

const GELU_K = Math.sqrt(2 / Math.PI)
const LN_EPS = 1e-5

/* ---------------------------------------------------------------- kernels */

/** out[N,O] = inp[N,I] @ W[I,O] + b[O]. Processes 4 rows at a time so each W row is read once per tile. */
function matmulForward(out: Arr, inp: Arr, W: Arr, b: Arr | null, N: number, I: number, O: number) {
  let n = 0
  for (; n + 4 <= N; n += 4) {
    const o0 = n * O, o1 = o0 + O, o2 = o1 + O, o3 = o2 + O
    for (let o = 0; o < O; o++) {
      const bv = b ? b[o] : 0
      out[o0 + o] = bv
      out[o1 + o] = bv
      out[o2 + o] = bv
      out[o3 + o] = bv
    }
    const i0 = n * I, i1 = i0 + I, i2 = i1 + I, i3 = i2 + I
    for (let i = 0; i < I; i++) {
      const a0 = inp[i0 + i], a1 = inp[i1 + i], a2 = inp[i2 + i], a3 = inp[i3 + i]
      const wr = i * O
      for (let o = 0; o < O; o++) {
        const w = W[wr + o]
        out[o0 + o] += a0 * w
        out[o1 + o] += a1 * w
        out[o2 + o] += a2 * w
        out[o3 + o] += a3 * w
      }
    }
  }
  for (; n < N; n++) {
    const oo = n * O
    for (let o = 0; o < O; o++) out[oo + o] = b ? b[o] : 0
    const ii = n * I
    for (let i = 0; i < I; i++) {
      const a = inp[ii + i]
      const wr = i * O
      for (let o = 0; o < O; o++) out[oo + o] += a * W[wr + o]
    }
  }
}

/** Given dout[N,O]: sets dinp[N,I]; accumulates dW[I,O] and db[O]. */
function matmulBackward(
  dinp: Arr | null,
  dW: Arr,
  db: Arr | null,
  dout: Arr,
  inp: Arr,
  W: Arr,
  N: number,
  I: number,
  O: number,
) {
  let n = 0
  for (; n + 4 <= N; n += 4) {
    const o0 = n * O, o1 = o0 + O, o2 = o1 + O, o3 = o2 + O
    const i0 = n * I, i1 = i0 + I, i2 = i1 + I, i3 = i2 + I
    for (let i = 0; i < I; i++) {
      const wr = i * O
      const a0 = inp[i0 + i], a1 = inp[i1 + i], a2 = inp[i2 + i], a3 = inp[i3 + i]
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0
      for (let o = 0; o < O; o++) {
        const w = W[wr + o]
        const d0 = dout[o0 + o], d1 = dout[o1 + o], d2 = dout[o2 + o], d3 = dout[o3 + o]
        s0 += d0 * w
        s1 += d1 * w
        s2 += d2 * w
        s3 += d3 * w
        dW[wr + o] += a0 * d0 + a1 * d1 + a2 * d2 + a3 * d3
      }
      if (dinp) {
        dinp[i0 + i] = s0
        dinp[i1 + i] = s1
        dinp[i2 + i] = s2
        dinp[i3 + i] = s3
      }
    }
    if (db) for (let o = 0; o < O; o++) db[o] += dout[o0 + o] + dout[o1 + o] + dout[o2 + o] + dout[o3 + o]
  }
  for (; n < N; n++) {
    const oo = n * O
    const ii = n * I
    for (let i = 0; i < I; i++) {
      const wr = i * O
      const a = inp[ii + i]
      let s = 0
      for (let o = 0; o < O; o++) {
        const d = dout[oo + o]
        s += d * W[wr + o]
        dW[wr + o] += a * d
      }
      if (dinp) dinp[ii + i] = s
    }
    if (db) for (let o = 0; o < O; o++) db[o] += dout[oo + o]
  }
}

function layernormForward(out: Arr, mean: Arr, rstd: Arr, inp: Arr, g: Arr, b: Arr, N: number, C: number) {
  for (let n = 0; n < N; n++) {
    const off = n * C
    let m = 0
    for (let i = 0; i < C; i++) m += inp[off + i]
    m /= C
    let v = 0
    for (let i = 0; i < C; i++) {
      const x = inp[off + i] - m
      v += x * x
    }
    v /= C
    const s = 1 / Math.sqrt(v + LN_EPS)
    for (let i = 0; i < C; i++) out[off + i] = (inp[off + i] - m) * s * g[i] + b[i]
    mean[n] = m
    rstd[n] = s
  }
}

/** Accumulates into dinp, dg, db. */
function layernormBackward(
  dinp: Arr,
  dg: Arr,
  db: Arr,
  dout: Arr,
  inp: Arr,
  g: Arr,
  mean: Arr,
  rstd: Arr,
  N: number,
  C: number,
) {
  for (let n = 0; n < N; n++) {
    const off = n * C
    const m = mean[n]
    const s = rstd[n]
    let dnormMean = 0
    let dnormNormMean = 0
    for (let i = 0; i < C; i++) {
      const norm = (inp[off + i] - m) * s
      const dnorm = g[i] * dout[off + i]
      dnormMean += dnorm
      dnormNormMean += dnorm * norm
    }
    dnormMean /= C
    dnormNormMean /= C
    for (let i = 0; i < C; i++) {
      const norm = (inp[off + i] - m) * s
      const d = dout[off + i]
      const dnorm = g[i] * d
      db[i] += d
      dg[i] += norm * d
      dinp[off + i] += (dnorm - dnormMean - norm * dnormNormMean) * s
    }
  }
}

function geluForward(out: Arr, inp: Arr, n: number) {
  for (let i = 0; i < n; i++) {
    const x = inp[i]
    out[i] = 0.5 * x * (1 + Math.tanh(GELU_K * (x + 0.044715 * x * x * x)))
  }
}

function geluBackward(dinp: Arr, inp: Arr, dout: Arr, n: number) {
  for (let i = 0; i < n; i++) {
    const x = inp[i]
    const u = GELU_K * (x + 0.044715 * x * x * x)
    const th = Math.tanh(u)
    const du = GELU_K * (1 + 3 * 0.044715 * x * x)
    dinp[i] = (0.5 * (1 + th) + 0.5 * x * (1 - th * th) * du) * dout[i]
  }
}

/** Causal multi-head self-attention. qkv rows are [q | k | v], each d wide; head h uses columns h*hs..(h+1)*hs. */
function attentionForward(out: Arr, att: Arr, qkv: Arr, B: number, T: number, d: number, H: number) {
  const hs = d / H
  const scale = 1 / Math.sqrt(hs)
  const C3 = 3 * d
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < T; t++) {
      const outOff = (b * T + t) * d
      for (let i = 0; i < d; i++) out[outOff + i] = 0
      for (let h = 0; h < H; h++) {
        const qOff = (b * T + t) * C3 + h * hs
        const aOff = ((b * H + h) * T + t) * T
        let maxv = -Infinity
        for (let t2 = 0; t2 <= t; t2++) {
          const kOff = (b * T + t2) * C3 + d + h * hs
          let s = 0
          for (let i = 0; i < hs; i++) s += qkv[qOff + i] * qkv[kOff + i]
          s *= scale
          att[aOff + t2] = s
          if (s > maxv) maxv = s
        }
        let sum = 0
        for (let t2 = 0; t2 <= t; t2++) {
          const e = Math.exp(att[aOff + t2] - maxv)
          att[aOff + t2] = e
          sum += e
        }
        const inv = 1 / sum
        for (let t2 = 0; t2 <= t; t2++) att[aOff + t2] *= inv
        for (let t2 = t + 1; t2 < T; t2++) att[aOff + t2] = 0
        const oh = outOff + h * hs
        for (let t2 = 0; t2 <= t; t2++) {
          const p = att[aOff + t2]
          const vOff = (b * T + t2) * C3 + 2 * d + h * hs
          for (let i = 0; i < hs; i++) out[oh + i] += p * qkv[vOff + i]
        }
      }
    }
  }
}

/** dqkv must be zeroed by the caller; it is accumulated. */
function attentionBackward(
  dqkv: Arr,
  dout: Arr,
  qkv: Arr,
  att: Arr,
  scratch: Arr,
  B: number,
  T: number,
  d: number,
  H: number,
) {
  const hs = d / H
  const scale = 1 / Math.sqrt(hs)
  const C3 = 3 * d
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < T; t++) {
      for (let h = 0; h < H; h++) {
        const qOff = (b * T + t) * C3 + h * hs
        const aOff = ((b * H + h) * T + t) * T
        const dOff = (b * T + t) * d + h * hs
        let dotSum = 0
        for (let t2 = 0; t2 <= t; t2++) {
          const vOff = (b * T + t2) * C3 + 2 * d + h * hs
          const p = att[aOff + t2]
          let dp = 0
          for (let i = 0; i < hs; i++) {
            const g = dout[dOff + i]
            dp += g * qkv[vOff + i]
            dqkv[vOff + i] += p * g
          }
          scratch[t2] = dp
          dotSum += p * dp
        }
        for (let t2 = 0; t2 <= t; t2++) {
          const ds = att[aOff + t2] * (scratch[t2] - dotSum) * scale
          if (ds === 0) continue
          const kOff = (b * T + t2) * C3 + d + h * hs
          for (let i = 0; i < hs; i++) {
            dqkv[qOff + i] += ds * qkv[kOff + i]
            dqkv[kOff + i] += ds * qkv[qOff + i]
          }
        }
      }
    }
  }
}

function softmaxRow(out: Arr | number[], logits: Arr, off: number, V: number, invTemp = 1) {
  let maxv = -Infinity
  for (let i = 0; i < V; i++) if (logits[off + i] > maxv) maxv = logits[off + i]
  let sum = 0
  for (let i = 0; i < V; i++) {
    const e = Math.exp((logits[off + i] - maxv) * invTemp)
    out[i] = e
    sum += e
  }
  for (let i = 0; i < V; i++) out[i] /= sum
}

/* ------------------------------------------------------------------ model */

export interface AdamW {
  lr: number
  beta1: number
  beta2: number
  eps: number
  weightDecay: number
}

export class GPT {
  readonly cfg: GPTConfig
  readonly params: Arr
  readonly grads: Arr
  readonly p: Params
  readonly g: Params
  readonly numParams: number
  private readonly specs: Spec[]
  private readonly ctor: ArrCtor
  private m: Arr | null = null
  private v: Arr | null = null
  private t = 0
  private acts = new Map<string, Acts>()
  private last: { a: Acts; idx: ArrayLike<number>; targets: ArrayLike<number> } | null = null

  constructor(cfg: GPTConfig, seed = 1337, ctor: ArrCtor = Float32Array) {
    if (cfg.dModel % cfg.nHead !== 0) throw new Error('dModel must be divisible by nHead')
    this.cfg = cfg
    this.ctor = ctor
    const { vocabSize: V, ctx, nLayer: L, dModel: d } = cfg
    const std = 0.02
    const projStd = 0.02 / Math.sqrt(2 * L)
    const specs: Spec[] = []
    let offset = 0
    const add = (key: string, size: number, init: Spec['init'], decay: boolean, s = std) => {
      specs.push({ key, size, init, decay, std: s, offset })
      offset += size
    }
    add('tok', V * d, 'normal', true)
    add('pos', ctx * d, 'normal', true, 0.01)
    for (let l = 0; l < L; l++) {
      add(`${l}.ln1g`, d, 'ones', false)
      add(`${l}.ln1b`, d, 'zeros', false)
      add(`${l}.wqkv`, d * 3 * d, 'normal', true)
      add(`${l}.bqkv`, 3 * d, 'zeros', false)
      add(`${l}.wo`, d * d, 'normal', true, projStd)
      add(`${l}.bo`, d, 'zeros', false)
      add(`${l}.ln2g`, d, 'ones', false)
      add(`${l}.ln2b`, d, 'zeros', false)
      add(`${l}.wfc`, d * 4 * d, 'normal', true)
      add(`${l}.bfc`, 4 * d, 'zeros', false)
      add(`${l}.wproj`, 4 * d * d, 'normal', true, projStd)
      add(`${l}.bproj`, d, 'zeros', false)
    }
    add('lnfg', d, 'ones', false)
    add('lnfb', d, 'zeros', false)
    add('wout', d * V, 'normal', true)
    add('bout', V, 'zeros', false)
    this.specs = specs
    this.numParams = offset
    this.params = new ctor(offset)
    this.grads = new ctor(offset)

    const rng = new Rng(seed)
    for (const s of specs) {
      const view = this.params.subarray(s.offset, s.offset + s.size)
      if (s.init === 'ones') view.fill(1)
      else if (s.init === 'normal') for (let i = 0; i < s.size; i++) view[i] = rng.normal() * s.std
    }
    this.p = this.views(this.params)
    this.g = this.views(this.grads)
  }

  /** Every parameter tensor in storage order: name, element count and offset into `params`. */
  layout(): { name: string; size: number; offset: number }[] {
    return this.specs.map((s) => ({ name: s.key, size: s.size, offset: s.offset }))
  }

  private views(buf: Arr): Params {
    const get = (key: string) => {
      const s = this.specs.find((x) => x.key === key)!
      return buf.subarray(s.offset, s.offset + s.size)
    }
    const layers: LayerParams[] = []
    for (let l = 0; l < this.cfg.nLayer; l++) {
      layers.push({
        ln1g: get(`${l}.ln1g`),
        ln1b: get(`${l}.ln1b`),
        wqkv: get(`${l}.wqkv`),
        bqkv: get(`${l}.bqkv`),
        wo: get(`${l}.wo`),
        bo: get(`${l}.bo`),
        ln2g: get(`${l}.ln2g`),
        ln2b: get(`${l}.ln2b`),
        wfc: get(`${l}.wfc`),
        bfc: get(`${l}.bfc`),
        wproj: get(`${l}.wproj`),
        bproj: get(`${l}.bproj`),
      })
    }
    return {
      tok: get('tok'),
      pos: get('pos'),
      layers,
      lnfg: get('lnfg'),
      lnfb: get('lnfb'),
      wout: get('wout'),
      bout: get('bout'),
    }
  }

  private getActs(B: number, T: number): Acts {
    const key = `${B}x${T}`
    let a = this.acts.get(key)
    if (a) return a
    if (this.acts.size > 6) this.acts.clear()
    const { dModel: d, nHead: H, vocabSize: V, nLayer: L } = this.cfg
    const N = B * T
    const C = this.ctor
    const layers: LayerActs[] = []
    for (let l = 0; l < L; l++) {
      layers.push({
        ln1: new C(N * d),
        ln1m: new C(N),
        ln1r: new C(N),
        qkv: new C(N * 3 * d),
        att: new C(B * H * T * T),
        atty: new C(N * d),
        res1: new C(N * d),
        ln2: new C(N * d),
        ln2m: new C(N),
        ln2r: new C(N),
        fch: new C(N * 4 * d),
        fcg: new C(N * 4 * d),
        res2: new C(N * d),
      })
    }
    a = { B, T, x0: new C(N * d), layers, lnf: new C(N * d), lnfm: new C(N), lnfr: new C(N), probs: new C(N * V) }
    this.acts.set(key, a)
    return a
  }

  /**
   * Forward pass over idx[B*T]. If targets are given, returns the mean cross-entropy
   * loss and remembers activations for backward(); otherwise returns NaN.
   */
  forward(idx: ArrayLike<number>, targets: ArrayLike<number> | null, B: number, T: number): number {
    const { dModel: d, nHead: H, vocabSize: V, ctx } = this.cfg
    if (T > ctx) throw new Error('sequence longer than context')
    const N = B * T
    const a = this.getActs(B, T)
    const p = this.p
    // embeddings
    for (let n = 0; n < N; n++) {
      const t = n % T
      const to = idx[n] * d
      const po = t * d
      const xo = n * d
      for (let i = 0; i < d; i++) a.x0[xo + i] = p.tok[to + i] + p.pos[po + i]
    }
    let x = a.x0
    for (let l = 0; l < this.cfg.nLayer; l++) {
      const lp = p.layers[l]
      const la = a.layers[l]
      layernormForward(la.ln1, la.ln1m, la.ln1r, x, lp.ln1g, lp.ln1b, N, d)
      matmulForward(la.qkv, la.ln1, lp.wqkv, lp.bqkv, N, d, 3 * d)
      attentionForward(la.atty, la.att, la.qkv, B, T, d, H)
      matmulForward(la.res1, la.atty, lp.wo, lp.bo, N, d, d)
      for (let i = 0; i < N * d; i++) la.res1[i] += x[i]
      layernormForward(la.ln2, la.ln2m, la.ln2r, la.res1, lp.ln2g, lp.ln2b, N, d)
      matmulForward(la.fch, la.ln2, lp.wfc, lp.bfc, N, d, 4 * d)
      geluForward(la.fcg, la.fch, N * 4 * d)
      matmulForward(la.res2, la.fcg, lp.wproj, lp.bproj, N, 4 * d, d)
      for (let i = 0; i < N * d; i++) la.res2[i] += la.res1[i]
      x = la.res2
    }
    layernormForward(a.lnf, a.lnfm, a.lnfr, x, p.lnfg, p.lnfb, N, d)
    matmulForward(a.probs, a.lnf, p.wout, p.bout, N, d, V)
    if (!targets) {
      this.last = null
      return NaN
    }
    let loss = 0
    for (let n = 0; n < N; n++) {
      const off = n * V
      softmaxRow(a.probs.subarray(off, off + V), a.probs, off, V)
      loss -= Math.log(Math.max(a.probs[off + targets[n]], 1e-30))
    }
    this.last = { a, idx, targets }
    return loss / N
  }

  /** Raw logits of the last forward pass without targets (row n). */
  logitsRow(B: number, T: number, n: number): Arr {
    const V = this.cfg.vocabSize
    return this.getActs(B, T).probs.subarray(n * V, (n + 1) * V)
  }

  /** Attention probabilities of the last forward pass: [layer][head][T*T] for batch row 0. */
  attention(B: number, T: number): Arr[][] {
    const a = this.getActs(B, T)
    const H = this.cfg.nHead
    return a.layers.map((la) => {
      const heads: Arr[] = []
      for (let h = 0; h < H; h++) heads.push(la.att.slice(h * T * T, (h + 1) * T * T))
      return heads
    })
  }

  zeroGrad() {
    this.grads.fill(0)
  }

  /** Backprop the loss of the last forward(idx, targets) into this.grads (accumulates). */
  backward() {
    if (!this.last) throw new Error('call forward with targets first')
    const { a, idx, targets } = this.last
    const { B, T } = a
    const { dModel: d, nHead: H, vocabSize: V, nLayer: L } = this.cfg
    const N = B * T
    const C = this.ctor
    if (!a.g) {
      a.g = {
        dres: new C(N * d),
        dln: new C(N * d),
        dqkv: new C(N * 3 * d),
        datty: new C(N * d),
        dfcg: new C(N * 4 * d),
        dfch: new C(N * 4 * d),
        dlnf: new C(N * d),
        dlogits: new C(N * V),
        scratch: new C(T),
      }
    }
    const gb = a.g
    const p = this.p
    const g = this.g

    // d loss / d logits = (probs - onehot) / N
    const invN = 1 / N
    for (let n = 0; n < N; n++) {
      const off = n * V
      for (let i = 0; i < V; i++) gb.dlogits[off + i] = a.probs[off + i] * invN
      gb.dlogits[off + targets[n]] -= invN
    }
    matmulBackward(gb.dlnf, g.wout, g.bout, gb.dlogits, a.lnf, p.wout, N, d, V)
    const xLast = L > 0 ? a.layers[L - 1].res2 : a.x0
    gb.dres.fill(0)
    layernormBackward(gb.dres, g.lnfg, g.lnfb, gb.dlnf, xLast, p.lnfg, a.lnfm, a.lnfr, N, d)

    for (let l = L - 1; l >= 0; l--) {
      const lp = p.layers[l]
      const lg = g.layers[l]
      const la = a.layers[l]
      const xin = l === 0 ? a.x0 : a.layers[l - 1].res2
      // MLP branch: res2 = res1 + proj(gelu(fc(ln2(res1))))
      matmulBackward(gb.dfcg, lg.wproj, lg.bproj, gb.dres, la.fcg, lp.wproj, N, 4 * d, d)
      geluBackward(gb.dfch, la.fch, gb.dfcg, N * 4 * d)
      matmulBackward(gb.dln, lg.wfc, lg.bfc, gb.dfch, la.ln2, lp.wfc, N, d, 4 * d)
      layernormBackward(gb.dres, lg.ln2g, lg.ln2b, gb.dln, la.res1, lp.ln2g, la.ln2m, la.ln2r, N, d)
      // attention branch: res1 = xin + wo(attn(qkv(ln1(xin))))
      matmulBackward(gb.datty, lg.wo, lg.bo, gb.dres, la.atty, lp.wo, N, d, d)
      gb.dqkv.fill(0)
      attentionBackward(gb.dqkv, gb.datty, la.qkv, la.att, gb.scratch, B, T, d, H)
      matmulBackward(gb.dln, lg.wqkv, lg.bqkv, gb.dqkv, la.ln1, lp.wqkv, N, d, 3 * d)
      layernormBackward(gb.dres, lg.ln1g, lg.ln1b, gb.dln, xin, lp.ln1g, la.ln1m, la.ln1r, N, d)
    }
    // embeddings
    for (let n = 0; n < N; n++) {
      const t = n % T
      const to = idx[n] * d
      const po = t * d
      const xo = n * d
      for (let i = 0; i < d; i++) {
        const gr = gb.dres[xo + i]
        g.tok[to + i] += gr
        g.pos[po + i] += gr
      }
    }
  }

  /** Clip the global grad norm to maxNorm; returns the norm before clipping. */
  clipGrads(maxNorm: number): number {
    let s = 0
    const gr = this.grads
    for (let i = 0; i < gr.length; i++) s += gr[i] * gr[i]
    const norm = Math.sqrt(s)
    if (norm > maxNorm && norm > 0) {
      const k = maxNorm / norm
      for (let i = 0; i < gr.length; i++) gr[i] *= k
    }
    return norm
  }

  /** One AdamW update (decoupled weight decay on matrices and embeddings only). */
  step(o: AdamW) {
    if (!this.m) {
      this.m = new this.ctor(this.numParams)
      this.v = new this.ctor(this.numParams)
    }
    const m = this.m
    const v = this.v!
    const w = this.params
    const gr = this.grads
    this.t++
    const bc1 = 1 - Math.pow(o.beta1, this.t)
    const bc2 = 1 - Math.pow(o.beta2, this.t)
    const b1 = o.beta1
    const b2 = o.beta2
    for (const s of this.specs) {
      const decay = s.decay ? o.lr * o.weightDecay : 0
      const end = s.offset + s.size
      for (let i = s.offset; i < end; i++) {
        const gi = gr[i]
        const mi = (m[i] = b1 * m[i] + (1 - b1) * gi)
        const vi = (v[i] = b2 * v[i] + (1 - b2) * gi * gi)
        w[i] -= o.lr * (mi / bc1 / (Math.sqrt(vi / bc2) + o.eps)) + decay * w[i]
      }
    }
  }

  /** Token embedding table as rows [V][d] (copy). */
  embeddings(): number[][] {
    const { vocabSize: V, dModel: d } = this.cfg
    const rows: number[][] = []
    for (let i = 0; i < V; i++) rows.push(Array.from(this.p.tok.subarray(i * d, (i + 1) * d)))
    return rows
  }
}

/* ------------------------------------------------------- KV-cached decoding */

/**
 * Incremental decoder: feeds one token at a time, caching each layer's keys and
 * values so generating a character costs one position, not the whole window.
 * When the window fills up, it re-primes on the most recent half of the text.
 */
export class Decoder {
  private readonly model: GPT
  private readonly kc: Arr[]
  private readonly vc: Arr[]
  private readonly x: Arr
  private readonly ln: Arr
  private readonly mean: Arr
  private readonly rstd: Arr
  private readonly qkv: Arr
  private readonly atty: Arr
  private readonly tmp: Arr
  private readonly fch: Arr
  private readonly att: Arr
  readonly logits: Arr
  /** tokens currently held in the cache, in order */
  private window: number[] = []

  constructor(model: GPT) {
    this.model = model
    const { dModel: d, ctx, nLayer: L, vocabSize: V } = model.cfg
    const C = Float64Array
    this.kc = Array.from({ length: L }, () => new C(ctx * d))
    this.vc = Array.from({ length: L }, () => new C(ctx * d))
    this.x = new C(d)
    this.ln = new C(d)
    this.mean = new C(1)
    this.rstd = new C(1)
    this.qkv = new C(3 * d)
    this.atty = new C(d)
    this.tmp = new C(d)
    this.fch = new C(4 * d)
    this.att = new C(ctx)
    this.logits = new C(V)
  }

  /** Reset and feed a whole prompt (keeps the last ctx tokens). */
  prime(tokens: number[]) {
    const ctx = this.model.cfg.ctx
    const keep = tokens.slice(-ctx)
    this.window = []
    for (const t of keep) this.feed(t)
  }

  /** Append one token; afterwards this.logits holds the next-token logits. */
  push(token: number) {
    const ctx = this.model.cfg.ctx
    if (this.window.length >= ctx) {
      const recent = this.window.slice(-Math.floor(ctx / 2))
      this.prime(recent)
    }
    this.feed(token)
  }

  private feed(token: number) {
    const m = this.model
    const { dModel: d, nHead: H, nLayer: L, vocabSize: V } = m.cfg
    const p = m.p
    const pos = this.window.length
    this.window.push(token)
    const hs = d / H
    const scale = 1 / Math.sqrt(hs)
    const x = this.x
    for (let i = 0; i < d; i++) x[i] = p.tok[token * d + i] + p.pos[pos * d + i]
    for (let l = 0; l < L; l++) {
      const lp = p.layers[l]
      layernormForward(this.ln, this.mean, this.rstd, x, lp.ln1g, lp.ln1b, 1, d)
      matmulForward(this.qkv, this.ln, lp.wqkv, lp.bqkv, 1, d, 3 * d)
      const kc = this.kc[l]
      const vc = this.vc[l]
      for (let i = 0; i < d; i++) {
        kc[pos * d + i] = this.qkv[d + i]
        vc[pos * d + i] = this.qkv[2 * d + i]
      }
      for (let h = 0; h < H; h++) {
        const ho = h * hs
        let maxv = -Infinity
        for (let t2 = 0; t2 <= pos; t2++) {
          let s = 0
          for (let i = 0; i < hs; i++) s += this.qkv[ho + i] * kc[t2 * d + ho + i]
          s *= scale
          this.att[t2] = s
          if (s > maxv) maxv = s
        }
        let sum = 0
        for (let t2 = 0; t2 <= pos; t2++) {
          const e = Math.exp(this.att[t2] - maxv)
          this.att[t2] = e
          sum += e
        }
        for (let i = 0; i < hs; i++) this.atty[ho + i] = 0
        for (let t2 = 0; t2 <= pos; t2++) {
          const pr = this.att[t2] / sum
          for (let i = 0; i < hs; i++) this.atty[ho + i] += pr * vc[t2 * d + ho + i]
        }
      }
      matmulForward(this.tmp, this.atty, lp.wo, lp.bo, 1, d, d)
      for (let i = 0; i < d; i++) x[i] += this.tmp[i]
      layernormForward(this.ln, this.mean, this.rstd, x, lp.ln2g, lp.ln2b, 1, d)
      matmulForward(this.fch, this.ln, lp.wfc, lp.bfc, 1, d, 4 * d)
      geluForward(this.fch, this.fch, 4 * d)
      matmulForward(this.tmp, this.fch, lp.wproj, lp.bproj, 1, 4 * d, d)
      for (let i = 0; i < d; i++) x[i] += this.tmp[i]
    }
    layernormForward(this.ln, this.mean, this.rstd, x, p.lnfg, p.lnfb, 1, d)
    matmulForward(this.logits, this.ln, p.wout, p.bout, 1, d, V)
  }

  /** Temperature-scaled next-token distribution from the current logits. */
  probs(temperature: number): number[] {
    const V = this.model.cfg.vocabSize
    const out = new Array<number>(V)
    softmaxRow(out, this.logits, 0, V, 1 / Math.max(temperature, 1e-3))
    return out
  }
}

/** Draw an index from a probability vector. */
export function sampleFrom(probs: number[], rng: Rng): number {
  const r = rng.next()
  let c = 0
  for (let i = 0; i < probs.length; i++) {
    c += probs[i]
    if (r < c) return i
  }
  return probs.length - 1
}

/** Indices of the k largest entries, largest first. */
export function topK(probs: number[], k: number): number[] {
  const idx = probs.map((_, i) => i)
  idx.sort((a, b) => probs[b] - probs[a])
  return idx.slice(0, k)
}
