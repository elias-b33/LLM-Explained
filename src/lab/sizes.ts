export type SizeId = 'tiny' | 'small' | 'medium' | 'large' | 'huge' | 'xl'
export type StartFrom = 'scratch' | 'pretrained'
export type Device = 'cpu' | 'gpu'

export interface ModelSize {
  id: SizeId
  label: string
  nLayer: number
  dModel: number
  nHead: number
  ctx: number
  batch: number
  lr: number
  blurb: string
  /** plain-language GPU requirement */
  requirement: string
  /** big enough that the CPU engine is impractical */
  gpuRecommended: boolean
  /** too big for the CPU engine at all: only offered on the GPU */
  gpuOnly?: boolean
  /** a pretrained checkpoint can exist for this size (public/models/<id>.bin) */
  pretrainable?: boolean
}

export const SIZES: ModelSize[] = [
  {
    id: 'tiny',
    label: 'Tiny',
    nLayer: 1,
    dModel: 32,
    nHead: 2,
    ctx: 32,
    batch: 32,
    lr: 5e-3,
    blurb: 'Learns fast. Words within a minute.',
    requirement: 'Any GPU',
    gpuRecommended: false,
  },
  {
    id: 'small',
    label: 'Small',
    nLayer: 2,
    dModel: 64,
    nHead: 4,
    ctx: 64,
    batch: 12,
    lr: 4e-3,
    blurb: 'Slower, but remembers more.',
    requirement: 'Any GPU',
    gpuRecommended: false,
  },
  {
    id: 'medium',
    label: 'Medium',
    nLayer: 3,
    dModel: 96,
    nHead: 4,
    ctx: 64,
    batch: 8,
    lr: 3e-3,
    blurb: 'Patient visitors only (on the CPU).',
    requirement: 'Any GPU',
    gpuRecommended: false,
  },
  {
    id: 'large',
    label: 'Large',
    nLayer: 4,
    dModel: 192,
    nHead: 6,
    ctx: 128,
    batch: 16,
    lr: 2e-3,
    blurb: 'Sees 128 characters back. Built for the GPU.',
    requirement: 'Dedicated GPU recommended, 2 GB+ VRAM',
    gpuRecommended: true,
    pretrainable: true,
  },
  {
    id: 'huge',
    label: 'Huge',
    nLayer: 6,
    dModel: 384,
    nHead: 6,
    ctx: 256,
    batch: 8,
    lr: 1e-3,
    blurb: 'About 10 million knobs. A real workout for a graphics card.',
    requirement: 'Dedicated GPU, 4 GB+ VRAM',
    gpuRecommended: true,
    pretrainable: true,
  },
  {
    id: 'xl',
    label: 'XL',
    nLayer: 8,
    dModel: 512,
    nHead: 8,
    ctx: 256,
    batch: 8,
    lr: 6e-4,
    blurb: 'About 25 million knobs, eight layers deep. GPU only.',
    requirement: 'GPU only: dedicated GPU, 4 GB+ VRAM',
    gpuRecommended: true,
    gpuOnly: true,
    pretrainable: true,
  },
]

/**
 * Fine-tuning a pretrained model. The learning rate is 1/50 of what the same
 * size uses from scratch (= its pretraining peak). Measured with the PyTorch
 * twin on Macbeth (Large, same batch and schedule as the page): at 1/5 the
 * validation loss bottoms out after ~100 steps and then climbs as the model
 * memorises the play (1.56 -> 1.32 -> 2.16 per byte by step 900); at 1/50 it
 * falls to ~1.28 and stays there for the first several minutes. The warmup
 * ramps the rate up while AdamW's running averages, which start at zero, settle.
 */
export const FINETUNE = {
  lrScale: 0.02,
  warmupSteps: 100,
  weightDecay: 0.01,
}
/** From scratch: a short warmup and the size's full learning rate (unchanged since the first Workshop). */
export const SCRATCH_WARMUP = 30

/** Learning rate at a (0-based) step. */
export function lrAt(s: ModelSize, start: StartFrom, step: number): number {
  const base = start === 'pretrained' ? s.lr * FINETUNE.lrScale : s.lr
  const warm = start === 'pretrained' ? FINETUNE.warmupSteps : SCRATCH_WARMUP
  return base * Math.min(1, (step + 1) / warm)
}

export function getSize(id: SizeId): ModelSize {
  return SIZES.find((s) => s.id === id) ?? SIZES[0]
}

/** Exact parameter count of the GPT in engine/gpt.ts for a given vocabulary size. */
export function paramCount(s: ModelSize, vocabSize: number): number {
  const d = s.dModel
  const perLayer = 2 * d + (d * 3 * d + 3 * d) + (d * d + d) + 2 * d + (d * 4 * d + 4 * d) + (4 * d * d + d)
  return vocabSize * d + s.ctx * d + s.nLayer * perLayer + 2 * d + d * vocabSize + vocabSize
}

/**
 * GPU memory the WebGPU engine allocates (engine/gpu/gpuGpt.ts), in bytes:
 * weights + gradients + two AdamW moments (fp32), every activation the backward
 * pass needs (including the B x heads x T x T attention matrices of each layer),
 * the backward scratch buffers, and small extras. Also returns the largest
 * single buffer, which is what WebGPU's per-buffer limits are checked against.
 */
export function gpuMemory(s: ModelSize, vocabSize: number): { total: number; largestBuffer: number } {
  const P = paramCount(s, vocabSize)
  const { dModel: d, nHead: H, ctx: T, batch: B, nLayer: L } = s
  const V = vocabSize
  const N = B * T
  const att = B * H * T * T
  const optimizer = 4 * P + P / 32 // params, grads, m, v, decay mask
  const perLayer = 16 * N * d + 4 * N + att
  const global = 2 * N * d + 2 * N + 2 * N * V + 3 * N // x0, lnf, ln stats, logits, dlogits, row loss, token ids
  const scratch = 15 * N * d + att
  const readback = P // staging copy when weights are read back for sampling
  const floats = optimizer + L * perLayer + global + scratch + readback
  const extras = 256 * 1024 // uniforms, norm partials
  const largest = Math.max(P, att, 4 * N * d, N * V) * 4
  return { total: floats * 4 + extras, largestBuffer: largest }
}

/** Approximate floating-point operations in one training step (forward + backward). */
export function stepFlops(s: ModelSize, vocabSize: number): number {
  const { dModel: d, ctx: T, batch: B, nLayer: L } = s
  const N = B * T
  const matmul = 6 * N * (12 * L * d * d + d * vocabSize)
  const attention = 6 * L * B * T * T * d // causal: about half of the full T x T, times fwd + 2x bwd
  return matmul + attention
}

/*
 * Speed estimates. CPU: about 1.1 GFLOP/s for one thread of the TypeScript
 * engine (Medium runs ~1 step/s). GPU: measured on a desktop RTX 4080-class
 * card: big models reach ~1.2 TFLOP/s, small ones add a fixed cost per step
 * (~0.3 ms plus ~0.6 ms per layer of kernel launches). Real numbers vary a lot
 * between GPUs, so the UI labels these as estimates.
 */
const CPU_FLOPS = 1.1e9
const GPU_FLOPS = 1.2e12
const GPU_BASE_S = 0.0003
const GPU_PER_LAYER_S = 0.0006

/** Rough steps per second on a device. */
export function estStepsPerSec(s: ModelSize, vocabSize: number, device: Device): number {
  const fl = stepFlops(s, vocabSize)
  if (device === 'cpu') return CPU_FLOPS / fl
  return 1 / (fl / GPU_FLOPS + GPU_BASE_S + GPU_PER_LAYER_S * s.nLayer)
}

export function fmtSpeed(sps: number): string {
  if (sps >= 10) return `~${Math.round(sps / (sps >= 100 ? 10 : 1)) * (sps >= 100 ? 10 : 1)} steps/s`
  if (sps >= 1) return `~${sps.toFixed(1).replace(/\.0$/, '')} steps/s`
  const sec = 1 / sps
  if (sec < 60) return `~${Math.round(sec)} s per step`
  return `~${Math.round(sec / 60)} min per step`
}

export function fmtBytes(b: number): string {
  if (b >= 1e9) return `${(b / 1e9).toFixed(1)} GB`
  if (b >= 1e6) return `${Math.round(b / 1e6)} MB`
  return `${Math.max(1, Math.round(b / 1e3))} kB`
}

/** Frontier LLMs are commonly estimated at around a trillion parameters (exact sizes are not published). */
export const FRONTIER_PARAMS = 1e12

export function frontierRatio(params: number): string {
  const r = FRONTIER_PARAMS / params
  if (r >= 1e6) return `${Math.round(r / 1e6)} million`
  return `${Math.round(r / 1e3)} thousand`
}
