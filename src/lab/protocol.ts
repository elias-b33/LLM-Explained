import type { Device, SizeId, StartFrom } from './sizes'
import type { GpuInfo } from './engine/gpu/device.ts'
import type { SelfTestResult } from './engine/gpu/selftest.ts'

export type { GpuInfo, SelfTestResult }
import type { GPTConfig } from './engine/gpt.ts'
import type { WeightsHeader } from './engine/weights.ts'

export type { WeightsHeader }

/* main thread -> trainer worker */
export type ToTrainer =
  | { type: 'connect'; helpers: MessagePort[] }
  | {
      type: 'init'
      text: string
      size: SizeId
      prompt: string
      seed: number
      device: Device
      /** scratch: per-text character vocab, random weights. pretrained: byte vocab, weights from `weights` */
      start: StartFrom
      /** the raw .bin file (pretrained only) */
      weights?: ArrayBuffer
    }
  | { type: 'start' }
  | { type: 'pause' }
  | { type: 'reset' }
  | { type: 'generate'; id: number; prompt: string; temperature: number; length: number }
  | { type: 'stopGenerate' }
  | { type: 'insight'; prompt: string }
  /** weights: optionally also check the GPU against the CPU on these pretrained weights */
  | { type: 'gpuSelfTest'; weights?: ArrayBuffer }

export interface ReadyMsg {
  type: 'ready'
  /** display label per token id */
  vocab: string[]
  vocabKind: 'chars' | 'bytes'
  /** header of the loaded weights file when fine-tuning a pretrained model */
  pretrained: WeightsHeader | null
  params: number
  trainChars: number
  valChars: number
  prompt: string
  threadsAvailable: number
  /** the engine actually in use (GPU can fall back to CPU) */
  device: Device
  /** GPU memory allocated by the model, in bytes (GPU only) */
  gpuBytes: number
}

export interface ProgressMsg {
  type: 'progress'
  step: number
  /** per-step training losses for steps [fromStep, step) */
  fromStep: number
  losses: number[]
  /** [step, val loss] pairs measured since the last message */
  val: [number, number][]
  charsPerSec: number
  trainMs: number
  running: boolean
  backend: string
  sampleEvery: number
}

export interface SampleMsg {
  type: 'sample'
  step: number
  loss: number
  prompt: string
  text: string
  /** written by the pretrained model before any fine-tuning */
  before?: boolean
}

export interface GenMsg {
  type: 'gen'
  id: number
  /** text completed by the newly generated token ('' while a multi-byte character is unfinished) */
  ch: string
  /** top-5 next-token probabilities (temperature-scaled) before it was drawn: [label, p, label is a byte placeholder] */
  top: [string, number, boolean][]
}

export interface InsightMsg {
  type: 'insight'
  /** 2D PCA of the character embeddings */
  points: { ch: string; x: number; y: number }[]
  /** attention of the last layer for the prompt tail: heads x T x T */
  attn: { tokens: string[]; heads: number[][] } | null
}

export type FromTrainer =
  | ReadyMsg
  | ProgressMsg
  | SampleMsg
  | GenMsg
  /** tail: leftover text of an unfinished multi-byte character (U+FFFD) */
  | { type: 'genDone'; id: number; tail?: string }
  | InsightMsg
  | { type: 'error'; message: string }
  | { type: 'gpuInfo'; info: GpuInfo }
  | { type: 'selfTest'; result: SelfTestResult }

/* trainer <-> helper (data-parallel gradient workers) */
export type ToHelper =
  | { type: 'helperInit'; port: MessagePort }
  | { type: 'cfg'; cfg: GPTConfig }
  | { type: 'job'; seq: number; params: Float32Array; x: Uint16Array; y: Uint16Array; B: number; T: number }

export type FromHelper = { type: 'grads'; seq: number; grads: Float32Array; loss: number } | { type: 'helperError'; message: string }
