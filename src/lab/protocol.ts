import type { SizeId } from './sizes'
import type { GPTConfig } from './engine/gpt.ts'

/* main thread -> trainer worker */
export type ToTrainer =
  | { type: 'connect'; helpers: MessagePort[] }
  | { type: 'init'; text: string; size: SizeId; prompt: string; seed: number }
  | { type: 'start' }
  | { type: 'pause' }
  | { type: 'reset' }
  | { type: 'generate'; id: number; prompt: string; temperature: number; length: number }
  | { type: 'stopGenerate' }
  | { type: 'insight'; prompt: string }

export interface ReadyMsg {
  type: 'ready'
  vocab: string[]
  params: number
  trainChars: number
  valChars: number
  prompt: string
  threadsAvailable: number
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
}

export interface GenMsg {
  type: 'gen'
  id: number
  /** newly generated character */
  ch: string
  /** top-5 next-character probabilities (temperature-scaled) before this character was drawn */
  top: [string, number][]
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
  | { type: 'genDone'; id: number }
  | InsightMsg
  | { type: 'error'; message: string }

/* trainer <-> helper (data-parallel gradient workers) */
export type ToHelper =
  | { type: 'helperInit'; port: MessagePort }
  | { type: 'cfg'; cfg: GPTConfig }
  | { type: 'job'; seq: number; params: Float32Array; x: Uint16Array; y: Uint16Array; B: number; T: number }

export type FromHelper = { type: 'grads'; seq: number; grads: Float32Array; loss: number } | { type: 'helperError'; message: string }
