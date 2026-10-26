/// <reference types="@webgpu/types" />
/** The "keep the model" preference, storage cleanup re-exports, and the WebGPU probe (page side). */
import { adapterLabel, probeGpu } from '@/lab/engine/gpu/device'
import type { GpuProbe } from './protocol'
export { clearModelStorage as clearModelCache, storedModelEntries as cachedModelFiles } from './wipe'

const KEEP_KEY = 'llm-explainer:ask:keep-model'

export function readKeep(): boolean {
  try {
    return localStorage.getItem(KEEP_KEY) === '1'
  } catch {
    return false
  }
}

export function writeKeep(v: boolean) {
  try {
    if (v) localStorage.setItem(KEEP_KEY, '1')
    else localStorage.removeItem(KEEP_KEY)
  } catch {
    /* private mode etc.: the choice just won't persist */
  }
}

export async function probe(forceNoGpu: boolean): Promise<GpuProbe> {
  if (forceNoGpu) return { status: 'none', f16: false, adapter: '', reason: 'WebGPU disabled by the ?nogpu dev flag.' }
  const info = await probeGpu()
  if (info.status !== 'ok') return { status: info.status, f16: false, adapter: adapterLabel(info), reason: info.reason }
  let f16: boolean
  try {
    const a = await (navigator as Navigator & { gpu?: GPU }).gpu?.requestAdapter({ powerPreference: 'high-performance' })
    f16 = Boolean(a?.features.has('shader-f16'))
  } catch {
    f16 = false
  }
  return { status: 'ok', f16, adapter: adapterLabel(info), reason: '' }
}
