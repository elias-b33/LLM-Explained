/// <reference types="@webgpu/types" />
/** WebGPU adapter detection and a cached device, used from inside the training worker. */

export interface GpuInfo {
  /** ok = hardware WebGPU usable; none = no WebGPU / no adapter; software = only a fallback (CPU-emulated) adapter */
  status: 'ok' | 'none' | 'software'
  reason: string
  vendor: string
  architecture: string
  description: string
  isFallback: boolean
  maxBufferSize: number
  maxStorageBufferBindingSize: number
}

const SOFTWARE_RE = /swiftshader|llvmpipe|lavapipe|basic render|software|\bwarp\b/i

let adapterP: Promise<GPUAdapter | null> | null = null
let deviceP: Promise<GPUDevice> | null = null

function requestAdapter(): Promise<GPUAdapter | null> {
  if (!adapterP) {
    const gpu = (navigator as Navigator & { gpu?: GPU }).gpu
    adapterP = gpu ? gpu.requestAdapter({ powerPreference: 'high-performance' }).catch(() => null) : Promise.resolve(null)
  }
  return adapterP
}

function isFallback(a: GPUAdapter): boolean {
  const info = a.info as GPUAdapterInfo & { isFallbackAdapter?: boolean }
  const legacy = (a as GPUAdapter & { isFallbackAdapter?: boolean }).isFallbackAdapter
  return Boolean(info?.isFallbackAdapter ?? legacy ?? false)
}

export async function probeGpu(): Promise<GpuInfo> {
  const empty: GpuInfo = {
    status: 'none',
    reason: '',
    vendor: '',
    architecture: '',
    description: '',
    isFallback: false,
    maxBufferSize: 0,
    maxStorageBufferBindingSize: 0,
  }
  if (!(navigator as Navigator & { gpu?: GPU }).gpu) {
    return { ...empty, reason: 'This browser has no WebGPU.' }
  }
  const a = await requestAdapter()
  if (!a) {
    return { ...empty, reason: 'WebGPU is present, but no GPU adapter was offered.' }
  }
  const info = a.info
  const out: GpuInfo = {
    status: 'ok',
    reason: '',
    vendor: info?.vendor ?? '',
    architecture: info?.architecture ?? '',
    description: info?.description ?? '',
    isFallback: isFallback(a),
    maxBufferSize: a.limits.maxBufferSize,
    maxStorageBufferBindingSize: a.limits.maxStorageBufferBindingSize,
  }
  const text = `${out.vendor} ${out.architecture} ${out.description}`
  if (out.isFallback || SOFTWARE_RE.test(text)) {
    out.status = 'software'
    out.reason = 'Only a software (fallback) WebGPU adapter is available. It runs on the CPU and would be slower than the CPU engine.'
  }
  return out
}

/** The worker's GPU device (created once; recreated after a device loss). */
export function getDevice(onLost: (msg: string) => void): Promise<GPUDevice> {
  if (!deviceP) {
    deviceP = (async () => {
      const a = await requestAdapter()
      if (!a) throw new Error('No WebGPU adapter')
      const dev = await a.requestDevice({
        requiredLimits: {
          maxBufferSize: a.limits.maxBufferSize,
          maxStorageBufferBindingSize: a.limits.maxStorageBufferBindingSize,
          maxStorageBuffersPerShaderStage: Math.min(8, a.limits.maxStorageBuffersPerShaderStage),
        },
      })
      void dev.lost.then((e) => {
        deviceP = null
        adapterP = null
        if (e.reason !== 'destroyed') onLost(e.message || 'The GPU device was lost.')
      })
      return dev
    })()
    deviceP.catch(() => {
      deviceP = null
    })
  }
  return deviceP
}

/** Human-readable adapter name, e.g. "nvidia · lovelace". */
export function adapterLabel(i: Pick<GpuInfo, 'vendor' | 'architecture' | 'description'>): string {
  const parts = [i.description, i.vendor, i.architecture].filter(Boolean)
  // description often repeats the vendor; keep it short
  const uniq = parts.filter((p, k) => parts.findIndex((q) => q.toLowerCase().includes(p.toLowerCase())) === k)
  return uniq.join(' · ') || 'unnamed GPU'
}
