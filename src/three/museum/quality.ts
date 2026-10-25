/**
 * Museum graphics quality. "high" loads the 2K texture set, renders at up to
 * 2× device pixels and uses 16× anisotropy and sharper canvas boards;
 * "standard" is the lighter 1K / 1.5× profile. Software-rendered WebGL
 * (SwiftShader, WARP, llvmpipe, Microsoft Basic Render) defaults to standard.
 *
 * The active profile is module-level (only one hall is ever mounted) and is set
 * by HallScene before its Canvas children render; the Canvas is keyed by the
 * quality so switching remounts everything with the new profile.
 */
export type Quality = 'standard' | 'high'

interface Profile {
  /** R3F dpr. A [min, max] range only clamps devicePixelRatio, so High uses a
   *  fixed 2 to actually supersample on 1× (e.g. 1080p) screens. */
  dpr: number | [number, number]
  anisotropy: number
  /** EffectComposer MSAA samples (the canvas itself is created without AA). */
  msaa: number
  res: '1k' | '2k'
  /** Backing-store scale for canvas-texture boards. */
  boardScale: number
}

export const QUALITY_PROFILES: Record<Quality, Profile> = {
  standard: { dpr: [1, 1.5], anisotropy: 8, msaa: 4, res: '1k', boardScale: 1 },
  high: { dpr: 2, anisotropy: 16, msaa: 4, res: '2k', boardScale: 1.6 },
}

const KEY = 'museum-quality'

interface GpuInfo {
  software: boolean
  maxAnisotropy: number
}

let gpu: GpuInfo | null = null

/** Probe the GPU once with a throwaway WebGL context. */
export function gpuInfo(): GpuInfo {
  if (gpu) return gpu
  gpu = { software: false, maxAnisotropy: 8 }
  try {
    const c = document.createElement('canvas')
    const gl = (c.getContext('webgl2') || c.getContext('webgl')) as WebGLRenderingContext | null
    if (gl) {
      const dbg = gl.getExtension('WEBGL_debug_renderer_info')
      const renderer = dbg ? String(gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL)) : ''
      gpu.software = /SwiftShader|Basic Render|llvmpipe|WARP/i.test(renderer)
      const aniso = gl.getExtension('EXT_texture_filter_anisotropic')
      gpu.maxAnisotropy = aniso ? Number(gl.getParameter(aniso.MAX_TEXTURE_MAX_ANISOTROPY_EXT)) || 1 : 1
      gl.getExtension('WEBGL_lose_context')?.loseContext()
    }
  } catch {
    /* keep defaults */
  }
  return gpu
}

export function loadQuality(): Quality {
  try {
    const v = localStorage.getItem(KEY)
    if (v === 'standard' || v === 'high') return v
  } catch {
    /* storage blocked */
  }
  return gpuInfo().software ? 'standard' : 'high'
}

export function saveQuality(q: Quality) {
  try {
    localStorage.setItem(KEY, q)
  } catch {
    /* storage blocked */
  }
}

let active: Quality = 'high'

export function setActiveQuality(q: Quality) {
  active = q
}

export function activeProfile() {
  return QUALITY_PROFILES[active]
}

/** Anisotropy for the active profile, capped at what the GPU supports. */
export function textureAnisotropy(): number {
  return Math.max(1, Math.min(activeProfile().anisotropy, gpuInfo().maxAnisotropy))
}
