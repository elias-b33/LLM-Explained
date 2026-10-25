import { CanvasTexture, SRGBColorSpace } from 'three'
import { activeProfile, textureAnisotropy } from './quality'

/**
 * A canvas-backed texture drawn in logical pixels. The backing canvas is
 * scaled up at High quality so boards stay crisp at dpr 2. Redraw only when
 * content changes: every `draw()` re-uploads the texture.
 */
export interface Board {
  tex: CanvasTexture
  w: number
  h: number
  draw: (fn: (ctx: CanvasRenderingContext2D) => void) => void
}

export function makeBoard(w: number, h: number, scale: number = activeProfile().boardScale): Board {
  const c = document.createElement('canvas')
  c.width = Math.round(w * scale)
  c.height = Math.round(h * scale)
  const ctx = c.getContext('2d')!
  const tex = new CanvasTexture(c)
  tex.colorSpace = SRGBColorSpace
  tex.anisotropy = textureAnisotropy()
  return {
    tex,
    w,
    h,
    draw(fn) {
      ctx.setTransform(scale, 0, 0, scale, 0, 0)
      ctx.clearRect(0, 0, w, h)
      fn(ctx)
      tex.needsUpdate = true
    },
  }
}

/** One-shot static board. */
export function staticBoard(w: number, h: number, fn: (ctx: CanvasRenderingContext2D) => void, scale?: number) {
  const b = makeBoard(w, h, scale)
  b.draw(fn)
  return b.tex
}

let measureCtx: CanvasRenderingContext2D | null = null
export function measure(text: string, font: string): number {
  if (!measureCtx) measureCtx = document.createElement('canvas').getContext('2d')!
  measureCtx.font = font
  return measureCtx.measureText(text).width
}

export function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.arcTo(x + w, y, x + w, y + h, r)
  ctx.arcTo(x + w, y + h, x, y + h, r)
  ctx.arcTo(x, y + h, x, y, r)
  ctx.arcTo(x, y, x + w, y, r)
  ctx.closePath()
}

export const VIOLET = '#8b7bff'
export const CYAN = '#2bd3e0'
export const MUSTARD = '#ffc94a'
export const MONO = '"JetBrains Mono", ui-monospace, monospace'
export const SANS = 'Inter, system-ui, sans-serif'
