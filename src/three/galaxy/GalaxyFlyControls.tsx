import { useEffect, type RefObject } from 'react'
import { useThree } from '@react-three/fiber'
import type CameraControlsImpl from 'camera-controls'

const MAP: Record<string, string> = {
  KeyW: 'fwd',
  ArrowUp: 'fwd',
  KeyS: 'back',
  ArrowDown: 'back',
  KeyA: 'left',
  ArrowLeft: 'left',
  KeyD: 'right',
  ArrowRight: 'right',
  KeyE: 'up',
  Space: 'up',
  KeyQ: 'down',
}

/**
 * Free-fly through the galaxy with WASD (+ Q/E or Space for down/up, Shift to
 * boost), while CameraControls still handles drag-to-look and scroll-to-zoom and
 * the cursor stays free for hovering/clicking words. Movement drives camera-controls
 * forward()/truck() on a self-paced rAF loop and invalidates so it renders on demand.
 */
export function GalaxyFlyControls({ controls }: { controls: RefObject<CameraControlsImpl | null> }) {
  const invalidate = useThree((s) => s.invalidate)

  useEffect(() => {
    const keys: Record<string, boolean> = {}
    let raf = 0
    let last = 0
    let boost = false

    const moving = () => keys.fwd || keys.back || keys.left || keys.right || keys.up || keys.down

    const tick = (t: number) => {
      const c = controls.current
      const dt = last ? Math.min((t - last) / 1000, 0.05) : 0.016
      last = t
      if (c) {
        const v = (boost ? 34 : 14) * dt
        if (keys.fwd) c.forward(v, false)
        if (keys.back) c.forward(-v, false)
        if (keys.left) c.truck(-v, 0, false)
        if (keys.right) c.truck(v, 0, false)
        if (keys.up) c.truck(0, v, false)
        if (keys.down) c.truck(0, -v, false)
        invalidate()
      }
      if (moving()) raf = requestAnimationFrame(tick)
      else {
        raf = 0
        last = 0
      }
    }

    const typing = () => {
      const tag = document.activeElement?.tagName
      return tag === 'INPUT' || tag === 'TEXTAREA'
    }

    const onDown = (e: KeyboardEvent) => {
      if (typing()) return
      if (e.code === 'ShiftLeft' || e.code === 'ShiftRight') {
        boost = true
        return
      }
      const k = MAP[e.code]
      if (!k) return
      if (e.code === 'Space') e.preventDefault()
      keys[k] = true
      if (!raf) {
        last = 0
        raf = requestAnimationFrame(tick)
      }
    }
    const onUp = (e: KeyboardEvent) => {
      if (e.code === 'ShiftLeft' || e.code === 'ShiftRight') {
        boost = false
        return
      }
      const k = MAP[e.code]
      if (k) keys[k] = false
    }

    window.addEventListener('keydown', onDown)
    window.addEventListener('keyup', onUp)
    return () => {
      window.removeEventListener('keydown', onDown)
      window.removeEventListener('keyup', onUp)
      if (raf) cancelAnimationFrame(raf)
    }
  }, [controls, invalidate])

  return null
}
