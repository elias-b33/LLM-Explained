import { useEffect, useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html } from '@react-three/drei'
import type { GalaxyData } from './types'

const NEAR_N = 12
const LABEL_RANGE_SQ = 28 * 28
/** Minimum seconds between near-field scans. */
const SCAN_INTERVAL = 0.12

/**
 * Labels only the words nearest the camera (recomputed a few times a second) plus
 * any "forced" words (hovered / selected / neighbors). A single-pass top-N keeps it
 * cheap even at tens of thousands of points; labels are DOM (drei Html) so they're
 * crisp, render every script (incl. Arabic), and don't fight the canvas capture.
 */
export function GalaxyLabels({ data, forced }: { data: GalaxyData; forced: number[] }) {
  const [near, setNear] = useState<number[]>([])
  const bufIdx = useRef(new Int32Array(NEAR_N))
  const bufDist = useRef(new Float64Array(NEAR_N))

  const lastScan = useRef(-Infinity)
  const trailing = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(() => () => {
    if (trailing.current) clearTimeout(trailing.current)
  }, [])

  // Runs only on rendered frames (the canvas is frameloop="demand"). The scan walks
  // every word, so throttle it to ~8 Hz; when a frame is skipped, schedule a trailing
  // invalidate so the camera's final resting position always gets scanned.
  useFrame((state) => {
    const now = state.clock.elapsedTime
    if (now - lastScan.current < SCAN_INTERVAL) {
      if (!trailing.current) {
        trailing.current = setTimeout(() => {
          trailing.current = null
          state.invalidate()
        }, SCAN_INTERVAL * 1000 + 10)
      }
      return
    }
    lastScan.current = now
    const cam = state.camera.position
    const words = data.words
    const idx = bufIdx.current
    const dist = bufDist.current
    idx.fill(-1)
    dist.fill(Infinity)
    let worst = Infinity
    let worstPos = 0
    for (let i = 0; i < words.length; i++) {
      const p = words[i].p
      const dx = p[0] - cam.x
      const dy = p[1] - cam.y
      const dz = p[2] - cam.z
      const d = dx * dx + dy * dy + dz * dz
      if (d < worst) {
        idx[worstPos] = i
        dist[worstPos] = d
        worst = -Infinity
        for (let k = 0; k < NEAR_N; k++) {
          if (dist[k] > worst) {
            worst = dist[k]
            worstPos = k
          }
        }
      }
    }
    // Only label stars that are genuinely close, so labels fade in as you fly
    // into a cloud and the zoomed-out overview stays clean.
    const next: number[] = []
    for (let k = 0; k < NEAR_N; k++) if (idx[k] >= 0 && dist[k] < LABEL_RANGE_SQ) next.push(idx[k])
    next.sort((a, b) => a - b)
    setNear((prev) =>
      prev.length === next.length && prev.every((v, i) => v === next[i]) ? prev : next,
    )
  })

  const forcedSet = new Set(forced)
  const shown = Array.from(new Set<number>([...near, ...forced]))

  return (
    <>
      {shown.map((i) => {
        const w = data.words[i]
        const isForced = forcedSet.has(i)
        return (
          <Html
            key={i}
            position={[w.p[0], w.p[1] + 0.5, w.p[2]]}
            center
            style={{ pointerEvents: 'none' }}
            zIndexRange={[10, 0]}
          >
            <span
              dir="auto"
              style={{
                color: isForced ? '#ffffff' : '#cdd5e3',
                fontFamily: 'var(--font-mono)',
                fontSize: isForced ? 14 : 12,
                fontWeight: isForced ? 600 : 400,
                whiteSpace: 'nowrap',
                textShadow: '0 1px 5px #000, 0 0 3px #000',
                opacity: isForced ? 1 : 0.8,
              }}
            >
              {w.w}
            </span>
          </Html>
        )
      })}
    </>
  )
}
