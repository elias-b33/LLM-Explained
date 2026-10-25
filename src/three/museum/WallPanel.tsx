import { useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html } from '@react-three/drei'

/**
 * Fixed light-mode tokens for the placards. Set inline on the card so the
 * lesson primitives (which only read CSS variables) render identically
 * whether the app is in light or dark mode.
 */
const PLACARD_TOKENS = {
  '--bg': '#f3ecdf',
  '--surface': '#fbf7ef',
  '--surface-2': '#e9dfcc',
  '--border': '#d9cbb1',
  '--border-strong': '#c4b190',
  '--fg': '#1d1711',
  '--fg-muted': '#3f362b',
  '--fg-subtle': '#7a6c58',
  '--accent': '#5a48e0',
  '--accent-2': '#0a8794',
  '--success': '#1f8a4f',
  '--warning': '#a8660f',
  '--danger': '#c2334b',
} as CSSProperties

const DISTANCE_FACTOR = 3.2
const PX = DISTANCE_FACTOR / 400 // world metres per CSS pixel for <Html transform>

/**
 * A museum lightbox placard fixed to a wall: a slim oak-framed light panel
 * (real geometry, so it reads from across the hall) with the lesson rendered
 * as crisp DOM via drei <Html transform> on a warm cream card. The text fades
 * in as you approach (and out as you leave) so the hall reads cleanly from a
 * distance instead of a wall of overlapping text.
 */
export function WallPanel({
  position,
  rotationY,
  width,
  height = 2.5,
  children,
}: {
  position: [number, number, number]
  rotationY: number
  width: number
  height?: number
  accent?: string
  children: ReactNode
}) {
  const card = useRef<HTMLDivElement>(null)
  // Size the physical frame to the card's real rendered height.
  const [cardH, setCardH] = useState<number | null>(null)
  useLayoutEffect(() => {
    const el = card.current
    if (!el) return
    const measure = () => el.offsetHeight > 0 && setCardH(el.offsetHeight * PX)
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  useFrame((state) => {
    const dx = state.camera.position.x - position[0]
    const dz = state.camera.position.z - position[2]
    const d = Math.sqrt(dx * dx + dz * dz)
    const o = Math.max(0, Math.min(1, 1 - (d - 7) / 8)) // full within 7m, gone by ~15m
    if (card.current) card.current.style.opacity = (0.02 + 0.98 * o).toFixed(3)
  })

  const fw = Math.round(width * 132) * PX + 0.18
  const fh = (cardH ?? height) + 0.18
  return (
    <group position={position} rotation={[0, rotationY, 0]}>
      {/* oak frame + the glowing lightbox face behind the card */}
      <mesh position={[0, 0, -0.07]} castShadow>
        <boxGeometry args={[fw, fh, 0.07]} />
        <meshStandardMaterial color="#7a5a36" roughness={0.6} />
      </mesh>
      <mesh position={[0, 0, -0.03]}>
        <planeGeometry args={[fw - 0.1, fh - 0.1]} />
        <meshStandardMaterial color="#f3e9d6" emissive="#fff1da" emissiveIntensity={0.18} roughness={0.9} />
      </mesh>
      <Html
        transform
        distanceFactor={DISTANCE_FACTOR}
        position={[0, 0, 0.01]}
        zIndexRange={[10, 0]}
        style={{ width: `${Math.round(width * 132)}px`, pointerEvents: 'none' }}
      >
        <div
          ref={card}
          style={{
            ...PLACARD_TOKENS,
            width: '100%',
            borderRadius: 4,
            borderTop: '3px solid #b8892f',
            background: 'linear-gradient(180deg, #fbf6ec, #f2e9d8)',
            boxShadow: 'inset 0 0 40px rgba(180,140,80,0.12)',
            padding: '24px 28px',
            color: 'var(--fg)',
            fontFamily: 'var(--font-sans)',
          }}
        >
          {children}
        </div>
      </Html>
    </group>
  )
}
