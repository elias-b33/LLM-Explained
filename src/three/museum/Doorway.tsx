import { useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Sparkles } from '@react-three/drei'
import {
  AdditiveBlending,
  DoubleSide,
  Vector3,
  type CanvasTexture,
  type Group,
  type Mesh,
  type MeshBasicMaterial,
} from 'three'
import { keys } from './input'

const PROBE = new Vector3()
const FW = 2.6 // opening width
const FH = 3.3 // opening height
const CY = FH / 2

/**
 * A glowing portal set into the end wall: a classical plaster architrave with a
 * thin lit reveal around a pulsing energy field, counter-rotating rings and
 * particles streaming through.
 * Walk up + press E to travel onward.
 */
export function Doorway({
  position,
  label,
  accent,
  accent2,
  sprite,
  onTravel,
}: {
  position: [number, number, number]
  label: string
  accent: string
  accent2: string
  sprite: CanvasTexture
  onTravel: () => void
}) {
  const [near, setNear] = useState(false)
  const fired = useRef(false)
  const field = useRef<Mesh>(null)
  const core = useRef<Mesh>(null)
  const ringA = useRef<Group>(null)
  const ringB = useRef<Group>(null)
  const ringC = useRef<Group>(null)

  useFrame((state) => {
    PROBE.set(position[0], 1.6, position[2])
    const inRange = state.camera.position.distanceTo(PROBE) < 3.8
    if (inRange !== near) setNear(inRange)
    if (inRange && keys.interact && !fired.current) {
      fired.current = true
      onTravel()
    }
    if (!keys.interact) fired.current = false

    const t = state.clock.elapsedTime
    const boost = near ? 1.8 : 1
    if (ringA.current) ringA.current.rotation.z = t * 0.5 * boost
    if (ringB.current) ringB.current.rotation.z = -t * 0.8 * boost
    if (ringC.current) ringC.current.rotation.z = t * 1.3 * boost
    const pulse = 0.5 + 0.5 * Math.sin(t * 2.2)
    if (field.current) (field.current.material as MeshBasicMaterial).opacity = (near ? 0.42 : 0.26) + pulse * 0.1
    if (core.current) core.current.scale.setScalar(0.9 + pulse * 0.15 * boost)
  })

  const edge = near ? accent2 : accent

  return (
    <group position={position}>
      {/* energy field filling the opening */}
      <mesh ref={field} position={[0, CY, 0]}>
        <planeGeometry args={[FW, FH]} />
        <meshBasicMaterial
          map={sprite}
          color={accent}
          transparent
          opacity={0.34}
          depthWrite={false}
          blending={AdditiveBlending}
          toneMapped={false}
        />
      </mesh>
      {/* bright vortex core */}
      <mesh ref={core} position={[0, CY, 0.02]}>
        <circleGeometry args={[0.9, 48]} />
        <meshBasicMaterial
          map={sprite}
          color={accent2}
          transparent
          opacity={0.7}
          depthWrite={false}
          blending={AdditiveBlending}
          toneMapped={false}
          side={DoubleSide}
        />
      </mesh>

      {/* counter-rotating rings */}
      <group position={[0, CY, 0.05]}>
        <group ref={ringA}>
          <mesh>
            <torusGeometry args={[1.35, 0.018, 8, 96, Math.PI * 1.6]} />
            <meshBasicMaterial color={accent2} toneMapped={false} />
          </mesh>
        </group>
        <group ref={ringB}>
          <mesh>
            <torusGeometry args={[1.08, 0.012, 8, 96, Math.PI * 1.2]} />
            <meshBasicMaterial color={accent} toneMapped={false} />
          </mesh>
        </group>
        <group ref={ringC}>
          <mesh>
            <torusGeometry args={[0.78, 0.01, 8, 64, Math.PI * 0.8]} />
            <meshBasicMaterial color="#ffffff" toneMapped={false} transparent opacity={0.8} />
          </mesh>
        </group>
      </group>

      {/* particles streaming through the portal */}
      <Sparkles
        count={70}
        scale={[FW * 0.9, FH * 0.9, 1.5]}
        position={[0, CY, 0.4]}
        size={2.4}
        speed={0.9}
        opacity={0.85}
        color={accent2}
      />

      {/* dark reveal behind the field so the opening reads as depth */}
      <mesh position={[0, CY, -0.3]}>
        <planeGeometry args={[FW, FH]} />
        <meshBasicMaterial color="#07060a" />
      </mesh>
      {/* classical plaster architrave: pilasters, lintel, cornice, edge light */}
      {[-1, 1].map((s) => (
        <group key={s} position={[s * (FW / 2 + 0.22), CY, 0]}>
          <mesh castShadow>
            <boxGeometry args={[0.44, FH, 0.4]} />
            <meshStandardMaterial color="#efe2c6" roughness={0.9} />
          </mesh>
          <mesh position={[0, -FH / 2 + 0.15, 0.02]}>
            <boxGeometry args={[0.54, 0.3, 0.46]} />
            <meshStandardMaterial color="#d9c8a8" roughness={0.9} />
          </mesh>
          <mesh position={[-s * 0.23, 0, 0.12]}>
            <boxGeometry args={[0.025, FH - 0.1, 0.025]} />
            <meshStandardMaterial color="#000000" emissive={edge} emissiveIntensity={1.6} toneMapped={false} />
          </mesh>
        </group>
      ))}
      <mesh position={[0, FH + 0.22, 0]}>
        <boxGeometry args={[FW + 0.88, 0.44, 0.4]} />
        <meshStandardMaterial color="#efe2c6" roughness={0.9} />
      </mesh>
      <mesh position={[0, FH + 0.5, 0.04]}>
        <boxGeometry args={[FW + 1.12, 0.12, 0.5]} />
        <meshStandardMaterial color="#d9c8a8" roughness={0.9} />
      </mesh>
      <mesh position={[0, FH + 0.02, 0.12]}>
        <boxGeometry args={[FW, 0.025, 0.025]} />
        <meshStandardMaterial color="#000000" emissive={edge} emissiveIntensity={1.6} toneMapped={false} />
      </mesh>
      {/* threshold glow on the floor */}
      <mesh position={[0, 0.008, 0.6]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[FW + 0.4, 1.4]} />
        <meshBasicMaterial
          map={sprite}
          color={accent2}
          transparent
          opacity={0.45}
          depthWrite={false}
          blending={AdditiveBlending}
          toneMapped={false}
        />
      </mesh>

      {/* label */}
      <Html
        position={[0, FH + 1.0, 0.2]}
        center
        distanceFactor={11}
        style={{ pointerEvents: 'none' }}
        zIndexRange={[15, 0]}
      >
        <div
          className="flex select-none flex-col items-center"
          style={{
            textShadow: '0 1px 6px #000, 0 0 3px #000',
            whiteSpace: 'nowrap',
          }}
        >
          <span
            style={{
              color: accent2,
              fontSize: 11,
              fontFamily: 'var(--font-mono)',
              letterSpacing: 1,
            }}
          >
            {near ? 'PRESS E TO ENTER' : 'NEXT EXHIBIT'}
          </span>
          <span
            style={{
              color: '#fff',
              fontWeight: 600,
              fontSize: 15,
              marginTop: 2,
            }}
          >
            {label}
          </span>
        </div>
      </Html>
    </group>
  )
}
