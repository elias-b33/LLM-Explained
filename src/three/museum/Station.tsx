import { useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html } from '@react-three/drei'
import { AdditiveBlending, Vector3, type CanvasTexture, type Group, type Mesh, type MeshBasicMaterial } from 'three'
import { keys } from './input'
import { useMuseumTextures } from './materials'
import { NeuralSphere, TokenBlocks, Vitrine } from './Exhibits'

const PROBE = new Vector3()

/**
 * An interactive station: a glass vitrine on an oak-and-mustard plinth holding
 * a hologram (floating token blocks for the predictor, a neural sphere for the
 * maths). Walk up and press E to open the demo/math in a focused 2D overlay.
 * Proximity brightens the hologram, turns the lid light to the accent and
 * swaps the label hint. Edge-triggered so holding E doesn't re-fire.
 */
export function Station({
  position,
  title,
  accent,
  accent2,
  sprite,
  variant = 'network',
  onOpen,
}: {
  position: [number, number, number]
  title: string
  accent: string
  accent2: string
  sprite: CanvasTexture
  variant?: 'tokens' | 'network'
  onOpen: () => void
}) {
  const tex = useMuseumTextures()
  const [near, setNear] = useState(false)
  const fired = useRef(false)
  const holo = useRef<Group>(null)
  const ring = useRef<Mesh>(null)

  useFrame((state) => {
    // proximity + interact
    PROBE.set(position[0], 1.6, position[2])
    const inRange = state.camera.position.distanceTo(PROBE) < 3.4
    if (inRange !== near) setNear(inRange)
    if (inRange && keys.interact && !fired.current) {
      fired.current = true
      onOpen()
    }
    if (!keys.interact) fired.current = false

    const t = state.clock.elapsedTime
    if (holo.current) holo.current.position.y = 1.47 + Math.sin(t * 1.6) * 0.025
    if (ring.current) {
      ring.current.rotation.z = t * 0.6
      const m = ring.current.material as MeshBasicMaterial
      m.opacity = near ? 0.9 : 0.45
    }
  })

  const boost = near ? 1.6 : 1

  return (
    <group position={position}>
      <Vitrine tex={tex} size={[1.0, 1.05, 1.0]} plinthH={0.92} edge={near ? accent2 : '#ffe6c4'}>
        <group ref={holo} position={[0, 1.47, 0]} scale={near ? 1.06 : 1}>
          {variant === 'tokens' ? (
            <TokenBlocks boost={boost} span={0.82} />
          ) : (
            <NeuralSphere sprite={sprite} boost={boost} />
          )}
        </group>
        {/* projector ring on the deck */}
        <mesh ref={ring} position={[0, 0.935, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.3, 0.33, 48, 1, 0, Math.PI * 1.6]} />
          <meshBasicMaterial
            color={accent}
            transparent
            opacity={0.5}
            toneMapped={false}
            blending={AdditiveBlending}
            depthWrite={false}
          />
        </mesh>
        <mesh position={[0, 0.94, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <circleGeometry args={[0.24, 32]} />
          <meshBasicMaterial
            map={sprite}
            color={accent2}
            transparent
            opacity={near ? 0.7 : 0.4}
            toneMapped={false}
            blending={AdditiveBlending}
            depthWrite={false}
          />
        </mesh>
      </Vitrine>

      {/* label */}
      <Html position={[0, 2.45, 0]} center distanceFactor={9} style={{ pointerEvents: 'none' }} zIndexRange={[15, 0]}>
        <div
          className="flex select-none flex-col items-center"
          style={{
            textShadow: '0 1px 6px #000, 0 0 3px #000',
            whiteSpace: 'nowrap',
          }}
        >
          <span style={{ color: '#fff7ea', fontWeight: 600, fontSize: 14 }}>{title}</span>
          <span
            style={{
              color: near ? accent2 : 'rgba(255,240,220,0.6)',
              fontSize: 11.5,
              fontFamily: 'var(--font-mono)',
              marginTop: 2,
            }}
          >
            {near ? 'press E to open' : 'interactive'}
          </span>
        </div>
      </Html>
    </group>
  )
}
