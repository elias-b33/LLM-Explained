import { useEffect, useMemo, useRef, type RefObject } from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import { CameraControls, Html, Line } from '@react-three/drei'
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing'
import { AdditiveBlending, Color, Sphere, Vector3 } from 'three'
import type CameraControlsImpl from 'camera-controls'
import galaxyRaw from '@/data/galaxy.json'
import type { GalaxyData } from './types'
import { GalaxyPoints, makeDotTexture } from './GalaxyPoints'
import { GalaxyLabels } from './GalaxyLabels'
import { GalaxyFlyControls } from './GalaxyFlyControls'

const data = galaxyRaw as unknown as GalaxyData
type ControlsRef = RefObject<CameraControlsImpl | null>
const UNI_FIT = 17

const ALL_CENTER = (() => {
  const n = data.words.length || 1
  let x = 0, y = 0, z = 0
  data.words.forEach((w) => {
    x += w.p[0]
    y += w.p[1]
    z += w.p[2]
  })
  return new Vector3(x / n, y / n, z / n)
})()
const ALL_RADIUS = (() => {
  let r = 1
  data.words.forEach((w) => {
    r = Math.max(r, ALL_CENTER.distanceTo(new Vector3(w.p[0], w.p[1], w.p[2])))
  })
  return Math.min(r, 60) // ignore far UMAP outliers when framing all three
})()

function RaycastConfig() {
  const raycaster = useThree((s) => s.raycaster)
  useEffect(() => {
    raycaster.params.Points = { threshold: 0.9 }
  }, [raycaster])
  return null
}

/** Flies the camera: to a selected word, else to the active universe, else frames all three. */
function CameraDirector({
  controls,
  selected,
  activeUniverse,
  reducedMotion,
}: {
  controls: ControlsRef
  selected: number | null
  activeUniverse: number | null
  reducedMotion: boolean
}) {
  const inited = useRef(false)
  useEffect(() => {
    let raf = 0
    const run = () => {
      const c = controls.current
      if (!c) {
        raf = requestAnimationFrame(run)
        return
      }
      const animate = inited.current && !reducedMotion
      inited.current = true
      if (selected != null) {
        const p = data.words[selected].p
        void c.setLookAt(p[0], p[1], p[2] + 9, p[0], p[1], p[2], animate)
      } else if (activeUniverse != null && data.clusters[activeUniverse]?.center) {
        const ctr = data.clusters[activeUniverse].center as [number, number, number]
        void c.fitToSphere(new Sphere(new Vector3(ctr[0], ctr[1], ctr[2]), UNI_FIT), animate)
      } else {
        void c.fitToSphere(new Sphere(ALL_CENTER.clone(), ALL_RADIUS * 0.95), animate)
      }
    }
    run()
    return () => cancelAnimationFrame(raf)
  }, [controls, selected, activeUniverse, reducedMotion])
  return null
}

function Highlights({ forced }: { forced: number[] }) {
  const sprite = useMemo(makeDotTexture, [])
  const geo = useMemo(() => {
    const pos = new Float32Array(forced.length * 3)
    const col = new Float32Array(forced.length * 3)
    const cc = data.clusters.map((c) => new Color(c.color))
    forced.forEach((idx, k) => {
      const w = data.words[idx]
      pos[k * 3] = w.p[0]
      pos[k * 3 + 1] = w.p[1]
      pos[k * 3 + 2] = w.p[2]
      const c = (cc[w.c] ?? new Color('#fff')).clone().lerp(new Color('#ffffff'), 0.4)
      col[k * 3] = c.r
      col[k * 3 + 1] = c.g
      col[k * 3 + 2] = c.b
    })
    return { pos, col }
  }, [forced])
  if (forced.length === 0) return null
  return (
    <points key={forced.join(',')}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[geo.pos, 3]} />
        <bufferAttribute attach="attributes-color" args={[geo.col, 3]} />
      </bufferGeometry>
      <pointsMaterial
        map={sprite}
        size={1.8}
        vertexColors
        sizeAttenuation
        transparent
        depthWrite={false}
        blending={AdditiveBlending}
      />
    </points>
  )
}

function NeighborLines({ selected }: { selected: number | null }) {
  if (selected == null) return null
  const from = data.words[selected].p
  return (
    <>
      {data.words[selected].nn.map((n) => (
        <Line key={n} points={[from, data.words[n].p]} color="#ffffff" lineWidth={1.2} transparent opacity={0.4} />
      ))}
    </>
  )
}

/** Deterministic 0..1 hash (no Math.random, so the scene is identical every load). */
function hash(n: number): number {
  const h = Math.sin(n * 12.9898 + 78.233) * 43758.5453
  return h - Math.floor(h)
}

/** Soft additive glow clouds around each universe, so it reads as a nebula. */
function Nebulae() {
  const sprite = useMemo(makeDotTexture, [])
  const puffs = useMemo(
    () =>
      data.clusters.flatMap((c, ci) => {
        const ctr = c.center ?? [0, 0, 0]
        // Many small, scattered, unevenly bright puffs read as wispy gas rather
        // than one round halo; a few are tinted toward white or a neighbour hue.
        return Array.from({ length: 22 }, (_, k) => {
          const s = ci * 53 + k * 11
          const spread = k < 6 ? 12 : 30
          const tint = hash(s + 5)
          return {
            key: `${ci}-${k}`,
            color: tint > 0.85 ? '#ffffff' : tint > 0.7 ? data.clusters[(ci + 1) % data.clusters.length].color : c.color,
            pos: [
              ctr[0] + (hash(s) - 0.5) * spread,
              ctr[1] + (hash(s + 1) - 0.5) * spread * 0.6,
              ctr[2] + (hash(s + 2) - 0.5) * spread,
            ] as [number, number, number],
            scale: 7 + hash(s + 3) ** 2 * 24,
            opacity: 0.02 + hash(s + 4) * 0.045,
          }
        })
      }),
    [],
  )
  return (
    <>
      {puffs.map((p) => (
        <sprite key={p.key} position={p.pos} scale={[p.scale, p.scale, 1]}>
          <spriteMaterial map={sprite} color={p.color} transparent opacity={p.opacity} depthWrite={false} blending={AdditiveBlending} fog={false} />
        </sprite>
      ))}
    </>
  )
}

/** A distant, unfogged background starfield for depth. */
function DeepStars() {
  const { pos, col } = useMemo(() => {
    const n = 5000
    const pos = new Float32Array(n * 3)
    const col = new Float32Array(n * 3)
    const tints = [new Color('#ffffff'), new Color('#aac4ff'), new Color('#ffd9b0'), new Color('#c9b8ff')]
    for (let i = 0; i < n; i++) {
      const u = hash(i * 3 + 1) * 2 - 1
      const th = hash(i * 3 + 2) * Math.PI * 2
      const r = 420 + hash(i * 3 + 3) * 260
      const s = Math.sqrt(1 - u * u)
      pos[i * 3] = r * s * Math.cos(th)
      pos[i * 3 + 1] = r * u
      pos[i * 3 + 2] = r * s * Math.sin(th)
      const c = tints[i % tints.length]
      const b = 0.25 + 0.75 * hash(i * 7 + 5) ** 3
      col[i * 3] = c.r * b
      col[i * 3 + 1] = c.g * b
      col[i * 3 + 2] = c.b * b
    }
    return { pos, col }
  }, [])
  return (
    <points>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[pos, 3]} />
        <bufferAttribute attach="attributes-color" args={[col, 3]} />
      </bufferGeometry>
      <pointsMaterial size={1.6} sizeAttenuation={false} vertexColors transparent depthWrite={false} fog={false} />
    </points>
  )
}

/** Big floating universe names, shown only in the zoomed-out overview. */
function UniverseTitles({ visible }: { visible: boolean }) {
  if (!visible) return null
  return (
    <>
      {data.clusters.map((c) => {
        const ctr = c.center ?? [0, 0, 0]
        return (
          <Html key={c.name} position={[ctr[0], ctr[1] + 17, ctr[2]]} center style={{ pointerEvents: 'none' }} zIndexRange={[5, 0]}>
            <div
              style={{
                color: c.color,
                fontFamily: 'var(--font-sans)',
                fontWeight: 600,
                fontSize: 13,
                letterSpacing: '0.32em',
                textTransform: 'uppercase',
                whiteSpace: 'nowrap',
                textShadow: `0 0 18px ${c.color}, 0 1px 4px #000`,
              }}
            >
              {c.name}
            </div>
          </Html>
        )
      })}
    </>
  )
}

export interface GalaxySceneProps {
  controls: ControlsRef
  selected: number | null
  activeUniverse: number | null
  forced: number[]
  onHover: (i: number | null) => void
  onSelect: (i: number) => void
  reducedMotion: boolean
}

export default function GalaxyScene({
  controls,
  selected,
  activeUniverse,
  forced,
  onHover,
  onSelect,
  reducedMotion,
}: GalaxySceneProps) {
  useEffect(() => {
    const raf = requestAnimationFrame(() => window.dispatchEvent(new Event('resize')))
    const t = setTimeout(() => window.dispatchEvent(new Event('resize')), 150)
    return () => {
      cancelAnimationFrame(raf)
      clearTimeout(t)
    }
  }, [])

  return (
    <Canvas
      frameloop="demand"
      camera={{ position: [0, 18, 100], fov: 55 }}
      gl={{ alpha: false, antialias: true, preserveDrawingBuffer: import.meta.env.DEV, powerPreference: 'high-performance' }}
      dpr={[1, 1.5]}
    >
      <color attach="background" args={['#04060b']} />
      <fog attach="fog" args={['#060912', 50, 190]} />
      <DeepStars />
      <Nebulae />
      <UniverseTitles visible={selected == null && activeUniverse == null} />
      <CameraControls ref={controls} makeDefault minDistance={1} maxDistance={220} smoothTime={0.4} />
      <GalaxyFlyControls controls={controls} />
      <RaycastConfig />
      <CameraDirector controls={controls} selected={selected} activeUniverse={activeUniverse} reducedMotion={reducedMotion} />
      <GalaxyPoints key={activeUniverse ?? 'all'} data={data} activeUniverse={activeUniverse} onHover={onHover} onSelect={onSelect} />
      <Highlights forced={forced} />
      <NeighborLines selected={selected} />
      <GalaxyLabels data={data} forced={forced} />
      <EffectComposer>
        <Bloom intensity={0.85} luminanceThreshold={0.42} luminanceSmoothing={0.9} mipmapBlur />
        <Vignette eskil={false} offset={0.25} darkness={0.7} />
      </EffectComposer>
    </Canvas>
  )
}
