import { useMemo, useRef, type ReactNode } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html } from '@react-three/drei'
import {
  AdditiveBlending,
  BufferGeometry,
  CanvasTexture,
  Color,
  DoubleSide,
  Float32BufferAttribute,
  InstancedMesh,
  Object3D,
  SRGBColorSpace,
  Vector3,
  type CanvasTexture as CanvasTextureT,
  type Group,
  type Mesh,
  type MeshBasicMaterial,
} from 'three'
import type { Exhibit } from './halls'
import { tile, type MuseumTextures } from './materials'
import { activeProfile, textureAnisotropy } from './quality'

export const VIOLET = '#8b7bff'
export const CYAN = '#2bd3e0'
const MUSTARD = '#ffc94a'
const LED = '#ffe6c4'
const BRASS = '#8a6a3a'

/* ------------------------------------------------------------------------ */
/* Shared bits                                                              */
/* ------------------------------------------------------------------------ */

/** Cheap museum glass: a faint transparent sheet that picks up env reflections. */
export function GlassMaterial({ opacity = 0.12 }: { opacity?: number }) {
  return (
    <meshPhysicalMaterial
      color="#e9f3f1"
      transparent
      opacity={opacity}
      roughness={0.04}
      metalness={0}
      envMapIntensity={2.2}
      specularIntensity={1}
      clearcoat={1}
      depthWrite={false}
      side={DoubleSide}
    />
  )
}

/** A canvas texture drawn in logical px; the backing store is scaled up at High quality. */
function canvasTex(w: number, h: number, draw: (ctx: CanvasRenderingContext2D) => void): CanvasTextureT {
  const k = activeProfile().boardScale
  const c = document.createElement('canvas')
  c.width = Math.round(w * k)
  c.height = Math.round(h * k)
  const ctx = c.getContext('2d')!
  ctx.scale(k, k)
  draw(ctx)
  const t = new CanvasTexture(c)
  t.colorSpace = SRGBColorSpace
  t.anisotropy = textureAnisotropy()
  return t
}

/** A causal (lower-triangular) attention heat-map. */
export function makeAttentionTexture(n = 12): CanvasTextureT {
  return canvasTex(512, 512, (ctx) => {
    ctx.fillStyle = '#05060c'
    ctx.fillRect(0, 0, 512, 512)
    const cell = 512 / n
    const v = new Color(VIOLET)
    const c2 = new Color(CYAN)
    let seed = 7
    const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647
    for (let r = 0; r < n; r++) {
      // softmax-ish row: strong on self + a couple of earlier tokens
      const w: number[] = []
      for (let c = 0; c <= r; c++) w.push(c === r ? 2.2 : c === 0 ? 1.4 : rnd() * rnd() * 3)
      const sum = w.reduce((a, b) => a + b, 0)
      for (let c = 0; c <= r; c++) {
        const p = Math.min(1, (w[c] / sum) * 2.2)
        const col = v.clone().lerp(c2, p)
        ctx.fillStyle = `rgba(${(col.r * 255) | 0},${(col.g * 255) | 0},${(col.b * 255) | 0},${0.12 + p * 0.88})`
        ctx.fillRect(c * cell + 2, r * cell + 2, cell - 4, cell - 4)
      }
    }
  })
}

/** Tiny lit caption that fades in as you approach. */
export function Caption({ position, children }: { position: [number, number, number]; children: ReactNode }) {
  const el = useRef<HTMLDivElement>(null)
  const g = useRef<Group>(null)
  useFrame((state) => {
    if (!el.current || !g.current) return
    const p = g.current.getWorldPosition(TMP)
    const d = state.camera.position.distanceTo(p)
    el.current.style.opacity = Math.max(0, Math.min(1, 1 - (d - 4.5) / 3)).toFixed(3)
  })
  return (
    <group ref={g} position={position}>
      <Html center distanceFactor={5} zIndexRange={[12, 0]} style={{ pointerEvents: 'none' }}>
        <div
          ref={el}
          style={{
            whiteSpace: 'nowrap',
            fontFamily: 'var(--font-sans)',
            fontSize: 12,
            fontWeight: 500,
            color: '#2a2118',
            background: '#efe6d4',
            border: '1px solid #cdbb98',
            borderRadius: 3,
            padding: '5px 10px',
            boxShadow: '0 2px 10px rgba(0,0,0,0.35)',
            opacity: 0,
          }}
        >
          {children}
        </div>
      </Html>
    </group>
  )
}
const TMP = new Vector3()

/* ------------------------------------------------------------------------ */
/* Vitrine                                                                  */
/* ------------------------------------------------------------------------ */

/**
 * A museum display case: light-oak plinth with a mustard band, an LED toe-line,
 * a glass hood with slim bronze edges and a light strip in the lid.
 */
export function Vitrine({
  tex,
  size = [1.0, 1.0, 1.0],
  plinthH = 0.9,
  edge = LED,
  edgeIntensity = 1.2,
  children,
}: {
  tex: MuseumTextures
  size?: [number, number, number]
  plinthH?: number
  edge?: string
  edgeIntensity?: number
  children?: ReactNode
}) {
  const [w, h, d] = size
  const oak = useMemo(() => tile(tex.oak, 1, 1), [tex])
  const top = plinthH + h
  const edges = useMemo(() => {
    const out: { p: [number, number, number]; s: [number, number, number] }[] = []
    const t = 0.018
    for (const sx of [-1, 1])
      for (const sz of [-1, 1])
        out.push({
          p: [(sx * w) / 2, plinthH + h / 2, (sz * d) / 2],
          s: [t, h, t],
        })
    for (const sz of [-1, 1]) out.push({ p: [0, top, (sz * d) / 2], s: [w, t, t] })
    for (const sx of [-1, 1]) out.push({ p: [(sx * w) / 2, top, 0], s: [t, t, d] })
    return out
  }, [w, h, d, plinthH, top])
  return (
    <group>
      {/* plinth */}
      <mesh position={[0, plinthH / 2 + 0.03, 0]} castShadow receiveShadow>
        <boxGeometry args={[w + 0.1, plinthH - 0.06, d + 0.1]} />
        <meshStandardMaterial {...oak} color="#f0dcb4" roughness={0.7} />
      </mesh>
      <mesh position={[0, plinthH - 0.06, 0]} castShadow>
        <boxGeometry args={[w + 0.12, 0.1, d + 0.12]} />
        <meshStandardMaterial {...oak} color={MUSTARD} roughness={0.6} />
      </mesh>
      {/* recessed toe + LED line */}
      <mesh position={[0, 0.015, 0]}>
        <boxGeometry args={[w + 0.02, 0.03, d + 0.02]} />
        <meshBasicMaterial color={LED} toneMapped={false} />
      </mesh>
      {/* deck the exhibit sits on */}
      <mesh position={[0, plinthH + 0.005, 0]} receiveShadow>
        <boxGeometry args={[w - 0.02, 0.01, d - 0.02]} />
        <meshStandardMaterial color="#e8dcc6" roughness={0.9} />
      </mesh>
      {/* glass hood */}
      <mesh position={[0, plinthH + h / 2, 0]} renderOrder={2}>
        <boxGeometry args={[w, h, d]} />
        <GlassMaterial />
      </mesh>
      {edges.map((e, i) => (
        <mesh key={i} position={e.p}>
          <boxGeometry args={e.s} />
          <meshStandardMaterial color={BRASS} metalness={0.8} roughness={0.35} />
        </mesh>
      ))}
      {/* lid light strips */}
      {[-1, 1].map((s) => (
        <mesh key={s} position={[(s * w) / 2 - s * 0.03, top - 0.02, 0]}>
          <boxGeometry args={[0.02, 0.012, d - 0.06]} />
          <meshBasicMaterial color={edge} toneMapped={false} transparent opacity={Math.min(1, edgeIntensity)} />
        </mesh>
      ))}
      {children}
    </group>
  )
}

/* ------------------------------------------------------------------------ */
/* Holograms                                                                */
/* ------------------------------------------------------------------------ */

/** A holographic neural network: nodes on a sphere with nearest-neighbour links. */
export function NeuralSphere({
  sprite,
  boost = 1,
  radius = 0.32,
}: {
  sprite: CanvasTextureT
  boost?: number
  radius?: number
}) {
  const group = useRef<Group>(null)
  const inst = useRef<InstancedMesh>(null)
  const N = 70
  const { pts, lines } = useMemo(() => {
    const pts: [number, number, number][] = []
    const ga = Math.PI * (3 - Math.sqrt(5))
    for (let i = 0; i < N; i++) {
      const y = 1 - (i / (N - 1)) * 2
      const r = Math.sqrt(1 - y * y)
      const th = ga * i
      const j = 0.85 + ((i * 37) % 10) / 60
      pts.push([Math.cos(th) * r * radius * j, y * radius * j, Math.sin(th) * r * radius * j])
    }
    const pos: number[] = []
    for (let i = 0; i < N; i++) {
      const d = pts
        .map((p, k) => [k, (p[0] - pts[i][0]) ** 2 + (p[1] - pts[i][1]) ** 2 + (p[2] - pts[i][2]) ** 2] as const)
        .sort((a, b) => a[1] - b[1])
      for (let n = 1; n <= 3; n++) pos.push(...pts[i], ...pts[d[n][0]])
      if (i % 5 === 0) pos.push(...pts[i], 0, 0, 0) // spokes to the core
    }
    const lines = new BufferGeometry()
    lines.setAttribute('position', new Float32BufferAttribute(pos, 3))
    return { pts, lines }
  }, [radius])

  useFrame((state) => {
    const t = state.clock.elapsedTime
    if (group.current) {
      group.current.rotation.y = t * 0.25 * boost
      group.current.rotation.x = Math.sin(t * 0.3) * 0.15
    }
  })

  return (
    <group ref={group}>
      <sprite scale={[radius * 4.2, radius * 4.2, 1]}>
        <spriteMaterial
          map={sprite}
          color={CYAN}
          transparent
          opacity={0.22 * boost}
          depthWrite={false}
          blending={AdditiveBlending}
        />
      </sprite>
      <mesh>
        <icosahedronGeometry args={[radius * 0.22, 1]} />
        <meshBasicMaterial color={CYAN} toneMapped={false} />
      </mesh>
      <lineSegments geometry={lines}>
        <lineBasicMaterial
          color={VIOLET}
          transparent
          opacity={0.55 * Math.min(1.4, boost)}
          toneMapped={false}
          blending={AdditiveBlending}
          depthWrite={false}
        />
      </lineSegments>
      <instancedMesh
        ref={(m: InstancedMesh | null) => {
          inst.current = m
          if (!m) return
          const o = new Object3D()
          pts.forEach((p, i) => {
            o.position.set(...p)
            o.updateMatrix()
            m.setMatrixAt(i, o.matrix)
            m.setColorAt(i, new Color(i % 4 === 0 ? VIOLET : CYAN))
          })
          m.instanceMatrix.needsUpdate = true
          if (m.instanceColor) m.instanceColor.needsUpdate = true
        }}
        args={[undefined, undefined, N]}
      >
        <sphereGeometry args={[0.014, 8, 6]} />
        <meshBasicMaterial toneMapped={false} />
      </instancedMesh>
    </group>
  )
}

function tokenTexture(text: string, hot: boolean): CanvasTextureT {
  return canvasTex(256, 128, (ctx) => {
    ctx.fillStyle = hot ? 'rgba(43,211,224,0.28)' : 'rgba(139,123,255,0.22)'
    ctx.fillRect(0, 0, 256, 128)
    ctx.strokeStyle = hot ? CYAN : VIOLET
    ctx.lineWidth = 6
    ctx.strokeRect(3, 3, 250, 122)
    ctx.fillStyle = '#ffffff'
    ctx.font = '600 52px Inter, system-ui, sans-serif'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(text, 128, 66)
  })
}

/** Floating token blocks: a prompt, with the predicted next token glowing. */
export function TokenBlocks({
  words = ['The', 'capital', 'of', 'France', 'is', 'Paris'],
  boost = 1,
  span = 0.8,
}: {
  words?: string[]
  boost?: number
  span?: number
}) {
  const refs = useRef<(Mesh | null)[]>([])
  const texs = useMemo(() => words.map((w, i) => tokenTexture(w, i === words.length - 1)), [words])
  useFrame((state) => {
    const t = state.clock.elapsedTime
    refs.current.forEach((m, i) => {
      if (!m) return
      m.position.y = Math.sin(t * 1.2 + i * 0.9) * 0.025
      m.rotation.y = Math.sin(t * 0.5 + i) * 0.25
      if (i === words.length - 1) {
        const mat = m.material as MeshBasicMaterial
        mat.opacity = 0.65 + 0.35 * Math.sin(t * 2.4) * boost
      }
    })
  })
  const n = words.length
  return (
    <group>
      {words.map((_, i) => {
        // arranged on a gentle arc
        const a = (i / (n - 1) - 0.5) * 1.6
        const x = Math.sin(a) * span * 0.5
        const z = -Math.cos(a) * span * 0.25 + span * 0.18
        const y = (i % 2) * 0.12 - 0.06 + (i === n - 1 ? 0.16 : 0)
        return (
          <group key={i} position={[x, y, z]} rotation={[0, -a * 0.6, 0]}>
            <mesh ref={(m) => void (refs.current[i] = m)}>
              <planeGeometry args={[0.2, 0.1]} />
              <meshBasicMaterial
                map={texs[i]}
                transparent
                side={DoubleSide}
                toneMapped={false}
                depthWrite={false}
                blending={AdditiveBlending}
              />
            </mesh>
          </group>
        )
      })}
    </group>
  )
}

/** A stylised accelerator package: substrate, glowing die, HBM stacks, traces. */
export function GpuChip({ boost = 1 }: { boost?: number }) {
  const g = useRef<Group>(null)
  const die = useMemo(
    () =>
      canvasTex(256, 256, (ctx) => {
        ctx.fillStyle = '#0a0c18'
        ctx.fillRect(0, 0, 256, 256)
        for (let y = 0; y < 16; y++)
          for (let x = 0; x < 16; x++) {
            const hot = (x * 7 + y * 13) % 9 === 0
            ctx.fillStyle = hot ? CYAN : (x + y) % 3 === 0 ? '#5b4fd6' : '#2b2770'
            ctx.fillRect(x * 16 + 2, y * 16 + 2, 12, 12)
          }
      }),
    [],
  )
  const traces = useMemo(
    () =>
      canvasTex(256, 256, (ctx) => {
        ctx.fillStyle = '#0d1510'
        ctx.fillRect(0, 0, 256, 256)
        ctx.strokeStyle = 'rgba(43,211,224,0.55)'
        ctx.lineWidth = 1.5
        for (let i = 0; i < 40; i++) {
          const x = (i * 53) % 256
          const y = (i * 97) % 256
          ctx.beginPath()
          ctx.moveTo(x, y)
          ctx.lineTo(x + ((i % 3) - 1) * 40, y)
          ctx.lineTo(x + ((i % 3) - 1) * 40, y + 30)
          ctx.stroke()
        }
      }),
    [],
  )
  useFrame((state) => {
    if (g.current) g.current.rotation.y = state.clock.elapsedTime * 0.2
  })
  return (
    <group ref={g} rotation={[0.35, 0, 0]}>
      <mesh>
        <boxGeometry args={[0.5, 0.03, 0.5]} />
        <meshStandardMaterial
          map={traces}
          emissiveMap={traces}
          emissive="#ffffff"
          emissiveIntensity={0.6 * boost}
          roughness={0.5}
          metalness={0.3}
        />
      </mesh>
      <mesh position={[0, 0.03, 0]}>
        <boxGeometry args={[0.24, 0.03, 0.24]} />
        <meshStandardMaterial
          map={die}
          emissiveMap={die}
          emissive="#ffffff"
          emissiveIntensity={1.4 * boost}
          roughness={0.2}
          metalness={0.6}
          toneMapped={false}
        />
      </mesh>
      {[
        [-0.17, -0.09],
        [-0.17, 0.09],
        [0.17, -0.09],
        [0.17, 0.09],
      ].map(([x, z], i) => (
        <mesh key={i} position={[x, 0.035, z]}>
          <boxGeometry args={[0.07, 0.04, 0.12]} />
          <meshStandardMaterial color="#1b1d24" metalness={0.8} roughness={0.25} />
        </mesh>
      ))}
    </group>
  )
}

/** A glass slab hovering upright, showing a causal attention matrix. */
export function AttentionPane({ boost = 1 }: { boost?: number }) {
  const tex = useMemo(() => makeAttentionTexture(12), [])
  const g = useRef<Group>(null)
  useFrame((state) => {
    if (g.current) g.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.35) * 0.35
  })
  return (
    <group ref={g}>
      <mesh>
        <planeGeometry args={[0.62, 0.62]} />
        <meshBasicMaterial
          map={tex}
          transparent
          opacity={0.92}
          side={DoubleSide}
          toneMapped={false}
          blending={AdditiveBlending}
          depthWrite={false}
          color={new Color(1.2 * boost, 1.2 * boost, 1.2 * boost)}
        />
      </mesh>
      <mesh position={[0, 0, -0.01]}>
        <planeGeometry args={[0.68, 0.68]} />
        <meshBasicMaterial
          color={VIOLET}
          transparent
          opacity={0.08}
          side={DoubleSide}
          depthWrite={false}
          blending={AdditiveBlending}
        />
      </mesh>
    </group>
  )
}

/** A rotating point cloud: words as positions, clustered by meaning. */
export function EmbeddingCloud({ sprite, boost = 1 }: { sprite: CanvasTextureT; boost?: number }) {
  const g = useRef<Group>(null)
  const geo = useMemo(() => {
    const pos: number[] = []
    const col: number[] = []
    const centres: [number, number, number, string][] = [
      [-0.16, 0.05, 0, VIOLET],
      [0.14, -0.08, 0.1, CYAN],
      [0.05, 0.14, -0.14, '#ffd38a'],
    ]
    let s = 3
    const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647 - 0.5
    for (const [cx, cy, cz, c] of centres) {
      const cc = new Color(c)
      for (let i = 0; i < 120; i++) {
        pos.push(cx + rnd() * 0.16, cy + rnd() * 0.16, cz + rnd() * 0.16)
        col.push(cc.r, cc.g, cc.b)
      }
    }
    const b = new BufferGeometry()
    b.setAttribute('position', new Float32BufferAttribute(pos, 3))
    b.setAttribute('color', new Float32BufferAttribute(col, 3))
    return b
  }, [])
  useFrame((state) => {
    if (g.current) g.current.rotation.y = state.clock.elapsedTime * 0.18
  })
  return (
    <group ref={g}>
      <points geometry={geo}>
        <pointsMaterial
          map={sprite}
          vertexColors
          size={0.035 * boost}
          transparent
          depthWrite={false}
          blending={AdditiveBlending}
          toneMapped={false}
        />
      </points>
      {/* faint axes */}
      {[0, 1, 2].map((k) => (
        <mesh key={k} rotation={k === 0 ? [0, 0, Math.PI / 2] : k === 1 ? [0, 0, 0] : [Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.0015, 0.0015, 0.6, 4]} />
          <meshBasicMaterial color="#ffffff" transparent opacity={0.25} />
        </mesh>
      ))}
    </group>
  )
}

export function ExhibitObject({
  exhibit,
  sprite,
  boost = 1,
}: {
  exhibit: Exhibit
  sprite: CanvasTextureT
  boost?: number
}) {
  if (exhibit === 'gpu') return <GpuChip boost={boost} />
  if (exhibit === 'tokens') return <TokenBlocks boost={boost} />
  if (exhibit === 'attention') return <AttentionPane boost={boost} />
  return <EmbeddingCloud sprite={sprite} boost={boost} />
}

/** A decorative (non-interactive) vitrine with a hologram and a caption. */
export function DecorVitrine({
  tex,
  sprite,
  position,
  exhibit,
  caption,
}: {
  tex: MuseumTextures
  sprite: CanvasTextureT
  position: [number, number, number]
  exhibit: Exhibit
  caption: string
}) {
  // face the nave: rotate so the caption side points at x = 0
  const rotY = position[0] < 0 ? Math.PI / 2 : -Math.PI / 2
  return (
    <group position={position} rotation={[0, rotY, 0]}>
      <Vitrine tex={tex} size={[0.86, 0.8, 0.86]} plinthH={0.95}>
        <group position={[0, 1.36, 0]}>
          <ExhibitObject exhibit={exhibit} sprite={sprite} />
        </group>
      </Vitrine>
      <Caption position={[0, 0.72, 0.5]}>{caption}</Caption>
    </group>
  )
}

/* ------------------------------------------------------------------------ */
/* Central display table                                                    */
/* ------------------------------------------------------------------------ */

const STREAM_VERT = /* glsl */ `
varying vec2 vUv;
void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
`

/** Backlit table screen: tokens streaming as glowing cells. */
const TABLE_FRAG = /* glsl */ `
uniform float uTime;
varying vec2 vUv;
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
void main() {
  vec2 g = vec2(vUv.x * 9.0, vUv.y * 48.0 - uTime * 1.2);
  vec2 id = floor(g);
  vec2 f = fract(g);
  float h = hash(id);
  float on = step(0.55, h);
  float cell = smoothstep(0.5, 0.42, max(abs(f.x - 0.5), abs(f.y - 0.5) * 0.9));
  vec3 violet = vec3(0.545, 0.482, 1.0);
  vec3 cyan = vec3(0.169, 0.827, 0.878);
  vec3 c = mix(violet, cyan, hash(id + 3.1));
  float pulse = 0.55 + 0.45 * sin(uTime * 2.0 + h * 30.0);
  vec3 base = vec3(0.05, 0.08, 0.16);
  vec3 col = base + c * cell * on * pulse * 0.55;
  gl_FragColor = vec4(col, 1.0);
}
`

export function DisplayTable({ tex, z0, z1 }: { tex: MuseumTextures; z0: number; z1: number }) {
  const len = Math.abs(z1 - z0)
  const mid = (z0 + z1) / 2
  const W = 1.35
  const H = 0.86
  const oak = useMemo(() => tile(tex.oak, 1, len / 1.2), [tex, len])
  const marble = useMemo(() => tile(tex.marble, 1, len / 1.4), [tex, len])
  const uniforms = useMemo(() => ({ uTime: { value: 0 } }), [])
  useFrame((state) => {
    uniforms.uTime.value = state.clock.elapsedTime
  })
  return (
    <group position={[0, 0, mid]}>
      {/* mustard base */}
      <mesh position={[0, H / 2 + 0.03, 0]} castShadow receiveShadow>
        <boxGeometry args={[W - 0.05, H - 0.06, len - 0.05]} />
        <meshStandardMaterial {...oak} color={MUSTARD} roughness={0.65} />
      </mesh>
      <mesh position={[0, 0.015, 0]}>
        <boxGeometry args={[W - 0.08, 0.03, len - 0.08]} />
        <meshBasicMaterial color={LED} toneMapped={false} />
      </mesh>
      {/* marble frame around the backlit screen */}
      {[-1, 1].map((s) => (
        <mesh key={`x${s}`} position={[(s * (W - 0.16)) / 2, H + 0.02, 0]} castShadow receiveShadow>
          <boxGeometry args={[0.16, 0.04, len]} />
          <meshStandardMaterial {...marble} color="#f4efe6" roughness={0.35} />
        </mesh>
      ))}
      {[-1, 1].map((s) => (
        <mesh key={`z${s}`} position={[0, H + 0.02, (s * (len - 0.3)) / 2]} castShadow receiveShadow>
          <boxGeometry args={[W, 0.04, 0.3]} />
          <meshStandardMaterial {...marble} color="#f4efe6" roughness={0.35} />
        </mesh>
      ))}
      <mesh position={[0, H + 0.005, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[W - 0.32, len - 0.6]} />
        <shaderMaterial vertexShader={STREAM_VERT} fragmentShader={TABLE_FRAG} uniforms={uniforms} toneMapped={false} />
      </mesh>
      {/* glass top */}
      <mesh position={[0, H + 0.045, 0]} renderOrder={2}>
        <boxGeometry args={[W, 0.012, len]} />
        <GlassMaterial opacity={0.05} />
      </mesh>
      {/* two small glass cubes with exhibits standing on the table */}
      {[
        [0.25, len / 2 - 0.6, 'tokens'],
        [-0.25, -len / 2 + 0.7, 'attention'],
      ].map(([x, z, k]) => (
        <group key={k as string} position={[x as number, H + 0.05, z as number]}>
          <mesh position={[0, 0.2, 0]} renderOrder={2}>
            <boxGeometry args={[0.42, 0.4, 0.42]} />
            <GlassMaterial />
          </mesh>
          <group position={[0, 0.2, 0]} scale={0.5}>
            {k === 'tokens' ? <TokenBlocks words={['to', 'be', 'or', 'not']} span={0.7} /> : <AttentionPane />}
          </group>
        </group>
      ))}
      <Caption position={[0.72, 0.62, 0]}>The context window, flowing</Caption>
    </group>
  )
}
