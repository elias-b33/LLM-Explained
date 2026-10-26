import { useEffect, useMemo, useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing'
import { AdditiveBlending, BufferAttribute, Color, MathUtils, type BufferGeometry, type Group } from 'three'
import { makeDotTexture } from '@/three/galaxy/textures'
import { useOnScreenFrameloop } from '@/hooks/useOnScreenFrameloop'

const NODES = 220
const RADIUS = 2.35
const PULSES = 150
const DISC = 1100
const C1 = new Color('#8b7bff')
const C2 = new Color('#2bd3e0')

/** Deterministic 0..1 hash so the network is identical on every load. */
function hash(n: number): number {
  const h = Math.sin(n * 12.9898 + 78.233) * 43758.5453
  return h - Math.floor(h)
}

function buildNetwork() {
  const pts: [number, number, number][] = []
  const nodePos = new Float32Array(NODES * 3)
  const nodeCol = new Float32Array(NODES * 3)
  for (let i = 0; i < NODES; i++) {
    // golden-spiral shell with radial jitter → an organic sphere of neurons
    const phi = Math.acos(1 - 2 * ((i + 0.5) / NODES))
    const theta = Math.PI * (1 + Math.sqrt(5)) * i
    const r = RADIUS * (0.72 + 0.28 * hash(i))
    const p: [number, number, number] = [
      r * Math.sin(phi) * Math.cos(theta),
      r * Math.sin(phi) * Math.sin(theta) * 0.88,
      r * Math.cos(phi),
    ]
    pts.push(p)
    nodePos.set(p, i * 3)
    const c = C1.clone().lerp(C2, (p[2] + RADIUS) / (2 * RADIUS))
    nodeCol.set([c.r, c.g, c.b], i * 3)
  }

  // Each node links to its 3 nearest neighbours → synapses.
  const edges: [number, number][] = []
  for (let i = 0; i < NODES; i++) {
    const near = pts
      .map((p, j) => ({ j, d: (p[0] - pts[i][0]) ** 2 + (p[1] - pts[i][1]) ** 2 + (p[2] - pts[i][2]) ** 2 }))
      .filter((o) => o.j !== i && o.d < 0.95)
      .sort((a, b) => a.d - b.d)
      .slice(0, 3)
    for (const n of near) if (n.j > i) edges.push([i, n.j])
  }
  const linePos = new Float32Array(edges.length * 6)
  const lineCol = new Float32Array(edges.length * 6)
  edges.forEach(([a, b], k) => {
    linePos.set(pts[a], k * 6)
    linePos.set(pts[b], k * 6 + 3)
    lineCol.set(nodeCol.subarray(a * 3, a * 3 + 3), k * 6)
    lineCol.set(nodeCol.subarray(b * 3, b * 3 + 3), k * 6 + 3)
  })

  // Hubs: every 9th node is drawn larger and brighter.
  const hubIdx = Array.from({ length: NODES }, (_, i) => i).filter((i) => i % 9 === 0)
  const hubPos = new Float32Array(hubIdx.length * 3)
  const hubCol = new Float32Array(hubIdx.length * 3)
  hubIdx.forEach((i, k) => {
    hubPos.set(pts[i], k * 3)
    const c = new Color(nodeCol[i * 3], nodeCol[i * 3 + 1], nodeCol[i * 3 + 2]).lerp(new Color('#ffffff'), 0.35)
    hubCol.set([c.r, c.g, c.b], k * 3)
  })

  // Signal pulses ride a random subset of synapses.
  const pulseEdges = Array.from({ length: PULSES }, (_, k) => edges[Math.floor(hash(k * 5 + 3) * edges.length)])
  const pulseOffset = Array.from({ length: PULSES }, (_, k) => hash(k * 7 + 1))
  const pulseSpeed = Array.from({ length: PULSES }, (_, k) => 0.35 + hash(k * 11 + 2) * 0.7)

  return { pts, nodePos, nodeCol, linePos, lineCol, hubPos, hubCol, pulseEdges, pulseOffset, pulseSpeed }
}

function buildDisc() {
  const pos = new Float32Array(DISC * 3)
  const col = new Float32Array(DISC * 3)
  for (let i = 0; i < DISC; i++) {
    const a = hash(i * 3 + 1) * Math.PI * 2
    const r = 3.1 + hash(i * 3 + 2) ** 1.6 * 1.9
    pos.set([Math.cos(a) * r, (hash(i * 3 + 3) - 0.5) * 0.18 * (r - 2.6), Math.sin(a) * r], i * 3)
    const c = C1.clone().lerp(C2, hash(i * 5 + 4))
    const b = 0.35 + 0.65 * hash(i * 9 + 5)
    col.set([c.r * b, c.g * b, c.b * b], i * 3)
  }
  return { pos, col }
}

function buildStars() {
  const n = 700
  const pos = new Float32Array(n * 3)
  for (let i = 0; i < n; i++) {
    const u = hash(i * 3 + 1) * 2 - 1
    const t = hash(i * 3 + 2) * Math.PI * 2
    const r = 14 + hash(i * 3 + 3) * 10
    const s = Math.sqrt(1 - u * u)
    pos.set([r * s * Math.cos(t), r * u, r * s * Math.sin(t) - 6], i * 3)
  }
  return pos
}

function Network() {
  const group = useRef<Group>(null)
  const disc = useRef<Group>(null)
  const pulseGeo = useRef<BufferGeometry>(null)
  const sprite = useMemo(() => makeDotTexture(), [])
  const net = useMemo(() => buildNetwork(), [])
  const ring = useMemo(() => buildDisc(), [])
  const stars = useMemo(() => buildStars(), [])
  const pulsePos = useMemo(() => new Float32Array(PULSES * 3), [])
  const mouse = useRef({ x: 0, y: 0 })

  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      mouse.current.x = (e.clientX / window.innerWidth) * 2 - 1
      mouse.current.y = (e.clientY / window.innerHeight) * 2 - 1
    }
    window.addEventListener('pointermove', onMove)
    return () => window.removeEventListener('pointermove', onMove)
  }, [])

  useFrame((state, rawDelta) => {
    // Clamp: after the loop is paused off-screen, the first frame back can carry a huge delta.
    const delta = Math.min(rawDelta, 0.05)
    const t = state.clock.elapsedTime
    const g = group.current
    if (g) {
      g.rotation.y += delta * 0.06
      g.rotation.x = MathUtils.lerp(g.rotation.x, mouse.current.y * 0.25, 0.04)
      g.rotation.z = MathUtils.lerp(g.rotation.z, mouse.current.x * 0.1, 0.04)
    }
    if (disc.current) disc.current.rotation.y -= delta * 0.045

    // Move each pulse along its synapse (writing into the mounted attribute's buffer).
    const attr = pulseGeo.current?.getAttribute('position') as BufferAttribute | undefined
    if (!attr) return
    const out = attr.array as Float32Array
    for (let k = 0; k < PULSES; k++) {
      const [a, b] = net.pulseEdges[k]
      const f = (t * net.pulseSpeed[k] + net.pulseOffset[k]) % 1
      const pa = net.pts[a]
      const pb = net.pts[b]
      out[k * 3] = pa[0] + (pb[0] - pa[0]) * f
      out[k * 3 + 1] = pa[1] + (pb[1] - pa[1]) * f
      out[k * 3 + 2] = pa[2] + (pb[2] - pa[2]) * f
    }
    attr.needsUpdate = true
  })

  return (
    <>
      {/* deep stars */}
      <points>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[stars, 3]} />
        </bufferGeometry>
        <pointsMaterial size={1.4} sizeAttenuation={false} color="#eadcc2" transparent opacity={0.4} depthWrite={false} />
      </points>

      <group ref={group}>
        {/* core glow */}
        <sprite scale={[3.4, 3.4, 1]}>
          <spriteMaterial map={sprite} color="#6d5efc" transparent opacity={0.28} depthWrite={false} blending={AdditiveBlending} />
        </sprite>
        <sprite scale={[1.3, 1.3, 1]}>
          <spriteMaterial map={sprite} color="#bfe9ff" transparent opacity={0.5} depthWrite={false} blending={AdditiveBlending} />
        </sprite>

        {/* synapses */}
        <lineSegments>
          <bufferGeometry>
            <bufferAttribute attach="attributes-position" args={[net.linePos, 3]} />
            <bufferAttribute attach="attributes-color" args={[net.lineCol, 3]} />
          </bufferGeometry>
          <lineBasicMaterial vertexColors transparent opacity={0.22} blending={AdditiveBlending} depthWrite={false} />
        </lineSegments>

        {/* neurons */}
        <points>
          <bufferGeometry>
            <bufferAttribute attach="attributes-position" args={[net.nodePos, 3]} />
            <bufferAttribute attach="attributes-color" args={[net.nodeCol, 3]} />
          </bufferGeometry>
          <pointsMaterial map={sprite} size={0.11} vertexColors sizeAttenuation transparent depthWrite={false} blending={AdditiveBlending} />
        </points>
        <points>
          <bufferGeometry>
            <bufferAttribute attach="attributes-position" args={[net.hubPos, 3]} />
            <bufferAttribute attach="attributes-color" args={[net.hubCol, 3]} />
          </bufferGeometry>
          <pointsMaterial map={sprite} size={0.3} vertexColors sizeAttenuation transparent depthWrite={false} blending={AdditiveBlending} />
        </points>

        {/* signal pulses */}
        <points>
          <bufferGeometry ref={pulseGeo}>
            <bufferAttribute attach="attributes-position" args={[pulsePos, 3]} />
          </bufferGeometry>
          <pointsMaterial map={sprite} size={0.13} color="#e6fbff" sizeAttenuation transparent depthWrite={false} blending={AdditiveBlending} />
        </points>
      </group>

      {/* orbiting particle disc */}
      <group rotation={[0.42, 0, 0.12]}>
        <group ref={disc}>
          <points>
            <bufferGeometry>
              <bufferAttribute attach="attributes-position" args={[ring.pos, 3]} />
              <bufferAttribute attach="attributes-color" args={[ring.col, 3]} />
            </bufferGeometry>
            <pointsMaterial map={sprite} size={0.05} vertexColors sizeAttenuation transparent depthWrite={false} blending={AdditiveBlending} />
          </points>
        </group>
      </group>
    </>
  )
}

export default function Hero3D() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  // The lazy canvas can mount before the hero has laid out, leaving R3F measuring
  // a zero-size container. Nudge a resize after first paint so it fills its parent.
  useEffect(() => {
    const raf = requestAnimationFrame(() => window.dispatchEvent(new Event('resize')))
    const t = setTimeout(() => window.dispatchEvent(new Event('resize')), 150)
    return () => {
      cancelAnimationFrame(raf)
      clearTimeout(t)
    }
  }, [])

  const frameloop = useOnScreenFrameloop(canvasRef)

  return (
    <Canvas
      ref={canvasRef}
      frameloop={frameloop}
      // Framed inside a 5:4 display case on the landing, so pull back a touch.
      camera={{ position: [0, 0.3, 7.4], fov: 56 }}
      gl={{ alpha: false, antialias: true, powerPreference: 'high-performance', preserveDrawingBuffer: import.meta.env.DEV }}
      dpr={[1, 1.5]}
      style={{ pointerEvents: 'none' }}
    >
      {/* warm black velvet: the inside of the museum case (matches --case-inside) */}
      <color attach="background" args={['#0f0a07']} />
      <Network />
      <EffectComposer>
        <Bloom intensity={1.2} luminanceThreshold={0.08} luminanceSmoothing={0.9} mipmapBlur />
        <Vignette eskil={false} offset={0.3} darkness={0.75} />
      </EffectComposer>
    </Canvas>
  )
}
