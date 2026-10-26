import { useEffect, useMemo, useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import {
  AdditiveBlending,
  Color,
  QuadraticBezierCurve3,
  TubeGeometry,
  Vector3,
  type Group,
  type Mesh,
  type MeshBasicMaterial,
  type ShaderMaterial,
} from 'three'
import { useInteractable, markExplored } from '../interaction'
import { sfx } from '../audio'
import { CYAN, MONO, MUSTARD, SANS, makeBoard, measure, roundRect, staticBoard } from '../board'

const WORDS = ['The', 'animal', "didn't", 'cross', 'the', 'street', 'because', 'it', 'was', 'too', 'tired']
/** Hand-authored, illustrative causal attention: each word only looks back. */
const WEIGHTS: Record<number, [number, number][]> = {
  1: [[0, 0.6]],
  2: [[1, 0.55], [0, 0.15]],
  3: [[2, 0.45], [1, 0.35]],
  4: [[3, 0.5], [0, 0.1]],
  5: [[4, 0.45], [3, 0.4]],
  6: [[2, 0.3], [3, 0.3], [5, 0.15]],
  7: [[1, 0.62], [5, 0.18], [6, 0.08], [0, 0.06]],
  8: [[7, 0.55], [1, 0.2]],
  9: [[8, 0.5], [7, 0.15]],
  10: [[1, 0.45], [7, 0.3], [9, 0.15], [8, 0.1]],
}
const IT = 7
const DEPTH = 0.36
const GAP = 0.06
const PX = 270
const FONT = `600 46px ${SANS}`

const ARC_VERT = /* glsl */ `
varying vec2 vUv;
void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
`
const ARC_FRAG = /* glsl */ `
uniform float uProg;
uniform float uW;
uniform float uTime;
varying vec2 vUv;
void main() {
  if (vUv.x > uProg) discard;
  vec3 a = vec3(1.0, 0.79, 0.29);
  vec3 b = vec3(0.169, 0.827, 0.878);
  vec3 col = mix(a, b, vUv.x);
  float head = smoothstep(uProg - 0.08, uProg, vUv.x) * step(uProg, 0.999);
  float flow = 0.75 + 0.25 * sin(vUv.x * 30.0 - uTime * 6.0);
  gl_FragColor = vec4(col * (1.0 + head * 2.0) * 1.4, (0.35 + 0.65 * uW) * flow);
}
`

/**
 * Attention walkway: a sentence laid across the nave as glowing floor tiles.
 * Look at a word and press E: light arcs fly back to the earlier words it
 * attends to, thickness = (illustrative) attention weight.
 */
export function AttentionWalk({ z }: { z: number }) {
  const tiles = useMemo(() => {
    const ws = WORDS.map((w) => measure(w, FONT) / PX + 0.16)
    const total = ws.reduce((a, b) => a + b, 0) + GAP * (ws.length - 1)
    const centres: number[] = []
    for (let i = 0, x = -total / 2; i < ws.length; x += ws[i] + GAP, i++) centres.push(x + ws[i] / 2)
    return WORDS.map((w, i) => {
      const cx = centres[i]
      const cw = Math.round(ws[i] * PX)
      const ch = Math.round(DEPTH * PX)
      const tex = staticBoard(cw, ch, (ctx) => {
        // glowing rim so the tiles read as lit glass
        ctx.strokeStyle = 'rgba(190,180,255,0.9)'
        ctx.lineWidth = 5
        roundRect(ctx, 4, 4, cw - 8, ch - 8, 12)
        ctx.stroke()
        ctx.font = FONT
        ctx.fillStyle = '#fff7ea'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText(w, cw / 2, (DEPTH * PX) / 2 + 2)
      })
      return { w, x: cx, width: ws[i], tex }
    })
  }, [])
  useEffect(() => () => tiles.forEach((t) => t.tex.dispose()), [tiles])

  const [sel, setSel] = useState<number | null>(null)
  const selT = useRef(0)
  const clock = useRef(0)
  const hover = useRef(-1)
  const group = useRef<Group>(null)
  const tileMats = useRef<(MeshBasicMaterial | null)[]>([])
  const arcMeshes = useRef<(Mesh | null)[]>([])

  const arcs = useMemo(() => {
    if (sel === null) return []
    const from = tiles[sel]
    return (WEIGHTS[sel] ?? [])
      .slice()
      .sort((a, b) => b[1] - a[1])
      .map(([j, w]) => {
        const to = tiles[j]
        const dx = Math.abs(from.x - to.x)
        const curve = new QuadraticBezierCurve3(
          new Vector3(from.x, 0.07, 0),
          new Vector3((from.x + to.x) / 2, 0.35 + dx * 0.5, 0),
          new Vector3(to.x, 0.07, 0),
        )
        const geo = new TubeGeometry(curve, 56, 0.006 + 0.04 * w, 8, false)
        return { j, w, geo, uniforms: { uProg: { value: 0 }, uW: { value: w }, uTime: { value: 0 } } }
      })
  }, [sel, tiles])
  useEffect(() => () => arcs.forEach((a) => a.geo.dispose()), [arcs])

  const sign = useMemo(() => makeBoard(1200, 210), [])
  useEffect(() => {
    sign.draw((ctx) => {
      ctx.fillStyle = 'rgba(11,15,26,0.92)'
      ctx.fillRect(0, 0, 1200, 210)
      ctx.strokeStyle = 'rgba(139,123,255,0.4)'
      ctx.lineWidth = 3
      ctx.strokeRect(8, 8, 1184, 194)
      ctx.fillStyle = CYAN
      ctx.font = `600 28px ${MONO}`
      ctx.fillText('ATTENTION', 40, 62)
      ctx.fillStyle = 'rgba(233,236,243,0.45)'
      ctx.font = `400 20px ${MONO}`
      ctx.fillText('illustrative weights · each word only looks back', 230, 62)
      if (sel === null) {
        ctx.fillStyle = '#f2f4f8'
        ctx.font = `500 46px ${SANS}`
        ctx.fillText('Which words does “it” look at?', 40, 140)
        ctx.fillStyle = 'rgba(233,236,243,0.6)'
        ctx.font = `400 26px ${SANS}`
        ctx.fillText('Look at a word on the floor and press E.', 40, 182)
      } else {
        ctx.font = `600 46px ${SANS}`
        ctx.fillStyle = MUSTARD
        const head = `“${WORDS[sel]}”`
        ctx.fillText(head, 40, 140)
        let x = 40 + ctx.measureText(head).width + 20
        ctx.font = `400 36px ${SANS}`
        ctx.fillStyle = 'rgba(233,236,243,0.75)'
        const list = (WEIGHTS[sel] ?? []).slice().sort((a, b) => b[1] - a[1])
        const rest = list.length ? 'looks at' : 'is first: nothing earlier to look at'
        ctx.fillText(rest, x, 140)
        x += ctx.measureText(rest).width + 22
        for (const [j, w] of list) {
          ctx.fillStyle = CYAN
          ctx.font = `600 36px ${SANS}`
          const s = `${WORDS[j]} ${Math.round(w * 100)}%`
          ctx.fillText(s, x, 140)
          x += ctx.measureText(s).width + 26
        }
        ctx.fillStyle = 'rgba(233,236,243,0.5)'
        ctx.font = `400 24px ${SANS}`
        ctx.fillText('Thicker light = more attention. This is how “it” learns it means the animal.', 40, 184)
      }
    })
  }, [sel, sign])

  useInteractable({
    targets: [group],
    reach: 3.4,
    verb: (h) => {
      const i = h.userData.i as number | undefined
      return i === undefined ? 'trace attention' : `see what “${WORDS[i]}” looks at`
    },
    onInteract: (h) => {
      const i = h.userData.i as number | undefined
      if (i === undefined) return
      markExplored('attention')
      setSel(i)
      selT.current = clock.current
      sfx('chime', { v: 0.6, pitch: 0.75 })
      sfx('whoosh', { v: 0.5 })
    },
    setHighlight: (h) => void (hover.current = h ? ((h.userData.i as number | undefined) ?? -1) : -1),
  })

  const base = useMemo(() => new Color('#3d3690'), [])
  const cyan = useMemo(() => new Color(CYAN), [])
  const mustard = useMemo(() => new Color(MUSTARD), [])
  const hoverC = useMemo(() => new Color('#8b7bff'), [])

  useFrame((state) => {
    const t = (clock.current = state.clock.elapsedTime)
    const since = t - selT.current
    const weights = sel === null ? [] : (WEIGHTS[sel] ?? [])
    tileMats.current.forEach((m, i) => {
      if (!m) return
      m.color.copy(base)
      if (i === hover.current) m.color.lerp(hoverC, 0.8)
      if (sel === null && i === IT) m.color.lerp(mustard, 0.25 + 0.2 * Math.sin(t * 2.5))
      if (sel === i) m.color.copy(mustard).multiplyScalar(0.5)
      const w = weights.find((x) => x[0] === i)?.[1]
      if (w !== undefined) m.color.lerp(cyan, Math.min(1, since / 0.8) * (0.2 + w * 0.6))
    })
    arcMeshes.current.forEach((mesh, k) => {
      if (!mesh) return
      const u = (mesh.material as ShaderMaterial).uniforms
      u.uProg.value = Math.min(1, Math.max(0, (since - k * 0.12) / 0.7))
      u.uTime.value = t
    })
  })

  return (
    <group position={[0, 0, z]}>
      <group ref={group}>
        {tiles.map((tl, i) => (
          <group key={i} position={[tl.x, 0, 0]}>
            <mesh position={[0, 0.025, 0]} userData={{ i }}>
              <boxGeometry args={[tl.width, 0.05, DEPTH]} />
              <meshBasicMaterial ref={(m) => void (tileMats.current[i] = m)} color="#3d3690" toneMapped={false} />
            </mesh>
            <mesh position={[0, 0.051, 0]} rotation={[-Math.PI / 2, 0, 0]} userData={{ i }}>
              <planeGeometry args={[tl.width, DEPTH]} />
              <meshBasicMaterial map={tl.tex} transparent toneMapped={false} depthWrite={false} />
            </mesh>
          </group>
        ))}
      </group>
      {arcs.map((a, k) => (
        <mesh key={`${sel}-${a.j}`} ref={(m) => void (arcMeshes.current[k] = m)} geometry={a.geo} renderOrder={3}>
          <shaderMaterial
            vertexShader={ARC_VERT}
            fragmentShader={ARC_FRAG}
            uniforms={a.uniforms}
            transparent
            depthWrite={false}
            blending={AdditiveBlending}
            toneMapped={false}
          />
        </mesh>
      ))}
      {/* overhead sign (above head height, so you can walk under it) */}
      <mesh position={[0, 2.78, -0.7]}>
        <planeGeometry args={[2.8, 0.49]} />
        <meshBasicMaterial map={sign.tex} toneMapped={false} transparent />
      </mesh>
      {[-1.3, 1.3].map((x) => (
        <mesh key={x} position={[x, (3.03 + 4.5) / 2, -0.7]}>
          <cylinderGeometry args={[0.004, 0.004, 4.5 - 3.03, 4]} />
          <meshBasicMaterial color="#6d655a" />
        </mesh>
      ))}
    </group>
  )
}

