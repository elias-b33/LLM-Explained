import { useEffect, useMemo, useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import {
  AdditiveBlending,
  BufferGeometry,
  Color,
  Float32BufferAttribute,
  Quaternion,
  Vector3,
  type CanvasTexture,
  type Group,
  type Mesh,
  type MeshBasicMaterial,
  type Sprite,
} from 'three'
import { tile, type MuseumTextures } from '../materials'
import { useInteractable, markExplored } from '../interaction'
import { sfx } from '../audio'
import { CYAN, MONO, MUSTARD, SANS, VIOLET, makeBoard, measure, staticBoard } from '../board'

type V = [number, number, number]
// Local frame: +x = viewer's right, +y up, +z toward the viewer.
const man: V = [-1.05, 1.25, 0.1]
const woman: V = [-0.6, 1.32, -0.12]
const king: V = [-0.95, 1.95, 0.0]
const france: V = [0.55, 1.2, 0.15]
const italy: V = [1.05, 1.15, -0.1]
const paris: V = [0.6, 1.72, 0.05]
const add = (a: V, b: V, c: V): V => [a[0] - b[0] + c[0], a[1] - b[1] + c[1], a[2] - b[2] + c[2]]

const WORDS: Record<string, { p: V; c: string }> = {
  king: { p: king, c: VIOLET },
  queen: { p: add(king, man, woman), c: VIOLET },
  prince: { p: [-1.15, 1.74, 0.22], c: VIOLET },
  princess: { p: [-0.7, 1.81, 0.0], c: VIOLET },
  man: { p: man, c: '#c9b8ff' },
  woman: { p: woman, c: '#c9b8ff' },
  boy: { p: [-1.2, 1.04, 0.26], c: '#c9b8ff' },
  girl: { p: [-0.75, 1.1, 0.04], c: '#c9b8ff' },
  France: { p: france, c: CYAN },
  Italy: { p: italy, c: CYAN },
  Germany: { p: [0.3, 1.36, -0.22], c: CYAN },
  Paris: { p: paris, c: '#8ff0f7' },
  Rome: { p: add(paris, france, italy), c: '#8ff0f7' },
  Berlin: { p: [0.35, 1.88, -0.32], c: '#8ff0f7' },
  cat: { p: [-0.12, 1.0, 0.3], c: '#ffb38a' },
  kitten: { p: [0.06, 1.12, 0.12], c: '#ffb38a' },
  dog: { p: [-0.3, 1.18, -0.14], c: '#ffb38a' },
  puppy: { p: [-0.12, 1.3, -0.3], c: '#ffb38a' },
  pizza: { p: [0.05, 2.02, 0.16], c: MUSTARD },
  pasta: { p: [-0.2, 1.9, -0.12], c: MUSTARD },
  apple: { p: [0.33, 1.96, -0.08], c: MUSTARD },
}

const ANALOGIES: [string, string, string, string][] = [
  ['king', 'man', 'woman', 'queen'],
  ['Paris', 'France', 'Italy', 'Rome'],
]

const UP = new Vector3(0, 1, 0)
const DIR = new Vector3()
const Q = new Quaternion()

/** Point an arrow (group with a unit shaft + head) from a toward b, drawn to fraction k. */
function aim(g: Group | null, a: V, b: V, k: number) {
  if (!g) return
  DIR.set(b[0] - a[0], b[1] - a[1], b[2] - a[2])
  const len = DIR.length()
  g.visible = k > 0.001
  if (!g.visible) return
  DIR.normalize()
  Q.setFromUnitVectors(UP, DIR)
  g.position.set(...a)
  g.quaternion.copy(Q)
  const L = Math.max(0.001, (len - 0.08) * k) // stop just short of the word
  const head = Math.min(0.07, L * 0.5)
  const shaft = g.children[0]
  const tip = g.children[1]
  shaft.scale.set(1, Math.max(0.001, L - head), 1)
  shaft.position.y = (L - head) / 2
  tip.position.y = L - head / 2
  tip.scale.setScalar(head / 0.07)
}

function Arrow({ color, groupRef }: { color: string; groupRef: React.RefObject<Group | null> }) {
  return (
    <group ref={groupRef} visible={false}>
      <mesh>
        <cylinderGeometry args={[0.009, 0.009, 1, 8]} />
        <meshBasicMaterial color={color} toneMapped={false} transparent opacity={0.95} blending={AdditiveBlending} depthWrite={false} />
      </mesh>
      <mesh>
        <coneGeometry args={[0.03, 0.07, 12]} />
        <meshBasicMaterial color={color} toneMapped={false} transparent opacity={0.95} blending={AdditiveBlending} depthWrite={false} />
      </mesh>
    </group>
  )
}

function wordTex(w: string, c: string): { tex: CanvasTexture; aspect: number } {
  const font = `600 44px ${SANS}`
  const cw = Math.ceil(measure(w, font)) + 40
  const tex = staticBoard(cw, 72, (ctx) => {
    ctx.font = font
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.shadowColor = c
    ctx.shadowBlur = 14
    ctx.fillStyle = '#ffffff'
    ctx.fillText(w, cw / 2, 38)
    ctx.shadowBlur = 0
    ctx.fillStyle = c
    ctx.beginPath()
    ctx.arc(10, 38, 5, 0, Math.PI * 2)
    ctx.fill()
  })
  return { tex, aspect: cw / 72 }
}

/**
 * Embedding alcove: words float in a 3D "meaning space" where related words
 * cluster. E plays vector arithmetic: the man→woman arrow is copied onto
 * "king" and lands on "queen" (then Paris − France + Italy ≈ Rome).
 */
export function EmbeddingAlcove({ tex, position }: { tex: MuseumTextures; position: [number, number, number] }) {
  const names = Object.keys(WORDS)
  const labels = useMemo(() => names.map((n) => wordTex(n, WORDS[n].c)), []) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => () => labels.forEach((l) => l.tex.dispose()), [labels])
  const sprites = useRef<(Sprite | null)[]>([])
  const arrowA = useRef<Group>(null)
  const arrowB = useRef<Group>(null)
  const hit = useRef<Mesh>(null)
  const ring = useRef<Mesh>(null)
  const cloud = useRef<Group>(null)
  const hl = useRef(false)
  const [an, setAn] = useState(0) // which analogy plays next
  const play = useRef<{ i: number; t0: number } | null>(null)
  const [step, setStep] = useState(-1) // equation progress shown on the sign
  const shown = useRef(-1)
  const clock = useRef(0)
  const oak = useMemo(() => tile(tex.oak, 2, 2), [tex])

  // faint background dust around the clusters
  const dust = useMemo(() => {
    const pos: number[] = []
    const col: number[] = []
    let s = 11
    const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647 - 0.5)
    for (const n of names) {
      const { p, c } = WORDS[n]
      const cc = new Color(c)
      for (let k = 0; k < 7; k++) {
        pos.push(p[0] + rnd() * 0.3, p[1] + rnd() * 0.25, p[2] + rnd() * 0.3)
        col.push(cc.r, cc.g, cc.b)
      }
    }
    const g = new BufferGeometry()
    g.setAttribute('position', new Float32BufferAttribute(pos, 3))
    g.setAttribute('color', new Float32BufferAttribute(col, 3))
    return g
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const sign = useMemo(() => makeBoard(1000, 200), [])
  useEffect(() => {
    const [a, b, c, d] = ANALOGIES[play.current?.i ?? an]
    sign.draw((ctx) => {
      ctx.fillStyle = 'rgba(11,15,26,0.9)'
      ctx.fillRect(0, 0, 1000, 200)
      ctx.strokeStyle = 'rgba(43,211,224,0.35)'
      ctx.lineWidth = 3
      ctx.strokeRect(8, 8, 984, 184)
      ctx.fillStyle = CYAN
      ctx.font = `600 26px ${MONO}`
      ctx.fillText('MEANING AS POSITION', 36, 56)
      ctx.font = `500 50px ${SANS}`
      if (step < 0) {
        ctx.fillStyle = '#f2f4f8'
        ctx.font = `500 40px ${SANS}`
        ctx.fillText('Similar words sit close together.', 36, 122)
        ctx.fillStyle = 'rgba(233,236,243,0.55)'
        ctx.font = `400 26px ${SANS}`
        ctx.fillText('Directions mean something too. Press E to do maths with words.', 36, 166)
        return
      }
      const parts: [string, string][] = [
        [a, MUSTARD],
        [` − ${b}`, step >= 1 ? '#c9b8ff' : 'transparent'],
        [` + ${c}`, step >= 1 ? '#c9b8ff' : 'transparent'],
        [` ≈ ${d}`, step >= 3 ? MUSTARD : 'transparent'],
      ]
      let x = 36
      for (const [s, col] of parts) {
        ctx.fillStyle = col
        ctx.fillText(s, x, 130)
        x += ctx.measureText(s).width
      }
      ctx.fillStyle = 'rgba(233,236,243,0.5)'
      ctx.font = `400 24px ${SANS}`
      ctx.fillText(step >= 3 ? 'Same arrow, new starting point. (Illustrative 3D map of a ~1000-D space.)' : `The ${b} → ${c} direction…`, 36, 174)
    })
  }, [step, an, sign])

  useInteractable({
    targets: [hit],
    reach: 3.0,
    verb: () => {
      const [a, b, c] = ANALOGIES[an]
      return `${a} − ${b} + ${c}`
    },
    onInteract: () => {
      if (play.current && clock.current - play.current.t0 < 2.6) return
      markExplored('embeddings')
      play.current = { i: an, t0: clock.current }
      setAn((x) => (x + 1) % ANALOGIES.length)
      shown.current = 0
      setStep(0)
      sfx('click')
    },
    setHighlight: (h) => void (hl.current = !!h),
  })

  useFrame((state) => {
    const t = (clock.current = state.clock.elapsedTime)
    if (cloud.current) cloud.current.position.y = Math.sin(t * 0.6) * 0.015
    const pl = play.current
    let active: string[] = []
    let goal: string | null = null
    if (pl) {
      const [a, b, c, d] = ANALOGIES[pl.i]
      const dt = t - pl.t0
      const pa = WORDS[a].p
      const pb = WORDS[b].p
      const pc = WORDS[c].p
      const kA = Math.min(1, Math.max(0, (dt - 0.5) / 0.7))
      const kB = Math.min(1, Math.max(0, (dt - 1.5) / 0.9))
      aim(arrowA.current, pb, pc, kA)
      aim(arrowB.current, pa, add(pa, pb, pc), kB)
      const s = dt < 0.5 ? 0 : dt < 1.5 ? 1 : dt < 2.4 ? 2 : 3
      if (s !== shown.current) {
        shown.current = s
        setStep(s)
        if (s === 1) sfx('whoosh', { v: 0.5 })
        if (s === 3) sfx('chime')
      }
      active = s >= 1 ? [a, b, c] : [a]
      if (s >= 3) goal = d
    }
    names.forEach((n, i) => {
      const sp = sprites.current[i]
      if (!sp) return
      const { aspect } = labels[i]
      const isGoal = n === goal
      const on = active.includes(n) || isGoal
      const s = 0.1 * (on ? 1.3 + (isGoal ? 0.15 * Math.sin(t * 5) : 0) : 1)
      sp.scale.set(s * aspect, s, 1)
      const m = sp.material
      m.opacity = !pl || on ? 1 : 0.4
      m.color.setScalar(hl.current && !pl ? 1.2 : 1)
    })
    if (ring.current) (ring.current.material as MeshBasicMaterial).opacity = (hl.current ? 0.85 : 0.4) + Math.sin(t * 1.7) * 0.07
  })

  return (
    <group position={position} rotation={[0, -Math.PI / 2, 0]}>
      {/* dais */}
      <mesh position={[0, 0.06, 0]} receiveShadow castShadow>
        <cylinderGeometry args={[1.05, 1.1, 0.12, 48]} />
        <meshStandardMaterial {...oak} color="#f0dcb4" roughness={0.7} />
      </mesh>
      <mesh ref={ring} position={[0, 0.125, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.92, 0.97, 64]} />
        <meshBasicMaterial color={CYAN} transparent opacity={0.4} toneMapped={false} blending={AdditiveBlending} depthWrite={false} />
      </mesh>
      <group ref={cloud}>
        <points geometry={dust}>
          <pointsMaterial size={0.012} vertexColors transparent opacity={0.7} depthWrite={false} blending={AdditiveBlending} toneMapped={false} />
        </points>
        {names.map((n, i) => (
          <sprite key={n} ref={(s) => void (sprites.current[i] = s)} position={WORDS[n].p}>
            <spriteMaterial map={labels[i].tex} transparent depthWrite={false} toneMapped={false} />
          </sprite>
        ))}
        <Arrow color={CYAN} groupRef={arrowA} />
        <Arrow color={MUSTARD} groupRef={arrowB} />
      </group>
      {/* sign behind/above the cloud, tilted down so it reads from the dais edge */}
      <mesh position={[0, 2.6, -0.6]} rotation={[0.4, 0, 0]}>
        <planeGeometry args={[1.85, 0.37]} />
        <meshBasicMaterial map={sign.tex} transparent toneMapped={false} />
      </mesh>
      <mesh ref={hit} position={[0, 1.6, 0]}>
        <boxGeometry args={[2.7, 1.6, 0.9]} />
        <meshBasicMaterial visible={false} />
      </mesh>
    </group>
  )
}
