import { useEffect, useMemo, useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import { AdditiveBlending, type Group, type Mesh, type MeshBasicMaterial } from 'three'
import { tile, type MuseumTextures } from '../materials'
import { useInteractable, markExplored } from '../interaction'
import { sfx } from '../audio'
import { CYAN, MONO, MUSTARD, SANS, VIOLET, makeBoard, staticBoard } from '../board'
import { MAX_STEPS, PROMPTS, TEMPERATURES, distFor, isEnd, sample, shape, type Dist } from './autoregressData'

/* Board geometry (canvas logical px) */
const BW = 1280
const BH = 720
const W = 3.6
const H = (W * BH) / BW
const CY = 3.2 // board centre height; its bottom (≈2.19 m) clears your head
const BAR_X = 330
const BAR_MAX = 700
const ROW0 = 330
const ROW = 66
const KNOB = '#d9452f'
const DIAL_ANGLES = [-1.75, -0.6, 0.6, 1.75] // radians, clockwise from "toward the board"

const toWorld = (px: number, py: number): [number, number] => [(px / BW - 0.5) * W, CY + (0.5 - py / BH) * H]

interface Sim {
  prompt: number
  gen: string[]
  done: boolean
  T: number
  dist: Dist
  shown: number[]
  from: number[]
  to: number[]
  morph0: number
  mode: 'idle' | 'roll' | 'slide' | 'landed'
  seq: number[]
  hops: number[] // cumulative end time of each hop
  t0: number
  pick: number
  row: number
  lastDraw: number
}

function wrap(ctx: CanvasRenderingContext2D, parts: [string, string][], x0: number, y0: number, maxW: number, lh: number) {
  let x = x0
  let y = y0
  for (const [text, col] of parts) {
    // split each part on spaces but keep them attached to the following word
    const words = text.match(/\s*\S+/g) ?? [text]
    for (const w of words) {
      const ww = ctx.measureText(w).width
      if (x + ww > x0 + maxW && x > x0) {
        x = x0
        y += lh
      }
      const draw = x === x0 ? w.trimStart() : w
      ctx.fillStyle = col
      ctx.fillText(draw, x, y)
      x += ctx.measureText(draw).width
    }
  }
  return { x, y }
}

function drawBoard(ctx: CanvasRenderingContext2D, s: Sim) {
  ctx.fillStyle = '#0b0f1a'
  ctx.fillRect(0, 0, BW, BH)
  ctx.strokeStyle = 'rgba(139,123,255,0.35)'
  ctx.lineWidth = 3
  ctx.strokeRect(14, 14, BW - 28, BH - 28)
  ctx.textBaseline = 'alphabetic'
  ctx.fillStyle = CYAN
  ctx.font = `600 30px ${MONO}`
  ctx.fillText('AUTOREGRESSION', 60, 76)
  ctx.textAlign = 'right'
  ctx.fillStyle = MUSTARD
  ctx.fillText(`temperature ${TEMPERATURES[s.T].toFixed(1)}`, BW - 60, 76)
  ctx.textAlign = 'left'

  // the growing sentence
  ctx.font = `500 50px ${SANS}`
  const parts: [string, string][] = [[PROMPTS[s.prompt].text, '#f2f4f8']]
  s.gen.forEach((g, i) => parts.push([g, i === s.gen.length - 1 && s.mode === 'idle' ? '#ffe39a' : MUSTARD]))
  const end = wrap(ctx, parts, 60, 160, BW - 140, 64)
  if (!s.done) {
    ctx.fillStyle = 'rgba(43,211,224,0.8)'
    ctx.fillRect(end.x + 8, end.y - 40, 22, 48)
  }

  if (s.done) {
    ctx.font = `500 40px ${SANS}`
    ctx.fillStyle = '#f2f4f8'
    ctx.fillText('The model picked an ending, so it stops.', 60, 400)
    ctx.fillStyle = 'rgba(233,236,243,0.6)'
    ctx.font = `400 30px ${SANS}`
    ctx.fillText('Every word was one roll of the dice, fed back in.', 60, 456)
    ctx.fillStyle = MUSTARD
    ctx.fillText('Pull the lever for a new prompt.', 60, 530)
  } else {
    s.dist.forEach(([tok], i) => {
      const y = ROW0 + i * ROW
      const picked = s.mode === 'landed' && i === s.pick
      const lit = i === s.row || picked
      ctx.font = `${lit ? 600 : 400} 34px ${MONO}`
      ctx.fillStyle = lit ? '#ffffff' : 'rgba(233,236,243,0.7)'
      const label = tok.startsWith(' ') ? '␣' + tok.slice(1) : tok
      ctx.fillText(label.length > 12 ? label.slice(0, 11) + '…' : label, 60, y + 40)
      ctx.fillStyle = 'rgba(255,255,255,0.06)'
      ctx.fillRect(BAR_X, y + 8, BAR_MAX, 42)
      const p = s.shown[i] ?? 0
      ctx.fillStyle = picked ? MUSTARD : lit ? CYAN : VIOLET
      ctx.globalAlpha = lit ? 1 : 0.8
      ctx.fillRect(BAR_X, y + 8, Math.max(4, BAR_MAX * p), 42)
      ctx.globalAlpha = 1
      ctx.font = `400 30px ${MONO}`
      ctx.fillStyle = 'rgba(233,236,243,0.75)'
      const pct = p * 100
      ctx.fillText(pct >= 1 ? `${pct.toFixed(0)}%` : `${pct.toFixed(1)}%`, BAR_X + BAR_MAX + 24, y + 40)
    })
  }
  ctx.fillStyle = 'rgba(233,236,243,0.32)'
  ctx.font = `400 22px ${SANS}`
  ctx.fillText(
    `Illustrative probabilities · token ${Math.min(s.gen.length, MAX_STEPS)}/${MAX_STEPS} · pull → sample → append → repeat`,
    60,
    BH - 40,
  )
}

function glowTexture() {
  return staticBoard(64, 64, (ctx) => {
    const g = ctx.createRadialGradient(32, 32, 0, 32, 32, 32)
    g.addColorStop(0, 'rgba(255,240,200,1)')
    g.addColorStop(0.3, 'rgba(255,201,74,0.6)')
    g.addColorStop(1, 'rgba(255,201,74,0)')
    ctx.fillStyle = g
    ctx.fillRect(0, 0, 64, 64)
  }, 1)
}

/**
 * The autoregression machine: an overhead board plus a lever console. Each
 * pull samples the next token from the shown distribution (a light ball
 * rattles over the bars and lands on the pick), appends it to the sentence and
 * shows the next distribution. The dial sets the temperature, reshaping the
 * bars as softmax(log p / T).
 */
export function AutoregressMachine({ tex, z, consoleZ }: { tex: MuseumTextures; z: number; consoleZ: number }) {
  const board = useMemo(() => makeBoard(BW, BH), [])
  const glow = useMemo(() => glowTexture(), [])
  const oak = useMemo(() => tile(tex.oak, 1.5, 1), [tex])
  const [tIdx, setTIdx] = useState(1)
  const sim = useRef<Sim>(null as unknown as Sim)
  if (sim.current == null) {
    const dist = distFor(PROMPTS[0], null, 0)
    const p = shape(dist, TEMPERATURES[1])
    sim.current = {
      prompt: 0, gen: [], done: false, T: 1, dist, shown: p, from: p, to: p, morph0: -10,
      mode: 'idle', seq: [], hops: [], t0: 0, pick: 0, row: -1, lastDraw: 0,
    }
  }
  const clock = useRef(0)
  const redraw = () => board.draw((ctx) => drawBoard(ctx, sim.current))
  useEffect(redraw, []) // eslint-disable-line react-hooks/exhaustive-deps

  const ball = useRef<Group>(null)
  const arm = useRef<Group>(null)
  const knob = useRef<Mesh>(null)
  const dial = useRef<Group>(null)
  const dialRing = useRef<Mesh>(null)
  const leverHit = useRef<Mesh>(null)
  const dialHit = useRef<Mesh>(null)
  const pullT = useRef(-10)
  const hlLever = useRef(false)
  const hlDial = useRef(false)
  const dialAngle = useRef(DIAL_ANGLES[1])

  const morphTo = (to: number[], from?: number[]) => {
    const s = sim.current
    s.from = from ?? s.shown.slice()
    s.to = to
    s.morph0 = clock.current
  }

  const face = useMemo(() => makeBoard(1024, 288), [])
  useEffect(() => {
    face.draw((ctx) => {
      ctx.fillStyle = '#15130f'
      ctx.fillRect(0, 0, 1024, 288)
      ctx.strokeStyle = 'rgba(184,137,47,0.7)'
      ctx.lineWidth = 4
      ctx.strokeRect(6, 6, 1012, 276)
      ctx.textAlign = 'center'
      ctx.fillStyle = MUSTARD
      ctx.font = `600 26px ${MONO}`
      ctx.fillText('PULL TO SAMPLE', 192, 262)
      ctx.fillText('TEMPERATURE', 800, 262)
      ctx.fillStyle = 'rgba(233,236,243,0.7)'
      ctx.font = `500 30px ${SANS}`
      ctx.fillText('sample → append → repeat', 500, 132)
      ctx.fillStyle = 'rgba(233,236,243,0.42)'
      ctx.font = `400 22px ${SANS}`
      ctx.fillText(['nearly always the top token', 'mostly likely tokens', 'the plain probabilities', 'wild, surprising picks'][tIdx], 800, 232)
      DIAL_ANGLES.forEach((a, i) => {
        const x = 800 + Math.sin(a) * 122
        const y = 140 - Math.cos(a) * 100
        ctx.fillStyle = i === tIdx ? MUSTARD : 'rgba(233,236,243,0.55)'
        ctx.font = `${i === tIdx ? 700 : 500} 26px ${MONO}`
        ctx.fillText(TEMPERATURES[i].toFixed(1), x, y + 9)
      })
      ctx.strokeStyle = 'rgba(184,137,47,0.5)'
      ctx.lineWidth = 3
      ctx.beginPath()
      ctx.arc(800, 136, 66, 0, Math.PI * 2)
      ctx.stroke()
    })
  }, [tIdx, face])

  const pull = () => {
    const s = sim.current
    if (s.mode !== 'idle' || clock.current - pullT.current < 0.55) return
    markExplored('lever')
    pullT.current = clock.current
    sfx('clunk')
    if (s.done) {
      s.prompt = (s.prompt + 1) % PROMPTS.length
      s.gen = []
      s.done = false
      s.dist = distFor(PROMPTS[s.prompt], null, 0)
      morphTo(shape(s.dist, TEMPERATURES[s.T]), s.dist.map(() => 0))
      return
    }
    const p = shape(s.dist, TEMPERATURES[s.T])
    s.shown = p.slice()
    s.to = p
    s.morph0 = -10
    const k = sample(p)
    // rattle up and down the rows, slowing, and stop on the pick
    const n = s.dist.length
    const seq: number[] = []
    if (n === 1) seq.push(0, 0, 0)
    else {
      let r = 0
      let dir = 1
      for (let i = 0; i < 40; i++) {
        seq.push(r)
        if (seq.length >= 7 && r === k) break
        if (r + dir < 0 || r + dir >= n) dir = -dir
        r += dir
      }
    }
    if (seq[seq.length - 1] !== k) seq.push(k)
    let acc = 0
    s.hops = seq.map((_, h) => (acc += 0.055 + 0.13 * (h / seq.length) ** 2))
    s.seq = seq
    s.pick = k
    s.mode = 'roll'
    s.t0 = clock.current + 0.25
    s.row = -1
  }

  const turnDial = () => {
    const s = sim.current
    if (s.mode === 'roll' || s.mode === 'slide') return
    markExplored('temperature')
    s.T = (s.T + 1) % TEMPERATURES.length
    setTIdx(s.T)
    sfx('click')
    if (!s.done) morphTo(shape(s.dist, TEMPERATURES[s.T]))
    else redraw()
  }

  useInteractable({
    targets: [leverHit],
    reach: 2.6,
    verb: () => (sim.current.done ? 'pull for a new prompt' : 'pull the lever · sample a token'),
    onInteract: pull,
    setHighlight: (h) => void (hlLever.current = !!h),
  })
  useInteractable({
    targets: [dialHit],
    reach: 2.6,
    verb: () => `temperature ${TEMPERATURES[sim.current.T].toFixed(1)} → ${TEMPERATURES[(sim.current.T + 1) % TEMPERATURES.length].toFixed(1)}`,
    onInteract: turnDial,
    setHighlight: (h) => void (hlDial.current = !!h),
  })

  useFrame((state, dt) => {
    const t = (clock.current = state.clock.elapsedTime)
    const s = sim.current

    // bar morph (temperature change / new distribution), redrawn at ≤30 Hz
    const mk = (t - s.morph0) / 0.4
    if (mk >= 0 && mk <= 1.2) {
      const e = 1 - (1 - Math.min(1, mk)) ** 3
      s.shown = s.to.map((v, i) => (s.from[i] ?? 0) + (v - (s.from[i] ?? 0)) * e)
      if (t - s.lastDraw > 1 / 30 || mk > 1) {
        s.lastDraw = t
        redraw()
      }
      if (mk > 1) s.morph0 = -10
    }

    // the sampling ball
    const b = ball.current
    if (b) {
      let vis = false
      if (s.mode === 'roll') {
        const el = t - s.t0
        vis = el > -0.25
        let h = s.hops.findIndex((x) => x > el)
        if (h < 0) {
          s.mode = 'slide'
          s.t0 = t
          h = s.seq.length - 1
        }
        const row = s.seq[Math.max(0, h)]
        if (row !== s.row && el >= 0) {
          s.row = row
          sfx('tick')
          redraw()
        }
        const [x, y] = toWorld(BAR_X - 26, ROW0 + Math.max(0, row) * ROW + 29)
        b.position.set(x, el < 0 ? y + 0.25 * (-el / 0.25) : y, 0.05)
      } else if (s.mode === 'slide') {
        vis = true
        const k = Math.min(1, (t - s.t0) / 0.35)
        const e = 1 - (1 - k) ** 2
        const [x0, y] = toWorld(BAR_X - 26, ROW0 + s.pick * ROW + 29)
        const [x1] = toWorld(BAR_X + Math.max(4, BAR_MAX * s.shown[s.pick]), 0)
        b.position.set(x0 + (x1 - x0) * e, y, 0.05)
        if (k >= 1) {
          s.mode = 'landed'
          s.t0 = t
          s.row = -1
          const tok = s.dist[s.pick][0]
          s.gen.push(tok)
          sfx('chime', { pitch: 0.85 + 0.35 * (1 - s.shown[s.pick]) })
          redraw()
        }
      } else if (s.mode === 'landed') {
        vis = t - s.t0 < 0.5
        b.scale.setScalar(1 + (t - s.t0) * 2)
        if (t - s.t0 > 0.55) {
          b.scale.setScalar(1)
          const tok = s.gen[s.gen.length - 1]
          s.mode = 'idle'
          if (isEnd(tok) || s.gen.length >= MAX_STEPS) {
            s.done = true
            redraw()
          } else {
            s.dist = distFor(PROMPTS[s.prompt], tok, s.gen.length)
            morphTo(shape(s.dist, TEMPERATURES[s.T]), s.dist.map(() => 0))
          }
        }
      }
      b.visible = vis
      if (s.mode !== 'landed') b.scale.setScalar(1)
    }

    // lever: rest tilted back, swing toward you on a pull
    if (arm.current) {
      const k = (t - pullT.current) / 0.5
      const swing = k >= 0 && k <= 1 ? Math.sin(Math.PI * k) : 0
      arm.current.rotation.x = -0.45 + swing * 1.05
    }
    if (knob.current) (knob.current.material as MeshBasicMaterial).color.set(KNOB).multiplyScalar(hlLever.current ? 1.8 : 1)

    // dial
    if (dial.current) {
      const target = -DIAL_ANGLES[s.T]
      dialAngle.current += (target - dialAngle.current) * Math.min(1, dt * 12)
      dial.current.rotation.y = dialAngle.current
    }
    if (dialRing.current) {
      const m = dialRing.current.material as MeshBasicMaterial
      m.opacity = (hlDial.current ? 0.9 : 0.35) + Math.sin(t * 2) * 0.05
    }
  })

  return (
    <group>
      {/* overhead board */}
      <group position={[0, 0, z]}>
        <mesh position={[0, CY, -0.03]} castShadow>
          <boxGeometry args={[W + 0.08, H + 0.08, 0.05]} />
          <meshStandardMaterial color="#1c1a17" metalness={0.5} roughness={0.45} />
        </mesh>
        <mesh position={[0, CY, 0]}>
          <planeGeometry args={[W, H]} />
          <meshBasicMaterial map={board.tex} toneMapped={false} />
        </mesh>
        {[CY + H / 2 + 0.02, CY - H / 2 - 0.02].map((y) => (
          <mesh key={y} position={[0, y, 0]}>
            <boxGeometry args={[W + 0.06, 0.04, 0.04]} />
            <meshStandardMaterial color="#2a2621" metalness={0.6} roughness={0.4} />
          </mesh>
        ))}
        {[-W / 2 + 0.1, W / 2 - 0.1].map((x) => (
          <mesh key={x} position={[x, (CY + H / 2 + 6.1) / 2, 0]}>
            <cylinderGeometry args={[0.004, 0.004, 6.1 - (CY + H / 2), 4]} />
            <meshBasicMaterial color="#6d655a" />
          </mesh>
        ))}
        <group ref={ball} visible={false}>
          <mesh>
            <sphereGeometry args={[0.035, 16, 12]} />
            <meshBasicMaterial color="#fff2c8" toneMapped={false} />
          </mesh>
          <sprite scale={[0.3, 0.3, 1]}>
            <spriteMaterial map={glow} transparent depthWrite={false} blending={AdditiveBlending} toneMapped={false} />
          </sprite>
        </group>
      </group>

      {/* console */}
      <group position={[0, 0, consoleZ]}>
        <mesh position={[0, 0.47, 0]} castShadow receiveShadow>
          <boxGeometry args={[1.7, 0.9, 0.55]} />
          <meshStandardMaterial {...oak} color="#f0dcb4" roughness={0.7} />
        </mesh>
        <mesh position={[0, 0.95, 0]} castShadow>
          <boxGeometry args={[1.74, 0.08, 0.59]} />
          <meshStandardMaterial {...oak} color={MUSTARD} roughness={0.6} />
        </mesh>
        <mesh position={[0, 0.015, 0]}>
          <boxGeometry args={[1.72, 0.03, 0.57]} />
          <meshBasicMaterial color="#ffe6c4" toneMapped={false} />
        </mesh>
        <mesh position={[0, 0.992, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <planeGeometry args={[1.6, 0.45]} />
          <meshBasicMaterial map={face.tex} toneMapped={false} />
        </mesh>

        {/* lever (animated: no shadow) */}
        <mesh position={[-0.5, 1.02, 0]}>
          <boxGeometry args={[0.12, 0.05, 0.3]} />
          <meshStandardMaterial color="#2a2621" metalness={0.7} roughness={0.35} />
        </mesh>
        <group ref={arm} position={[-0.5, 1.04, 0]}>
          <mesh position={[0, 0.19, 0]}>
            <cylinderGeometry args={[0.016, 0.02, 0.38, 10]} />
            <meshStandardMaterial color="#b08a4a" metalness={0.85} roughness={0.3} />
          </mesh>
          <mesh ref={knob} position={[0, 0.39, 0]}>
            <sphereGeometry args={[0.055, 20, 14]} />
            <meshBasicMaterial color={KNOB} toneMapped={false} />
          </mesh>
        </group>
        <mesh ref={leverHit} position={[-0.5, 1.22, 0.05]}>
          <boxGeometry args={[0.32, 0.55, 0.5]} />
          <meshBasicMaterial visible={false} />
        </mesh>

        {/* temperature dial */}
        <group ref={dial} position={[0.45, 1.0, 0]}>
          <mesh position={[0, 0.03, 0]}>
            <cylinderGeometry args={[0.075, 0.085, 0.06, 32]} />
            <meshStandardMaterial color="#2a2621" metalness={0.75} roughness={0.3} />
          </mesh>
          <mesh position={[0, 0.062, -0.045]}>
            <boxGeometry args={[0.014, 0.008, 0.06]} />
            <meshBasicMaterial color={MUSTARD} toneMapped={false} />
          </mesh>
        </group>
        <mesh ref={dialRing} position={[0.45, 0.996, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.092, 0.104, 40]} />
          <meshBasicMaterial color={CYAN} transparent opacity={0.35} toneMapped={false} blending={AdditiveBlending} depthWrite={false} />
        </mesh>
        <mesh ref={dialHit} position={[0.45, 1.06, 0]}>
          <boxGeometry args={[0.4, 0.2, 0.4]} />
          <meshBasicMaterial visible={false} />
        </mesh>
      </group>
    </group>
  )
}

