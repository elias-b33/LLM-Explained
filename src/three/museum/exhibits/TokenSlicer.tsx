import { useEffect, useMemo, useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import { AdditiveBlending, type Group, type Mesh, type MeshBasicMaterial } from 'three'
import { tile, type MuseumTextures } from '../materials'
import { useInteractable, markExplored } from '../interaction'
import { sfx } from '../audio'
import { CYAN, MONO, MUSTARD, SANS, VIOLET, makeBoard, measure, roundRect, staticBoard } from '../board'

/** Real o200k_base (GPT-4o) splits + IDs, precomputed with gpt-tokenizer so the
 *  2 MB encoder never loads in the museum. */
const SENTENCES: [string, number][][] = [
  [['The', 976], [' capital', 9029], [' of', 328], [' France', 10128], [' is', 382], [' Paris', 12650], ['.', 13]],
  [['Str', 3504], ['aw', 1134], ['berry', 19772], [' has', 853], [' three', 3407], [' r', 428], ["'s", 885], [' in', 306], [' it', 480], ['.', 13]],
  [['Token', 4421], ['ization', 2860], [" isn't", 12471], [' magic', 19745], [',', 11], [" it's", 4275], [' chunk', 23958], ['ing', 289], ['!', 0]],
  [['In', 637], [' ', 220], ['202', 1323], ['5', 20], [',', 11], [' GPT', 174803], ['-', 12], ['4', 19], ['o', 78], [' read', 1729], [' ', 220], ['123', 7633], ['456', 19354], ['7', 22], [' tokens', 20290], ['.', 13]],
]

const PX = 400 // canvas px per metre
const FONT = `500 46px ${SANS}`
const Y = 1.62
const ROW_MAX = 2.7
const COLORS = [VIOLET, CYAN, MUSTARD, '#ff8fb1']

interface Block {
  tok: string
  id: number
  w: number // metres
  start: [number, number]
  target: [number, number]
  tex: ReturnType<typeof staticBoard>
}

function layout(sentence: [string, number][]) {
  const text = sentence.map((s) => s[0]).join('')
  const textW = measure(text, FONT) / PX
  const blocks: Block[] = []
  let cum = 0
  for (const [i, [tok, id]] of sentence.entries()) {
    const tw = measure(tok, FONT) / PX
    const shown = tok.startsWith(' ') ? '␣' + tok.slice(1) : tok
    const sw = measure(shown, FONT)
    const cw = Math.max(sw, measure(String(id), `500 24px ${MONO}`)) + 30
    const col = COLORS[i % COLORS.length]
    const tex = staticBoard(cw, 112, (ctx) => {
      ctx.fillStyle = col + '38'
      roundRect(ctx, 3, 3, cw - 6, 106, 10)
      ctx.fill()
      ctx.strokeStyle = col
      ctx.lineWidth = 4
      ctx.stroke()
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.font = FONT
      if (shown.startsWith('␣')) {
        const lw = measure('␣', FONT)
        const x0 = cw / 2 - sw / 2
        ctx.textAlign = 'left'
        ctx.fillStyle = 'rgba(255,255,255,0.35)'
        ctx.fillText('␣', x0, 44)
        ctx.fillStyle = '#ffffff'
        ctx.fillText(shown.slice(1), x0 + lw, 44)
        ctx.textAlign = 'center'
      } else {
        ctx.fillStyle = '#ffffff'
        ctx.fillText(shown, cw / 2, 44)
      }
      ctx.font = `500 24px ${MONO}`
      ctx.fillStyle = col
      ctx.fillText(String(id), cw / 2, 88)
    })
    blocks.push({ tok, id, w: cw / PX, start: [-textW / 2 + cum + tw / 2, Y], target: [0, 0], tex })
    cum += tw
  }
  // spread into rows no wider than ROW_MAX, centred
  const GAP = 0.05
  const rows: Block[][] = [[]]
  let rw = 0
  for (const b of blocks) {
    if (rw + b.w > ROW_MAX && rows[rows.length - 1].length) {
      rows.push([])
      rw = 0
    }
    rows[rows.length - 1].push(b)
    rw += b.w + GAP
  }
  rows.forEach((row, r) => {
    const total = row.reduce((a, b) => a + b.w, 0) + GAP * (row.length - 1)
    let x = -total / 2
    const y = Y + 0.2 + (rows.length - 1 - r) * 0.4 - (rows.length - 1) * 0.14
    row.forEach((b, k) => {
      b.target = [x + b.w / 2, y + (k % 2 ? 0.03 : -0.03)]
      x += b.w + GAP
    })
  })
  const strip = staticBoard(textW * PX + 60, 90, (ctx) => {
    ctx.fillStyle = 'rgba(139,123,255,0.16)'
    roundRect(ctx, 2, 2, textW * PX + 56, 86, 14)
    ctx.fill()
    ctx.strokeStyle = 'rgba(255,233,190,0.55)'
    ctx.lineWidth = 3
    ctx.stroke()
    ctx.font = FONT
    ctx.fillStyle = '#fff6e6'
    ctx.textBaseline = 'middle'
    ctx.fillText(text, 30, 47)
  })
  return { text, textW, blocks, strip }
}

const easeOutBack = (k: number) => 1 + 2.2 * (k - 1) ** 3 + 1.2 * (k - 1) ** 2

/**
 * Token slicer: a lectern with a glowing sentence floating above it. E splits
 * the sentence into its real GPT-4o tokens (leading spaces shown as ␣) with
 * their IDs; E again merges it back and slices the next sentence.
 */
export function TokenSlicer({ tex, position }: { tex: MuseumTextures; position: [number, number, number] }) {
  const [idx, setIdx] = useState(0)
  const L = useMemo(() => layout(SENTENCES[idx]), [idx])
  const lay = useRef(L)
  lay.current = L
  useEffect(
    () => () => {
      L.strip.dispose()
      L.blocks.forEach((b) => b.tex.dispose())
    },
    [L],
  )
  const phase = useRef<{ p: 'whole' | 'split' | 'merge'; t0: number; auto: number | null }>({ p: 'whole', t0: 0, auto: null })
  const clock = useRef(0)
  const hl = useRef(false)
  const [split, setSplit] = useState(false) // only drives the reader text

  const blockRefs = useRef<(Mesh | null)[]>([])
  const strip = useRef<Mesh>(null)
  const ring = useRef<Mesh>(null)
  const blocksGroup = useRef<Group>(null)
  const hit = useRef<Mesh>(null)

  const oak = useMemo(() => tile(tex.oak, 1, 1), [tex])
  const reader = useMemo(() => makeBoard(720, 220), [])
  useEffect(() => {
    const s = SENTENCES[idx]
    const chars = s.map((t) => t[0]).join('').length
    reader.draw((ctx) => {
      ctx.fillStyle = '#0b0f1a'
      ctx.fillRect(0, 0, 720, 220)
      ctx.fillStyle = CYAN
      ctx.font = `600 26px ${MONO}`
      ctx.fillText('TOKENIZER', 34, 52)
      ctx.fillStyle = 'rgba(233,236,243,0.45)'
      ctx.font = `400 20px ${MONO}`
      ctx.fillText('GPT-4o · o200k vocabulary', 220, 52)
      ctx.font = `600 44px ${SANS}`
      ctx.fillStyle = '#f2f4f8'
      ctx.fillText(split ? `${chars} characters → ${s.length} tokens` : 'Models never see letters.', 34, 124)
      ctx.font = `400 24px ${SANS}`
      ctx.fillStyle = MUSTARD
      ctx.fillText(split ? 'Each piece is just an ID number.' : 'They see these pieces.', 34, 176)
    })
  }, [idx, split, reader])

  const begin = (p: 'split' | 'merge') => {
    phase.current = { p, t0: clock.current, auto: null }
    if (p === 'split') {
      setSplit(true)
      sfx('slice')
    } else sfx('whoosh', { v: 0.7 })
  }

  useInteractable({
    targets: [hit],
    reach: 2.9,
    verb: () => (phase.current.p === 'whole' ? 'slice into tokens' : 'next sentence'),
    onInteract: () => {
      markExplored('tokens')
      const ph = phase.current
      if (ph.p === 'whole') begin('split')
      else if (ph.p === 'split' && clock.current - ph.t0 > 0.4) begin('merge')
    },
    setHighlight: (h) => void (hl.current = !!h),
  })

  useFrame((state) => {
    const t = (clock.current = state.clock.elapsedTime)
    const ph = phase.current
    const dt = t - ph.t0
    const { blocks } = lay.current
    const stripMat = strip.current?.material as MeshBasicMaterial | undefined
    let stripO = 1
    let blockO = 0
    let k = 0

    if (ph.p === 'whole') {
      if (ph.auto !== null && t >= ph.auto) begin('split')
      stripO = Math.min(1, (stripMat?.opacity ?? 0) + 0.08)
    } else if (ph.p === 'split') {
      stripO = Math.max(0, 1 - dt * 5)
      blockO = Math.min(1, dt * 6)
      k = 1
    } else {
      const m = Math.min(1, dt / 0.35)
      k = 1 - m
      blockO = 1 - m * m
      stripO = 0
      if (m >= 1) {
        phase.current = { p: 'whole', t0: t, auto: t + 0.55 }
        setSplit(false)
        setIdx((i) => (i + 1) % SENTENCES.length)
      }
    }

    if (stripMat) {
      stripMat.opacity = stripO
      strip.current!.visible = stripO > 0.01
      const b = hl.current && ph.p === 'whole' ? 1.35 : 1
      stripMat.color.setScalar(b)
    }
    if (blocksGroup.current) blocksGroup.current.visible = blockO > 0.01
    if (blockO > 0.01) {
      blocks.forEach((b, i) => {
        const m = blockRefs.current[i]
        if (!m) return
        let e = k
        if (ph.p === 'split') e = easeOutBack(Math.max(0, Math.min(1, (dt - i * 0.035) / 0.55)))
        const bob = ph.p === 'split' ? Math.sin(t * 1.3 + i * 0.8) * 0.012 * Math.min(1, dt) : 0
        m.position.x = b.start[0] + (b.target[0] - b.start[0]) * e
        m.position.y = b.start[1] + (b.target[1] - b.start[1]) * e + bob
        const s = 0.42 + 0.58 * Math.min(1, e)
        m.scale.set(1, s, 1)
        const mat = m.material as MeshBasicMaterial
        mat.opacity = blockO
        mat.color.setScalar(hl.current ? 1.25 : 1)
      })
    }
    if (ring.current) {
      const mat = ring.current.material as MeshBasicMaterial
      mat.opacity = (hl.current ? 0.85 : 0.4) + Math.sin(t * 2) * 0.08
    }
  })

  const { blocks, textW } = L
  return (
    <group position={position} rotation={[0, Math.PI / 2, 0]}>
      {/* lectern */}
      <mesh position={[0, 0.47, 0]} castShadow receiveShadow>
        <boxGeometry args={[0.9, 0.9, 0.5]} />
        <meshStandardMaterial {...oak} color="#f0dcb4" roughness={0.7} />
      </mesh>
      <mesh position={[0, 0.93, 0]} castShadow>
        <boxGeometry args={[0.94, 0.08, 0.54]} />
        <meshStandardMaterial {...oak} color={MUSTARD} roughness={0.6} />
      </mesh>
      <mesh position={[0, 0.015, 0]}>
        <boxGeometry args={[0.92, 0.03, 0.52]} />
        <meshBasicMaterial color="#ffe6c4" toneMapped={false} />
      </mesh>
      {/* slanted reader screen */}
      <group position={[0, 1.04, 0.02]} rotation={[-0.75, 0, 0]}>
        <mesh position={[0, 0, -0.012]}>
          <boxGeometry args={[0.86, 0.29, 0.02]} />
          <meshStandardMaterial color="#1c1a17" metalness={0.5} roughness={0.45} />
        </mesh>
        <mesh>
          <planeGeometry args={[0.82, 0.25]} />
          <meshBasicMaterial map={reader.tex} toneMapped={false} />
        </mesh>
      </group>
      {/* projector ring under the hologram */}
      <mesh ref={ring} position={[0, 1.2, -0.05]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.2, 0.235, 48]} />
        <meshBasicMaterial color={CYAN} transparent opacity={0.4} toneMapped={false} blending={AdditiveBlending} depthWrite={false} />
      </mesh>

      {/* the whole sentence */}
      <mesh ref={strip} position={[0, Y, 0]}>
        <planeGeometry args={[textW + 60 / PX, 90 / PX]} />
        <meshBasicMaterial map={L.strip} transparent toneMapped={false} depthWrite={false} />
      </mesh>
      {/* the token blocks */}
      <group ref={blocksGroup} visible={false}>
        {blocks.map((b, i) => (
          <mesh key={`${idx}-${i}`} ref={(m) => void (blockRefs.current[i] = m)} position={[b.start[0], b.start[1], 0]}>
            <planeGeometry args={[b.w, 112 / PX]} />
            <meshBasicMaterial map={b.tex} transparent toneMapped={false} depthWrite={false} opacity={0} />
          </mesh>
        ))}
      </group>

      {/* hit volume: lectern + hologram */}
      <mesh ref={hit} position={[0, 1.15, 0]}>
        <boxGeometry args={[2.9, 1.5, 0.7]} />
        <meshBasicMaterial visible={false} />
      </mesh>
    </group>
  )
}
