import { useRef, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { Html, Line, OrbitControls } from '@react-three/drei'
import {
  EMBEDDING_WORDS,
  ANALOGIES,
  GROUP_COLORS,
  GROUP_Z,
  type WordPoint,
} from '@/data/embeddings'
import { Button } from '@/primitives/Button'
import { useOnScreenFrameloop } from '@/hooks/useOnScreenFrameloop'

const SCALE = 4.2

type Vec3 = [number, number, number]

const pos = (w: WordPoint): Vec3 => [
  (w.x - 0.5) * SCALE,
  (0.5 - w.y) * SCALE,
  (GROUP_Z[w.group] ?? 0) * SCALE,
]

function WordNode({ w, dim }: { w: WordPoint; dim: boolean }) {
  const color = GROUP_COLORS[w.group] ?? '#8b7bff'
  return (
    <group position={pos(w)}>
      <mesh>
        <sphereGeometry args={[0.07, 16, 16]} />
        <meshBasicMaterial color={color} transparent opacity={dim ? 0.18 : 1} />
      </mesh>
      <Html position={[0, 0.16, 0]} center distanceFactor={9} style={{ pointerEvents: 'none' }}>
        <span
          style={{
            color: dim ? '#586074' : '#e8ecf3',
            fontFamily: 'var(--font-mono)',
            fontSize: 13,
            whiteSpace: 'nowrap',
            textShadow: '0 1px 4px #000, 0 0 2px #000',
          }}
        >
          {w.word}
        </span>
      </Html>
    </group>
  )
}

export function EmbeddingSpace3D() {
  const [mode, setMode] = useState<'explore' | number>('explore')
  // Stop the auto-rotating render loop while the demo is off-screen or the tab is hidden.
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const frameloop = useOnScreenFrameloop(canvasRef)
  const analogy = typeof mode === 'number' ? ANALOGIES[mode] : null

  const highlight = new Set<string>()
  if (analogy) [analogy.a, analogy.b, analogy.c, analogy.d].forEach((w) => highlight.add(w))
  const byWord = Object.fromEntries(EMBEDDING_WORDS.map((w) => [w.word, w] as const))

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        <Button variant={mode === 'explore' ? 'secondary' : 'ghost'} className="!px-2.5 !py-1 text-xs" onClick={() => setMode('explore')}>
          Free orbit
        </Button>
        {ANALOGIES.map((a, i) => (
          <Button key={i} variant={mode === i ? 'secondary' : 'ghost'} className="!px-2.5 !py-1 font-mono text-xs" onClick={() => setMode(i)}>
            {a.a} − {a.b} + {a.c}
          </Button>
        ))}
      </div>

      <div className="h-80 overflow-hidden rounded-lg border border-border" style={{ background: '#070a0f' }}>
        <Canvas
          ref={canvasRef}
          frameloop={frameloop}
          camera={{ position: [0, 0.6, 7], fov: 50 }}
          dpr={[1, 1.5]}
          gl={{ alpha: false, antialias: true, preserveDrawingBuffer: import.meta.env.DEV }}
        >
          <color attach="background" args={['#070a0f']} />
          <OrbitControls enablePan={false} enableZoom={false} autoRotate autoRotateSpeed={0.55} />
          {EMBEDDING_WORDS.map((w) => (
            <WordNode key={w.word} w={w} dim={highlight.size > 0 && !highlight.has(w.word)} />
          ))}
          {analogy && (
            <>
              <Line points={[pos(byWord[analogy.b]), pos(byWord[analogy.a])]} color="#8b7bff" lineWidth={3} />
              <Line
                points={[pos(byWord[analogy.c]), pos(byWord[analogy.d])]}
                color="#2bd3e0"
                lineWidth={3}
                dashed
                dashSize={0.22}
                gapSize={0.12}
              />
            </>
          )}
        </Canvas>
      </div>

      {analogy ? (
        <p className="text-sm leading-relaxed text-fg">
          <span className="font-mono">
            {analogy.a} − {analogy.b} + {analogy.c}
          </span>{' '}
          ≈{' '}
          <span className="font-mono font-semibold" style={{ color: 'var(--accent)' }}>
            {analogy.d}
          </span>{' '}
          — and the two arrows stay parallel even as you orbit in 3-D.
        </p>
      ) : (
        <p className="text-xs text-fg-subtle">
          Drag to orbit. Related words cluster together, and each cluster floats at its own depth —
          a small window into what is really a many-hundred-dimensional space.
        </p>
      )}
    </div>
  )
}
