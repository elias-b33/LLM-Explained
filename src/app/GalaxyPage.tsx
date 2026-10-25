import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type CameraControlsImpl from 'camera-controls'
import galaxyRaw from '@/data/galaxy.json'
import type { GalaxyData } from '@/three/galaxy/types'
import { usePrefersReducedMotion } from '@/hooks/usePrefersReducedMotion'
import { GalaxyOverlay } from '@/three/galaxy/GalaxyOverlay'

const GalaxyScene = lazy(() => import('@/three/galaxy/GalaxyScene'))
const data = galaxyRaw as unknown as GalaxyData

export function GalaxyPage() {
  const navigate = useNavigate()
  const reducedMotion = usePrefersReducedMotion()
  const controls = useRef<CameraControlsImpl | null>(null)
  const tipRef = useRef<HTMLDivElement>(null)
  const mouse = useRef({ x: 0, y: 0 })

  const [selected, setSelected] = useState<number | null>(null)
  const [hovered, setHovered] = useState<number | null>(null)
  const [activeUniverse, setActiveUniverse] = useState<number | null>(null)

  const indexByName = useMemo(() => {
    const m: Record<string, number> = {}
    data.words.forEach((w, i) => {
      m[w.w] = i
    })
    return m
  }, [])

  const forced = useMemo(() => {
    const s = new Set<number>()
    if (selected != null) {
      s.add(selected)
      data.words[selected].nn.forEach((n) => s.add(n))
    }
    if (hovered != null) s.add(hovered)
    return Array.from(s)
  }, [selected, hovered])

  useEffect(() => {
    const prev = document.title
    document.title = 'Word Universes · LLM Explainer'
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') navigate(-1)
    }
    const onMove = (e: PointerEvent) => {
      mouse.current.x = e.clientX
      mouse.current.y = e.clientY
      if (tipRef.current) {
        tipRef.current.style.transform = `translate(${e.clientX + 16}px, ${e.clientY + 18}px)`
      }
    }
    window.addEventListener('keydown', onKey)
    window.addEventListener('pointermove', onMove)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('pointermove', onMove)
      document.title = prev
    }
  }, [navigate])

  const handleHover = (i: number | null) => setHovered((prev) => (prev === i ? prev : i))
  const handleSelect = (i: number) => {
    setSelected(i)
    setActiveUniverse(data.words[i].c) // focus the word's universe
  }
  const selectByName = (name: string) => {
    const i = indexByName[name.trim().toLowerCase()] ?? indexByName[name.trim()]
    if (i != null) {
      handleSelect(i)
      return true
    }
    return false
  }
  const selectUniverse = (i: number | null) => {
    setSelected(null)
    setHovered(null)
    setActiveUniverse(i)
  }

  return (
    <div className="fixed inset-0" style={{ background: '#070a0f' }}>
      <Suspense
        fallback={
          <div className="flex h-full items-center justify-center text-sm text-white/50">
            Entering the universes…
          </div>
        }
      >
        <GalaxyScene
          controls={controls}
          selected={selected}
          activeUniverse={activeUniverse}
          forced={forced}
          onHover={handleHover}
          onSelect={handleSelect}
          reducedMotion={reducedMotion}
        />
      </Suspense>
      <GalaxyOverlay
        data={data}
        selected={selected}
        onSelect={handleSelect}
        selectByName={selectByName}
        activeUniverse={activeUniverse}
        selectUniverse={selectUniverse}
        onExit={() => navigate(-1)}
      />

      {/* cursor tooltip: shows the exact word you'll select before you click */}
      {hovered != null && hovered !== selected && (
        <div
          ref={tipRef}
          dir="auto"
          className="pointer-events-none fixed top-0 left-0 z-50 flex items-center gap-1.5 rounded-md border border-white/15 bg-black/75 px-2.5 py-1 text-sm text-white shadow-lg backdrop-blur"
          style={{ transform: `translate(${mouse.current.x + 16}px, ${mouse.current.y + 18}px)` }}
        >
          <span
            className="h-2 w-2 shrink-0 rounded-full"
            style={{ backgroundColor: data.clusters[data.words[hovered].c]?.color }}
          />
          {data.words[hovered].w}
        </div>
      )}
    </div>
  )
}
