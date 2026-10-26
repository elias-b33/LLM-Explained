import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type CameraControlsImpl from 'camera-controls'
import { galaxyData } from '@/three/galaxy/data'
import { usePrefersReducedMotion } from '@/hooks/usePrefersReducedMotion'
import { GalaxyOverlay } from '@/three/galaxy/GalaxyOverlay'

const GalaxyScene = lazy(() => import('@/three/galaxy/GalaxyScene'))

/** Rendered by the /galaxy route once its lazy loader has fetched the word data (router.tsx). */
export function GalaxyPage() {
  const data = galaxyData()
  const navigate = useNavigate()
  const reducedMotion = usePrefersReducedMotion()
  const controls = useRef<CameraControlsImpl | null>(null)
  const tipRef = useRef<HTMLDivElement>(null)
  const mouse = useRef({ x: 0, y: 0 })
  // Place the tooltip at the last pointer position as it mounts; pointermove moves it after.
  const attachTip = useCallback((el: HTMLDivElement | null) => {
    tipRef.current = el
    if (el) el.style.transform = `translate(${mouse.current.x + 16}px, ${mouse.current.y + 18}px)`
  }, [])

  const [selected, setSelected] = useState<number | null>(null)
  const [hovered, setHovered] = useState<number | null>(null)
  const [activeUniverse, setActiveUniverse] = useState<number | null>(null)

  const indexByName = useMemo(() => {
    const m: Record<string, number> = {}
    data.words.forEach((w, i) => {
      m[w.w] = i
    })
    return m
  }, [data])

  const forced = useMemo(() => {
    const s = new Set<number>()
    if (selected != null) {
      s.add(selected)
      data.words[selected].nn.forEach((n) => s.add(n))
    }
    if (hovered != null) s.add(hovered)
    return Array.from(s)
  }, [data, selected, hovered])

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
          <div className="flex h-full flex-col items-center justify-center gap-2 text-center">
            <p className="text-[10px] font-semibold tracking-[0.24em] text-[#d4b062] uppercase">The Planetarium</p>
            <p className="font-display text-2xl text-[#f1e9da]/80 italic">Dimming the house lights…</p>
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
          ref={attachTip}
          dir="auto"
          className="pointer-events-none fixed top-0 left-0 z-50 flex items-center gap-1.5 rounded-sm border border-[#b8924f]/45 bg-[#1a140e]/85 px-2.5 py-1 text-sm text-[#f6efe0] shadow-lg"
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
