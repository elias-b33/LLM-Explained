import { lazy, Suspense, useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react'
import { Navigate, useNavigate, useParams } from 'react-router-dom'
import { motion } from 'motion/react'
import type { PointerLockControls as PointerLockControlsImpl } from 'three-stdlib'
import { getAdjacent, getModule, moduleLabel } from '@/modules/registry'
import { usePrefersReducedMotion } from '@/hooks/usePrefersReducedMotion'
import { useProgress } from '@/progress/ProgressProvider'
import { HALLS, type FocusTarget } from '@/three/museum/halls'
import { museumUi, resetExplored } from '@/three/museum/interaction'
import { isMuted, setMuted, startAudio, stopAudio } from '@/three/museum/audio'
import { gpuInfo, loadQuality, saveQuality, type Quality } from '@/three/museum/quality'
import { MuseumHUD } from './MuseumHUD'
import { StationFocus } from './StationFocus'

const HallScene = lazy(() => import('@/three/museum/HallScene'))

function hasWebGL(): boolean {
  try {
    const c = document.createElement('canvas')
    return !!(c.getContext('webgl2') || c.getContext('webgl'))
  } catch {
    return false
  }
}

/**
 * The walkable first-person exhibit at /museum/:slug. Owns all state that must
 * cross the Canvas boundary (pointer-lock, focused station) and renders the
 * lazy HallScene plus the DOM overlays. Falls back to the 2D reading page when
 * the exhibit has no authored hall, when WebGL is unavailable, or (optionally)
 * when the user prefers reduced motion.
 */
export function MuseumPage() {
  const { slug } = useParams()
  const navigate = useNavigate()
  const reducedMotion = usePrefersReducedMotion()
  const { markDone } = useProgress()
  const controlsRef = useRef<PointerLockControlsImpl | null>(null)
  const [locked, setLocked] = useState(false)
  const [hasEntered, setHasEntered] = useState(false)
  const [force, setForce] = useState(false)
  const [focus, setFocus] = useState<FocusTarget | null>(null)
  const [traveling, setTraveling] = useState(false)
  const webgl = useMemo(hasWebGL, [])
  const [quality, setQuality] = useState<Quality>(loadQuality)
  const [muted, setMutedState] = useState(isMuted)
  const ui = useSyncExternalStore(museumUi.subscribe, museumUi.get)
  const software = useMemo(() => webgl && gpuInfo().software, [webgl])

  // Fresh tracker per hall visit; audio lives only while the museum is open.
  useEffect(() => {
    resetExplored()
  }, [slug])
  useEffect(() => () => stopAudio(), [])

  const meta = getModule(slug)
  const hall = slug ? HALLS[slug] : undefined

  useEffect(() => {
    if (!meta) return
    const prev = document.title
    document.title = `${meta.title} · Museum · LLM Explainer`
    return () => {
      document.title = prev
    }
  }, [meta])

  // Opening a station frees the cursor so you can use the demo.
  useEffect(() => {
    if (focus) controlsRef.current?.unlock()
  }, [focus])

  if (!meta) return <Navigate to="/" replace />
  if (!hall || !webgl) return <Navigate to={`/module/${meta.slug}`} replace />

  if (reducedMotion && !force) {
    return (
      <div className="fixed inset-0 grid place-items-center bg-bg px-6 text-center">
        <div className="max-w-md">
          <p className="exhibit-label">Exhibit {moduleLabel(meta.index)}</p>
          <h1 className="font-display mt-2 text-3xl text-fg">{meta.title}</h1>
          <p className="mt-3 leading-relaxed text-fg-muted">
            The walkable 3D exhibit moves continuously. Since your system prefers reduced motion, the
            calm reading version may be more comfortable.
          </p>
          <div className="mt-6 flex flex-wrap justify-center gap-3">
            <button onClick={() => navigate(`/module/${meta.slug}`)} className="btn-museum">
              Open the reading version
            </button>
            <button onClick={() => setForce(true)} className="btn-outline">
              Enter in 3D anyway
            </button>
          </div>
        </div>
      </div>
    )
  }

  const { prev, next } = getAdjacent(meta.slug)
  const handleTravel = (dir: 'next' | 'prev') => {
    const target = dir === 'next' ? next : prev
    if (!target) return
    markDone(meta.slug)
    const go = () => navigate(HALLS[target.slug] ? `/museum/${target.slug}` : `/module/${target.slug}`)
    if (reducedMotion) {
      go()
      return
    }
    setTraveling(true)
    window.setTimeout(go, 300)
  }

  return (
    <div className="fixed inset-0" style={{ background: '#140e09' }}>
      <Suspense
        fallback={
          <div className="dark grid h-full place-items-center">
            <p className="exhibit-label">Entering the exhibit…</p>
          </div>
        }
      >
        <HallScene
          hall={hall}
          controlsRef={controlsRef}
          playing={!focus}
          focusOpen={!!focus}
          onLock={() => {
            startAudio() // a click anywhere can lock; it still carries the user gesture
            setLocked(true)
            setHasEntered(true)
          }}
          onUnlock={() => setLocked(false)}
          onOpenStation={setFocus}
          onTravel={handleTravel}
          nextLabel={next?.title}
          prevLabel={prev?.title}
          quality={quality}
        />
      </Suspense>
      <MuseumHUD
        locked={locked}
        hasEntered={hasEntered}
        focusOpen={!!focus}
        title={meta.title}
        exhibitLabel={moduleLabel(meta.index)}
        onEnter={() => {
          startAudio() // needs this user gesture
          controlsRef.current?.lock()
        }}
        onExit={() => navigate(`/module/${meta.slug}`)}
        prompt={ui.prompt}
        explored={ui.explored.length}
        tourTotal={hall.tour?.length ?? 0}
        muted={muted}
        onToggleMute={() => {
          setMuted(!muted)
          setMutedState(!muted)
        }}
        quality={quality}
        onQuality={(q) => {
          saveQuality(q)
          setQuality(q)
        }}
        softwareRendering={software}
      />
      {focus && <StationFocus target={focus} onClose={() => setFocus(null)} />}
      {traveling && (
        <motion.div
          className="fixed inset-0 z-[60] bg-black"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3 }}
        />
      )}
    </div>
  )
}
