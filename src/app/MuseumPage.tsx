import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react'
import { Navigate, useNavigate, useParams } from 'react-router-dom'
import { motion } from 'motion/react'
import type { PointerLockControls as PointerLockControlsImpl } from 'three-stdlib'
import { getAdjacent, getModule, moduleLabel } from '@/modules/registry'
import { usePrefersReducedMotion } from '@/hooks/usePrefersReducedMotion'
import { useProgress } from '@/progress/ProgressProvider'
import { HALLS, type FocusTarget } from '@/three/museum/halls'
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
          <p className="font-mono text-xs tracking-wider text-fg-subtle">
            EXHIBIT {moduleLabel(meta.index)}
          </p>
          <h1 className="mt-2 text-2xl font-semibold text-fg">{meta.title}</h1>
          <p className="mt-3 leading-relaxed text-fg-muted">
            The walkable 3D exhibit moves continuously. Since your system prefers reduced motion, the
            calm reading version may be more comfortable.
          </p>
          <div className="mt-6 flex flex-wrap justify-center gap-3">
            <button
              onClick={() => navigate(`/module/${meta.slug}`)}
              className="rounded-xl px-5 py-3 font-medium text-white"
              style={{ backgroundImage: 'linear-gradient(120deg, var(--accent), var(--accent-2))' }}
            >
              Open the reading version
            </button>
            <button
              onClick={() => setForce(true)}
              className="rounded-xl border border-border px-5 py-3 font-medium text-fg-muted transition-colors hover:text-fg"
            >
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
          <div className="grid h-full place-items-center text-sm text-[#e9dcc4]/60">
            Entering the exhibit…
          </div>
        }
      >
        <HallScene
          hall={hall}
          controlsRef={controlsRef}
          playing={!focus}
          focusOpen={!!focus}
          onLock={() => {
            setLocked(true)
            setHasEntered(true)
          }}
          onUnlock={() => setLocked(false)}
          onOpenStation={setFocus}
          onTravel={handleTravel}
          nextLabel={next?.title}
          prevLabel={prev?.title}
        />
      </Suspense>
      <MuseumHUD
        locked={locked}
        hasEntered={hasEntered}
        focusOpen={!!focus}
        title={meta.title}
        exhibitLabel={moduleLabel(meta.index)}
        onEnter={() => controlsRef.current?.lock()}
        onExit={() => navigate(`/module/${meta.slug}`)}
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
