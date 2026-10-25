import { Suspense, useEffect } from 'react'
import { Link, Navigate, useNavigate, useParams } from 'react-router-dom'
import { motion } from 'motion/react'
import { getAdjacent, getModule, moduleLabel } from '@/modules/registry'
import { MODULE_CONTENT } from '@/modules/content'
import { useProgress } from '@/progress/ProgressProvider'
import { fadeRise } from '@/theme/tokens'
import type { ModuleMeta } from '@/types'
import { useInViewOnce } from '@/hooks/useInViewOnce'
import { HALL_SLUGS } from '@/three/museum/hallSlugs'
import { ArrowLeftIcon, ArrowRightIcon, SparklesIcon } from './icons'

function LessonLoading() {
  return (
    <div className="space-y-4" aria-label="Loading lesson" aria-busy="true">
      <div className="h-4 w-2/3 animate-pulse bg-surface-2" />
      <div className="h-4 w-full animate-pulse bg-surface-2" />
      <div className="h-4 w-5/6 animate-pulse bg-surface-2" />
      <div className="h-40 w-full animate-pulse bg-surface-2" />
    </div>
  )
}

function Placeholder() {
  return (
    <div className="placard-frame px-6 py-14 text-center">
      <p className="exhibit-label">Gallery closed for installation</p>
      <p className="mt-2 font-display text-2xl text-fg">This exhibit is being assembled.</p>
      <p className="mt-1 text-sm text-fg-muted">The interactive content for this module is coming together.</p>
    </div>
  )
}

function NavCard({ meta, dir }: { meta: ModuleMeta; dir: 'prev' | 'next' }) {
  const isNext = dir === 'next'
  return (
    <Link
      to={`/module/${meta.slug}`}
      className={[
        'group flex flex-1 items-center gap-4 border border-border bg-surface/70 p-4 transition-colors hover:border-brass hover:bg-surface',
        isNext ? 'flex-row-reverse text-right' : '',
      ].join(' ')}
    >
      <span className="shrink-0 text-brass transition-transform group-hover:translate-x-0.5">
        {isNext ? <ArrowRightIcon size={18} /> : <ArrowLeftIcon size={18} />}
      </span>
      <span className="min-w-0">
        <span className="exhibit-label block !text-[10px]">
          {isNext ? 'Next' : 'Previous'} · Exhibit {moduleLabel(meta.index)}
        </span>
        <span className="mt-0.5 block truncate font-display text-lg text-fg">{meta.title}</span>
      </span>
    </Link>
  )
}

function ModuleNav({ prev, next }: { prev?: ModuleMeta; next?: ModuleMeta }) {
  return (
    <nav aria-label="Exhibit navigation" className="mt-10 flex flex-col gap-3 border-t border-border pt-8 sm:flex-row">
      {prev ? <NavCard meta={prev} dir="prev" /> : <div className="flex-1" />}
      {next ? <NavCard meta={next} dir="next" /> : <div className="flex-1" />}
    </nav>
  )
}

/** "This way to the next gallery" sign at the end of the room. */
function ContinueCTA({ next }: { next?: ModuleMeta }) {
  const navigate = useNavigate()
  if (!next) return null
  return (
    <div className="mt-20 flex flex-col items-center gap-3 text-center">
      <div className="ornament w-40" aria-hidden="true">
        <span />
      </div>
      <span className="exhibit-label mt-2">This way to Exhibit {moduleLabel(next.index)}</span>
      <p className="max-w-md font-display text-3xl leading-tight text-fg">{next.title}</p>
      <button onClick={() => navigate(`/module/${next.slug}`)} className="btn-museum group mt-3">
        Continue the tour
        <ArrowRightIcon size={16} className="transition-transform group-hover:translate-x-0.5" />
      </button>
    </div>
  )
}

/**
 * Keyed per-slug by the parent, so it remounts on every navigation — which
 * re-arms useInViewOnce and marks *this* exhibit explored only when its end
 * actually scrolls into view (avoids marking the next module done on arrival).
 */
function EndSentinel({ onReach }: { onReach: () => void }) {
  const [ref, reached] = useInViewOnce<HTMLDivElement>()
  useEffect(() => {
    if (reached) onReach()
  }, [reached, onReach])
  return <div ref={ref} aria-hidden="true" className="h-px" />
}

export function ModulePage() {
  const { slug } = useParams()
  const meta = getModule(slug)
  const { markInProgress, markDone } = useProgress()

  useEffect(() => {
    if (meta) {
      markInProgress(meta.slug)
      document.title = `${meta.title} · LLM Explainer`
    }
  }, [meta, markInProgress])

  if (!meta) return <Navigate to="/" replace />

  const Content = MODULE_CONTENT[meta.slug]
  const { prev, next } = getAdjacent(meta.slug)

  return (
    <article className="mx-auto max-w-[46rem] px-5 py-10 sm:px-8 sm:py-16">
      {/* The room's entrance placard */}
      <motion.header initial="hidden" animate="visible" variants={fadeRise} className="relative mb-14 text-center">
        <span
          aria-hidden="true"
          className="pointer-events-none absolute inset-x-0 -top-10 font-display text-[10rem] leading-none font-semibold text-brass opacity-[0.11] select-none sm:-top-14 sm:text-[14rem]"
        >
          {moduleLabel(meta.index)}
        </span>
        <div className="relative flex items-center justify-center gap-3">
          <span className="exhibit-label">Exhibit {moduleLabel(meta.index)}</span>
          <span className="h-1 w-1 rotate-45 bg-gold" aria-hidden="true" />
          <span className="text-[11px] font-semibold tracking-[0.2em] text-fg-subtle uppercase">
            {meta.minutes} min visit
          </span>
        </div>
        <h1 className="relative mt-5 font-display text-[2.75rem] leading-[1.04] text-fg sm:text-6xl">{meta.title}</h1>
        <div className="ornament relative mx-auto mt-6 w-48" aria-hidden="true">
          <span />
        </div>
        <p className="relative mx-auto mt-5 max-w-xl font-display text-xl leading-snug text-fg-muted italic sm:text-[1.4rem]">
          {meta.subtitle}
        </p>
        {HALL_SLUGS.has(meta.slug) && (
          <Link to={`/museum/${meta.slug}`} className="btn-outline relative mt-7">
            <SparklesIcon size={15} className="text-accent" />
            Enter this gallery in 3D
            <span className="tag text-accent">New</span>
          </Link>
        )}
      </motion.header>

      {Content ? (
        <Suspense fallback={<LessonLoading />}>
          <div className="drop-cap space-y-8">
            <Content />
          </div>
        </Suspense>
      ) : (
        <Placeholder />
      )}

      <EndSentinel key={meta.slug} onReach={() => markDone(meta.slug)} />

      <ContinueCTA next={next} />
      <ModuleNav prev={prev} next={next} />
    </article>
  )
}
