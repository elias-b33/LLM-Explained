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
      <div className="h-4 w-2/3 animate-pulse rounded bg-surface-2" />
      <div className="h-4 w-full animate-pulse rounded bg-surface-2" />
      <div className="h-4 w-5/6 animate-pulse rounded bg-surface-2" />
      <div className="h-40 w-full animate-pulse rounded-xl bg-surface-2" />
    </div>
  )
}

function Placeholder() {
  return (
    <div className="rounded-2xl border border-dashed border-border-strong bg-surface/50 px-6 py-14 text-center">
      <span
        className="mx-auto mb-4 flex h-11 w-11 items-center justify-center rounded-xl text-white"
        style={{ backgroundImage: 'linear-gradient(135deg, var(--accent), var(--accent-2))' }}
      >
        <SparklesIcon />
      </span>
      <p className="text-fg">This lesson is being assembled.</p>
      <p className="mt-1 text-sm text-fg-muted">
        The interactive content for this module is coming together.
      </p>
    </div>
  )
}

function NavCard({ meta, dir }: { meta: ModuleMeta; dir: 'prev' | 'next' }) {
  const isNext = dir === 'next'
  return (
    <Link
      to={`/module/${meta.slug}`}
      className={[
        'group flex flex-1 items-center gap-3 rounded-2xl border border-border bg-surface/60 p-4 transition-colors hover:border-border-strong hover:bg-surface',
        isNext ? 'flex-row-reverse text-right' : '',
      ].join(' ')}
    >
      <span className="shrink-0 text-fg-subtle transition-colors group-hover:text-accent">
        {isNext ? <ArrowRightIcon /> : <ArrowLeftIcon />}
      </span>
      <span className="min-w-0">
        <span className="block text-xs text-fg-subtle">{isNext ? 'Next' : 'Previous'}</span>
        <span className="block truncate text-sm font-medium text-fg">{meta.title}</span>
      </span>
    </Link>
  )
}

function ModuleNav({ prev, next }: { prev?: ModuleMeta; next?: ModuleMeta }) {
  return (
    <div className="mt-10 flex flex-col gap-3 border-t border-border pt-8 sm:flex-row">
      {prev ? <NavCard meta={prev} dir="prev" /> : <div className="flex-1" />}
      {next ? <NavCard meta={next} dir="next" /> : <div className="flex-1" />}
    </div>
  )
}

/** Prominent "go to the next exhibit" button. */
function ContinueCTA({ next }: { next?: ModuleMeta }) {
  const navigate = useNavigate()
  if (!next) return null
  return (
    <div className="mt-16 flex flex-col items-center gap-2.5">
      <span className="font-mono text-[11px] tracking-wider text-fg-subtle uppercase">
        Next exhibit
      </span>
      <button
        onClick={() => navigate(`/module/${next.slug}`)}
        className="group inline-flex items-center gap-2 rounded-xl px-6 py-3 font-medium text-white shadow-lg transition-transform hover:-translate-y-0.5"
        style={{
          backgroundImage: 'linear-gradient(120deg, var(--accent), var(--accent-2))',
          boxShadow: '0 12px 34px -12px var(--accent)',
        }}
      >
        Continue to {next.title}
        <ArrowRightIcon size={18} className="transition-transform group-hover:translate-x-0.5" />
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
    <article className="mx-auto max-w-[46rem] px-5 py-10 sm:px-8 sm:py-14">
      <motion.header initial="hidden" animate="visible" variants={fadeRise} className="relative mb-12 pt-6">
        <span
          aria-hidden="true"
          className="pointer-events-none absolute -top-6 -right-2 font-mono text-[9rem] leading-none font-bold tracking-tighter select-none sm:-right-6 sm:text-[12rem]"
          style={{
            backgroundImage:
              'linear-gradient(160deg, color-mix(in srgb, var(--accent) 26%, transparent), color-mix(in srgb, var(--accent-2) 6%, transparent) 70%)',
            WebkitBackgroundClip: 'text',
            backgroundClip: 'text',
            color: 'transparent',
          }}
        >
          {moduleLabel(meta.index)}
        </span>
        <div className="relative mb-4 flex items-center gap-3 font-mono text-xs text-fg-subtle">
          <span
            className="rounded-full border px-2.5 py-1 tracking-[0.18em]"
            style={{
              color: 'var(--accent)',
              borderColor: 'color-mix(in srgb, var(--accent) 40%, transparent)',
              backgroundColor: 'color-mix(in srgb, var(--accent) 10%, transparent)',
              boxShadow: '0 0 20px -6px var(--accent)',
            }}
          >
            EXHIBIT {moduleLabel(meta.index)}
          </span>
          <span className="h-1 w-1 rounded-full bg-current opacity-50" />
          <span>{meta.minutes} min read</span>
        </div>
        <h1 className="relative text-4xl font-semibold tracking-tight text-fg sm:text-5xl sm:leading-[1.05]">
          {meta.title}
        </h1>
        <span
          aria-hidden="true"
          className="relative mt-5 block h-px w-28"
          style={{ backgroundImage: 'linear-gradient(90deg, var(--accent), var(--accent-2), transparent)' }}
        />
        <p className="mt-4 text-lg leading-relaxed text-fg-muted">{meta.subtitle}</p>
        {HALL_SLUGS.has(meta.slug) && (
          <Link
            to={`/museum/${meta.slug}`}
            className="mt-5 inline-flex items-center gap-2 rounded-xl border border-border bg-surface/60 px-4 py-2.5 text-sm font-medium text-fg transition-colors hover:border-border-strong hover:bg-surface"
          >
            <SparklesIcon size={15} />
            Explore this exhibit in 3D
            <span className="rounded-full border border-border-strong px-1.5 py-0.5 font-mono text-[10px] tracking-wide text-fg-subtle">
              NEW
            </span>
          </Link>
        )}
      </motion.header>

      {Content ? (
        <Suspense fallback={<LessonLoading />}>
          <div className="space-y-8">
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
