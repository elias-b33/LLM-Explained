import { lazy, Suspense, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { motion } from 'motion/react'
import { MODULES, TOTAL_MINUTES, moduleLabel } from '@/modules/registry'
import { useProgress } from '@/progress/ProgressProvider'
import { fadeRise, staggerParent } from '@/theme/tokens'
import { ArrowRightIcon, SparklesIcon } from './icons'
import { StatusDot } from './StatusDot'

const Hero3D = lazy(() => import('@/three/Hero3D'))

export function Landing() {
  const navigate = useNavigate()
  const { getStatus, completedCount } = useProgress()

  useEffect(() => {
    document.title = 'LLM Explainer — How Language Models Actually Work'
  }, [])

  const resumeTarget = MODULES.find((m) => getStatus(m.slug) !== 'done') ?? MODULES[0]
  const started = completedCount > 0

  return (
    <div>
      {/* Hero with 3D backdrop */}
      <div className="relative flex min-h-[82vh] items-center overflow-hidden bg-aurora">
        <div className="pointer-events-none absolute inset-0">
          <Suspense fallback={null}>
            <Hero3D />
          </Suspense>
        </div>
        <div
          className="pointer-events-none absolute inset-0"
          style={{
            background:
              'radial-gradient(ellipse 55% 45% at center, rgba(5,7,12,0.7), rgba(5,7,12,0.2) 60%, transparent 85%)',
          }}
        />
        <section className="relative z-10 mx-auto w-full max-w-3xl px-5 pt-16 pb-20 text-center sm:px-8">
        <motion.div initial="hidden" animate="visible" variants={staggerParent}>
          <motion.div
            variants={fadeRise}
            className="mb-6 inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/5 px-3.5 py-1.5 text-xs font-medium text-white/70 backdrop-blur"
          >
            <SparklesIcon size={14} />
            An interactive, visual deep-dive
          </motion.div>

          <motion.h1
            variants={fadeRise}
            className="text-5xl font-semibold tracking-tight text-white sm:text-7xl sm:leading-[1.02]"
            style={{ textShadow: '0 4px 40px rgba(109,94,252,0.35)' }}
          >
            How Language Models <span className="text-shimmer">Actually Work</span>
          </motion.h1>

          <motion.p
            variants={fadeRise}
            className="mx-auto mt-6 max-w-xl text-lg leading-relaxed text-white/70"
          >
            Build a real, deep understanding of LLMs — from raw text becoming tokens, to attention,
            to training and generation. Every idea paired with something you can poke at.
          </motion.p>

          <motion.div
            variants={fadeRise}
            className="mt-9 flex flex-col items-center justify-center gap-3 sm:flex-row"
          >
            <button
              onClick={() => navigate(`/module/${resumeTarget.slug}`)}
              className="cta-glow group inline-flex items-center gap-2 rounded-xl px-6 py-3 font-medium text-white shadow-lg transition-transform hover:-translate-y-0.5"
              style={{
                backgroundImage: 'linear-gradient(120deg, var(--accent), var(--accent-2))',
                boxShadow: '0 10px 30px -10px var(--accent)',
              }}
            >
              {started ? 'Resume' : 'Begin the journey'}
              <ArrowRightIcon size={18} className="transition-transform group-hover:translate-x-0.5" />
            </button>
            <button
              onClick={() => navigate('/museum/orientation')}
              className="inline-flex items-center gap-2 rounded-xl border border-white/15 bg-white/5 px-6 py-3 font-medium text-white/90 backdrop-blur transition-colors hover:bg-white/10"
            >
              <SparklesIcon size={16} />
              Walk the museum
              <span className="rounded-full border border-white/15 px-1.5 py-0.5 text-[10px] tracking-wide text-white/60">
                3D
              </span>
            </button>
          </motion.div>

          <motion.div variants={fadeRise} className="mt-4">
            <span className="text-sm text-white/45">
              {MODULES.length} modules · ~{TOTAL_MINUTES} min
            </span>
          </motion.div>
        </motion.div>
        </section>
        <div
          className="pointer-events-none absolute inset-x-0 bottom-0 h-24"
          style={{ backgroundImage: 'linear-gradient(to bottom, transparent, var(--bg))' }}
        />
      </div>

      {/* Module list */}
      <section className="mx-auto max-w-3xl px-5 pb-24 sm:px-8">
        <div className="mb-6 flex items-end justify-between gap-4">
          <div>
            <p className="font-mono text-[11px] tracking-[0.3em] text-fg-subtle uppercase">The collection</p>
            <h2 className="mt-1 text-2xl font-semibold tracking-tight text-fg">Eleven exhibits, one idea</h2>
          </div>
          <span className="hidden text-sm text-fg-subtle sm:block">tokens in → next token out</span>
        </div>
        <motion.ol
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.05 }}
          variants={staggerParent}
          className="space-y-3"
        >
          {MODULES.map((m) => {
            const status = getStatus(m.slug)
            return (
              <motion.li key={m.id} variants={fadeRise}>
                <Link
                  to={`/module/${m.slug}`}
                  className="glow-card group flex items-center gap-4 rounded-2xl bg-surface/55 p-4 backdrop-blur-md transition-all duration-300 hover:-translate-y-0.5 hover:bg-surface/80 sm:gap-5 sm:p-5"
                >
                  <span
                    className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl font-mono text-sm font-semibold text-white tabular-nums transition-transform duration-300 group-hover:scale-105"
                    style={{
                      backgroundImage:
                        'linear-gradient(140deg, color-mix(in srgb, var(--accent) 85%, #000), color-mix(in srgb, var(--accent-2) 70%, #000))',
                      boxShadow: '0 6px 22px -8px var(--accent), inset 0 1px 0 rgba(255,255,255,0.25)',
                    }}
                  >
                    {moduleLabel(m.index)}
                  </span>
                  <div className="min-w-0 flex-1">
                    <h3 className="font-medium text-fg">{m.title}</h3>
                    <p className="mt-0.5 line-clamp-2 text-sm text-fg-muted">{m.subtitle}</p>
                  </div>
                  <StatusDot status={status} className="shrink-0" />
                  <ArrowRightIcon
                    size={18}
                    className="shrink-0 text-fg-subtle transition-all group-hover:translate-x-0.5 group-hover:text-accent"
                  />
                </Link>
              </motion.li>
            )
          })}
        </motion.ol>
      </section>
    </div>
  )
}
