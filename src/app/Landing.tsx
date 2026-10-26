import { lazy, Suspense, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { motion } from 'motion/react'
import { MODULES, TOTAL_MINUTES, moduleLabel } from '@/modules/registry'
import { useProgress } from '@/progress/ProgressProvider'
import { fadeRise, staggerParent } from '@/theme/tokens'
import { ArrowRightIcon, BookIcon, ChatIcon, PlanetIcon, SparklesIcon, WrenchIcon } from './icons'
import { StatusDot } from './StatusDot'

const Hero3D = lazy(() => import('@/three/Hero3D'))

/** The centrepiece: the neural sphere shown inside a lit glass case on a stone plinth. */
function HeroExhibit() {
  return (
    <figure className="relative mx-auto w-full max-w-[34rem]">
      {/* ceiling spotlight falling onto the case */}
      <div className="spotlight pointer-events-none absolute -inset-x-16 -top-24 h-72" aria-hidden="true" />
      <div className="display-case relative aspect-[5/4] w-full">
        <div className="pointer-events-none absolute inset-0">
          <Suspense fallback={null}>
            <Hero3D />
          </Suspense>
        </div>
        {/* small engraved maker's plate inside the glass */}
        <span className="absolute bottom-3 left-3 z-10 font-mono text-[10px] tracking-[0.2em] text-[#d9c49a]/70 uppercase">
          Live · rendered in your browser
        </span>
      </div>
      <div className="plinth mx-[-4%]" aria-hidden="true" />
      <div className="mx-auto h-3 w-[96%] bg-[radial-gradient(50%_100%_at_50%_0%,rgb(var(--shadow-color)/0.35),transparent)]" aria-hidden="true" />

      {/* museum label beside the object */}
      <figcaption className="placard-frame relative mx-auto mt-4 max-w-sm px-5 py-4 text-left sm:absolute sm:right-[-1.5rem] sm:bottom-[-2.5rem] sm:mt-0 sm:w-64">
        <p className="exhibit-label !text-[10px]">Exhibit 00 · Centrepiece</p>
        <p className="mt-1 font-display text-xl leading-tight text-fg">The Neural Sphere</p>
        <p className="mt-1 text-xs leading-relaxed text-fg-muted">
          220 neurons, light, and signal pulses. An artist’s impression of a network thinking —
          illustrative, not a real model.
        </p>
      </figcaption>
    </figure>
  )
}

/** "Plan your visit" — the five ways into the collection. */
function VisitOptions({ resumeSlug }: { resumeSlug: string }) {
  const items = [
    {
      to: `/module/${resumeSlug}`,
      Icon: BookIcon,
      kicker: 'Read',
      title: 'The gallery guide',
      body: 'Eleven exhibits of wall text, each paired with an interactive object you can handle.',
      tone: 'var(--brass-ink)',
    },
    {
      to: '/museum/orientation',
      Icon: SparklesIcon,
      kicker: 'Walk',
      title: 'The 3D museum',
      body: 'Stroll through vaulted halls and lean into glass cases where the ideas glow.',
      tone: 'var(--accent)',
    },
    {
      to: '/galaxy',
      Icon: PlanetIcon,
      kicker: 'Gaze',
      title: 'The planetarium',
      body: 'Fly through 45,000 words arranged by meaning — embeddings as a night sky.',
      tone: 'var(--accent-2)',
    },
    {
      to: '/workshop',
      Icon: WrenchIcon,
      kicker: 'Build',
      title: 'The workshop',
      body: 'Train a tiny language model in your own browser and watch gibberish turn into words.',
      tone: 'var(--brass-ink)',
    },
    {
      to: '/ask',
      Icon: ChatIcon,
      kicker: 'Ask',
      title: 'The reading room',
      body: 'Download a real chat model into the tab and question a book. An experiment, with its score shown.',
      tone: 'var(--accent)',
    },
  ]
  return (
    <section aria-labelledby="visit-heading" className="mx-auto max-w-6xl px-5 pt-20 sm:px-8">
      <div className="mb-8 text-center">
        <p className="exhibit-label">Plan your visit</p>
        <h2 id="visit-heading" className="mt-2 font-display text-3xl text-fg sm:text-4xl">
          Five ways through the collection
        </h2>
      </div>
      <div className="grid gap-px overflow-hidden border border-border bg-border sm:grid-cols-2 xl:grid-cols-5">
        {items.map(({ to, Icon, kicker, title, body, tone }) => (
          <Link key={to} to={to} className="group bg-surface p-6 transition-colors hover:bg-bg sm:p-7 sm:last:col-span-2 xl:last:col-span-1">
            <span className="flex items-center gap-2" style={{ color: tone }}>
              <Icon size={18} />
              <span className="text-[11px] font-semibold tracking-[0.2em] uppercase">{kicker}</span>
            </span>
            <span className="mt-3 block font-display text-2xl text-fg">{title}</span>
            <span className="mt-2 block text-sm leading-relaxed text-fg-muted">{body}</span>
            <span className="mt-4 inline-flex items-center gap-1.5 text-xs font-semibold tracking-[0.14em] text-fg uppercase">
              Enter
              <ArrowRightIcon size={14} className="text-brass transition-transform group-hover:translate-x-1" />
            </span>
          </Link>
        ))}
      </div>
    </section>
  )
}

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
      {/* Hero: limestone gallery wall, text on the left, the exhibit on the right */}
      <div className="limestone-wall overflow-hidden border-b border-border-strong/60">
        <section className="mx-auto grid max-w-6xl items-center gap-14 px-5 pt-14 pb-12 sm:px-8 sm:pb-20 lg:grid-cols-[1.05fr_1fr] lg:gap-10 lg:pt-20 lg:pb-28">
          <motion.div initial="hidden" animate="visible" variants={staggerParent} className="text-center lg:text-left">
            <motion.p variants={fadeRise} className="exhibit-label">
              Permanent exhibition · Interactive
            </motion.p>
            <motion.div variants={fadeRise} className="brass-rule-left mx-auto mt-4 w-24 lg:mx-0" aria-hidden="true" />

            <motion.h1
              variants={fadeRise}
              className="mt-6 font-display text-[2.9rem] leading-[1.02] font-semibold text-fg sm:text-7xl"
            >
              How Language Models <em className="holo-text pr-1 font-semibold italic">Actually Work</em>
            </motion.h1>

            <motion.p
              variants={fadeRise}
              className="mx-auto mt-6 max-w-xl text-lg leading-relaxed text-fg-muted lg:mx-0"
            >
              A museum of the machine that writes. Follow text as it becomes tokens, meets attention,
              learns from the internet and speaks again — every idea displayed beside an exhibit you
              can touch.
            </motion.p>

            <motion.div
              variants={fadeRise}
              className="mt-9 flex flex-col items-center justify-center gap-3 sm:flex-row lg:justify-start"
            >
              <button onClick={() => navigate(`/module/${resumeTarget.slug}`)} className="btn-museum group">
                {started ? 'Resume your visit' : 'Begin the tour'}
                <ArrowRightIcon size={16} className="transition-transform group-hover:translate-x-0.5" />
              </button>
              <button onClick={() => navigate('/museum/orientation')} className="btn-outline">
                <SparklesIcon size={15} className="text-accent" />
                Walk the museum
                <span className="tag text-accent">3D</span>
              </button>
            </motion.div>

            <motion.dl
              variants={fadeRise}
              className="mt-10 flex justify-center gap-8 text-left lg:justify-start"
            >
              {[
                [String(MODULES.length), 'Exhibits'],
                [`~${TOTAL_MINUTES}`, 'Minutes'],
                ['4', 'Floors'],
              ].map(([n, l]) => (
                <div key={l}>
                  <dt className="sr-only">{l}</dt>
                  <dd className="font-display text-3xl leading-none text-fg">{n}</dd>
                  <dd className="mt-1 text-[10px] font-semibold tracking-[0.2em] text-fg-subtle uppercase">{l}</dd>
                </div>
              ))}
            </motion.dl>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.15, ease: [0.22, 1, 0.36, 1] }}
            className="pb-6 sm:pb-10"
          >
            <HeroExhibit />
          </motion.div>
        </section>
      </div>

      <VisitOptions resumeSlug={resumeTarget.slug} />

      {/* The collection: framed exhibit plaques */}
      <section aria-labelledby="collection-heading" className="mx-auto max-w-6xl px-5 pt-20 pb-20 sm:px-8">
        <div className="mb-8 flex flex-col items-center gap-2 text-center">
          <p className="exhibit-label">The collection</p>
          <h2 id="collection-heading" className="font-display text-3xl text-fg sm:text-4xl">
            Eleven exhibits, one idea
          </h2>
          <p className="font-display text-lg text-fg-muted italic">tokens in → next token out</p>
        </div>
        <motion.ol
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.05 }}
          variants={staggerParent}
          className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3"
        >
          {MODULES.map((m) => {
            const status = getStatus(m.slug)
            return (
              <motion.li key={m.id} variants={fadeRise} className="flex">
                <Link
                  to={`/module/${m.slug}`}
                  className="placard-frame lift-on-hover group relative flex w-full flex-col overflow-hidden p-6 hover:shadow-[inset_0_0_0_4px_var(--surface),inset_0_0_0_5px_var(--gold),0_26px_44px_-26px_rgb(var(--shadow-color)/0.55)]"
                >
                  {/* oversized engraved numeral */}
                  <span
                    aria-hidden="true"
                    className="pointer-events-none absolute -top-3 right-3 font-display text-[6.5rem] leading-none font-semibold text-brass opacity-[0.16] transition-opacity select-none group-hover:opacity-30"
                  >
                    {moduleLabel(m.index)}
                  </span>
                  <span className="relative flex items-center justify-between gap-3">
                    <span className="exhibit-label !text-[10px]">Exhibit {moduleLabel(m.index)}</span>
                    <StatusDot status={status} className="shrink-0" />
                  </span>
                  <h3 className="relative mt-3 font-display text-[1.65rem] leading-tight text-fg">{m.title}</h3>
                  <p className="relative mt-2 flex-1 text-sm leading-relaxed text-fg-muted">{m.subtitle}</p>
                  <span className="relative mt-5 flex items-center justify-between border-t border-border pt-3 text-xs text-fg-subtle">
                    <span>{m.minutes} min visit</span>
                    <span className="inline-flex items-center gap-1.5 font-semibold tracking-[0.14em] text-fg uppercase">
                      View
                      <ArrowRightIcon size={14} className="text-brass transition-transform group-hover:translate-x-1" />
                    </span>
                  </span>
                  {/* the AI light: a faint violet line that wakes on hover */}
                  <span
                    aria-hidden="true"
                    className="absolute inset-x-6 top-0 h-px opacity-0 transition-opacity group-hover:opacity-100"
                    style={{
                      backgroundImage: 'linear-gradient(90deg, transparent, var(--accent), var(--accent-2), transparent)',
                      boxShadow: '0 0 10px var(--accent)',
                    }}
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
