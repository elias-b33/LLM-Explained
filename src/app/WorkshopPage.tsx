import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { motion } from 'motion/react'
import { Button } from '@/primitives/Button'
import { DemoFrame } from '@/primitives/DemoFrame'
import { DeepDive } from '@/primitives/DeepDive'
import { cn } from '@/lib/cn'
import { CORPORA, CUSTOM_MAX, CUSTOM_MIN, type CorpusId } from '@/lab/corpora'
import {
  FINETUNE,
  SIZES,
  estStepsPerSec,
  fmtBytes,
  fmtSpeed,
  frontierRatio,
  gpuMemory,
  paramCount,
  type Device,
  type ModelSize,
  type SizeId,
  type StartFrom,
} from '@/lab/sizes'
import { fetchWeights, type PretrainedState } from '@/lab/pretrained'
import { DevicePicker, SelfTestPanel } from '@/lab/DevicePicker'
import type { GpuInfo, SelfTestResult, WeightsHeader } from '@/lab/protocol'
import { useTrainer } from '@/lab/useTrainer'
import { LossChart } from '@/lab/LossChart'
import { SampleTimeline } from '@/lab/SampleTimeline'
import { Playground } from '@/lab/Playground'
import { InsightPanel } from '@/lab/InsightPanel'

const fmtTime = (ms: number) => {
  const s = Math.floor(ms / 1000)
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}
const fmtInt = (n: number) => Math.round(n).toLocaleString('en-US')

function Step({ n, title, hint, children }: { n: string; title: string; hint: string; children: ReactNode }) {
  return (
    <section className="min-w-0 space-y-3">
      <div>
        <p className="exhibit-label">Step {n}</p>
        <h2 className="mt-0.5 font-display text-[1.7rem] leading-tight text-fg">{title}</h2>
        <p className="mt-1 text-sm text-fg-muted">{hint}</p>
      </div>
      {children}
    </section>
  )
}

function Choice({
  selected,
  onClick,
  children,
  disabled,
}: {
  selected: boolean
  onClick: () => void
  children: ReactNode
  disabled?: boolean
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={selected}
      disabled={disabled}
      onClick={onClick}
      className={cn(
        'relative flex h-full w-full min-w-0 flex-col items-start border p-3.5 text-left transition-colors enabled:cursor-pointer disabled:opacity-50',
        selected
          ? 'border-accent bg-surface shadow-[0_0_0_3px_color-mix(in_srgb,var(--accent)_14%,transparent)]'
          : 'border-border bg-surface/60 hover:border-brass hover:bg-surface',
      )}
    >
      {children}
    </button>
  )
}

function Stat({ label, value, mono = true }: { label: string; value: ReactNode; mono?: boolean }) {
  return (
    <div className="min-w-0 border border-border bg-surface/70 px-3 py-2">
      <div className="text-[10px] font-semibold tracking-[0.16em] text-fg-subtle uppercase">{label}</div>
      <div className={cn('mt-0.5 truncate text-[15px] text-fg tabular-nums', mono && 'font-mono')}>{value}</div>
    </div>
  )
}

export function WorkshopPage() {
  const trainer = useTrainer()
  const { view } = trainer
  const [corpusId, setCorpusId] = useState<CorpusId>('shakespeare')
  const [sizeId, setSizeId] = useState<SizeId>('tiny')
  const [device, setDevice] = useState<Device>('cpu')
  const [draft, setDraft] = useState('')
  const [customText, setCustomText] = useState<string | null>(null)
  const [loadingText, setLoadingText] = useState(false)
  const [logScale, setLogScale] = useState(false)
  const [playPrompt, setPlayPrompt] = useState('')
  const [startFrom, setStartFrom] = useState<StartFrom>('scratch')
  /** download state of each size's pretrained weights (fetched lazily, kept for the session) */
  const [weights, setWeights] = useState<Partial<Record<SizeId, PretrainedState>>>({})
  const textCache = useRef(new Map<string, string>())
  const size = SIZES.find((s) => s.id === sizeId)!
  const wantPretrained = startFrom === 'pretrained' && !!size.pretrainable
  const wState = wantPretrained ? weights[sizeId] : undefined
  /** what the model is actually built from right now: pretrained only once its weights are here */
  const effectiveStart: StartFrom = wantPretrained && wState?.status === 'ready' ? 'pretrained' : 'scratch'
  const waitingForWeights = wantPretrained && (!wState || wState.status === 'loading')

  // fetch the pretrained weights the first time they are asked for
  useEffect(() => {
    if (!wantPretrained || weights[sizeId]) return
    const ctl = new AbortController()
    let last = 0
    fetchWeights(
      sizeId,
      (loaded, total) => {
        const now = performance.now()
        if (now - last < 100 && loaded !== total) return
        last = now
        setWeights((w) => ({ ...w, [sizeId]: { status: 'loading', loaded, total } }))
      },
      ctl.signal,
    )
      .then((st) => setWeights((w) => ({ ...w, [sizeId]: st })))
      .catch(() => {
        /* aborted: forget the partial download */
        setWeights((w) => {
          const n = { ...w }
          delete n[sizeId]
          return n
        })
      })
    return () => ctl.abort()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wantPretrained, sizeId])

  // XL is GPU only: leaving the GPU steps down to the biggest CPU-capable size
  const changeDevice = (d: Device) => {
    setDevice(d)
    if (d === 'cpu' && size.gpuOnly) setSizeId('huge')
  }

  useEffect(() => {
    document.title = 'The Workshop · LLM Explainer'
  }, [])

  // (re)build the model whenever the text or size changes
  useEffect(() => {
    let cancelled = false
    async function go() {
      let text: string
      let prompt: string
      if (corpusId === 'custom') {
        if (!customText) return
        text = customText
        prompt = (customText.trimStart().split('\n')[0] ?? '').slice(0, 10)
      } else {
        const c = CORPORA.find((x) => x.id === corpusId)!
        prompt = c.prompt
        const cached = textCache.current.get(c.id)
        if (cached) text = cached
        else {
          setLoadingText(true)
          try {
            text = await c.load()
          } finally {
            setLoadingText(false)
          }
          textCache.current.set(c.id, text)
          if (cancelled) return
        }
      }
      if (cancelled) return
      if (device === 'cpu' && size.gpuOnly) return
      const w = effectiveStart === 'pretrained' && wState?.status === 'ready' ? wState.buf : undefined
      trainer.init(text, sizeId, prompt, device, effectiveStart, w)
    }
    if (waitingForWeights) return
    void go()
    return () => {
      cancelled = true
    }
    // trainer.init is stable
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [corpusId, sizeId, customText, device, effectiveStart, waitingForWeights])

  // the worker falls back to the CPU engine if the GPU fails to start: reflect that in the switch
  const [fallbackSeen, setFallbackSeen] = useState<unknown>(null)
  if (view.info && view.info.device === 'cpu' && device === 'gpu' && view.error && fallbackSeen !== view.info) {
    setFallbackSeen(view.info)
    changeDevice('cpu')
  }

  // dev-only: expose the live view for debugging (loss history etc.)
  useEffect(() => {
    if (import.meta.env.DEV) (window as unknown as { __workshopView?: unknown }).__workshopView = view
  }, [view])

  // dev-only console hook: window.__gpuSelfTest() prints a per-tensor error table
  const { runSelfTest } = trainer
  useEffect(() => {
    if (!import.meta.env.DEV) return
    const w = window as unknown as { __gpuSelfTest?: (url?: string) => Promise<SelfTestResult> }
    // optional url of a .bin: also compare the engines on those pretrained weights
    w.__gpuSelfTest = async (url?: string) => {
      const buf = url ? await fetch(url).then((r) => r.arrayBuffer()) : undefined
      const r = await runSelfTest(buf)
      console.log(`GPU self-test: ${r.error ?? r.summary}`)
      for (const c of r.cases) {
        console.log(
          `${c.ok ? 'PASS' : 'FAIL'} ${c.label}: loss cpu ${c.lossCpu} gpu ${c.lossGpu} (rel ${c.lossRel.toExponential(2)}), max logit diff ${c.logitsMaxAbs?.toExponential(2)}`,
        )
        console.table(
          c.grads.map((g, i) => ({
            tensor: g.tensor,
            size: g.size,
            gradMaxAbs: g.maxAbs,
            gradRel: g.rel,
            paramDriftPerLr: c.params[i].rel,
            ok: g.ok && c.params[i].ok,
          })),
        )
        console.table(c.stepLosses)
      }
      return r
    }
    return () => {
      delete w.__gpuSelfTest
    }
  }, [runSelfTest])

  const vocabSize = view.info?.vocab.length ?? 70
  const training = view.phase === 'training'
  const canTrain =
    (view.phase === 'ready' || view.phase === 'paused') &&
    !loadingText &&
    !waitingForWeights &&
    !(device === 'cpu' && size.gpuOnly) &&
    !(corpusId === 'custom' && !customText)
  const recentLoss = useMemo(() => {
    const l = view.losses
    const n = Math.min(50, l.length)
    if (!n) return null
    let s = 0
    for (let i = l.length - n; i < l.length; i++) s += l[i]
    return s / n
  }, [view.losses])
  const lastVal = view.val.length ? view.val[view.val.length - 1][1] : null

  // keep "look inside" fresh while training, and whenever the prompt changes
  const { requestInsight } = trainer
  const hasModel = view.info !== null
  useEffect(() => {
    if (!hasModel) return
    const t = setTimeout(() => requestInsight(playPrompt), 250)
    return () => clearTimeout(t)
  }, [playPrompt, requestInsight, hasModel, view.info])
  useEffect(() => {
    if (!training) {
      if (hasModel && view.step > 0) requestInsight(playPrompt)
      return
    }
    const id = setInterval(() => {
      if (document.visibilityState === 'visible') requestInsight(playPrompt)
    }, 4000)
    return () => clearInterval(id)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [training, requestInsight, hasModel])

  const draftChars = draft.length
  const draftOk = draftChars >= CUSTOM_MIN
  const corpus = CORPORA.find((c) => c.id === corpusId)
  const defaultPlayPrompt = corpusId === 'custom' ? (view.info?.prompt ?? '') : (corpus?.prompt.trim() ?? '')

  return (
    <article className="mx-auto max-w-6xl px-4 py-10 sm:px-8 sm:py-14">
      <motion.header
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="mb-12 text-center"
      >
        <p className="exhibit-label">The Workshop</p>
        <h1 className="mt-4 font-display text-[2.6rem] leading-[1.05] text-fg sm:text-6xl">
          Train your own <span className="holo-text">model</span>
        </h1>
        <div className="ornament mx-auto mt-5 w-48" aria-hidden="true">
          <span />
        </div>
        <p className="mx-auto mt-5 max-w-2xl font-display text-xl leading-snug text-fg-muted italic sm:text-[1.35rem]">
          A real transformer, trained from scratch right here in your browser tab. No server, no download. Close the tab and it’s gone.
        </p>
      </motion.header>

      <div className="grid grid-cols-1 gap-10">
        <Step n="I" title="Pick the training text" hint="The model starts knowing nothing. Everything it learns comes from this text.">
          <div role="radiogroup" aria-label="Training text" className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
            {CORPORA.map((c) => (
              <Choice key={c.id} selected={corpusId === c.id} onClick={() => setCorpusId(c.id)}>
                <span className="font-display text-lg leading-tight text-fg">{c.title}</span>
                <span className="text-xs text-fg-muted">{c.author}</span>
                <span className="tag mt-2 text-brass-ink">{c.lang}</span>
              </Choice>
            ))}
            <Choice selected={corpusId === 'custom'} onClick={() => setCorpusId('custom')}>
              <span className="font-display text-lg leading-tight text-fg">Paste your own</span>
              <span className="text-xs text-fg-muted">Lyrics, code, your diary…</span>
              <span className="tag mt-2 text-accent">Your text</span>
            </Choice>
          </div>
          {corpusId === 'custom' && (
            <div className="space-y-2 border border-border bg-surface/70 p-3">
              <label htmlFor="ws-custom" className="text-sm text-fg-muted">
                Paste at least {CUSTOM_MIN.toLocaleString('en-US')} characters. It never leaves your browser.
              </label>
              <textarea
                id="ws-custom"
                value={draft}
                onChange={(e) => setDraft(e.target.value.slice(0, CUSTOM_MAX))}
                rows={6}
                className="w-full resize-y border border-border-strong bg-surface p-2 font-mono text-xs text-fg outline-none focus:border-accent"
                placeholder="Paste a long text here…"
              />
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className={cn('font-mono text-xs', draftOk ? 'text-success' : 'text-fg-subtle')}>
                  {fmtInt(draftChars)} / {fmtInt(CUSTOM_MIN)}+ characters
                  {draftChars >= CUSTOM_MAX && ' (capped at 1 MB)'}
                </span>
                <Button variant="secondary" disabled={!draftOk} onClick={() => setCustomText(draft)}>
                  {customText ? 'Use updated text' : 'Use this text'}
                </Button>
              </div>
            </div>
          )}
          <DeepDive title="What’s happening? The training text">
            <p>
              The model learns from this text and nothing else. It has never seen a dictionary, a website or a single other book, so
              whatever style, spelling and words it picks up come from here.
            </p>
            <p>
              Before training, the text is split in two: the first 90% is used for learning, and the last 10% is held out. The model
              never trains on that part, so we can use it later as a fair exam. More text, and more varied text, gives the model more
              examples to find real patterns in, so it generalises to new sentences instead of just memorising the ones it saw. With a
              very short text, it can only memorise.
            </p>
          </DeepDive>
        </Step>

        <Step n="II" title="Pick a model size" hint="Parameters are the knobs training adjusts. More knobs can learn more, but each step is slower.">
          <DevicePicker device={device} onChange={changeDevice} gpu={view.gpu} disabled={training} />
          <div role="radiogroup" aria-label="Model size" className="grid grid-cols-1 gap-2.5 md:grid-cols-2 2xl:grid-cols-3 md:[&>:last-child:nth-child(odd)]:col-span-2">
            {SIZES.map((s) => (
              <SizeCard
                key={s.id}
                size={s}
                vocabSize={vocabSize}
                device={device}
                gpu={view.gpu}
                selected={sizeId === s.id}
                onSelect={() => setSizeId(s.id)}
              />
            ))}
          </div>
          <p className="text-xs text-fg-subtle">
            For scale: frontier LLMs are estimated at around a trillion parameters, roughly{' '}
            <b className="text-fg-muted">{frontierRatio(paramCount(size, vocabSize))} times</b> more than
            this one. Speeds are rough estimates; the real number shows up under the training bench.
          </p>
          <StartFromPicker
            size={size}
            startFrom={startFrom}
            onChange={setStartFrom}
            state={wState}
            disabled={training}
          />
          {view.gpu?.status === 'ok' && (
            <SelfTestPanel
              result={view.selfTest}
              running={view.selfTestRunning}
              onRun={() => void runSelfTest(effectiveStart === 'pretrained' && wState?.status === 'ready' ? wState.buf : undefined)}
            />
          )}
          <DeepDive title="What’s happening? Model size">
            <p>
              <b>Parameters</b> are the adjustable numbers inside the model. Training is the process of tuning them. <b>Layers</b> are
              stacked processing stages: each one refines what the previous one worked out. <b>Heads</b> are parallel “attention”
              spotlights inside a layer, each free to look for a different kind of clue. <b>Width</b> is how many numbers describe each
              character internally, so a wider model can hold richer ideas. <b>Context</b> is how many previous characters it can see
              when guessing the next one.
            </p>
            <p>
              Almost all of training is multiplying big tables of numbers together: millions of multiply-adds per step, each one
              independent of the others. A <b>CPU</b> has a handful of cores that work through them a few at a time. A <b>GPU</b> has
              thousands of small cores that do thousands of multiply-adds at the same moment, which is exactly the shape of this work.
              That is why the Large and Huge models are practical on a graphics card and painfully slow without one, and why XL
              (25 million parameters) is offered on the GPU only.
            </p>
            <p>
              <b>VRAM</b> is the graphics card’s own memory. During training it has to hold every parameter, its gradient (which way to
              nudge it), two running averages the AdamW optimizer keeps for each parameter, and the intermediate results of every layer
              for the whole batch, including each head’s attention table (context × context numbers per sequence). The memory figure on
              each size adds all of that up. Browsers don’t reveal how much VRAM a card has, so the requirement line is a rule of thumb.
              What we can check is the largest single block of memory your GPU allows, and sizes that wouldn’t fit are switched off.
            </p>
          </DeepDive>
          <DeepDive title="Transfer learning: why start from a pretrained model?">
            <p>
              A model trained from scratch on one play has to learn everything from that play alone: spelling, grammar, what a word
              even is. A <b>pretrained</b> model has already read hundreds of megabytes of English and Swedish books, so it arrives
              knowing how words, sentences and punctuation work. <b>Fine-tuning</b> then only has to teach it what is special about
              your text, like its names, its rhythm and its style.
            </p>
            <p>
              This is how real language models are built: one long, expensive pretraining run, then many short, cheap fine-tunes.
              Fine-tuning uses a learning rate fifty times smaller and warms it up over the first {FINETUNE.warmupSteps}{' '}
              steps, so the first updates refine what the model knows instead of wiping it out. Pretrained models read raw bytes
              (256 symbols), so they can take any text, but their loss is per byte, not per character.
            </p>
          </DeepDive>
        </Step>
      </div>

      <div className="mt-12">
        <p className="exhibit-label">Step III</p>
        <h2 className="mt-0.5 mb-3 font-display text-[1.7rem] leading-tight text-fg">Train it</h2>
        <DemoFrame
          title="The training bench"
          status="real"
          caption="Each step, the model guesses the next character at hundreds of spots in the text, measures how wrong it was, and nudges every parameter slightly toward a better guess."
        >
          <div className="flex flex-wrap items-center gap-3">
            {training ? (
              <Button variant="primary" onClick={trainer.pause} className="min-w-28">
                <PauseIcon /> Pause
              </Button>
            ) : (
              <Button variant="primary" onClick={trainer.start} disabled={!canTrain} className="min-w-28">
                <PlayIcon /> {view.step > 0 ? 'Resume' : 'Train'}
              </Button>
            )}
            <Button variant="secondary" onClick={trainer.reset} disabled={!view.info || view.phase === 'loading'}>
              Reset
            </Button>
            <span className="text-xs text-fg-subtle">
              {device === 'cpu' && size.gpuOnly
                ? `${size.label} runs on the GPU only.`
                : waitingForWeights
                  ? 'Downloading the pretrained weights…'
                  : view.phase === 'booting' || view.phase === 'loading' || loadingText
                ? effectiveStart === 'pretrained'
                  ? 'Loading the pretrained model…'
                  : 'Building a fresh, random model…'
                : corpusId === 'custom' && !customText
                  ? 'Paste a text above first.'
                  : view.phase === 'ready'
                    ? view.info?.pretrained
                      ? 'A pretrained model, not yet fine-tuned on this text. Press Train to fine-tune.'
                      : 'A fresh model with random knobs. Press Train.'
                    : training
                      ? 'Training… you can keep using the page.'
                      : 'Paused. Try generating text below.'}
            </span>
          </div>

          <div className="mt-4">
            <DeepDive title="What’s happening? Training">
              <p>
                One <b>step</b> has three parts. The model guesses the next character at hundreds of spots in the text. Then we measure
                how wrong the guesses were. Finally, <b>gradient descent</b> nudges every parameter a tiny amount in whichever direction
                would have made those guesses a bit better. Then it repeats, thousands of times.
              </p>
              <p>
                <b>Chars / sec</b> is how many characters the model is learning from each second, a measure of your computer’s speed. On the <b>CPU</b> engine the arithmetic runs in JavaScript, split over a few threads; on the <b>GPU</b> engine every step is sent to the graphics card as a batch of small programs (compute shaders) and the weights never leave it, so only the loss comes back each step.
                At the start, every parameter is random, so the model is just guessing, and its output is gibberish. It only gets
                better because millions of tiny nudges add up.
              </p>
            </DeepDive>
          </div>

          <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
            <Stat label="Step" value={fmtInt(view.step)} />
            <Stat label="Chars / sec" value={training ? fmtInt(view.charsPerSec) : '–'} />
            <Stat label="Elapsed" value={fmtTime(view.trainMs)} />
            <Stat label="Train loss" value={recentLoss !== null ? recentLoss.toFixed(3) : '–'} />
            <Stat label="Val loss" value={lastVal !== null ? lastVal.toFixed(3) : '–'} />
            <Stat
              label="Vocabulary"
              value={view.info ? (view.info.vocabKind === 'bytes' ? '256 bytes' : `${view.info.vocab.length} symbols`) : '–'}
            />
          </div>
          <p className="mt-2 font-mono text-[11px] text-fg-subtle">
            Engine: {view.backend}
            {view.info && view.info.device === 'gpu' && ` · ${fmtBytes(view.info.gpuBytes)} on the GPU`}
            {view.info &&
              ` · ${fmtInt(view.info.params)} parameters · ${fmtInt(view.info.trainChars)} training ${view.info.vocabKind === 'bytes' ? 'bytes' : 'chars'} + ${fmtInt(view.info.valChars)} held out`}
            {view.info?.pretrained && ` · ${pretrainedSummary(view.info.pretrained)}`}
          </p>
          {view.error && (
            <p className="mt-2 text-sm" style={{ color: 'var(--danger)' }}>
              Something went wrong: {view.error}
            </p>
          )}

          <div className="mt-5 grid gap-6 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
            <div className="min-w-0">
              <div className="mb-2 flex flex-wrap items-end justify-between gap-2">
                <div>
                  <p className="font-display text-lg text-fg">Loss</p>
                  <p className="text-xs text-fg-muted">
                    How surprised the model is by the real next character. Lower is better.
                  </p>
                </div>
                <label className="flex cursor-pointer items-center gap-1.5 text-xs text-fg-muted">
                  <input type="checkbox" checked={logScale} onChange={(e) => setLogScale(e.target.checked)} />
                  log scale
                </label>
              </div>
              <LossChart losses={view.losses} val={view.val} vocabSize={vocabSize} logScale={logScale} />
              <p className="mt-1 text-xs text-fg-subtle">
                Validation loss is measured on the last 10% of the text, which it never trains on. If it tracks the training line, the
                model is learning patterns, not memorising.
              </p>
              <div className="mt-2">
                <DeepDive title="What’s happening? The loss chart">
                  <p>
                    Loss is a measure of <b>surprise</b>, in units called nats. If the model gave the real next character a high
                    probability, it’s barely surprised and the loss is low. If it gave it almost none, the loss is high.
                  </p>
                  <p>
                    The dashed <b>random-guessing line</b> sits at ln(vocabulary size): the loss of a model that thinks every symbol is
                    equally likely. Below it, the model has learned something. <b>Training</b> loss is measured on text it practises
                    on; <b>validation</b> loss on the held-out text. A small gap is normal. <b>Overfitting</b> is when training keeps
                    falling but validation flattens or rises: the model is memorising rather than learning.
                  </p>
                </DeepDive>
              </div>
            </div>
            <div className="min-w-0">
              <p className="font-display text-lg text-fg">What it writes</p>
              <p className="mb-2 text-xs text-fg-muted">
                Same prompt, same dice, every {view.sampleEvery ? `${view.sampleEvery} steps` : 'few seconds'}. Watch gibberish become
                words.
              </p>
              {view.before && view.samples.length > 1 && (
                <div className="mb-3 border border-dashed border-brass bg-surface/60 p-3">
                  <p className="exhibit-label !tracking-[0.14em]">Before fine-tuning · pretrained only</p>
                  <p className="mt-1.5 font-mono text-[12.5px] leading-relaxed break-words whitespace-pre-wrap text-fg">
                    <span className="text-brass-ink">{view.before.prompt}</span>
                    {view.before.text}
                  </p>
                </div>
              )}
              <SampleTimeline samples={view.samples} />
              <div className="mt-2">
                <DeepDive title="What’s happening? The samples">
                  <p>
                    Early on, the model has only learned which characters are common, so you get noise. Soon it picks up letter
                    patterns, like which letters tend to follow each other. Then whole words appear, then punctuation, line breaks and
                    the rhythm of the text, and finally something of its style.
                  </p>
                  <p>
                    All of this comes from one skill. At every moment the model only predicts a single next character, then repeats.
                    It never plans a sentence ahead.
                  </p>
                </DeepDive>
              </div>
            </div>
          </div>
        </DemoFrame>
      </div>

      <div className="mt-12">
        <p className="exhibit-label">Step IV</p>
        <h2 className="mt-0.5 mb-3 font-display text-[1.7rem] leading-tight text-fg">Play with your model</h2>
        <DemoFrame
          title="The writing desk"
          status="real"
          caption="It writes by predicting one character, adding it to the text, and predicting again. That loop is all a large language model does, just with tokens instead of characters, and at enormous scale."
        >
          <Playground
            defaultPrompt={defaultPlayPrompt}
            disabled={!view.info}
            byteMode={view.info?.vocabKind === 'bytes'}
            generate={trainer.generate}
            stop={trainer.stopGenerate}
            onPromptChange={setPlayPrompt}
          />
          <div className="mt-4">
            <DeepDive title="What’s happening? Generation and temperature">
              <p>
                For each new character, the model produces a probability for every symbol in its vocabulary. The bars show the most
                likely candidates. It then <b>samples</b>: it rolls weighted dice, so a character with 60% probability is picked about
                60% of the time.
              </p>
              <p>
                <b>Temperature</b> reshapes those odds. Low temperature favours the top choice nearly every time: safe, but repetitive
                and prone to loops. High temperature flattens the odds, so unlikely characters get picked: more creative, but also more
                chaotic and error-prone.
              </p>
            </DeepDive>
          </div>
        </DemoFrame>
      </div>

      <div className="mt-12">
        <p className="exhibit-label">Step V</p>
        <h2 className="mt-0.5 mb-3 font-display text-[1.7rem] leading-tight text-fg">Look inside</h2>
        <DemoFrame title="Under the glass" status="real" caption="Live views of your model’s own numbers. They refresh every few seconds while it trains.">
          <InsightPanel insight={view.insight} />
          <div className="mt-4">
            <DeepDive title="What’s happening? Under the glass">
              <p>
                <b>Embeddings:</b> inside the model, each character becomes a list of numbers, a point in space. The map squashes that
                space down to two dimensions. Nobody tells the model which characters are alike, but as it trains, similar ones, such
                as vowels, end up close together because they behave similarly in text.
              </p>
              <p>
                <b>Attention:</b> to guess the next character, the model looks back at earlier ones and decides how much each matters.
                The heatmap shows those weights: a stronger colour means “paying more attention”. The upper-right triangle is empty on purpose:
                the model can only look at what came before, never ahead, because the future is exactly what it is trying to predict.
              </p>
            </DeepDive>
          </div>
        </DemoFrame>
      </div>

      <div className="mt-12">
        <DeepDive title="How is this different from ChatGPT?">
          <p>
            Same core idea: <b>predict the next token</b>. The differences are mostly scale and polish.
          </p>
          <ul>
            <li>
              <b>Tokens, not characters.</b> Real models read chunks of words, so a sentence is a few dozen tokens instead of a few
              hundred characters.
            </li>
            <li>
              <b>Size.</b> Roughly a trillion parameters, against eighteen thousand to about twenty-five million here.
            </li>
            <li>
              <b>Data.</b> Trillions of words from books, websites and code, not one book.
            </li>
            <li>
              <b>Compute.</b> Thousands of GPUs running for months, not a browser tab for minutes.
            </li>
            <li>
              <b>Assistant training.</b> After that, <b>instruction tuning</b> and <b>RLHF</b> (reinforcement learning from human
              feedback) teach the model to answer questions helpfully instead of just continuing text.
            </li>
          </ul>
        </DeepDive>
      </div>

      <aside className="mt-12 border-t border-border pt-6 text-xs leading-relaxed text-fg-subtle">
        <p className="exhibit-label mb-2">Credits</p>
        <p className="mb-2">
          Training texts are public domain, from{' '}
          <a href="https://www.gutenberg.org" className="underline decoration-brass underline-offset-2 hover:text-fg" target="_blank" rel="noreferrer">
            Project Gutenberg
          </a>
          , with the Gutenberg header, footer and licence text removed:
        </p>
        <ul className="list-disc space-y-1 pl-5">
          {CORPORA.map((c) => (
            <li key={c.id}>
              <i>{c.title}</i>, {c.author}.{' '}
              <a href={c.url} className="underline decoration-brass underline-offset-2 hover:text-fg" target="_blank" rel="noreferrer">
                {c.source}
              </a>
              .
            </li>
          ))}
        </ul>
        <p className="mt-2">
          The pretrained Large, Huge and XL models first read about 380 MB of other public-domain Project Gutenberg books:
          810 English and 211 Swedish ones, taken from the{' '}
          <a
            href="https://huggingface.co/datasets/manu/project_gutenberg"
            className="underline decoration-brass underline-offset-2 hover:text-fg"
            target="_blank"
            rel="noreferrer"
          >
            manu/project_gutenberg
          </a>{' '}
          collection on Hugging Face, with the Gutenberg header, footer and licence text removed, duplicates dropped, and the three
          texts above left out, so fine-tuning on them is a fair test. That pretraining ran in PyTorch on one desktop GPU; the
          weights are the same numbers this page’s engine uses.
        </p>
        <p className="mt-2">
          The model is a small GPT-style transformer (token + position embeddings, causal self-attention, MLP, layer norm) written from
          scratch in TypeScript, trained with AdamW in a Web Worker, either on the CPU or on the GPU through WebGPU compute shaders
          (WGSL) that mirror the CPU engine kernel for kernel.
        </p>
      </aside>
    </article>
  )
}

/** One model size: architecture, GPU memory and requirement, and estimated speed on each device. */
function SizeCard({
  size: s,
  vocabSize,
  device,
  gpu,
  selected,
  onSelect,
}: {
  size: ModelSize
  vocabSize: number
  device: Device
  gpu: GpuInfo | null
  selected: boolean
  onSelect: () => void
}) {
  const pc = paramCount(s, vocabSize)
  const mem = gpuMemory(s, vocabSize)
  let blocked: string | null = null
  if (device === 'cpu' && s.gpuOnly) blocked = 'too big for the CPU engine. Switch to the GPU above.'
  if (device === 'gpu' && gpu?.status === 'ok') {
    const limit = Math.min(gpu.maxBufferSize, gpu.maxStorageBufferBindingSize)
    if (mem.largestBuffer > limit) blocked = `it needs a ${fmtBytes(mem.largestBuffer)} buffer and this GPU allows at most ${fmtBytes(limit)}.`
  }
  const cpuSlow = device === 'cpu' && s.gpuRecommended && !s.gpuOnly
  return (
    <Choice selected={selected} onClick={onSelect} disabled={!!blocked}>
      <div className="flex w-full flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5">
        <span className="font-display text-xl leading-tight text-fg">
          {s.label} <span className="font-mono text-sm text-accent">{fmtInt(pc)} params</span>
        </span>
        <span className="text-[11px] text-fg-subtle">
          {s.nLayer} layer{s.nLayer > 1 ? 's' : ''} · {s.nHead} heads · width {s.dModel} · sees {s.ctx} chars
        </span>
      </div>
      <span className="mt-1 text-xs text-fg-muted">{s.blurb}</span>
      <span className="mt-1.5 text-[11px] leading-snug text-fg-subtle">
        <b className="font-medium text-fg-muted">GPU:</b> ≈ {fmtBytes(mem.total)} of video memory · {s.requirement}
      </span>
      <span className="text-[11px] leading-snug text-fg-subtle">
        <b className="font-medium text-fg-muted">Est. speed:</b>{' '}
        {s.gpuOnly ? 'CPU: not offered' : `CPU ${fmtSpeed(estStepsPerSec(s, vocabSize, 'cpu'))} (one thread)`} · GPU{' '}
        {fmtSpeed(estStepsPerSec(s, vocabSize, 'gpu'))} (fast desktop GPU)
      </span>
      {cpuSlow && (
        <span className="mt-1 text-[11px] leading-snug" style={{ color: 'var(--danger)' }}>
          Very slow on the CPU: hours before it writes real words, and each step keeps the controls busy until it finishes.
          {gpu?.status === 'ok' ? ' Switch to the GPU above.' : ' A GPU is strongly recommended.'}
        </span>
      )}
      {blocked && (
        <span className="mt-1 text-[11px] leading-snug" style={{ color: 'var(--danger)' }}>
          {s.gpuOnly && device === 'cpu' ? 'GPU only: ' : 'Off on this GPU: '}
          {blocked}
        </span>
      )}
    </Choice>
  )
}

function pretrainedSummary(h: WeightsHeader): string {
  const t = h.training ?? {}
  const parts = ['pretrained']
  if (t.dataset) parts.push(`on ${t.dataset}`)
  if (t.tokensSeen) parts.push(`${fmtBytes(t.tokensSeen)} read`)
  if (typeof t.finalValLoss === 'number') parts.push(`val loss ${t.finalValLoss.toFixed(3)} per byte`)
  return parts.join(' · ')
}

/** From scratch vs pretrained, with the weights download (or "coming soon") underneath. */
function StartFromPicker({
  size,
  startFrom,
  onChange,
  state,
  disabled,
}: {
  size: ModelSize
  startFrom: StartFrom
  onChange: (s: StartFrom) => void
  state: PretrainedState | undefined
  disabled: boolean
}) {
  const can = !!size.pretrainable
  return (
    <div className="space-y-2">
      <p className="text-sm text-fg-muted">Start from</p>
      <div role="radiogroup" aria-label="Start from" className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
        <Choice selected={startFrom === 'scratch' || !can} onClick={() => onChange('scratch')} disabled={disabled}>
          <span className="font-display text-lg leading-tight text-fg">From scratch</span>
          <span className="text-xs text-fg-muted">Random knobs. Learns only from the text above, one character at a time.</span>
        </Choice>
        <Choice selected={startFrom === 'pretrained' && can} onClick={() => onChange('pretrained')} disabled={disabled || !can}>
          <span className="font-display text-lg leading-tight text-fg">Pretrained</span>
          <span className="text-xs text-fg-muted">
            {can
              ? 'Has already read hundreds of megabytes of English and Swedish books. You fine-tune it on your text.'
              : 'Available for Large, Huge and XL.'}
          </span>
        </Choice>
      </div>
      {startFrom === 'pretrained' && can && state && <PretrainedStatus size={size} state={state} />}
    </div>
  )
}

function PretrainedStatus({ size, state }: { size: ModelSize; state: PretrainedState }) {
  if (state.status === 'loading') {
    const pct = state.total ? Math.min(100, (state.loaded / state.total) * 100) : null
    return (
      <div className="border border-border bg-surface/70 p-2.5 text-xs text-fg-muted">
        <p>
          Downloading the pretrained {size.label} weights: {fmtBytes(state.loaded)}
          {state.total ? ` of ${fmtBytes(state.total)}` : ''}
        </p>
        <div
          className="mt-1.5 h-2 bg-surface-2"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={pct === null ? undefined : Math.round(pct)}
        >
          <div className={cn('h-full bg-accent', pct === null && 'w-1/3 animate-pulse')} style={pct === null ? undefined : { width: `${pct}%` }} />
        </div>
      </div>
    )
  }
  if (state.status === 'missing')
    return (
      <p className="border border-dashed border-brass bg-surface/70 p-2.5 text-xs text-fg-muted">
        <b className="text-fg">Pretrained weights coming soon.</b> The {size.label} model hasn’t finished its pretraining yet, so
        for now this size starts from scratch.
      </p>
    )
  if (state.status === 'error')
    return (
      <p className="border border-border bg-surface/70 p-2.5 text-xs" style={{ color: 'var(--danger)' }}>
        Could not load the pretrained weights: {state.message}. Starting from scratch instead.
      </p>
    )
  const t = state.header.training ?? {}
  return (
    <p className="text-xs text-fg-subtle">
      <span className="text-success">●</span> Pretrained {size.label} weights loaded ({fmtBytes(state.bytes)}, {state.header.dtype}).
      {t.dataset && <> Read {t.tokensSeen ? `${fmtBytes(t.tokensSeen)} of ` : ''}{t.dataset}.</>}
      {typeof t.finalValLoss === 'number' && <> Validation loss {t.finalValLoss.toFixed(3)} per byte.</>}
    </p>
  )
}

function PlayIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
      <path d="M2.5 1.5v9l8-4.5z" fill="currentColor" />
    </svg>
  )
}

function PauseIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
      <path d="M2.5 1.5h2.5v9H2.5zM7 1.5h2.5v9H7z" fill="currentColor" />
    </svg>
  )
}
