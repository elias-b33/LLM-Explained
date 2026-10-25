/**
 * All museum sound is synthesised with Web Audio (no sample files): a low,
 * slightly reverberant room tone, soft footstep ticks, and small UI sounds for
 * the exhibits. The context is created only from the "Click to enter" gesture
 * (autoplay policy) and closed when the museum unmounts.
 */
type Sfx = 'step' | 'clunk' | 'click' | 'chime' | 'tick' | 'slice' | 'whoosh'

const MUTE_KEY = 'museum-muted'

let ctx: AudioContext | null = null
let master: GainNode | null = null
let reverb: GainNode | null = null
let noise: AudioBuffer | null = null
let muted = readMuted()

function readMuted(): boolean {
  try {
    return localStorage.getItem(MUTE_KEY) === '1'
  } catch {
    return false
  }
}

export function isMuted() {
  return muted
}

export function setMuted(m: boolean) {
  muted = m
  try {
    localStorage.setItem(MUTE_KEY, m ? '1' : '0')
  } catch {
    /* storage blocked */
  }
  if (ctx && master) master.gain.setTargetAtTime(m ? 0 : 1, ctx.currentTime, 0.08)
}

function makeImpulse(c: AudioContext, seconds: number, decay: number): AudioBuffer {
  const len = Math.floor(c.sampleRate * seconds)
  const buf = c.createBuffer(2, len, c.sampleRate)
  for (let ch = 0; ch < 2; ch++) {
    const d = buf.getChannelData(ch)
    for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * (1 - i / len) ** decay
  }
  return buf
}

/** Start (or resume) audio. Must be called from a user gesture. */
export function startAudio() {
  try {
    if (ctx) {
      void ctx.resume()
      return
    }
    const AC = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    if (!AC) return
    const c = new AC()
    ctx = c
    master = c.createGain()
    master.gain.value = muted ? 0 : 1
    master.connect(c.destination)

    // Hall reverb: a generated decaying-noise impulse.
    const conv = c.createConvolver()
    conv.buffer = makeImpulse(c, 2.6, 3)
    reverb = c.createGain()
    reverb.gain.value = 0.5
    reverb.connect(conv)
    conv.connect(master)

    // 2 s of white noise reused by every noisy sound.
    noise = c.createBuffer(1, c.sampleRate * 2, c.sampleRate)
    const nd = noise.getChannelData(0)
    for (let i = 0; i < nd.length; i++) nd[i] = Math.random() * 2 - 1

    // Room tone: looped brown noise, low-passed, very quiet, with a slow filter drift.
    const brown = c.createBuffer(1, c.sampleRate * 4, c.sampleRate)
    const bd = brown.getChannelData(0)
    let last = 0
    for (let i = 0; i < bd.length; i++) {
      last = (last + 0.02 * (Math.random() * 2 - 1)) / 1.02
      bd[i] = last * 3.5
    }
    const room = c.createBufferSource()
    room.buffer = brown
    room.loop = true
    const lp = c.createBiquadFilter()
    lp.type = 'lowpass'
    lp.frequency.value = 220
    const lfo = c.createOscillator()
    lfo.frequency.value = 0.05
    const lfoGain = c.createGain()
    lfoGain.gain.value = 60
    lfo.connect(lfoGain)
    lfoGain.connect(lp.frequency)
    const roomGain = c.createGain()
    roomGain.gain.value = 0
    roomGain.gain.setTargetAtTime(0.11, c.currentTime, 1.5) // fade in
    room.connect(lp)
    lp.connect(roomGain)
    roomGain.connect(master)
    roomGain.connect(reverb)
    room.start()
    lfo.start()
  } catch {
    ctx = null
  }
}

export function stopAudio() {
  const c = ctx
  ctx = master = reverb = noise = null
  if (c) void c.close().catch(() => {})
}

function env(g: GainNode, t: number, peak: number, attack: number, release: number) {
  g.gain.setValueAtTime(0.0001, t)
  g.gain.exponentialRampToValueAtTime(peak, t + attack)
  g.gain.exponentialRampToValueAtTime(0.0001, t + attack + release)
}

function out(node: AudioNode, wet = 0.3) {
  node.connect(master!)
  if (wet > 0) {
    const s = ctx!.createGain()
    s.gain.value = wet
    node.connect(s)
    s.connect(reverb!)
  }
}

function noiseBurst(t: number, f: number, q: number, peak: number, release: number, wet: number) {
  const c = ctx!
  const src = c.createBufferSource()
  src.buffer = noise
  const bp = c.createBiquadFilter()
  bp.type = 'bandpass'
  bp.frequency.value = f
  bp.Q.value = q
  const g = c.createGain()
  env(g, t, peak, 0.004, release)
  src.connect(bp)
  bp.connect(g)
  out(g, wet)
  src.start(t, Math.random() * 1.5, release + 0.05)
}

function tone(t: number, freq: number, peak: number, release: number, type: OscillatorType = 'sine', wet = 0.4, glideTo?: number) {
  const c = ctx!
  const o = c.createOscillator()
  o.type = type
  o.frequency.setValueAtTime(freq, t)
  if (glideTo) o.frequency.exponentialRampToValueAtTime(glideTo, t + release)
  const g = c.createGain()
  env(g, t, peak, 0.006, release)
  o.connect(g)
  out(g, wet)
  o.start(t)
  o.stop(t + release + 0.1)
}

/** Play a one-shot effect. `v` scales volume, `pitch` scales chime pitch. */
export function sfx(name: Sfx, opts: { v?: number; pitch?: number } = {}) {
  if (!ctx || !master || !noise || muted) return
  const t = ctx.currentTime + 0.005
  const v = opts.v ?? 1
  const p = opts.pitch ?? 1
  switch (name) {
    case 'step':
      noiseBurst(t, 260 + Math.random() * 160, 1.2, 0.05 * v, 0.07, 0.15)
      break
    case 'tick':
      noiseBurst(t, 3200, 6, 0.03 * v, 0.03, 0.2)
      break
    case 'click':
      noiseBurst(t, 2400, 4, 0.07 * v, 0.04, 0.25)
      tone(t, 1800, 0.02 * v, 0.05, 'square', 0.1)
      break
    case 'clunk':
      tone(t, 110, 0.22 * v, 0.3, 'sine', 0.35, 48)
      noiseBurst(t, 900, 1.5, 0.08 * v, 0.09, 0.3)
      tone(t + 0.12, 70, 0.1 * v, 0.25, 'triangle', 0.3, 45)
      break
    case 'chime':
      tone(t, 880 * p, 0.07 * v, 1.4, 'sine', 0.6)
      tone(t, 1320 * p, 0.035 * v, 1.0, 'sine', 0.6)
      tone(t, 2640 * p, 0.012 * v, 0.5, 'sine', 0.6)
      break
    case 'slice':
      ;[0, 0.05, 0.1, 0.15].forEach((d, i) => tone(t + d, 660 * 1.25 ** i, 0.035 * v, 0.5, 'sine', 0.5))
      noiseBurst(t, 5000, 0.8, 0.02 * v, 0.2, 0.4)
      break
    case 'whoosh': {
      const c = ctx
      const src = c.createBufferSource()
      src.buffer = noise
      const bp = c.createBiquadFilter()
      bp.type = 'bandpass'
      bp.Q.value = 1.4
      bp.frequency.setValueAtTime(300, t)
      bp.frequency.exponentialRampToValueAtTime(2400, t + 0.5)
      const g = c.createGain()
      env(g, t, 0.05 * v, 0.2, 0.45)
      src.connect(bp)
      bp.connect(g)
      out(g, 0.5)
      src.start(t, 0, 0.8)
      break
    }
  }
}
