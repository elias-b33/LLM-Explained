import type { ReactNode } from 'react'
import type { DemoStatus } from '@/types'
import { SectionHeading } from '@/primitives/SectionHeading'

/**
 * A hall is an ordered spatial sequence placed down the room's depth (you walk
 * in −z). Wall panels carry only a headline + one line (the full text lives in
 * the 2D reading mode, src/modules/*); ideas are taught by in-world exhibits
 * you operate (decor kinds slicer / embedding / attention / machine); demos &
 * heavy math become stations that open a focused 2D overlay; a doorway sits at
 * the far end.
 * Authoring the other 10 halls is pure data against this model — no new code.
 */
export type Segment =
  | {
      kind: 'panel'
      side: 'left' | 'right' | 'back'
      z: number
      w?: number
      h?: number
      content: ReactNode
    }
  | {
      kind: 'station'
      /** Defaults to the nave centre line. */
      x?: number
      z: number
      demoKey: string
      title: string
      status: DemoStatus
    }
  | { kind: 'doorway'; z: number; dir: 'next' | 'prev' }

/** Non-interactive set dressing: vitrines with hologram exhibits, the central
 *  display table, the suspended scrim and sheer curtains. */
export type Exhibit = 'gpu' | 'tokens' | 'attention' | 'embedding'
export type Decor =
  | { kind: 'vitrine'; x: number; z: number; exhibit: Exhibit; caption: string }
  | { kind: 'table'; z0: number; z1: number }
  | { kind: 'screen'; z: number }
  /** Interactive exhibits (look + E). */
  | { kind: 'slicer'; x: number; z: number }
  | { kind: 'embedding'; x: number; z: number }
  | { kind: 'attention'; z: number }
  | { kind: 'machine'; z: number; consoleZ: number }
  | {
      kind: 'curtain'
      side: 'left' | 'right'
      z: number
      len: number
      mode: 'wall' | 'partition'
    }

/** A stop on the "exhibits explored" tracker; x/z is where to stand. */
export interface TourStop {
  id: string
  label: string
  x: number
  z: number
}

export interface Hall {
  room: { width: number; depth: number; height: number }
  accent: string
  accent2: string
  reflectiveFloor?: boolean
  segments: Segment[]
  decor?: Decor[]
  tour?: TourStop[]
}

/** A short wall placard: big headline + one line. */
function Headline({ title, line }: { title: string; line: string }) {
  return (
    <>
      <SectionHeading>{title}</SectionHeading>
      <p style={{ marginTop: 10, fontSize: 17, lineHeight: 1.45, color: 'var(--fg-muted)' }}>{line}</p>
    </>
  )
}

/** What a station opens in the focused 2D overlay. */
export interface FocusTarget {
  demoKey: string
  title: string
  status: DemoStatus
}

export const HALLS: Record<string, Hall> = {
  orientation: {
    room: { width: 14, depth: 48, height: 4.6 },
    accent: '#8b7bff',
    accent2: '#2bd3e0',
    segments: [
      {
        kind: 'panel',
        side: 'right',
        z: -15,
        w: 2.8,
        content: (
          <Headline title="Every word looks back" line="Each word gathers meaning from earlier words. Look at a floor tile and press E." />
        ),
      },
      {
        kind: 'station',
        x: -5.0,
        z: -15,
        demoKey: 'next-token',
        title: 'Next-token predictor',
        status: 'illustrative',
      },
      {
        kind: 'panel',
        side: 'left',
        z: -21,
        w: 2.8,
        content: (
          <Headline
            title="One token at a time"
            line="Pick a token, append it, predict again. That loop, autoregression, writes every reply."
          />
        ),
      },
      {
        kind: 'panel',
        side: 'right',
        z: -27,
        w: 2.8,
        content: (
          <Headline
            title="Where does the intelligence come from?"
            line="Predicting the next token well, across trillions of words, forces a model to learn grammar, facts and reasoning."
          />
        ),
      },
      {
        kind: 'station',
        z: -33,
        demoKey: 'orientation-math',
        title: 'The next-token objective',
        status: 'real',
      },
      {
        kind: 'panel',
        side: 'left',
        z: -39,
        w: 2.8,
        content: (
          <Headline
            title="Tokens in, probabilities out"
            line="An LLM only ever scores the next token. Run that in a loop and you get language."
          />
        ),
      },
      { kind: 'doorway', z: -47.6, dir: 'next' },
    ],
    decor: [
      {
        kind: 'vitrine',
        x: -5.3,
        z: -3.4,
        exhibit: 'gpu',
        caption: 'Accelerator die — where the matrix math runs',
      },
      {
        kind: 'vitrine',
        x: 5.3,
        z: -3.4,
        exhibit: 'tokens',
        caption: 'Tokens — text, cut into pieces',
      },
      { kind: 'slicer', x: -5.2, z: -9 },
      { kind: 'embedding', x: 5.3, z: -9 },
      { kind: 'attention', z: -15 },
      { kind: 'machine', z: -24, consoleZ: -21.7 },
      { kind: 'curtain', side: 'right', z: -21, len: 5.4, mode: 'wall' },
      { kind: 'curtain', side: 'left', z: -27, len: 5.4, mode: 'wall' },
      { kind: 'curtain', side: 'left', z: -30, len: 1.9, mode: 'partition' },
      { kind: 'curtain', side: 'right', z: -30, len: 1.9, mode: 'partition' },
      { kind: 'curtain', side: 'right', z: -33, len: 5.4, mode: 'wall' },
      { kind: 'curtain', side: 'left', z: -33, len: 5.4, mode: 'wall' },
      {
        kind: 'vitrine',
        x: 5.3,
        z: -39,
        exhibit: 'gpu',
        caption: 'A model is just numbers on silicon',
      },
      { kind: 'curtain', side: 'left', z: -45, len: 5.4, mode: 'wall' },
      { kind: 'curtain', side: 'right', z: -45, len: 5.4, mode: 'wall' },
    ],
    tour: [
      { id: 'tokens', label: 'Token slicer', x: -3.9, z: -9 },
      { id: 'embeddings', label: 'Meaning space', x: 3.8, z: -9 },
      { id: 'attention', label: 'Attention walkway', x: 0, z: -13.6 },
      { id: 'lever', label: 'Autoregression lever', x: -0.5, z: -20.9 },
      { id: 'temperature', label: 'Temperature dial', x: 0.45, z: -20.9 },
    ],
  },
}
