import type { ReactNode } from 'react'
import type { DemoStatus } from '@/types'
import { Prose } from '@/primitives/Prose'
import { SectionHeading } from '@/primitives/SectionHeading'
import { Callout } from '@/primitives/Callout'
import { Placard } from '@/primitives/Placard'
import { TAKEAWAYS } from '@/data/takeaways'

/**
 * A hall is an ordered spatial sequence placed down the room's depth (you walk
 * in −z). Prose/callouts/placards become wall panels; demos & heavy math become
 * stations that open a focused 2D overlay; a doorway sits at the far end.
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
  | {
      kind: 'curtain'
      side: 'left' | 'right'
      z: number
      len: number
      mode: 'wall' | 'partition'
    }

export interface Hall {
  room: { width: number; depth: number; height: number }
  accent: string
  accent2: string
  reflectiveFloor?: boolean
  segments: Segment[]
  decor?: Decor[]
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
        side: 'left',
        z: -9,
        content: (
          <>
            <SectionHeading>What is an LLM, really?</SectionHeading>
            <Prose>
              <p>
                In one sentence: an LLM is a function that takes a sequence of <strong>tokens</strong> and returns a{' '}
                <strong>probability for every token that could come next</strong>. Everything else exists to make that
                one prediction good.
              </p>
            </Prose>
          </>
        ),
      },
      {
        kind: 'panel',
        side: 'right',
        z: -15,
        content: (
          <>
            <SectionHeading>An extremely good autocomplete</SectionHeading>
            <Prose>
              <p>
                Give it <code>The capital of France is</code> and it spreads probability across its vocabulary —{' '}
                <code> Paris</code> very likely, <code> banana</code> almost impossible. It never looks anything up; it
                only ever predicts the next token.
              </p>
            </Prose>
          </>
        ),
      },
      {
        kind: 'station',
        z: -15,
        demoKey: 'next-token',
        title: 'Next-token predictor',
        status: 'illustrative',
      },
      {
        kind: 'panel',
        side: 'left',
        z: -21,
        content: (
          <>
            <SectionHeading>One prediction becomes a paragraph</SectionHeading>
            <Prose>
              <p>
                To write, it runs in a loop: pick a token, append it, feed everything back in, predict again. That loop
                is <strong>autoregression</strong>.
              </p>
            </Prose>
            <Callout variant="insight" title="Nothing plans ahead">
              There is no outline. Fluent paragraphs emerge from repeatedly answering one narrow question:{' '}
              <em>what token comes next?</em>
            </Callout>
          </>
        ),
      },
      {
        kind: 'panel',
        side: 'right',
        z: -27,
        content: (
          <>
            <SectionHeading>So where does the intelligence come from?</SectionHeading>
            <Prose>
              <p>
                Predicting the next token <em>well</em>, across trillions of words, secretly demands grammar, facts,
                tone, and reasoning. Competence is a side effect of <strong>compression</strong> — squeezing the
                regularities of human writing into a fixed set of weights.
              </p>
            </Prose>
          </>
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
        w: 4.6,
        h: 3.2,
        content: <Placard takeaways={TAKEAWAYS['orientation']} />,
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
      { kind: 'table', z0: -6.8, z1: -11.4 },
      {
        kind: 'vitrine',
        x: 5.3,
        z: -9,
        exhibit: 'attention',
        caption: 'Attention — which token looks at which',
      },
      {
        kind: 'vitrine',
        x: -5.3,
        z: -15,
        exhibit: 'embedding',
        caption: 'Embeddings — meaning as position',
      },
      { kind: 'screen', z: -24 },
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
  },
}
