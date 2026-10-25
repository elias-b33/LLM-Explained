import { DemoFrame } from '@/primitives/DemoFrame'
import { MathBlock } from '@/primitives/MathBlock'
import { Prose } from '@/primitives/Prose'
import { NextTokenPredictor } from '@/demos/NextTokenPredictor'
import type { FocusTarget } from '@/three/museum/halls'

/**
 * Maps a station's demoKey to its content. Lazily imported by StationFocus so the
 * demos (and KaTeX) only load when a station is first opened — never on the
 * museum's initial load.
 */
export function DemoBody({ target }: { target: FocusTarget }) {
  switch (target.demoKey) {
    case 'next-token':
      return (
        <DemoFrame
          title="Next-token predictor"
          status="illustrative"
          caption="A feel for the idea using curated probabilities — not a live model."
        >
          <NextTokenPredictor />
        </DemoFrame>
      )
    case 'orientation-math':
      return (
        <div className="space-y-1">
          <Prose>
            <p>Formally, the model defines a conditional distribution over the next token:</p>
          </Prose>
          <MathBlock display>{String.raw`P(x_t \mid x_1, x_2, \dots, x_{t-1})`}</MathBlock>
          <Prose>
            <p>
              The probability of an entire sequence is the product of these step-by-step predictions —
              the chain rule of probability:
            </p>
          </Prose>
          <MathBlock display>{String.raw`P(x_1, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})`}</MathBlock>
          <Prose>
            <p>
              Training simply nudges the weights so the tokens that actually appeared in real text are
              assigned high probability.
            </p>
          </Prose>
        </div>
      )
    default:
      return <p className="text-fg-muted">This exhibit is coming together.</p>
  }
}
