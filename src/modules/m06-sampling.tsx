import { Prose } from '@/primitives/Prose'
import { SectionHeading } from '@/primitives/SectionHeading'
import { Callout } from '@/primitives/Callout'
import { DeepDive } from '@/primitives/DeepDive'
import { MathBlock } from '@/primitives/MathBlock'
import { DemoFrame } from '@/primitives/DemoFrame'
import { SamplingPlayground } from '@/demos/SamplingPlayground'
import { Placard } from '@/primitives/Placard'
import { TAKEAWAYS } from '@/data/takeaways'

export default function Sampling() {
  return (
    <>
      <Prose>
        <p>
          The transformer hands us a vector of <strong>logits</strong> — one raw score per token in
          the vocabulary. Two steps remain: turn those scores into probabilities, then actually pick a
          token. How we pick is where “creativity” and “determinism” come from.
        </p>
      </Prose>

      <section className="space-y-3">
        <SectionHeading>Logits → probabilities with softmax</SectionHeading>
        <Prose>
          <p>
            Logits can be any real numbers. The <strong>softmax</strong> function exponentiates them
            (making everything positive and exaggerating differences) and normalizes so they sum to 1.
            Now we have a genuine probability distribution over the next token.
          </p>
        </Prose>
      </section>

      <section className="space-y-3">
        <SectionHeading>Choosing a token</SectionHeading>
        <Prose>
          <p>Given the distribution, several strategies exist:</p>
          <ul>
            <li>
              <strong>Greedy</strong> — always take the single most likely token. Deterministic, but
              often flat and repetitive.
            </li>
            <li>
              <strong>Temperature (τ)</strong> — divide logits by τ before softmax. Low τ sharpens
              toward the top token; high τ flattens the field and invites surprises. τ → 0 is greedy.
            </li>
            <li>
              <strong>Top-k</strong> — only consider the k most likely tokens, then sample among them.
            </li>
            <li>
              <strong>Top-p (nucleus)</strong> — consider the smallest group of top tokens whose
              probabilities add up to at least p, then sample. The number of candidates flexes with
              the model’s confidence.
            </li>
          </ul>
        </Prose>
        <DemoFrame
          title="Sampling playground"
          status="real"
          caption="The softmax, temperature, top-k, and top-p math here is exact — drag the controls and watch the distribution reshape."
        >
          <SamplingPlayground />
        </DemoFrame>
        <Callout variant="insight" title="Coherence vs. creativity">
          These knobs trade off the same axis. Low temperature and tight top-p give safe, focused
          text; higher values give variety and risk. There’s no single “correct” setting — it depends
          on the task.
        </Callout>
      </section>

      <DeepDive title="Show the math: softmax with temperature">
        <p>With temperature <MathBlock>{String.raw`\tau`}</MathBlock>, the probability of token <MathBlock>{String.raw`i`}</MathBlock> is:</p>
        <MathBlock display>
          {String.raw`p_i = \frac{e^{\,z_i / \tau}}{\sum_j e^{\,z_j / \tau}}`}
        </MathBlock>
        <p>
          As <MathBlock>{String.raw`\tau \to 0`}</MathBlock> the largest logit dominates completely
          (greedy); as <MathBlock>{String.raw`\tau \to \infty`}</MathBlock> all tokens approach equal
          probability (uniform random). <strong>Top-k</strong> zeroes out all but the{' '}
          <MathBlock>{String.raw`k`}</MathBlock> largest probabilities and renormalizes;{' '}
          <strong>top-p</strong> keeps the smallest set with cumulative probability{' '}
          <MathBlock>{String.raw`\ge p`}</MathBlock> and renormalizes the rest to zero.
        </p>
      </DeepDive>

      <Placard takeaways={TAKEAWAYS['sampling']} />
    </>
  )
}
