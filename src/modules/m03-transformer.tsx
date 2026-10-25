import { Prose } from '@/primitives/Prose'
import { SectionHeading } from '@/primitives/SectionHeading'
import { Callout } from '@/primitives/Callout'
import { DeepDive } from '@/primitives/DeepDive'
import { MathBlock } from '@/primitives/MathBlock'
import { DemoFrame } from '@/primitives/DemoFrame'
import { TransformerFlow } from '@/demos/TransformerFlow'
import { Placard } from '@/primitives/Placard'
import { TAKEAWAYS } from '@/data/takeaways'

export default function Transformer() {
  return (
    <>
      <Prose>
        <p>
          We now have a sequence of vectors — one per token, each carrying meaning and position. The
          <strong> transformer</strong> is the engine that refines those vectors until the last one
          is ready to predict what comes next.
        </p>
      </Prose>

      <section className="space-y-3">
        <SectionHeading>A stack of identical blocks</SectionHeading>
        <Prose>
          <p>
            The architecture is surprisingly repetitive: the same kind of <strong>block</strong> is
            stacked many times (a small model might have 12, a large one over 100). Each block takes
            the sequence of vectors in and produces an improved sequence of vectors out — same shape,
            richer content.
          </p>
        </Prose>
        <DemoFrame
          title="Data flow through the model"
          status="illustrative"
          caption="A schematic of the pipeline — step through to watch a sequence move up the stack."
        >
          <TransformerFlow />
        </DemoFrame>
      </section>

      <section className="space-y-3">
        <SectionHeading>What each block does</SectionHeading>
        <Prose>
          <p>Every block has two distinct jobs, done in order:</p>
          <ul>
            <li>
              <strong>Attention</strong> — let each token look at the other tokens and pull in
              whatever context it needs. This is communication <em>between</em> positions.
            </li>
            <li>
              <strong>Feed-forward network</strong> — process each token on its own, transforming what
              it just gathered. This is computation <em>within</em> a position.
            </li>
          </ul>
        </Prose>
        <Callout variant="insight" title="The two-step rhythm">
          Gather, then think. Attention <em>routes</em> information; the feed-forward layer{' '}
          <em>digests</em> it. Stacking this pair dozens of times is most of what a transformer is.
        </Callout>
      </section>

      <section className="space-y-3">
        <SectionHeading>From vectors back to a prediction</SectionHeading>
        <Prose>
          <p>
            After the final block, each token has a heavily refined vector. To predict the next
            token, the model takes the vector at the <em>last</em> position and applies one more
            linear step — the <strong>unembedding</strong> — turning it into a score (a “logit”) for
            every token in the vocabulary. The next module shows how those scores become probabilities.
          </p>
        </Prose>
      </section>

      <DeepDive title="Show the math: shapes and the stack">
        <p>
          Let the input be <MathBlock>{String.raw`T`}</MathBlock> tokens, each a vector of size{' '}
          <MathBlock>{String.raw`d_{\text{model}}`}</MathBlock>. The whole sequence is a matrix of
          shape <MathBlock>{String.raw`(T \times d_{\text{model}})`}</MathBlock>. A single block is a
          function that preserves that shape:
        </p>
        <MathBlock display>{String.raw`\text{Block}: \mathbb{R}^{T \times d_{\text{model}}} \to \mathbb{R}^{T \times d_{\text{model}}}`}</MathBlock>
        <p>
          Stack <MathBlock>{String.raw`L`}</MathBlock> of them, then multiply the final vectors by an
          unembedding matrix <MathBlock>{String.raw`U`}</MathBlock> of shape{' '}
          <MathBlock>{String.raw`(d_{\text{model}} \times V)`}</MathBlock> to get one logit per token
          in the vocabulary of size <MathBlock>{String.raw`V`}</MathBlock>.
        </p>
      </DeepDive>

      <Placard takeaways={TAKEAWAYS['transformer']} />
    </>
  )
}
