import { Prose } from '@/primitives/Prose'
import { SectionHeading } from '@/primitives/SectionHeading'
import { Callout } from '@/primitives/Callout'
import { DeepDive } from '@/primitives/DeepDive'
import { MathBlock } from '@/primitives/MathBlock'
import { DemoFrame } from '@/primitives/DemoFrame'
import { ResidualStream } from '@/demos/ResidualStream'
import { Placard } from '@/primitives/Placard'
import { TAKEAWAYS } from '@/data/takeaways'

export default function Layers() {
  return (
    <>
      <Prose>
        <p>
          Attention gets the attention, but it’s only half of a block. The other half — and a few
          quiet tricks around it — are what let a transformer be stacked dozens of layers deep
          without falling apart.
        </p>
      </Prose>

      <section className="space-y-3">
        <SectionHeading>The feed-forward network</SectionHeading>
        <Prose>
          <p>
            After attention gathers context, each token passes — independently — through a small{' '}
            <strong>feed-forward network</strong>: expand to a much wider hidden size, apply a
            non-linearity, then project back down. It’s simple, but it’s where a surprising amount of
            the model’s learned knowledge and “thinking” happens. Roughly two-thirds of a model’s
            parameters live in these layers.
          </p>
        </Prose>
        <DemoFrame
          title="The residual stream"
          status="illustrative"
          caption="A schematic of how each layer adds its contribution to the running representation."
        >
          <ResidualStream />
        </DemoFrame>
      </section>

      <section className="space-y-3">
        <SectionHeading>The residual stream: a highway</SectionHeading>
        <Prose>
          <p>
            Here’s the crucial design choice. Instead of replacing a token’s vector, each sub-layer{' '}
            <em>adds</em> to it: the output is <code>x + sublayer(x)</code>. Picture a central
            “residual stream” running straight up the stack; every attention and feed-forward layer
            reads from it and writes a small edit back. Because the original signal is always
            preserved by the <code>+ x</code>, information — and the gradients used in training — can
            travel through a hundred layers without dissolving.
          </p>
        </Prose>
        <Callout variant="insight" title="Working memory">
          Think of the residual stream as the model’s working memory. Each layer makes an incremental
          revision rather than starting over — which is exactly why depth keeps helping.
        </Callout>
      </section>

      <section className="space-y-3">
        <SectionHeading>Keeping the numbers sane</SectionHeading>
        <Prose>
          <p>
            Stack enough operations and values tend to explode or vanish. <strong>Layer
            normalization</strong> rescales each vector to a consistent range before each sub-layer,
            which is what makes training very deep networks stable and fast. It’s unglamorous
            plumbing, but without it the whole tower would wobble.
          </p>
        </Prose>
      </section>

      <DeepDive title="Show the math: one block, end to end">
        <p>A modern (“pre-norm”) block computes, for the sequence of vectors <MathBlock>{String.raw`x`}</MathBlock>:</p>
        <MathBlock display>{String.raw`h = x + \text{Attention}(\text{LN}(x))`}</MathBlock>
        <MathBlock display>{String.raw`y = h + \text{FFN}(\text{LN}(h))`}</MathBlock>
        <p>where the feed-forward network is two linear layers with a non-linearity in between:</p>
        <MathBlock display>{String.raw`\text{FFN}(z) = W_2 \,\sigma(W_1 z + b_1) + b_2`}</MathBlock>
        <p>
          The two <code>+</code> signs are the residual connections. <MathBlock>{String.raw`\text{LN}`}</MathBlock>{' '}
          is layer norm, and <MathBlock>{String.raw`\sigma`}</MathBlock> is an activation such as GELU.
          Notice the hidden size inside <MathBlock>{String.raw`W_1`}</MathBlock> is typically about 4×
          wider than <MathBlock>{String.raw`d_{\text{model}}`}</MathBlock>.
        </p>
      </DeepDive>

      <Placard takeaways={TAKEAWAYS['layers']} />
    </>
  )
}
