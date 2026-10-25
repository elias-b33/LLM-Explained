import { Prose } from '@/primitives/Prose'
import { SectionHeading } from '@/primitives/SectionHeading'
import { Callout } from '@/primitives/Callout'
import { DeepDive } from '@/primitives/DeepDive'
import { MathBlock } from '@/primitives/MathBlock'
import { DemoFrame } from '@/primitives/DemoFrame'
import { AttentionExplorer } from '@/demos/AttentionExplorer'
import { AttentionMath } from '@/demos/AttentionMath'
import { Placard } from '@/primitives/Placard'
import { TAKEAWAYS } from '@/data/takeaways'

export default function Attention() {
  return (
    <>
      <Prose>
        <p>
          Attention is the idea that made transformers possible — and it is the part most worth
          understanding deeply. It answers a single question for every token: <em>given what I am,
          which other tokens should I pay attention to, and what should I take from them?</em>
        </p>
      </Prose>

      <section className="space-y-3">
        <SectionHeading>The problem it solves</SectionHeading>
        <Prose>
          <p>
            Consider “The trophy didn’t fit in the suitcase because <strong>it</strong> was too big.”
            To understand <code>it</code>, the model must look back and connect it to{' '}
            <code>trophy</code>, not <code>suitcase</code>. Meaning depends on relationships between
            distant words — and attention is the mechanism that lets a token reach across the
            sequence to find exactly the context it needs.
          </p>
        </Prose>
      </section>

      <section className="space-y-3">
        <SectionHeading>Query, Key, Value</SectionHeading>
        <Prose>
          <p>From each token’s vector, the model computes three new vectors:</p>
          <ul>
            <li>
              <strong>Query</strong> — “what am I looking for?”
            </li>
            <li>
              <strong>Key</strong> — “what do I have to offer?”
            </li>
            <li>
              <strong>Value</strong> — “what information will I hand over if you attend to me?”
            </li>
          </ul>
          <p>
            A token compares its <em>query</em> against every other token’s <em>key</em> using a dot
            product. High match = high relevance. Those scores are squashed through a{' '}
            <strong>softmax</strong> into weights that sum to 1, and the token’s output is the
            weighted sum of all the <em>values</em>. In short: it gathers a custom blend of the
            sequence, tailored to what it was looking for.
          </p>
        </Prose>
        <DemoFrame
          title="Attention heatmap"
          status="illustrative"
          caption="Curated patterns that illustrate the kinds of links attention learns — not weights pulled from a live model."
        >
          <AttentionExplorer />
        </DemoFrame>
        <DemoFrame
          title="Q·K → softmax → weighted V, by hand"
          status="real"
          caption="A tiny example computed live with exact arithmetic, so you can see every number."
        >
          <AttentionMath />
        </DemoFrame>
      </section>

      <section className="space-y-3">
        <SectionHeading>Many heads, many relationships</SectionHeading>
        <Prose>
          <p>
            One attention pattern isn’t enough. Real models run several <strong>heads</strong> in
            parallel, each with its own Query/Key/Value projections. One head might track grammatical
            subject–verb links, another might resolve pronouns, another might follow quotation marks.
            Their outputs are combined, giving each token a rich, multi-angle view of its context.
          </p>
        </Prose>
        <Callout variant="info" title="Looking only backward">
          In a text-generating model, a token may attend only to tokens <em>before</em> it — never the
          future. This “causal mask” is what makes left-to-right generation honest: the model can’t
          peek at words it hasn’t written yet.
        </Callout>
      </section>

      <DeepDive title="Show the math: scaled dot-product attention">
        <p>Packing the queries, keys, and values into matrices, a single head computes:</p>
        <MathBlock display>
          {String.raw`\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V`}
        </MathBlock>
        <p>
          The <MathBlock>{String.raw`Q K^{\top}`}</MathBlock> term is every query dotted with every key
          (a <MathBlock>{String.raw`T \times T`}</MathBlock> grid of scores). Dividing by{' '}
          <MathBlock>{String.raw`\sqrt{d_k}`}</MathBlock> keeps those scores from growing too large as
          the dimension increases, which stabilizes the softmax. Multiple heads run this in parallel
          and their results are concatenated and mixed by one more linear layer.
        </p>
      </DeepDive>

      <Placard takeaways={TAKEAWAYS['attention']} />
    </>
  )
}
