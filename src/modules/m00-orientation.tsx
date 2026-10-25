import { Prose } from '@/primitives/Prose'
import { SectionHeading } from '@/primitives/SectionHeading'
import { Callout } from '@/primitives/Callout'
import { DeepDive } from '@/primitives/DeepDive'
import { MathBlock } from '@/primitives/MathBlock'
import { DemoFrame } from '@/primitives/DemoFrame'
import { NextTokenPredictor } from '@/demos/NextTokenPredictor'
import { Placard } from '@/primitives/Placard'
import { TAKEAWAYS } from '@/data/takeaways'

export default function Orientation() {
  return (
    <>
      <Prose>
        <p>
          Here is the whole idea in one sentence: <strong>a large language model is a function that
          takes a sequence of tokens and returns a probability for every token that could come
          next.</strong> Everything else — the billions of parameters, the attention layers, the
          training runs that cost millions — exists only to make that one prediction good.
        </p>
      </Prose>

      <section className="space-y-3">
        <SectionHeading>An extremely good autocomplete</SectionHeading>
        <Prose>
          <p>
            Give a model the text <code>The capital of France is</code> and it does not look up an
            answer in a table. It estimates that <code> Paris</code> is a very likely next token,
            <code> Lyon</code> much less likely, and <code> banana</code> almost impossible. The
            model never sees words the way you do — it only ever spreads probability across its fixed
            vocabulary of tokens.
          </p>
        </Prose>
        <DemoFrame
          title="Next-token predictor"
          status="illustrative"
          caption="A feel for the idea using curated probabilities — not a live model."
        >
          <NextTokenPredictor />
        </DemoFrame>
      </section>

      <section className="space-y-3">
        <SectionHeading>How one prediction becomes a paragraph</SectionHeading>
        <Prose>
          <p>
            To actually write, the model runs in a loop. It produces a distribution, one token is
            chosen from it, that token is appended to the input, and the whole sequence is fed back
            in to predict the next one. This is <strong>autoregression</strong>: generating one token
            at a time, each conditioned on everything written so far.
          </p>
        </Prose>
        <Callout variant="insight" title="Nothing plans ahead">
          The model has no outline of the sentence it is about to write. Fluent, coherent paragraphs
          emerge from repeatedly answering one narrow question: <em>what token comes next?</em>
        </Callout>
      </section>

      <section className="space-y-3">
        <SectionHeading>So where does the intelligence come from?</SectionHeading>
        <Prose>
          <p>
            If the task is so simple, why are these models so capable? Because predicting the next
            token <em>well</em>, across trillions of words of human text, secretly demands
            understanding a great deal: grammar, facts, tone, and chains of reasoning. To finish
            <code> The detective finally realised the murderer was…</code> the model has to have
            tracked the whole story. Competence is a side effect of compression — squeezing the
            regularities of human writing into a fixed set of weights.
          </p>
        </Prose>
      </section>

      <section className="space-y-3">
        <SectionHeading>The road ahead</SectionHeading>
        <Prose>
          <p>Over the next modules we will open the box, following the data as it flows through:</p>
          <ul>
            <li>
              <strong>Tokenization</strong> — turning text into tokens.
            </li>
            <li>
              <strong>Embeddings</strong> — turning tokens into vectors that carry meaning.
            </li>
            <li>
              <strong>The transformer &amp; attention</strong> — how the model mixes information
              between tokens.
            </li>
            <li>
              <strong>Logits &amp; sampling</strong> — turning the final vectors back into a choice
              of token.
            </li>
            <li>
              <strong>Training &amp; alignment</strong> — how it learned all this, and how it became
              a helpful assistant.
            </li>
          </ul>
        </Prose>
      </section>

      <DeepDive title="Show the math: the next-token objective">
        <p>Formally, the model defines a conditional distribution over the next token:</p>
        <MathBlock display>{String.raw`P(x_t \mid x_1, x_2, \dots, x_{t-1})`}</MathBlock>
        <p>
          The probability of an entire sequence is the product of these step-by-step predictions —
          the chain rule of probability:
        </p>
        <MathBlock display>{String.raw`P(x_1, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})`}</MathBlock>
        <p>
          Training simply nudges the weights so that the tokens which actually appeared in real text
          are assigned high probability.
        </p>
      </DeepDive>

      <Placard takeaways={TAKEAWAYS['orientation']} />
    </>
  )
}
