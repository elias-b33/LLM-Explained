import { Prose } from '@/primitives/Prose'
import { SectionHeading } from '@/primitives/SectionHeading'
import { Callout } from '@/primitives/Callout'
import { DeepDive } from '@/primitives/DeepDive'
import { MathBlock } from '@/primitives/MathBlock'
import { DemoFrame } from '@/primitives/DemoFrame'
import { AutoregressiveGen } from '@/demos/AutoregressiveGen'
import { Placard } from '@/primitives/Placard'
import { TAKEAWAYS } from '@/data/takeaways'

export default function Inference() {
  return (
    <>
      <Prose>
        <p>
          You now understand how the model computes a single prediction and how it learned to. This
          last technical module is about what happens when you actually <em>run</em> it — and where its
          limits come from.
        </p>
      </Prose>

      <section className="space-y-3">
        <SectionHeading>Generation, one token at a time</SectionHeading>
        <Prose>
          <p>
            Generation is the autoregressive loop from the very first module, now in full: feed in the
            prompt, predict a token, append it, feed everything back in, predict the next. The model
            never writes ahead — it commits to one token, then reconsiders the whole sequence to choose
            the following one.
          </p>
        </Prose>
        <DemoFrame
          title="Watch it generate"
          status="real"
          caption="Tokenization is real; the choice of each next token is simulated for illustration, not drawn from a live model."
        >
          <AutoregressiveGen />
        </DemoFrame>
      </section>

      <section className="space-y-3">
        <SectionHeading>The KV cache and the context window</SectionHeading>
        <Prose>
          <p>
            Naively, predicting each new token would recompute attention over the entire history — slow
            and wasteful. Instead, the keys and values of past tokens are saved in a{' '}
            <strong>KV cache</strong> and reused, so each step only computes the new token’s query and
            attends to what’s cached. The cost: that cache grows with every token, eating memory.
          </p>
          <p>
            The <strong>context window</strong> is the maximum number of tokens the model can hold at
            once. Run past it and the oldest tokens fall out of view. Bigger windows are powerful but
            expensive — attention cost grows with the <em>square</em> of the sequence length.
          </p>
        </Prose>
        <Callout variant="info" title="Why prompts cost what they do">
          Long inputs and long outputs both cost more, and the KV cache plus context length drive the
          memory a model needs. Tokens, again, are the unit of everything.
        </Callout>
      </section>

      <section className="space-y-3">
        <SectionHeading>Why models hallucinate</SectionHeading>
        <Prose>
          <p>
            A language model is a <strong>plausibility engine</strong>, not a fact database. It
            produces text that is <em>likely</em>, which usually overlaps with what’s true — but not
            always. With no built-in sense of certainty, it can state a confident, fluent falsehood the
            same way it states a fact. Understanding this one point explains most of a model’s failure
            modes.
          </p>
        </Prose>
      </section>

      <section className="space-y-3">
        <SectionHeading>“Reasoning” and chain-of-thought</SectionHeading>
        <Prose>
          <p>
            Ask a model to “think step by step” and it often does better on hard problems. Why?
            Because every token it writes becomes part of the context for the next one. Spelling out
            intermediate steps gives the model more room — more computation and more working memory —
            before it has to commit to a final answer. Modern “reasoning” models are trained to do
            exactly this, at length.
          </p>
        </Prose>
      </section>

      <DeepDive title="Show the math: why attention is the bottleneck">
        <p>
          For a sequence of length <MathBlock>{String.raw`T`}</MathBlock>, attention compares every
          token with every other token, so its cost scales as:
        </p>
        <MathBlock display>{String.raw`\mathcal{O}(T^2 \cdot d)`}</MathBlock>
        <p>
          That quadratic <MathBlock>{String.raw`T^2`}</MathBlock> is why doubling the context roughly
          quadruples the attention work. Two phases differ in character: <strong>prefill</strong>
          processes the whole prompt at once (compute-heavy), while <strong>decode</strong> generates
          one token at a time using the KV cache (memory-bandwidth-heavy). Much of modern
          inference engineering is about taming these two costs.
        </p>
      </DeepDive>

      <Placard takeaways={TAKEAWAYS['inference']} />
    </>
  )
}
