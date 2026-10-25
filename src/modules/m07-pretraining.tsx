import { Prose } from '@/primitives/Prose'
import { SectionHeading } from '@/primitives/SectionHeading'
import { Callout } from '@/primitives/Callout'
import { DeepDive } from '@/primitives/DeepDive'
import { MathBlock } from '@/primitives/MathBlock'
import { DemoFrame } from '@/primitives/DemoFrame'
import { GradientDescent } from '@/demos/GradientDescent'
import { Placard } from '@/primitives/Placard'
import { TAKEAWAYS } from '@/data/takeaways'

export default function Pretraining() {
  return (
    <>
      <Prose>
        <p>
          We’ve seen how a trained model computes. But how does it <em>become</em> trained? All of its
          knowledge gets poured in during <strong>pretraining</strong> — and the recipe is
          astonishingly simple: predict the next token, over and over, on a large fraction of the
          written internet.
        </p>
      </Prose>

      <section className="space-y-3">
        <SectionHeading>One objective, no labels needed</SectionHeading>
        <Prose>
          <p>
            Take a sentence, hide everything after some point, and ask the model to predict the next
            token. The correct answer is simply the token that really came next — so the data labels
            itself. This is called <strong>self-supervised</strong> learning, and it’s why models can
            train on trillions of tokens without an army of human annotators.
          </p>
        </Prose>
      </section>

      <section className="space-y-3">
        <SectionHeading>Loss = surprise</SectionHeading>
        <Prose>
          <p>
            How do we score a prediction? With <strong>cross-entropy loss</strong>, which you can read
            as <em>surprise</em>. If the model gave the true next token a high probability, the loss is
            low; if it confidently expected something else, the loss is high. The whole goal of
            training is to make the model less surprised by real text.
          </p>
        </Prose>
        <DemoFrame
          title="Gradient descent, rolling downhill"
          status="illustrative"
          caption="The descent dynamics are real arithmetic on a toy loss surface — the landscape is a teaching prop, not a real model’s."
        >
          <GradientDescent />
        </DemoFrame>
      </section>

      <section className="space-y-3">
        <SectionHeading>Learning by nudging weights</SectionHeading>
        <Prose>
          <p>
            Each prediction produces a loss. <strong>Backpropagation</strong> works out how much every
            one of the model’s billions of weights contributed to that loss, and{' '}
            <strong>gradient descent</strong> nudges each weight a tiny step in the direction that
            would have reduced it. Picture a ball rolling down a hill toward lower loss. Repeat across
            trillions of tokens and the weights slowly arrange themselves into something that
            understands language.
          </p>
        </Prose>
        <Callout variant="insight" title="Why scale matters">
          More data, more parameters, and more compute reliably produce better models — a relationship
          steady enough to be called <strong>scaling laws</strong>. Push far enough and new abilities
          (arithmetic, translation, coding) appear that weren’t explicitly trained for.
        </Callout>
      </section>

      <DeepDive title="Show the math: cross-entropy and the update rule">
        <p>
          For a single position, with the model’s predicted probability of the true token{' '}
          <MathBlock>{String.raw`p_{\text{true}}`}</MathBlock>, the loss is:
        </p>
        <MathBlock display>{String.raw`\mathcal{L} = -\log p_{\text{true}}`}</MathBlock>
        <p>
          A confident, correct prediction (<MathBlock>{String.raw`p_{\text{true}} \approx 1`}</MathBlock>)
          gives loss near 0; a near-zero probability gives a huge loss. Averaging over all positions
          gives the training loss. Gradient descent then updates every weight{' '}
          <MathBlock>{String.raw`\theta`}</MathBlock> using the gradient of the loss:
        </p>
        <MathBlock display>{String.raw`\theta \leftarrow \theta - \eta \, \nabla_{\theta}\mathcal{L}`}</MathBlock>
        <p>
          where <MathBlock>{String.raw`\eta`}</MathBlock> is the learning rate — the size of each
          downhill step.
        </p>
      </DeepDive>

      <Placard takeaways={TAKEAWAYS['pretraining']} />
    </>
  )
}
