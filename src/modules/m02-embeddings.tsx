import { Prose } from '@/primitives/Prose'
import { SectionHeading } from '@/primitives/SectionHeading'
import { Callout } from '@/primitives/Callout'
import { DeepDive } from '@/primitives/DeepDive'
import { MathBlock } from '@/primitives/MathBlock'
import { DemoFrame } from '@/primitives/DemoFrame'
import { EmbeddingMap } from '@/demos/EmbeddingMap'
import { EmbeddingSpace3D } from '@/three/EmbeddingSpace3D'
import { ImmersiveLink } from '@/app/ImmersiveLink'
import { Placard } from '@/primitives/Placard'
import { TAKEAWAYS } from '@/data/takeaways'

export default function Embeddings() {
  return (
    <>
      <Prose>
        <p>
          A token id like <code>1531</code> is just a name tag — on its own it carries no meaning
          (1531 isn’t “more” than 1530 in any useful sense). The model’s first real move is to replace
          each token id with a <strong>vector</strong>: a long list of numbers that <em>can</em> hold
          meaning.
        </p>
      </Prose>

      <section className="space-y-3">
        <SectionHeading>Meaning as a direction in space</SectionHeading>
        <Prose>
          <p>
            Each token gets a learned <strong>embedding</strong> — a point in a high-dimensional
            space (hundreds or thousands of dimensions). The magic is in the geometry: tokens with
            similar meanings end up close together, and clusters form for animals, countries, verbs,
            and so on. Nobody tells the model that <code>cat</code> and <code>dog</code> are related;
            it discovers this because they behave similarly when predicting text.
          </p>
        </Prose>
        <DemoFrame
          title="The embedding map"
          status="illustrative"
          caption="A 2-D projection of real high-dimensional structure: positions show clustering, distances are approximate."
        >
          <EmbeddingMap />
        </DemoFrame>
        <DemoFrame
          title="The same space, in 3-D"
          status="illustrative"
          caption="The same idea in three dimensions — drag to orbit. Real embeddings have hundreds of dimensions; this is just a window into that."
        >
          <EmbeddingSpace3D />
        </DemoFrame>
        <div className="flex justify-center pt-2">
          <ImmersiveLink />
        </div>
      </section>

      <section className="space-y-3">
        <SectionHeading>Analogies become arithmetic</SectionHeading>
        <Prose>
          <p>
            Because directions carry meaning, you can do algebra with concepts. The famous example:
            take the vector for <em>king</em>, subtract <em>man</em>, add <em>woman</em>, and you land
            right next to <em>queen</em>.
          </p>
        </Prose>
        <MathBlock display>{String.raw`\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}`}</MathBlock>
        <Prose>
          <p>
            The step from <em>man</em> to <em>woman</em> is roughly the same direction as from{' '}
            <em>king</em> to <em>queen</em> — a “gender” axis the model picked up entirely on its own.
          </p>
        </Prose>
        <Callout variant="insight" title="Learned, not designed">
          Nobody writes down what each dimension means. These directions fall out of the single goal
          of predicting text well — meaning is discovered, not programmed.
        </Callout>
      </section>

      <section className="space-y-3">
        <SectionHeading>Measuring closeness</SectionHeading>
        <Prose>
          <p>
            To ask “how similar are two tokens?”, we compare the <em>angle</em> between their vectors
            using <strong>cosine similarity</strong> — 1 means the same direction, 0 means unrelated,
            −1 means opposite. Length is ignored; only direction matters.
          </p>
        </Prose>
      </section>

      <DeepDive title="Show the math: embeddings and similarity">
        <p>
          All embeddings live in one matrix <MathBlock>{String.raw`E`}</MathBlock> of shape
          (vocabulary × <MathBlock>{String.raw`d_{\text{model}}`}</MathBlock>). Looking up a token is
          just selecting its row — equivalently, multiplying a one-hot vector by{' '}
          <MathBlock>{String.raw`E`}</MathBlock>. Cosine similarity between two vectors is:
        </p>
        <MathBlock display>
          {String.raw`\text{sim}(\vec{a}, \vec{b}) = \frac{\vec{a} \cdot \vec{b}}{\lVert \vec{a} \rVert \, \lVert \vec{b} \rVert}`}
        </MathBlock>
        <p>
          One more wrinkle: the transformer also needs to know token <em>order</em>, so a{' '}
          <strong>positional</strong> signal is added to each embedding. Meaning and position both
          ride inside the vectors that flow into the next stage.
        </p>
      </DeepDive>

      <Placard takeaways={TAKEAWAYS['embeddings']} />
    </>
  )
}
