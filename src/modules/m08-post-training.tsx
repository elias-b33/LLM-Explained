import { Prose } from '@/primitives/Prose'
import { SectionHeading } from '@/primitives/SectionHeading'
import { Callout } from '@/primitives/Callout'
import { DeepDive } from '@/primitives/DeepDive'
import { DemoFrame } from '@/primitives/DemoFrame'
import { BaseVsInstruct } from '@/demos/BaseVsInstruct'
import { Placard } from '@/primitives/Placard'
import { TAKEAWAYS } from '@/data/takeaways'

export default function PostTraining() {
  return (
    <>
      <Prose>
        <p>
          A freshly pretrained model is brilliant and almost unusable. It has read the internet, but
          it doesn’t know it’s supposed to be a helpful assistant. Ask <code>How do I bake bread?</code>{' '}
          and a raw base model might reply with a list of <em>other</em> bread questions — because that’s
          a plausible continuation of text it saw online. <strong>Post-training</strong> is what turns
          this text-completer into something you’d want to talk to.
        </p>
      </Prose>

      <section className="space-y-3">
        <SectionHeading>Step 1 — Supervised fine-tuning</SectionHeading>
        <Prose>
          <p>
            First, the model is fine-tuned on a curated set of example conversations: a prompt paired
            with a high-quality, helpful response, written or vetted by people. This{' '}
            <strong>supervised fine-tuning</strong> (SFT) teaches the <em>format</em> and{' '}
            <em>register</em> of being an assistant — answer the question, be clear, follow
            instructions. It’s not adding new facts so much as shaping how the existing knowledge comes
            out.
          </p>
        </Prose>
        <DemoFrame
          title="Base model vs. assistant"
          status="illustrative"
          caption="Curated examples of the same prompt before and after alignment — illustrative, not generated live."
        >
          <BaseVsInstruct />
        </DemoFrame>
      </section>

      <section className="space-y-3">
        <SectionHeading>Step 2 — Learning from preferences (RLHF)</SectionHeading>
        <Prose>
          <p>
            SFT gets you a decent assistant; preference learning makes it genuinely good. The model
            generates several candidate responses; humans (or a <strong>reward model</strong> trained
            to imitate human judgement) rank which is better. The model is then optimized to produce
            responses that score higher — a loop known as <strong>RLHF</strong> (reinforcement learning
            from human feedback). A popular, simpler variant called <strong>DPO</strong> achieves much
            the same thing by directly training on the preferred-vs-rejected pairs.
          </p>
        </Prose>
        <Callout variant="insight" title="Behavior, not knowledge">
          Alignment mostly reshapes <em>behavior</em> — helpfulness, honesty, harmlessness, tone. The
          raw knowledge was already baked in during pretraining; post-training decides how that
          knowledge is expressed.
        </Callout>
      </section>

      <section className="space-y-3">
        <SectionHeading>The system prompt</SectionHeading>
        <Prose>
          <p>
            One more lever: a hidden <strong>system prompt</strong> placed before the conversation sets
            the assistant’s persona and rules (“You are a helpful assistant. Be concise…”). Because the
            model conditions on everything in its context, those few instructions steer the entire
            exchange that follows.
          </p>
        </Prose>
      </section>

      <DeepDive title="Going deeper: the RLHF loop">
        <p>RLHF has three moving parts:</p>
        <ul>
          <li>
            <strong>Collect preferences.</strong> For many prompts, gather pairs of responses and a
            human judgement of which is better.
          </li>
          <li>
            <strong>Train a reward model.</strong> A separate model learns to predict that human
            preference — effectively a scorer for “how good is this response?”
          </li>
          <li>
            <strong>Optimize the policy.</strong> The language model is updated to maximize the reward,
            while a penalty keeps it from drifting too far from the sensible SFT model.
          </li>
        </ul>
        <p>
          DPO collapses this into a single training objective on the preference pairs, skipping the
          separate reward model — simpler and very effective in practice.
        </p>
      </DeepDive>

      <Placard takeaways={TAKEAWAYS['post-training']} />
    </>
  )
}
