import { Prose } from '@/primitives/Prose'
import { SectionHeading } from '@/primitives/SectionHeading'
import { Callout } from '@/primitives/Callout'
import { DeepDive } from '@/primitives/DeepDive'
import { DemoFrame } from '@/primitives/DemoFrame'
import { TokenizerPlayground } from '@/demos/TokenizerPlayground'
import { Placard } from '@/primitives/Placard'
import { TAKEAWAYS } from '@/data/takeaways'

export default function Tokenization() {
  return (
    <>
      <Prose>
        <p>
          A model can’t do arithmetic on letters. Before anything else, your text has to become
          numbers. That translation step is <strong>tokenization</strong>, and the choices it makes
          ripple through everything the model does — including how much your prompt costs and where
          its weaknesses show.
        </p>
      </Prose>

      <section className="space-y-3">
        <SectionHeading>Why not just use words?</SectionHeading>
        <Prose>
          <p>
            The obvious idea — one number per word — breaks down fast. A vocabulary of every English
            word is enormous, and it still chokes on names, typos, code, emoji, and other languages
            (the dreaded “out-of-vocabulary” problem). The opposite extreme — one token per character
            — never runs out of vocabulary, but it makes sequences painfully long and forces the
            model to relearn from scratch that <code>t-h-e</code> spells a word.
          </p>
        </Prose>
      </section>

      <section className="space-y-3">
        <SectionHeading>Sub-words: the happy medium</SectionHeading>
        <Prose>
          <p>
            Modern models use <strong>sub-word</strong> tokens. Common words like <code> the</code>
            get their own single token; rarer words are broken into reusable pieces. The word
            <code>tokenization</code> might split into <code>token</code> + <code>ization</code>. This
            keeps the vocabulary a manageable size — often 50,000 to 200,000 tokens — while still
            being able to spell out anything, one piece at a time.
          </p>
        </Prose>
        <DemoFrame
          title="BPE tokenizer playground"
          status="real"
          caption="This runs a real tokenizer in your browser, so every token and id below is genuine."
        >
          <TokenizerPlayground />
        </DemoFrame>
      </section>

      <section className="space-y-3">
        <SectionHeading>The things that surprise people</SectionHeading>
        <Prose>
          <ul>
            <li>
              <strong>Spaces matter.</strong> <code> dog</code> (with a leading space) and{' '}
              <code>dog</code> are usually different tokens.
            </li>
            <li>
              <strong>Case matters.</strong> <code>The</code> and <code>the</code> can tokenize
              differently.
            </li>
            <li>
              <strong>Numbers fragment.</strong> <code>1234</code> may be several tokens — part of
              why models can fumble arithmetic.
            </li>
            <li>
              <strong>Rule of thumb:</strong> one token ≈ 0.75 English words, or about 4 characters.
            </li>
          </ul>
        </Prose>
        <Callout variant="info">
          Token counts are why context limits and API pricing are measured in tokens, not words —
          and why the same sentence can cost more to express in some languages than others.
        </Callout>
      </section>

      <DeepDive title="Show the math: how BPE builds its vocabulary">
        <p>
          The most common scheme is <strong>Byte-Pair Encoding</strong>. It starts from raw
          bytes/characters and repeatedly applies a single rule:
        </p>
        <ul>
          <li>Scan the corpus and count every adjacent pair of symbols.</li>
          <li>Merge the most frequent pair into one new symbol, and add it to the vocabulary.</li>
          <li>Repeat — thousands of times.</li>
        </ul>
        <p>
          After enough merges, frequent sequences (<code>ing</code>, <code> the</code>, whole common
          words) become single tokens, while rare strings remain splittable into smaller parts.
          Encoding new text just replays the learned merges, in the order they were created. Working
          on raw <em>bytes</em> guarantees the tokenizer can encode literally any input — no
          character is ever truly unknown.
        </p>
      </DeepDive>

      <Placard takeaways={TAKEAWAYS['tokenization']} />
    </>
  )
}
