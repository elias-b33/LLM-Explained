/** One-line takeaways per exhibit, shown on the closing museum placard. */
export const TAKEAWAYS: Record<string, string[]> = {
  orientation: [
    'A language model outputs a probability for every possible next token — nothing more, nothing less.',
    'Fluent text emerges from doing that one step over and over, feeding each token back in (autoregression).',
    'Its broad competence is a side effect of compressing the patterns of human writing into fixed weights.',
  ],
  tokenization: [
    'Text is split into sub-word tokens before a model can read it.',
    'Sub-words balance a manageable vocabulary against the ability to spell anything.',
    'Spaces and capitalization are part of tokens — which is why token counts, not word counts, drive cost and context limits.',
  ],
  embeddings: [
    'Each token becomes a vector, and related meanings land near each other in the space.',
    'Relationships show up as consistent directions — so analogies become arithmetic (king − man + woman ≈ queen).',
    'These directions are learned from the prediction task; nobody programs them by hand.',
  ],
  transformer: [
    'A transformer is a tall stack of identical blocks that repeatedly refine the token vectors.',
    'Each block does two jobs: attention gathers context between tokens, then a feed-forward network processes each one.',
    'A final unembedding turns the last vector into a score for every token in the vocabulary.',
  ],
  attention: [
    'Attention lets every token pull in information from every other token.',
    'Each token forms a Query, Key, and Value; query·key scores become softmax weights over the values.',
    'Several heads run in parallel, each specializing in a different kind of relationship.',
  ],
  layers: [
    'The feed-forward network does per-token “thinking” and holds much of the model’s knowledge.',
    'The residual stream is a running sum every layer adds to — so signal and gradients are never lost.',
    'Layer normalization keeps the numbers stable enough to stack very deep.',
  ],
  sampling: [
    'Softmax turns raw logits into a probability distribution over the next token.',
    'Temperature sharpens (low) or flattens (high) that distribution; top-k and top-p trim the long tail.',
    'These knobs trade coherence against creativity — there is no single “correct” setting.',
  ],
  pretraining: [
    'Models learn by predicting the next token over enormous amounts of text — no human labels needed.',
    'Cross-entropy loss measures “surprise”; gradient descent nudges every weight to reduce it.',
    'More data, parameters, and compute reliably yield better models (scaling laws).',
  ],
  'post-training': [
    'A freshly pretrained model is a text-completer, not yet a helpful assistant.',
    'Supervised fine-tuning teaches the assistant format; preference learning (RLHF) makes it genuinely helpful.',
    'Alignment reshapes behavior — the knowledge was already there from pretraining.',
  ],
  inference: [
    'Generation is autoregressive: one token at a time, each fed back into a growing context.',
    'The KV cache makes this efficient; the context window and quadratic attention bound how much it can hold.',
    'A model is a plausibility engine, not a fact database — which is why it can hallucinate.',
  ],
  capstone: [
    'Text → tokens → embeddings → transformer blocks → logits → a sampled token, repeated.',
    'Attention moves information between tokens; feed-forward layers process each one.',
    'Pretraining builds the knowledge; post-training shapes it into a helpful assistant.',
    'It is all arithmetic at staggering scale — the wonder is that so simple a step, repeated, adds up to this.',
  ],
}
