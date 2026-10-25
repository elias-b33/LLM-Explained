import type { ModuleMeta } from '@/types'

/** The ordered curriculum. Drives routing, the sidebar, progress, and prev/next nav. */
export const MODULES: ModuleMeta[] = [
  {
    id: 'orientation',
    slug: 'orientation',
    index: 0,
    title: 'What Is an LLM, Really?',
    subtitle: 'The one-sentence truth: tokens go in, a probability distribution over the next token comes out.',
    minutes: 6,
  },
  {
    id: 'tokenization',
    slug: 'tokenization',
    index: 1,
    title: 'Tokenization',
    subtitle: 'How raw text becomes the numbered pieces a model can actually read.',
    minutes: 8,
  },
  {
    id: 'embeddings',
    slug: 'embeddings',
    index: 2,
    title: 'Embeddings',
    subtitle: 'Where meaning becomes geometry — and analogies become arithmetic.',
    minutes: 9,
  },
  {
    id: 'transformer',
    slug: 'transformer',
    index: 3,
    title: 'The Transformer',
    subtitle: 'The assembly line that turns a sequence of tokens into a prediction.',
    minutes: 8,
  },
  {
    id: 'attention',
    slug: 'attention',
    index: 4,
    title: 'Attention',
    subtitle: 'The mechanism that lets every token look at every other token. The heart of it all.',
    minutes: 12,
  },
  {
    id: 'layers',
    slug: 'layers',
    index: 5,
    title: 'Inside a Layer',
    subtitle: 'Feed-forward networks, the residual stream, layer norm — and why depth matters.',
    minutes: 9,
  },
  {
    id: 'sampling',
    slug: 'sampling',
    index: 6,
    title: 'From Logits to Words',
    subtitle: 'Softmax, temperature, and how the model actually chooses what to say next.',
    minutes: 9,
  },
  {
    id: 'pretraining',
    slug: 'pretraining',
    index: 7,
    title: 'How Models Learn',
    subtitle: 'Next-token prediction, loss as surprise, gradient descent, and the magic of scale.',
    minutes: 11,
  },
  {
    id: 'post-training',
    slug: 'post-training',
    index: 8,
    title: 'From Predictor to Assistant',
    subtitle: 'SFT and RLHF: the alignment step that turns a text-completer into a helpful assistant.',
    minutes: 9,
  },
  {
    id: 'inference',
    slug: 'inference',
    index: 9,
    title: 'Generation & Its Limits',
    subtitle: 'Autoregression, the context window, the KV cache, and why models hallucinate.',
    minutes: 10,
  },
  {
    id: 'capstone',
    slug: 'capstone',
    index: 10,
    title: 'The Whole Picture',
    subtitle: 'Everything, connected — a master map, a glossary, and where to go next.',
    minutes: 6,
  },
]

export const TOTAL_MINUTES = MODULES.reduce((sum, m) => sum + m.minutes, 0)

export function getModule(slug: string | undefined): ModuleMeta | undefined {
  return MODULES.find((m) => m.slug === slug)
}

export function getAdjacent(slug: string): { prev?: ModuleMeta; next?: ModuleMeta } {
  const i = MODULES.findIndex((m) => m.slug === slug)
  if (i === -1) return {}
  return {
    prev: i > 0 ? MODULES[i - 1] : undefined,
    next: i < MODULES.length - 1 ? MODULES[i + 1] : undefined,
  }
}

/** The two-digit label shown above a module title, e.g. "01". */
export const moduleLabel = (index: number): string => index.toString().padStart(2, '0')
