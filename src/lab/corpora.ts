/** Bundled public-domain training texts, each loaded lazily into its own chunk. */
export type CorpusId = 'shakespeare' | 'strindberg' | 'grimm' | 'custom'

export interface Corpus {
  id: Exclude<CorpusId, 'custom'>
  title: string
  author: string
  lang: string
  /** fixed prompt used for the evolving samples */
  prompt: string
  load: () => Promise<string>
  source: string
  url: string
}

export const CORPORA: Corpus[] = [
  {
    id: 'shakespeare',
    title: 'Macbeth',
    author: 'William Shakespeare',
    lang: 'English verse',
    prompt: 'MACBETH.\n',
    load: () => import('./corpora/shakespeare-macbeth.txt?raw').then((m) => m.default),
    source: 'Project Gutenberg eBook #1533, full play (~100 KB)',
    url: 'https://www.gutenberg.org/ebooks/1533',
  },
  {
    id: 'strindberg',
    title: 'Röda rummet',
    author: 'August Strindberg',
    lang: 'Svenska',
    prompt: 'Det var en ',
    load: () => import('./corpora/strindberg-roda-rummet.txt?raw').then((m) => m.default),
    source: 'Project Gutenberg eBook #57052, first ~150 KB (1879 novel; 1912 Samlade skrifter edition)',
    url: 'https://www.gutenberg.org/ebooks/57052',
  },
  {
    id: 'grimm',
    title: 'Grimms’ Fairy Tales',
    author: 'Brothers Grimm (tr. Taylor & Edwardes)',
    lang: 'English prose',
    prompt: 'Once upon a time ',
    load: () => import('./corpora/grimm-fairy-tales.txt?raw').then((m) => m.default),
    source: 'Project Gutenberg eBook #2591, first ~150 KB',
    url: 'https://www.gutenberg.org/ebooks/2591',
  },
]

export const CUSTOM_MIN = 2000
export const CUSTOM_MAX = 1_000_000
