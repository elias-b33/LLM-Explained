/**
 * Small evaluation set for "Ask the text" (dev only: window.__askEval()).
 * Each question lists the key facts a good answer contains, and keywords used as a rough
 * automatic hint; the real "grounded" verdict is made by reading the answer against the
 * retrieved passages.
 */
import type { Turn } from './useAsk'

export interface EvalItem {
  text: 'strindberg' | 'shakespeare'
  lang: 'sv' | 'en'
  question: string
  facts: string
  /** any of these (lower-case) in the answer counts as a hit for the automatic hint */
  keywords: string[]
  /** the evidence: a retrieved passage containing any of these strings counts as a retrieval hit */
  gold: string[]
  /** hold-out question, written before the round-3 tuning and never tuned on */
  holdout?: boolean
}

export const EVAL: EvalItem[] = [
  {
    text: 'strindberg',
    lang: 'sv',
    question: 'Vem är Arvid Falk?',
    facts: 'En ung tjänsteman (ämbetsman) som lämnar ämbetsmannabanan för att bli litteratör/författare.',
    keywords: ['ämbetsman', 'tjänsteman', 'litteratör', 'författare', 'ämbetsmannabanan', 'kollegi'],
    gold: ['numera litteratören Arvid Falk', 'övergivit ämbetsmannabanan'],
  },
  {
    text: 'strindberg',
    lang: 'sv',
    question: 'Vad är Röda rummet?',
    facts: 'Det inre schackrummet i Berns salong, uppkallat efter sin röda möblering, där bohemkotteriet träffas.',
    keywords: ['berns', 'schackrum', 'röda möblering', 'kotteri', 'träffa'],
    gold: ['antog namn av Röda Rummet'],
  },
  {
    text: 'strindberg',
    lang: 'sv',
    question: 'Vem är Olle Montanus?',
    facts: 'En av Falks bohemvänner; beskrivs som "ett snille, men han vill inte arbeta".',
    keywords: ['snille', 'arbeta', 'filosof', 'bildhuggare'],
    gold: ['Olle Montanus, kära du, är ett snille'],
  },
  {
    text: 'strindberg',
    lang: 'sv',
    question: 'Vad heter Arvid Falks bror och vad arbetar han med?',
    facts: 'Carl Nicolaus Falk, linkramhandlare (som sin far).',
    keywords: ['carl nicolaus', 'linkram'],
    gold: ['Linkramhandlaren Carl Nicolaus Falk'],
  },
  {
    text: 'shakespeare',
    lang: 'en',
    question: 'Why does Macbeth kill Duncan?',
    facts: "The witches' prophecy that he will be king, his ambition, and Lady Macbeth's urging.",
    keywords: ['prophec', 'witch', 'ambition', 'king', 'lady macbeth', 'crown'],
    gold: ['I have no spur', 'screw your courage', 'Was the hope drunk', 'Glamis thou art'],
  },
  {
    text: 'shakespeare',
    lang: 'en',
    question: 'Who is Banquo?',
    facts: "A Scottish general and Macbeth's fellow captain; the witches say his descendants will be kings; Macbeth has him murdered.",
    keywords: ['general', 'thane', 'captain', 'kings', 'murder', 'fleance', 'friend'],
    gold: ['Thou shalt get kings', 'Our captains, Macbeth and Banquo'],
  },
  {
    text: 'shakespeare',
    lang: 'en',
    question: 'What do the witches predict for Macbeth?',
    facts: 'That he will be Thane of Glamis, Thane of Cawdor, and "king hereafter".',
    keywords: ['cawdor', 'glamis', 'king hereafter', 'king'],
    gold: ['king hereafter'],
  },
  {
    text: 'shakespeare',
    lang: 'en',
    question: 'What does Lady Macbeth do while sleepwalking?',
    facts: 'She tries to wash imaginary blood from her hands ("Out, damned spot!").',
    keywords: ['spot', 'hand', 'wash', 'blood'],
    gold: ['damned spot', 'washing her hands'],
  },
  // hold-out: written before tuning, reported separately
  {
    text: 'strindberg',
    lang: 'sv',
    question: 'Vad målar Lundell?',
    facts: 'En skiss till en altartavla (för Träskåla); han är "figur- eller rättare kyrkmålare".',
    keywords: ['altartavla', 'kyrk'],
    gold: ['skiss till en altartavla', 'kyrkmålare'],
    holdout: true,
  },
  {
    text: 'strindberg',
    lang: 'sv',
    question: 'Var har Carl Nicolaus Falk sin affär?',
    facts: 'Vid Österlånggatan, snett emot Ferkens gränd.',
    keywords: ['österlånggatan', 'ferkens'],
    gold: ['Österlånggatan'],
    holdout: true,
  },
  {
    text: 'shakespeare',
    lang: 'en',
    question: 'Who kills Macbeth?',
    facts: 'Macduff, who enters with Macbeth’s head.',
    keywords: ['macduff'],
    gold: ['Macbeth’s head', 'untimely ripp’d', 'lay on, Macduff'],
    holdout: true,
  },
  {
    text: 'shakespeare',
    lang: 'en',
    question: 'What happens to Fleance when Banquo is attacked?',
    facts: 'He escapes ("Fly, good Fleance"; "Fleance is ’scap’d").',
    keywords: ['escape', 'flee', 'fled', 'fly', 'scap'],
    gold: ['Fly, good Fleance', 'Fleance is ’scap’d'],
    holdout: true,
  },
]

const SV_HINT = /[åäö]|\b(och|det|är|som|inte|han|hon|att|en|ett|av|på|för)\b/gi
const EN_HINT = /\b(the|and|is|was|he|she|that|not|of|to|his|her|a|in)\b/gi

export function answerLang(a: string): 'sv' | 'en' | '?' {
  const sv = (a.match(SV_HINT) ?? []).length
  const en = (a.match(EN_HINT) ?? []).length
  if (!sv && !en) return '?'
  return sv > en ? 'sv' : 'en'
}

export interface EvalRow {
  model: string
  holdout: boolean
  /** a retrieved passage contained the evidence */
  goldHit: boolean
  hyde: string | null
  retrievalMs: number
  question: string
  lang: 'sv' | 'en'
  answer: string
  langOk: boolean
  cites: boolean
  keywordHits: string[]
  passages: { n: number; score: number; start: string }[]
  tokens: number
  decodeTps: number
  firstTokenMs: number
}

export function scoreTurn(model: string, item: EvalItem, t: Turn): EvalRow {
  const a = t.answer.trim()
  const low = a.toLowerCase()
  return {
    model,
    holdout: Boolean(item.holdout),
    goldHit: t.passages.some((p) => item.gold.some((g) => p.text.includes(g))),
    hyde: t.hyde,
    retrievalMs: Math.round(t.retrievalMs),
    question: item.question,
    lang: item.lang,
    answer: a,
    langOk: answerLang(a) === item.lang,
    cites: /\[\d+(?:\s*,\s*\d+)*\]/.test(a),
    keywordHits: item.keywords.filter((k) => low.includes(k)),
    passages: t.passages.map((p) => ({ n: p.n, score: +p.score.toFixed(3), start: p.text.slice(0, 80).replace(/\s+/g, ' ') })),
    tokens: t.tokens,
    decodeTps: +t.decodeTps.toFixed(1),
    firstTokenMs: Math.round(t.firstTokenMs),
  }
}
