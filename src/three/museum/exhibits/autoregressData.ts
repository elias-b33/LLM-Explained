/**
 * Hand-authored, illustrative next-token tables for the lever machine. Each
 * prompt has its own little "model": the distribution depends only on the
 * last token (a toy bigram table), which is enough for plausible 6–10 token
 * runs. Probabilities are relative weights (renormalised over the shown
 * candidates); temperature reshapes them as softmax(log p / T).
 */
export type Dist = [token: string, weight: number][]

export interface Prompt {
  text: string
  start: Dist
  next: Record<string, Dist>
}

const END: Dist = [['.', 78], ['!', 14], [' forever', 8]]
export const FORCE_END: Dist = [['.', 88], ['!', 12]]
export const MAX_STEPS = 10

export function distFor(p: Prompt, last: string | null, steps: number): Dist {
  if (steps >= MAX_STEPS - 1) return FORCE_END
  if (last === null) return p.start
  if (last === ' forever') return FORCE_END
  return p.next[last] ?? END
}

export const isEnd = (tok: string) => tok === '.' || tok === '!'

const place: Dist = [[' and', 34], ['.', 28], [',', 16], [' all', 10], [' in', 8]]
const ending: Dist = [['.', 62], [',', 14], [' and', 12], ['!', 6], [' again', 6]]
const person: Dist = [[' princess', 28], [' fox', 22], [' girl', 20], [' dragon', 16], [' king', 14]]
const noun: Dist = [[' who', 46], ['.', 22], [' named', 16], [' with', 10], [' called', 6]]
const names: Dist = [[' Luna', 30], [' Rose', 25], [' Pip', 20], [' Ember', 15], [' Tom', 10]]
const food: Dist = [[',', 30], [' because', 28], ['.', 22], [' with', 12], [' and', 8]]

export const PROMPTS: Prompt[] = [
  {
    text: 'The cat sat on the',
    start: [[' mat', 46], [' sofa', 18], [' windowsill', 12], [' floor', 10], [' keyboard', 8]],
    next: {
      ' mat': place,
      ' sofa': place,
      ' windowsill': place,
      ' floor': place,
      ' keyboard': [[' and', 40], ['.', 24], [',', 16], [' again', 12], [' while', 8]],
      ' and': [[' purred', 30], [' watched', 24], [' fell', 18], [' stared', 14], [' refused', 10]],
      ',': [[' purring', 30], [' watching', 24], [' waiting', 20], [' ignoring', 14], [' half', 8]],
      ' all': [[' day', 62], [' afternoon', 20], [' morning', 10], [' night', 6], [' alone', 2]],
      ' in': [[' the', 70], [' a', 16], [' silence', 8], [' peace', 4], [' style', 2]],
      ' the': [[' sun', 38], [' corner', 20], [' rain', 14], [' dark', 14], [' morning', 10]],
      ' a': [[' patch', 34], [' sunbeam', 28], [' daze', 16], [' box', 12], [' hurry', 6]],
      ' patch': [[' of', 90], ['.', 10]],
      ' of': [[' sunlight', 70], [' light', 20], [' warmth', 10]],
      ' purred': [[' softly', 34], ['.', 30], [' happily', 14], [' loudly', 12], [' at', 8]],
      ' watched': [[' the', 62], [' birds', 20], [' me', 10], [' everything', 6], [' TV', 2]],
      ' fell': [[' asleep', 82], [' off', 10], [' silent', 4], [' over', 3], [' in', 1]],
      ' stared': [[' at', 60], [' out', 24], [' into', 10], [' back', 4], [' blankly', 2]],
      ' refused': [[' to', 92], [' dinner', 4], [' food', 2], [' politely', 1], [' everything', 1]],
      ' to': [[' move', 48], [' leave', 24], [' share', 12], [' budge', 10], [' eat', 6]],
      ' at': [[' the', 72], [' me', 16], [' nothing', 8], [' a', 4]],
      ' into': [[' the', 72], [' space', 18], [' nothing', 10]],
      ' out': [[' the', 70], [' of', 20], [' at', 10]],
      ' purring': [[' happily', 30], [' softly', 30], ['.', 20], [' loudly', 10], [' away', 10]],
      ' watching': [[' the', 60], [' birds', 25], [' me', 15]],
      ' waiting': [[' for', 70], ['.', 20], [' patiently', 10]],
      ' for': [[' dinner', 40], [' me', 25], [' the', 20], [' nothing', 10], [' a', 5]],
      ' ignoring': [[' me', 50], [' everyone', 25], [' the', 25]],
      ' half': [[' asleep', 85], [' awake', 10], [' hidden', 5]],
      ' asleep': ending,
      ' softly': ending,
      ' happily': ending,
    },
  },
  {
    text: 'Once upon a',
    start: [[' time', 94], [' midnight', 3], [' hill', 1.5], [' dream', 1], [' star', 0.5]],
    next: {
      ' time': [[',', 62], [' there', 22], [' in', 12], [' a', 3], [' long', 1]],
      ' midnight': [[',', 60], [' there', 30], [' in', 10]],
      ' hill': [[',', 50], [' there', 30], [' lived', 20]],
      ' dream': [[',', 60], [' there', 40]],
      ' star': [[',', 50], [' fell', 50]],
      ',': [[' there', 48], [' in', 22], [' a', 16], [' deep', 8], [' long', 6]],
      ' there': [[' lived', 64], [' was', 30], [' stood', 4], [' ruled', 2]],
      ' lived': [[' a', 72], [' an', 14], [' two', 8], [' the', 6]],
      ' was': [[' a', 76], [' an', 14], [' once', 6], [' nothing', 4]],
      ' stood': [[' a', 70], [' an', 20], [' the', 10]],
      ' ruled': [[' a', 70], [' an', 20], [' the', 10]],
      ' in': [[' a', 66], [' the', 28], [' faraway', 6]],
      ' a': [[' young', 26], [' little', 24], [' brave', 18], [' kingdom', 16], [' dragon', 16]],
      ' an': [[' old', 62], [' ancient', 22], [' owl', 10], [' empress', 6]],
      ' the': [[' forest', 34], [' kingdom', 30], [' village', 20], [' mountains', 16]],
      ' faraway': [[' land', 60], [' kingdom', 40]],
      ' two': [[' sisters', 40], [' brothers', 35], [' foxes', 25]],
      ' young': person,
      ' little': person,
      ' brave': person,
      ' old': person,
      ' ancient': [[' dragon', 50], [' king', 30], [' owl', 20]],
      ' deep': [[' forest', 60], [' sleep', 40]],
      ' long': [[' ago', 90], [' time', 10]],
      ' ago': [[',', 70], ['.', 30]],
      ' princess': noun,
      ' fox': noun,
      ' girl': noun,
      ' dragon': noun,
      ' king': noun,
      ' kingdom': noun,
      ' forest': noun,
      ' village': noun,
      ' mountains': noun,
      ' land': noun,
      ' sisters': noun,
      ' brothers': noun,
      ' foxes': noun,
      ' owl': noun,
      ' empress': noun,
      ' who': [[' loved', 36], [' dreamed', 22], [' could', 18], [' feared', 14], [' never', 10]],
      ' loved': [[' the', 40], [' to', 30], [' adventure', 20], [' stories', 10]],
      ' to': [[' explore', 40], [' sing', 30], [' read', 20], [' fly', 10]],
      ' dreamed': [[' of', 70], [' about', 30]],
      ' of': [[' flying', 40], [' the', 30], [' adventure', 30]],
      ' about': [[' flying', 40], [' the', 30], [' adventure', 30]],
      ' could': [[' talk', 40], [' fly', 30], [' sing', 20], [' read', 10]],
      ' never': [[' slept', 40], [' smiled', 30], [' gave', 30]],
      ' gave': [[' up', 100]],
      ' feared': [[' nothing', 60], [' the', 40]],
      ' named': names,
      ' called': names,
      ' with': [[' golden', 40], [' a', 30], [' silver', 30]],
      ' golden': [[' hair', 50], [' wings', 30], [' eyes', 20]],
      ' silver': [[' hair', 50], [' wings', 30], [' eyes', 20]],
    },
  },
  {
    text: 'My favourite food is',
    start: [[' pizza', 30], [' sushi', 22], [' pasta', 18], [' chocolate', 16], [' tacos', 14]],
    next: {
      ' pizza': food,
      ' sushi': food,
      ' pasta': food,
      ' chocolate': food,
      ' tacos': food,
      ',': [[' especially', 40], [' but', 30], [' hands', 30]],
      ' hands': [[' down', 100]],
      ' especially': [[' when', 55], [' with', 30], [' on', 15]],
      ' when': [[' it', 40], [' I', 35], [' someone', 25]],
      ' someone': [[' else', 70], [' cooks', 30]],
      ' else': [[' cooks', 60], [' makes', 40]],
      ' because': [[' it', 46], [' I', 30], [' my', 14], [' nothing', 10]],
      ' it': [["'s", 48], [' is', 30], [' reminds', 22]],
      "'s": [[' delicious', 30], [' comforting', 24], [' cheap', 18], [' quick', 16], [' always', 12]],
      ' is': [[' delicious', 30], [' comforting', 24], [' cheap', 18], [' quick', 16], [' always', 12]],
      ' always': [[' warm', 40], [' different', 30], [' there', 30]],
      ' reminds': [[' me', 90], [' us', 10]],
      ' me': [[' of', 80], ['.', 20]],
      ' us': [[' of', 100]],
      ' of': [[' home', 50], [' summer', 30], [' childhood', 20]],
      ' I': [[' grew', 40], [' can', 30], [' love', 30]],
      ' grew': [[' up', 90], [' tired', 10]],
      ' up': [[' eating', 50], [' with', 50]],
      ' eating': [[' it', 100]],
      ' can': [[' eat', 60], [' make', 40]],
      ' eat': [[' it', 70], [' a', 30]],
      ' make': [[' it', 70], [' a', 30]],
      ' my': [[' mom', 40], [' dad', 30], [' grandma', 30]],
      ' mom': [[' makes', 60], [' made', 40]],
      ' dad': [[' makes', 60], [' made', 40]],
      ' grandma': [[' makes', 60], [' made', 40]],
      ' makes': [[' it', 70], [' the', 30]],
      ' made': [[' it', 70], [' the', 30]],
      ' the': [[' best', 100]],
      ' with': [[' extra', 40], [' cheese', 30], [' friends', 30]],
      ' extra': [[' cheese', 60], [' spice', 40]],
      ' and': [[' pasta', 30], [' ice', 30], [' anything', 40]],
      ' ice': [[' cream', 100]],
      ' but': [[' only', 50], [' not', 50]],
      ' only': [[' on', 60], [' when', 40]],
      ' on': [[' Fridays', 60], [' weekends', 40]],
      ' not': [[' every', 60], [' cold', 40]],
      ' every': [[' day', 100]],
    },
  },
]

export const TEMPERATURES = [0.2, 0.7, 1.0, 1.5]

/** softmax(log w / T) over the candidates. */
export function shape(d: Dist, T: number): number[] {
  const ls = d.map(([, w]) => Math.log(w) / T)
  const m = Math.max(...ls)
  const e = ls.map((l) => Math.exp(l - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((x) => x / s)
}

export function sample(p: number[]): number {
  let r = Math.random()
  for (let i = 0; i < p.length; i++) {
    r -= p[i]
    if (r <= 0) return i
  }
  return p.length - 1
}
