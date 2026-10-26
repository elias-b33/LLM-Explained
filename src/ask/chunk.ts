/**
 * Structure-aware chunking.
 *
 * The text is first cut into sections it announces itself: acts and scenes of a play
 * ("ACT I", "SCENE VII. The same. A Lobby in the Castle."), chapters of a novel
 * ("ANDRA KAPITLET." + title line) or ALL-CAPS story titles ("HANSEL AND GRETEL").
 * Inside a section, passages of ~350–450 characters are built from whole speeches /
 * paragraphs (a long one is split on sentence ends), with a little overlap.
 *
 * Every passage carries a short header such as "Act I, Scene 7 (A Lobby in the Castle)"
 * or "Kapitel 2 · Bröder emellan", which is embedded with it and shown to the model, so
 * a passage of dialogue still says where it is. `section` lets the retriever add the
 * neighbouring passage from the same scene or chapter.
 */

export interface Chunk {
  /** 0-based index in the text */
  id: number
  /** where it sits, e.g. "Act I, Scene 7 (A Lobby in the Castle)"; may be '' */
  header: string
  text: string
  /** index of the section (scene / chapter / story) it belongs to */
  section: number
}

const MIN = 350
const TARGET = 400
const MAX = 450
const OVERLAP = 80

const ROMAN: Record<string, number> = { I: 1, V: 5, X: 10, L: 50 }
function roman(s: string): number {
  let n = 0
  for (let i = 0; i < s.length; i++) {
    const v = ROMAN[s[i]] ?? 0
    const next = ROMAN[s[i + 1]] ?? 0
    n += v < next ? -v : v
  }
  return n
}
const SV_ORD = ['FÖRSTA', 'ANDRA', 'TREDJE', 'FJÄRDE', 'FEMTE', 'SJÄTTE', 'SJUNDE', 'ÅTTONDE', 'NIONDE', 'TIONDE', 'ELFTE', 'TOLFTE']

interface Section {
  header: string
  body: string
}

/** split into sections with a human header; falls back to one header-less section */
function sections(text: string): Section[] {
  const lines = text.split('\n')
  const out: Section[] = []
  let act = ''
  let header = ''
  let buf: string[] = []
  const flush = () => {
    const body = buf.join('\n').trim()
    if (body) out.push({ header, body })
    buf = []
  }
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i].trim()
    let m: RegExpExecArray | null
    if ((m = /^ACT ([IVXL]+)\.?$/.exec(l))) {
      flush()
      act = `Act ${roman(m[1])}`
      header = act
      continue
    }
    if ((m = /^SCENE ([IVXL]+)\.\s*(.*)$/.exec(l))) {
      flush()
      const place = m[2].replace(/^The same\.\s*/i, '').replace(/\.$/, '')
      header = `${act ? act + ', ' : ''}Scene ${roman(m[1])}${place ? ` (${place})` : ''}`
      continue
    }
    if ((m = /^([A-ZÅÄÖ]+) KAPITLET\.$/.exec(l))) {
      flush()
      const n = SV_ORD.indexOf(m[1]) + 1
      // the chapter title is the next non-empty line
      let title = ''
      while (i + 1 < lines.length && !lines[i + 1].trim()) i++
      if (i + 1 < lines.length && lines[i + 1].trim().length < 60) title = lines[++i].trim().replace(/\.$/, '')
      header = `Kapitel ${n || m[1].toLowerCase()}${title ? ' · ' + title : ''}`
      continue
    }
    // ALL-CAPS story titles (fairy-tale collections): short, no full stop, surrounded by blanks
    if (
      /^[A-Z][A-Z ,’'-]{3,48}$/.test(l) &&
      !lines[i - 1]?.trim() &&
      !lines[i + 1]?.trim() &&
      !/^[A-Z]+\.$/.test(l)
    ) {
      flush()
      header = l.replace(/\b([A-Z])([A-Z’']+)\b/g, (_, a: string, b: string) => a + b.toLowerCase())
      continue
    }
    buf.push(lines[i])
  }
  flush()
  return out.length ? out : [{ header: '', body: text }]
}

/** split a long piece into sentence-ish units of at most MAX chars */
function sentences(s: string): string[] {
  const out: string[] = []
  const re = /[^.!?…\n]+(?:[.!?…]+["'’”»)]*|\n|$)\s*/g
  let m: RegExpExecArray | null
  while ((m = re.exec(s))) {
    if (!m[0]) {
      re.lastIndex++
      continue
    }
    let t = m[0]
    while (t.length > MAX) {
      let cut = t.lastIndexOf(' ', MAX - 50)
      if (cut < MAX / 2) cut = MAX - 50
      out.push(t.slice(0, cut))
      t = t.slice(cut)
    }
    out.push(t)
  }
  return out
}

/** the last ~OVERLAP chars of a passage, starting at a line/sentence start if possible */
function tailOf(s: string): string {
  const from = Math.max(0, s.length - OVERLAP)
  const rest = s.slice(from)
  const m = /\n|[.!?;:]\s/.exec(rest)
  if (m && m.index < rest.length - 20) return rest.slice(m.index + m[0].length)
  const w = /\s/.exec(rest)
  return w ? rest.slice(w.index + 1) : ''
}

interface Unit {
  t: string
  /** speaker of the speech this unit belongs to (plays), if any */
  speaker: string
  /** the unit itself starts with the speaker's name line */
  named: boolean
}

const SPEAKER_RE = /^([A-Z][A-Z ’'-]{1,30})\.\n/

/** passages of one section; `speaker` is set when a passage starts in the middle of a speech */
function chunkBody(body: string): { text: string; speaker: string }[] {
  const units: Unit[] = []
  for (const p of body.split(/\n\s*\n/)) {
    if (!p.trim()) continue
    const sp = SPEAKER_RE.exec(p.trimStart())?.[1] ?? ''
    if (p.length <= MAX) units.push({ t: p + '\n\n', speaker: sp, named: Boolean(sp) })
    else sentences(p).forEach((t, k) => units.push({ t, speaker: sp, named: Boolean(sp) && k === 0 }))
  }
  const out: { text: string; speaker: string }[] = []
  let cur = ''
  let curSpeaker = ''
  const start = (u: Unit, carried: boolean) => {
    // a passage that opens mid-speech (or with carried-over lines) names the speaker in its header
    curSpeaker = u.speaker && (carried || !u.named) ? u.speaker : ''
  }
  for (let i = 0; i < units.length; ) {
    const u = units[i]
    if (!cur) start(u, false)
    if (cur && cur.length + u.t.length > MAX && cur.length >= MIN * 0.6) {
      out.push({ text: cur.trim(), speaker: curSpeaker })
      const tail = tailOf(cur)
      cur = tail.trim() && tail.length + u.t.length <= MAX ? tail : ''
      start(units[i - 1] ?? u, Boolean(cur))
      continue
    }
    cur += u.t
    i++
    if (cur.length >= TARGET) {
      out.push({ text: cur.trim(), speaker: curSpeaker })
      const tail = tailOf(cur)
      cur = i < units.length && tail.trim() ? tail : ''
      if (cur) start(units[i - 1], true)
    }
  }
  if (cur.trim()) {
    const prev = out[out.length - 1]
    const t = cur.trim()
    if (prev && t.length < 150 && !prev.text.endsWith(t)) prev.text = `${prev.text}\n${t}`
    else if (!prev || !prev.text.endsWith(t)) out.push({ text: t, speaker: curSpeaker })
  }
  return out.map((o) => ({ ...o, text: o.text.replace(/\n{3,}/g, '\n\n') }))
}

export function chunkText(raw: string): Chunk[] {
  const text = raw.replace(/\r\n?/g, '\n')
  const chunks: Chunk[] = []
  sections(text).forEach((sec, s) => {
    for (const p of chunkBody(sec.body)) {
      const header = [sec.header, p.speaker].filter(Boolean).join(' · ')
      chunks.push({ id: chunks.length, header, text: p.text, section: s })
    }
  })
  return chunks
}

/** what the embedder, keyword index and model see: header + passage */
export function chunkWithHeader(c: Chunk): string {
  return c.header ? `${c.header}\n${c.text}` : c.text
}
