import { describe, expect, it } from 'vitest'
import macbeth from '@/lab/corpora/shakespeare-macbeth.txt?raw'
import strindberg from '@/lab/corpora/strindberg-roda-rummet.txt?raw'
import grimm from '@/lab/corpora/grimm-fairy-tales.txt?raw'
import { chunkText, chunkWithHeader } from './chunk'

const PLAY = `ACT I

SCENE I. A heath.

FIRST WITCH.
When shall we three meet again
In thunder, lightning, or in rain?

SECOND WITCH.
When the hurlyburly’s done.

SCENE II. The same. A Lobby in the Castle.

MACBETH.
${'So foul and fair a day I have not seen. '.repeat(30)}
`

const NOVEL = `FÖRSTA KAPITLET.

Stockholm i fågelperspektiv.

Det var en afton i början av maj.

ANDRA KAPITLET.

Bröder emellan.

Han gick.`

describe('structure-aware chunking', () => {
  it('play: act, scene and place become the header; mid-speech passages name the speaker', () => {
    const cs = chunkText(PLAY)
    expect(cs[0].header).toBe('Act 1, Scene 1 (A heath)')
    expect(cs[0].text).toContain('FIRST WITCH.')
    expect(cs[0].text).toContain('SECOND WITCH.')
    // "The same." is dropped from the place
    expect(cs[1].header).toBe('Act 1, Scene 2 (A Lobby in the Castle)')
    expect(cs[1].text.startsWith('MACBETH.')).toBe(true)
    // the long speech continues in later passages that carry the speaker
    const rest = cs.slice(2)
    expect(rest.length).toBeGreaterThan(0)
    for (const c of rest) expect(c.header).toBe('Act 1, Scene 2 (A Lobby in the Castle) · MACBETH')
    expect(new Set(cs.map((c) => c.section))).toEqual(new Set([0, 1]))
    expect(chunkWithHeader(cs[0]).split('\n')[0]).toBe(cs[0].header)
  })

  it('novel: Swedish ordinal chapters with their title line', () => {
    const cs = chunkText(NOVEL)
    expect(cs.map((c) => c.header)).toEqual(['Kapitel 1 · Stockholm i fågelperspektiv', 'Kapitel 2 · Bröder emellan'])
    expect(cs.map((c) => c.text)).toEqual(['Det var en afton i början av maj.', 'Han gick.'])
    expect(cs.map((c) => c.section)).toEqual([0, 1])
  })

  it('text without structure is one header-less section; CRLF is normalised', () => {
    const cs = chunkText('Just some text.\r\n\r\nAnd another paragraph.')
    expect(cs).toEqual([{ id: 0, header: '', text: 'Just some text.\n\nAnd another paragraph.', section: 0 }])
  })

  it('a long unpunctuated paragraph is split into bounded passages', () => {
    const cs = chunkText('word '.repeat(2000))
    expect(cs.length).toBeGreaterThan(15)
    for (const c of cs) expect(c.text.length).toBeLessThanOrEqual(500)
  })

  it.each([
    ['Macbeth', macbeth, 'Act 1, Scene 3 (A heath)'],
    ['Röda rummet', strindberg, 'Kapitel 2 · Bröder emellan'],
    ['Grimm', grimm, 'Hansel And Gretel'],
  ])('%s: sizes stay near the 350-450 character target', (_, text, header) => {
    const cs = chunkText(text)
    const lens = cs.map((c) => c.text.length).sort((a, b) => a - b)
    const q = (p: number) => lens[Math.floor(p * (lens.length - 1))]
    expect(cs.length).toBeGreaterThan(200)
    expect(q(0.05)).toBeGreaterThanOrEqual(150)
    expect(q(0.5)).toBeGreaterThanOrEqual(350)
    expect(q(0.9)).toBeLessThanOrEqual(450)
    expect(lens[lens.length - 1]).toBeLessThanOrEqual(700)
    expect(cs.every((c, i) => c.id === i && c.text.trim() === c.text && c.text.length > 0)).toBe(true)
    expect(cs.every((c, i) => i === 0 || c.section >= cs[i - 1].section)).toBe(true)
    expect(cs.some((c) => c.header.startsWith(header))).toBe(true)
  })
})
