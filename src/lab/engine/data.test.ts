import { describe, expect, it } from 'vitest'
import { TokenStream, buildVocab, byteLabel, byteVocab, decode, encode, incompleteTail, isBytePlaceholder, tokenLabels } from './data'

describe('byte vocabulary', () => {
  const v = byteVocab()

  it('has 256 tokens and ASCII is one byte per character', () => {
    expect(v.chars).toHaveLength(256)
    expect(Array.from(encode(v, 'Hi!'))).toEqual([72, 105, 33])
  })

  it('encodes multi-byte UTF-8 and decodes it back', () => {
    for (const s of ['å', 'Röda rummet', '€ 5', 'emoji 😀 ok', 'مرحبا']) {
      const ids = encode(v, s)
      expect(decode(v, ids)).toBe(s)
    }
    expect(Array.from(encode(v, 'å'))).toEqual([0xc3, 0xa5])
    expect(Array.from(encode(v, '€'))).toEqual([0xe2, 0x82, 0xac])
  })

  it('detects a character split at the end of a sequence', () => {
    const ids = Array.from(encode(v, 'ok😀'))
    expect(incompleteTail(ids)).toBe(0)
    expect(incompleteTail(ids.slice(0, -1))).toBe(3)
    expect(incompleteTail(ids.slice(0, -2))).toBe(2)
    expect(incompleteTail(ids.slice(0, -3))).toBe(1)
    expect(incompleteTail(Array.from(encode(v, 'aå')).slice(0, -1))).toBe(1)
    // a cut-off character is dropped on request, otherwise it becomes U+FFFD
    const cut = Array.from(encode(v, 'smörgås')).slice(0, -2) // ends in the lead byte of 'å'
    expect(decode(v, cut, true)).toBe('smörg')
    expect(decode(v, cut)).toBe('smörg�')
  })

  it('streams a multi-byte character split across two pushes', () => {
    const ts = new TokenStream(v)
    const [lead, cont] = Array.from(encode(v, 'ä'))
    expect(ts.push('h'.charCodeAt(0))).toBe('h')
    expect(ts.push(lead)).toBe('')
    expect(ts.pending).toEqual([lead])
    // the candidate that would finish the character previews it
    expect(ts.candidate(cont)).toEqual(['ä', false])
    expect(ts.push(cont)).toBe('ä')
    expect(ts.pending).toEqual([])
    // an unfinished character at the end flushes as U+FFFD
    ts.push(0xe2)
    expect(ts.flush()).toBe('�')
  })

  it('labels bytes and multi-byte tokens readably', () => {
    expect(byteLabel(65)).toBe('A')
    expect(byteLabel(10)).toBe('\n')
    expect(byteLabel(0xc3)).toBe('‹C3›')
    expect(isBytePlaceholder('‹C3›')).toBe(true)
    expect(isBytePlaceholder('A')).toBe(false)
    expect(tokenLabels(v, encode(v, 'aö'))).toEqual(['a', 'ö', '◦'])
  })
})

describe('character vocabulary', () => {
  it('maps every distinct character and round-trips', () => {
    const v = buildVocab('hello world')
    expect(v.kind).toBe('chars')
    expect(v.chars).toEqual([' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w'])
    expect(decode(v, encode(v, 'hello'))).toBe('hello')
    // unknown characters fall back to the unk id (the space here)
    expect(decode(v, encode(v, 'hz'))).toBe('h ')
  })

  it('folds rare characters into one unknown token beyond 120 symbols', () => {
    let text = ''
    for (let c = 0; c < 200; c++) text += String.fromCodePoint(0x4e00 + c).repeat(c < 100 ? 5 : 1)
    const v = buildVocab(text)
    expect(v.chars.length).toBe(120)
    expect(v.chars).toContain('¿')
    expect(v.chars[v.unk]).toBe('¿')
  })
})
