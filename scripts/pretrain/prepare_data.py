"""
Build the pretraining dataset: public-domain English and Swedish books from
Project Gutenberg, via the existing Hugging Face dataset `manu/project_gutenberg`
(streamed; nothing is crawled from gutenberg.org).

  python prepare_data.py --smoke                     # tiny slice for testing (~10 MB)
  python prepare_data.py                             # the B2 dataset (~420 MB)

Output (default C:/Users/elias/llm-pretrain/data/<name>/, outside OneDrive):
  train.bin, val.bin   raw UTF-8 bytes, books separated by a blank-line gap; the bytes ARE the tokens
  meta.json            sizes, per-language stats, the book list and the split of every book

Steps per book: drop the Project Gutenberg header, footer and licence text
(everything outside the *** START / *** END markers, plus "Produced by" credits),
normalise line endings, skip books that are too short or garbled, skip the
Workshop's own three texts (any edition) so fine-tuning on them is a fair test,
and drop exact and near duplicates (same opening 4 kB after normalisation).
The held-out split is whole books (~0.5% of the bytes of each language).

Licences: see README.md (the texts are public domain in the US; the HF
dataset itself declares no licence of its own).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import unicodedata
from pathlib import Path

DEFAULT_OUT = Path(os.environ.get('LLM_PRETRAIN_DATA', r'C:\Users\elias\llm-pretrain\data'))
HF_DATASET = 'manu/project_gutenberg'
SEP = '\n\n\n\n'

START_RE = re.compile(r'^\s*\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG E(?:BOOK|TEXT)[^\n]*$', re.I | re.M)
END_RE = re.compile(
    r'^\s*(?:\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG E(?:BOOK|TEXT)'
    r'|End of (?:the )?Project Gutenberg(?:\'s)? E(?:book|text)'
    r'|End of Project Gutenberg\'s'
    r'|\*\*\*END OF THE PROJECT GUTENBERG)',
    re.I | re.M,
)
OLD_SMALL_PRINT_RE = re.compile(r'\*END\*THE SMALL PRINT!.*?\n', re.I | re.S)
TITLE_RE = re.compile(r'^Title:\s*(.+)$', re.M)
CREDIT_RE = re.compile(
    r'^(?:Produced by|E-?text prepared by|Transcribed (?:from|by)|This e-?text was produced|Updated editions will replace'
    r'|Digitized by|Produced from images|Distributed Proofread|Online Distributed Proofreading)',
    re.I,
)
ILLUSTRATION_RE = re.compile(r'^\s*\[(?:Illustration|Bild|Ill\.)[^\]]*\]\s*$', re.I | re.M)
# the Workshop's bundled texts (src/lab/corpora): excluded in every edition
EXCLUDE_IDS = {'1533', '57052', '2591'}
EXCLUDE_TITLE_RE = re.compile(r'macbeth|röda rummet|grimm', re.I)


def clean_book(raw: str) -> tuple[str | None, str]:
    """Returns (body or None if unusable, title)."""
    text = raw.replace('\ufeff', '').replace('\r\n', '\n').replace('\r', '\n')
    m = TITLE_RE.search(text[:5000])
    title = m.group(1).strip() if m else ''
    s = START_RE.search(text)
    if not s:
        sp = OLD_SMALL_PRINT_RE.search(text)
        if not sp:
            return None, title
        start = sp.end()
    else:
        start = s.end()
    e = END_RE.search(text, start)
    body = text[start:e.start() if e else len(text)]
    # credits / transcriber lines near the top: drop those paragraphs
    paras = body.lstrip('\n').split('\n\n')
    head = []
    for p in paras[:12]:
        low = p.lower()
        if CREDIT_RE.match(p.strip()) or 'gutenberg' in low or 'http' in low or 'images of the original pages' in low:
            continue
        head.append(p)
    body = '\n\n'.join(head + paras[12:])
    body = ILLUSTRATION_RE.sub('', body)
    body = unicodedata.normalize('NFC', body)
    body = '\n'.join(line.rstrip() for line in body.split('\n'))
    body = re.sub(r'\n{4,}', '\n\n\n', body).strip('\n')
    if 'gutenberg' in body[-3000:].lower() and 'license' in body[-3000:].lower():
        return None, title  # footer not found cleanly: skip rather than train on licence text
    if len(body) < 20_000:
        return None, title  # pamphlets, indexes, stubs
    if body.count('\ufffd') > len(body) * 1e-4:
        return None, title  # mis-decoded file
    letters = sum(c.isalpha() for c in body[:20000])
    if letters < 0.6 * min(len(body), 20000):
        return None, title  # tables, number lists, music, code
    return body, title


def near_key(body: str) -> str:
    norm = re.sub(r'[\W_]+', '', body[:6000].lower())[:4000]
    return hashlib.sha1(norm.encode()).hexdigest()


def collect(lang: str, target_bytes: int, max_books: int | None, seed: int, stats: dict, seen: set) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split=lang, streaming=True)
    if lang == 'en':
        ds = ds.shuffle(seed=seed, buffer_size=64)  # mix shards, not just the lowest ebook numbers
    books, total, t0 = [], 0, time.time()
    for r in ds:
        stats['seen'] += 1
        pg_id = str(r['id']).split('-')[0]
        body, title = clean_book(r['text'])
        if pg_id in EXCLUDE_IDS or EXCLUDE_TITLE_RE.search(title):
            stats['excluded_workshop'] += 1
            continue
        if body is None:
            stats['dropped_unusable'] += 1
            continue
        k = near_key(body)
        full = hashlib.sha1(body.encode()).hexdigest()
        if k in seen or full in seen:
            stats['dropped_duplicate'] += 1
            continue
        seen.update((k, full))
        b = len(body.encode('utf-8'))
        books.append({'id': pg_id, 'lang': lang, 'title': title[:120], 'bytes': b, 'text': body})
        total += b
        if len(books) % 50 == 0:
            print(f'  {lang}: {len(books)} books, {total / 1e6:.0f} MB ({time.time() - t0:.0f} s)', flush=True)
        if total >= target_bytes or (max_books and len(books) >= max_books):
            break
    return books


def split_books(books: list[dict], val_frac: float) -> None:
    for lang in sorted({b['lang'] for b in books}):
        group = sorted((b for b in books if b['lang'] == lang), key=lambda b: hashlib.md5(b['id'].encode()).hexdigest())
        need = val_frac * sum(b['bytes'] for b in group)
        got = 0
        for b in group:
            b['split'] = 'val' if got < need or got == 0 else 'train'
            if b['split'] == 'val':
                got += b['bytes']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--en-mb', type=float, default=320, help='English bytes to collect (MB)')
    ap.add_argument('--sv-mb', type=float, default=200, help='Swedish bytes (MB); the dataset has ~110 MB, all of it is used')
    ap.add_argument('--val-frac', type=float, default=0.005)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--smoke', action='store_true', help='tiny slice: 6 Swedish + 25 English books')
    args = ap.parse_args()
    out = args.out or DEFAULT_OUT / ('smoke' if args.smoke else 'pg-en-sv')
    out.mkdir(parents=True, exist_ok=True)
    stats = {'seen': 0, 'excluded_workshop': 0, 'dropped_unusable': 0, 'dropped_duplicate': 0}
    seen: set[str] = set()
    t0 = time.time()
    books = []
    books += collect('sv', int(args.sv_mb * 1e6), 6 if args.smoke else None, args.seed, stats, seen)
    books += collect('en', int(args.en_mb * 1e6), 25 if args.smoke else None, args.seed, stats, seen)
    # smoke: a bigger held-out share so the tiny val set is not one book
    split_books(books, 0.08 if args.smoke else args.val_frac)

    sizes = {}
    for split in ('train', 'val'):
        part = [b for b in books if b['split'] == split]
        data = SEP.join(b['text'] for b in part).encode('utf-8')
        (out / f'{split}.bin').write_bytes(data)
        sizes[split] = len(data)
    by_lang = {
        lang: {
            'books': sum(1 for b in books if b['lang'] == lang),
            'bytes': sum(b['bytes'] for b in books if b['lang'] == lang),
            'val_books': sum(1 for b in books if b['lang'] == lang and b['split'] == 'val'),
        }
        for lang in ('en', 'sv')
    }
    meta = {
        'source': f'https://huggingface.co/datasets/{HF_DATASET}',
        'description': 'Project Gutenberg books (en + sv), PG header/footer/licence removed, deduplicated, Workshop texts excluded',
        'separator': SEP,
        'train_bytes': sizes['train'],
        'val_bytes': sizes['val'],
        'by_lang': by_lang,
        'filter_stats': stats,
        'seconds': round(time.time() - t0),
        'books': [{k: v for k, v in b.items() if k != 'text'} for b in books],
    }
    (out / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in meta.items() if k != 'books'}, ensure_ascii=False, indent=1))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
