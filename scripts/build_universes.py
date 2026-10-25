"""
build_universes.py — three language "universes" for the embedding galaxy.

Streams the most frequent words from fastText vectors for English, Swedish, and
Arabic (only reading the head of each .vec.gz, so the download stays small),
projects each language to 3D on its own with UMAP, and offsets the three clouds
into separate regions of space. Writes the same galaxy.json the app reads, with
each "cluster" being a language universe.

Run once:
    python -m pip install numpy umap-learn
    python scripts/build_universes.py

fastText vectors (Grave et al., 2018) are released under CC-BY-SA-3.0.
"""

import gzip
import json
import os
import urllib.request

import numpy as np
from sklearn.neighbors import NearestNeighbors

# Per-language: code, display name, color, and the center of its universe.
LANGS = [
    {"code": "en", "name": "English", "color": "#8b7bff", "center": [-34.0, 0.0, 0.0]},
    {"code": "sv", "name": "Swedish", "color": "#2bd3e0", "center": [34.0, 0.0, 0.0]},
    {"code": "ar", "name": "Arabic", "color": "#f0b429", "center": [0.0, 0.0, -38.0]},
]
N = 15000           # words per universe
SCAN = 42000        # max lines to scan per language to find N clean words
RADIUS = 13.0       # each universe's cloud radius
NEIGHBORS = 6
URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{}.300.vec.gz"

EN_STOP = set(
    "the of to and a in is was he for it with as his on be at by this had not are but from or have "
    "an they which one you were her all she there would their we him been has when who will more no "
    "if out so what up its about into than them can only other new some could time two may then do "
    "any now such over me even most made after also did before must back where much way well down "
    "because just people how too our man us between three never know".split()
)


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def stream(code):
    url = URL.format(code)
    print(f"  streaming {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=180)
    gz = gzip.GzipFile(fileobj=resp)
    gz.readline()  # header: "<count> <dim>"
    minlen = 2 if code == "ar" else 3
    words, vecs, read = [], [], 0
    while len(words) < N and read < SCAN:
        line = gz.readline()
        if not line:
            break
        read += 1
        try:
            parts = line.decode("utf-8").rstrip("\n").split(" ")
        except UnicodeDecodeError:
            continue
        w = parts[0]
        if len(w) < minlen or not w.isalpha():
            continue
        if code == "en" and (w in EN_STOP or not w.islower()):
            continue
        if len(parts) < 301:
            continue
        try:
            vecs.append(np.asarray(parts[1:301], dtype=np.float32))
        except ValueError:
            continue
        words.append(w)
    gz.close()
    resp.close()
    print(f"    kept {len(words)} words (scanned {read} lines)")
    return words, np.array(vecs, dtype=np.float32)


def project(vecs, center):
    import umap
    # No random_state -> UMAP runs multi-threaded (much faster at 10k words).
    emb = umap.UMAP(
        n_components=3, n_neighbors=15, min_dist=0.15, metric="cosine"
    ).fit_transform(vecs)
    emb = np.asarray(emb, dtype=np.float32)
    emb -= emb.mean(axis=0)
    emb *= RADIUS / (np.percentile(np.linalg.norm(emb, axis=1), 95) or 1.0)
    return emb + np.asarray(center, dtype=np.float32)


def main():
    all_words, clusters = [], []
    en = None
    offset = 0

    for li, lang in enumerate(LANGS):
        words, vecs = stream(lang["code"])
        if not len(words):
            raise SystemExit(f"no words for {lang['code']}")
        if li == 0:
            en = (words, vecs)
        emb = project(vecs, lang["center"])

        nbrs = NearestNeighbors(n_neighbors=NEIGHBORS + 1, metric="cosine").fit(vecs)
        _, ind = nbrs.kneighbors(vecs)
        nn = [[int(offset + j) for j in row[1 : NEIGHBORS + 1]] for row in ind]

        for i, w in enumerate(words):
            all_words.append({
                "w": w,
                "p": [round(float(emb[i][0]), 2), round(float(emb[i][1]), 2), round(float(emb[i][2]), 2)],
                "c": li,
                "nn": nn[i],
            })
        clusters.append({"name": lang["name"], "color": lang["color"], "center": [round(x, 1) for x in lang["center"]]})
        offset += len(words)
        print(f"  {lang['name']}: {len(words)} words")

    # Analogies (English universe).
    analogies = []
    ewords, evecs = en
    eidx = {w: i for i, w in enumerate(ewords)}
    en_normed = unit(evecs)
    for a, b, c, d in [("king", "man", "woman", "queen"), ("paris", "france", "japan", "tokyo"), ("rome", "italy", "france", "paris")]:
        if all(x in eidx for x in (a, b, c, d)):
            t = unit(evecs[eidx[a]] - evecs[eidx[b]] + evecs[eidx[c]])
            if eidx[d] in np.argsort(-(en_normed @ t))[:5]:
                analogies.append({"a": a, "b": b, "c": c, "d": d})
                print(f"  analogy OK: {a} - {b} + {c} ~= {d}")

    data = {"words": all_words, "clusters": clusters, "analogies": analogies}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "data", "galaxy.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
    print(f"\nWrote {len(all_words)} words across {len(clusters)} universes, {len(analogies)} analogies, {os.path.getsize(out)/1024:.1f} KB")


if __name__ == "__main__":
    main()
