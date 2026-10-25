"""
build_galaxy.py — offline data pipeline for the immersive embedding galaxy.

Pulls ~2000 of the most frequent real English words from pretrained GloVe vectors
(filtering out function words), projects them to 3D with UMAP, colors each word by
its nearest hand-named semantic category (so the legend stays clean), precomputes
nearest neighbors, verifies a few analogies, and writes a compact JSON the app ships.

Run once:
    python -m pip install gensim numpy scikit-learn umap-learn
    python scripts/build_galaxy.py

Word positions are derived from GloVe (Pennington et al., 2014); the pretrained
vectors are released into the public domain (PDDL).
"""

import json
import os

import numpy as np

TARGET = 2000
NEIGHBORS = 6
ASSIGN_THRESHOLD = 0.45  # below this cosine to every category -> "other"

# (name, color, seed words that anchor the category centroid)
SEED_GROUPS = [
    ("animals", "#35d07f", ["cat", "dog", "horse", "cow", "lion", "tiger", "bear", "wolf", "elephant", "rabbit", "bird", "fish", "snake", "deer", "monkey"]),
    ("food", "#ff9f5a", ["bread", "cheese", "meat", "rice", "soup", "fruit", "apple", "banana", "sugar", "coffee", "wine", "cake", "dinner", "kitchen", "cook"]),
    ("countries", "#f0b429", ["france", "germany", "spain", "italy", "china", "japan", "india", "russia", "brazil", "canada", "egypt", "mexico", "greece", "turkey"]),
    ("cities", "#4d9bff", ["paris", "london", "berlin", "madrid", "rome", "tokyo", "moscow", "beijing", "chicago", "boston", "sydney", "toronto", "vienna"]),
    ("people", "#fca5a5", ["man", "woman", "child", "friend", "person", "girl", "boy", "baby", "neighbor", "stranger", "crowd", "people"]),
    ("royalty", "#8b7bff", ["king", "queen", "prince", "princess", "emperor", "royal", "throne", "crown", "kingdom", "castle", "palace", "noble", "dynasty"]),
    ("colors", "#c084fc", ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "pink", "brown", "golden", "colored"]),
    ("numbers", "#2bd3e0", ["one", "two", "three", "four", "five", "ten", "hundred", "thousand", "million", "number", "dozen", "count"]),
    ("body", "#f472b6", ["head", "face", "eye", "hand", "arm", "leg", "heart", "blood", "brain", "bone", "skin", "foot", "finger", "shoulder"]),
    ("vehicles", "#60a5fa", ["car", "truck", "train", "plane", "ship", "boat", "bus", "bicycle", "engine", "wheel", "vehicle", "aircraft", "automobile"]),
    ("nature", "#38bdf8", ["rain", "snow", "sun", "wind", "storm", "cloud", "sky", "river", "mountain", "forest", "ocean", "tree", "lake", "valley"]),
    ("emotions", "#fb7185", ["happy", "sad", "angry", "fear", "love", "joy", "hope", "afraid", "proud", "lonely", "excited", "nervous", "emotion"]),
    ("work", "#a3e635", ["doctor", "teacher", "lawyer", "engineer", "worker", "manager", "company", "business", "office", "employee", "career", "job"]),
    ("arts", "#fbbf24", ["music", "song", "band", "guitar", "concert", "dance", "painting", "film", "movie", "theater", "artist", "album", "poetry"]),
    ("science", "#5eead4", ["science", "research", "technology", "computer", "data", "energy", "theory", "experiment", "chemical", "physics", "biology", "scientific"]),
    ("politics", "#e879f9", ["war", "army", "military", "government", "president", "political", "election", "parliament", "minister", "court", "treaty"]),
]
OTHER = {"name": "other", "color": "#5b6472"}

MUST_HAVE = ["king", "queen", "man", "woman", "paris", "france", "japan", "tokyo", "rome", "italy", "germany", "berlin", "london", "england"]

STOPWORDS = set("""
a an the of to in and or but is are was were be been being am has have had do does did having will would shall
should can could may might must this that these those i you he she it we they them him her his its our your their
my me us as at by for from on off out up down with within without into onto over under again further then once here
there when where why how all any both each few more most other some such no nor not only own same so than too very
just dont now also said say says like get got make made go goes went come came see seen know what which who whom whose
new old one two three four five first second third while about against between through during before after above below
per via etc among upon toward towards across behind beyond what when your you well back even still way around many much
""".split())

ANALOGY_CANDIDATES = [
    ("king", "man", "woman", "queen"),
    ("paris", "france", "japan", "tokyo"),
    ("rome", "italy", "france", "paris"),
    ("london", "england", "france", "paris"),
    ("queen", "woman", "man", "king"),
    ("tokyo", "japan", "germany", "berlin"),
]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "..", "src", "data", "galaxy.json")


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def main():
    import gensim.downloader as api
    print("Loading GloVe (glove-wiki-gigaword-50) ...")
    kv = api.load("glove-wiki-gigaword-50")

    # Category centroids in embedding space.
    centroids = []
    for _name, _color, seeds in SEED_GROUPS:
        sv = [kv[w] for w in seeds if w in kv.key_to_index]
        centroids.append(np.mean(sv, axis=0))
    centroids = unit(np.array(centroids, dtype=np.float32))

    # Select frequent content words (seeds + must-have guaranteed in).
    words, seen = [], set()
    for w in MUST_HAVE + [s for _n, _c, ss in SEED_GROUPS for s in ss]:
        if w in kv.key_to_index and w not in seen:
            words.append(w); seen.add(w)
    for w in kv.index_to_key:
        if len(words) >= TARGET:
            break
        if w in seen or w in STOPWORDS or not w.isalpha() or len(w) < 3:
            continue
        words.append(w); seen.add(w)

    vecs = np.array([kv[w] for w in words], dtype=np.float32)
    normed = unit(vecs)
    print(f"{len(words)} words selected")

    # Project to 3D.
    try:
        import umap
        print("Projecting with UMAP ...")
        emb = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.22, metric="cosine", random_state=42).fit_transform(vecs)
    except Exception as e:  # noqa: BLE001
        from sklearn.decomposition import PCA
        print(f"UMAP unavailable ({e}); PCA fallback.")
        emb = PCA(n_components=3, random_state=42).fit_transform(vecs)
    emb = np.array(emb, dtype=np.float32)
    emb -= emb.mean(axis=0)
    emb *= 17.0 / (np.percentile(np.linalg.norm(emb, axis=1), 95) or 1.0)

    # Assign each word to nearest category (or "other").
    sims = normed @ centroids.T  # N x 16
    best = sims.argmax(axis=1)
    bestsim = sims.max(axis=1)
    other_idx = len(SEED_GROUPS)
    labels = np.where(bestsim >= ASSIGN_THRESHOLD, best, other_idx)

    clusters = [{"name": n, "color": c} for n, c, _ in SEED_GROUPS] + [OTHER]
    sizes = np.bincount(labels, minlength=len(clusters))
    for i, cl in enumerate(clusters):
        print(f"  {cl['name']:10s}: {sizes[i]}")

    # Neighbors via cosine within the set.
    cos = normed @ normed.T
    np.fill_diagonal(cos, -2.0)
    nn_idx = np.argpartition(-cos, NEIGHBORS, axis=1)[:, :NEIGHBORS]
    nn = [list(map(int, row[np.argsort(-cos[i, row])])) for i, row in enumerate(nn_idx)]

    words_out = [
        {"w": w, "p": [round(float(emb[i][0]), 3), round(float(emb[i][1]), 3), round(float(emb[i][2]), 3)], "c": int(labels[i]), "nn": nn[i]}
        for i, w in enumerate(words)
    ]

    analogies = []
    index_of = {w: i for i, w in enumerate(words)}
    for a, b, c, d in ANALOGY_CANDIDATES:
        if not all(x in index_of for x in (a, b, c, d)):
            continue
        preds = [p for p, _ in kv.most_similar(positive=[a, c], negative=[b], topn=5)]
        if d in preds:
            analogies.append({"a": a, "b": b, "c": c, "d": d})

    data = {"words": words_out, "clusters": clusters, "analogies": analogies}
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
    print(f"\nWrote {OUT_PATH}: {len(words_out)} words, {len(clusters)} clusters, {len(analogies)} analogies, {os.path.getsize(OUT_PATH)/1024:.1f} KB")


if __name__ == "__main__":
    main()
