"""Sentence embeddings (the language side of EEG-to-language decoding)."""
import numpy as np
from pathlib import Path


def embed_sentences(texts, kind="sbert", model="sentence-transformers/all-MiniLM-L6-v2"):
    if kind == "sbert":
        from sentence_transformers import SentenceTransformer
        v = SentenceTransformer(model).encode(texts, batch_size=64, normalize_embeddings=True,
                                              show_progress_bar=False)
        return np.asarray(v, np.float32)
    if kind == "tfidf":  # offline fallback (also used for smoke tests)
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        X = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit_transform(texts)
        k = max(2, min(128, X.shape[1] - 1, X.shape[0] - 1))
        v = TruncatedSVD(k, random_state=0).fit_transform(X)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
        return v.astype(np.float32)
    raise ValueError(kind)


def text_vectors(meta, kind, out_dir):
    """dict: sentence-id -> L2-normalised embedding (cached)."""
    p = Path(out_dir) / f"text_{kind}.npz"
    u = meta.drop_duplicates("sid")[["sid", "text"]]
    if p.exists():
        z = np.load(p, allow_pickle=True)
        d = dict(zip(z["keys"], z["vecs"]))
        if all(s in d for s in u.sid):
            return d
    v = embed_sentences(list(u.text), kind)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.savez(p, keys=np.array(u.sid), vecs=v)
    return dict(zip(u.sid, v))
