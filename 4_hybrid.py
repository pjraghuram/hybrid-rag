"""
Step 4: Hybrid retrieval. Run BM25 + Dense, then fuse their rankings with
Reciprocal Rank Fusion (RRF). RRF ignores the raw scores (different scales)
and combines by RANK position instead.
"""

import json
from pathlib import Path
import numpy as np
import bm25s
from sentence_transformers import SentenceTransformer

# ---------- load everything once ----------
with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

bm25 = bm25s.BM25.load("indexes/bm25", load_corpus=False)
model = SentenceTransformer("all-mpnet-base-v2")
doc_embeddings = np.load("indexes/dense_embeddings.npy")


def normalize(vecs):
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


doc_norm = normalize(doc_embeddings)


# ---------- the two individual retrievers ----------
def search_bm25(query, k):
    """Return a list of chunk indices, best first."""
    q_tokens = bm25s.tokenize(query, stopwords="en")
    results, scores = bm25.retrieve(q_tokens, k=k)
    return list(results[0])  # just the ranked chunk indices


def search_dense(query, k):
    """Return a list of chunk indices, best first."""
    q_vec = model.encode([query], convert_to_numpy=True)
    q_norm = normalize(q_vec)[0]
    scores = doc_norm @ q_norm
    return list(np.argsort(-scores)[:k])


# ---------- RRF: the fusion ----------
def reciprocal_rank_fusion(rankings, k=60):
    """
    rankings = a list of ranked-ID lists (one from BM25, one from dense).
    For each doc: score = sum over rankings of 1 / (k + rank).
    Docs ranked high in BOTH lists rise to the top.
    """
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):  # rank starts at 1
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    # sort docs by fused score, highest first
    return sorted(scores.items(), key=lambda x: -x[1])


# ---------- the hybrid search ----------
def hybrid_search(query, k=5, candidates=50):
    """
    Over-fetch 'candidates' from each engine, fuse with RRF, return top-k.
    """
    bm25_ranking = search_bm25(query, candidates)
    dense_ranking = search_dense(query, candidates)
    fused = reciprocal_rank_fusion([bm25_ranking, dense_ranking])
    return fused[:k]  # list of (doc_id, fused_score)


# ======================================================================
# TEST: run the query that made the engines DIVERGE in the face-off
# ======================================================================
QUERY = "What's the safest way to store medicine so it doesn't go bad over time?"
print(f"QUERY: {QUERY}\n")
print("=" * 70)

results = hybrid_search(QUERY, k=5)

for rank, (doc_id, score) in enumerate(results, start=1):
    m = chunks[doc_id]
    print(f"#{rank}  rrf_score={score:.4f}  source={m['source']}  (id {m['id']})")
    print(f"     {m['text'][:140].strip()}...\n")
