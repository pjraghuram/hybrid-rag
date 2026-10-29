"""
Face-off: run the SAME query through BOTH BM25 (keyword) and Dense (semantic),
side by side. Change the QUERY line and re-run to probe different cases.
"""

import json
from pathlib import Path
import numpy as np
import bm25s
from sentence_transformers import SentenceTransformer

# ======================================================================
# CHANGE THIS LINE to test different queries
QUERY = "What's the safest way to store medicine so it doesn't go bad over time?"
TOP_K = 5
# ======================================================================

# --- load chunks (shared by both engines) ---
with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# ----------------------------------------------------------------------
# ENGINE 1: BM25 (keyword)
# ----------------------------------------------------------------------
bm25 = bm25s.BM25.load("indexes/bm25", load_corpus=False)
q_tokens = bm25s.tokenize(QUERY, stopwords="en")
bm25_results, bm25_scores = bm25.retrieve(q_tokens, k=TOP_K)

# ----------------------------------------------------------------------
# ENGINE 2: Dense (semantic)
# ----------------------------------------------------------------------
model = SentenceTransformer("all-mpnet-base-v2")
doc_embeddings = np.load("indexes/dense_embeddings.npy")


def normalize(vecs):
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


doc_norm = normalize(doc_embeddings)
q_vec = model.encode([QUERY], convert_to_numpy=True)
q_norm = normalize(q_vec)[0]
dense_scores_all = doc_norm @ q_norm
dense_top = np.argsort(-dense_scores_all)[:TOP_K]

# ----------------------------------------------------------------------
# PRINT SIDE BY SIDE
# ----------------------------------------------------------------------
print("=" * 70)
print(f"QUERY: {QUERY}")
print("=" * 70)

print("\n########## BM25 (keyword) ##########\n")
for rank, (idx, score) in enumerate(zip(bm25_results[0], bm25_scores[0]), start=1):
    m = chunks[idx]
    print(f"#{rank}  score={score:.2f}  source={m['source']}  (id {m['id']})")
    print(f"     {m['text'][:140].strip()}...\n")

print("\n########## DENSE (semantic) ##########\n")
for rank, idx in enumerate(dense_top, start=1):
    m = chunks[idx]
    print(
        f"#{rank}  score={dense_scores_all[idx]:.3f}  source={m['source']}  (id {m['id']})"
    )
    print(f"     {m['text'][:140].strip()}...\n")

# ----------------------------------------------------------------------
# QUICK OVERLAP CHECK: do the two engines agree?
# ----------------------------------------------------------------------
bm25_sources = {chunks[i]["source"] for i in bm25_results[0]}
dense_sources = {chunks[i]["source"] for i in dense_top}
shared = bm25_sources & dense_sources

print("=" * 70)
print(f"BM25 top sources:  {sorted(bm25_sources)}")
print(f"Dense top sources: {sorted(dense_sources)}")
print(
    f"Shared sources:    {sorted(shared)}  ({len(shared)}/{len(bm25_sources | dense_sources)} overlap)"
)
print("=" * 70)
