"""
Shared retrieval module. All the pieces that were copy-pasted across
2_bm25.py, 3_embed.py, 4_hybrid.py, 5_rerank.py, 7_evaluate.py, rag_pipeline.py
now live HERE, defined once. Every script imports from this file.

This also fixes the demo-vs-eval mismatch: there is now ONE definition of the
hybrid+rerank pipeline, so the method you demo is exactly the method you score.
"""

import os
import json
from pathlib import Path
import numpy as np
import bm25s
import cohere
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# ---------- config (one place to change these) ----------
CHUNKS_FILE = Path("data/processed/chunks.json")
BM25_DIR = Path("indexes/bm25")
EMB_FILE = Path("indexes/dense_embeddings.npy")
EMBED_MODEL = "all-mpnet-base-v2"
CANDIDATES = 50  # how many each engine fetches for fusion/rerank
RRF_K = 60  # RRF smoothing constant

# ---------- load everything ONCE at import ----------
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

bm25 = bm25s.BM25.load(str(BM25_DIR), load_corpus=False)
model = SentenceTransformer(EMBED_MODEL)
doc_embeddings = np.load(EMB_FILE)
co = cohere.Client(os.getenv("COHERE_API_KEY"))


def _normalize(vecs):
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


doc_norm = _normalize(doc_embeddings)


# ---------- the retrieval functions (ONE definition each) ----------
def search_bm25(query, k=CANDIDATES):
    """Keyword retrieval -> ranked list of chunk indices."""
    q_tokens = bm25s.tokenize(query, stopwords="en")
    results, _ = bm25.retrieve(q_tokens, k=k)
    return list(results[0])


def search_dense(query, k=CANDIDATES):
    """Semantic retrieval -> ranked list of chunk indices."""
    q_vec = model.encode([query], convert_to_numpy=True)
    q_norm = _normalize(q_vec)[0]
    scores = doc_norm @ q_norm
    return list(np.argsort(-scores)[:k])


def reciprocal_rank_fusion(rankings, k=RRF_K):
    """Fuse multiple ranked ID lists by rank (not score)."""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: -x[1])]


def search_hybrid(query, k=5, candidates=CANDIDATES):
    """BM25 + dense, fused with RRF -> top k chunk indices."""
    bm = search_bm25(query, candidates)
    de = search_dense(query, candidates)
    return reciprocal_rank_fusion([bm, de])[:k]


def search_hybrid_rerank(query, k=5, candidates=CANDIDATES):
    fused = reciprocal_rank_fusion(
        [search_bm25(query, candidates), search_dense(query, candidates)]
    )[:candidates]
    texts = [chunks[i]["text"] for i in fused]
    reranked = co.rerank(
        model="rerank-english-v3.0", query=query, documents=texts, top_n=k
    )
    return [fused[r.index] for r in reranked.results]
