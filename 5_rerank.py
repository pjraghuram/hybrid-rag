"""
Step 5: Re-ranking. Take the hybrid (RRF) candidates and reorder them with a
cross-encoder (Cohere rerank) that looks at the query + each document TOGETHER.
This is usually the single biggest quality jump in the whole pipeline.
"""

import os
import json
from pathlib import Path
import numpy as np
import bm25s
import cohere
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()  # load COHERE_API_KEY from .env
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# ---------- load everything once ----------
with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

bm25 = bm25s.BM25.load("indexes/bm25", load_corpus=False)
model = SentenceTransformer("all-mpnet-base-v2")
doc_embeddings = np.load("indexes/dense_embeddings.npy")


def normalize(vecs):
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


doc_norm = normalize(doc_embeddings)


# ---------- the two retrievers + RRF (same as Step 4) ----------
def search_bm25(query, k):
    q_tokens = bm25s.tokenize(query, stopwords="en")
    results, _ = bm25.retrieve(q_tokens, k=k)
    return list(results[0])


def search_dense(query, k):
    q_vec = model.encode([query], convert_to_numpy=True)
    q_norm = normalize(q_vec)[0]
    scores = doc_norm @ q_norm
    return list(np.argsort(-scores)[:k])


def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def hybrid_search(query, candidates=50):
    """Return a list of candidate chunk ids (not yet reranked)."""
    bm25_ranking = search_bm25(query, candidates)
    dense_ranking = search_dense(query, candidates)
    fused = reciprocal_rank_fusion([bm25_ranking, dense_ranking])
    return [doc_id for doc_id, _ in fused]


# ---------- the NEW part: re-ranking ----------
def hybrid_rerank_search(query, k=5, candidates=50):
    """
    1. Get hybrid candidates (RRF).
    2. Send query + candidate TEXTS to Cohere rerank.
    3. Return the top-k AFTER reranking.
    """
    candidate_ids = hybrid_search(query, candidates)  # e.g. 50 candidate ids
    candidate_texts = [chunks[i]["text"] for i in candidate_ids]

    # the cross-encoder reads query + each document together and scores relevance
    reranked = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=candidate_texts,
        top_n=k,
    )

    # reranked gives us positions INTO candidate_texts; map back to chunk ids
    results = []
    for r in reranked.results:
        chunk_id = candidate_ids[r.index]
        results.append((chunk_id, r.relevance_score))
    return results


# ======================================================================
# TEST: same storage query, compare hybrid vs hybrid+rerank
# ======================================================================
QUERY = "What's the safest way to store medicine so it doesn't go bad over time?"
print(f"QUERY: {QUERY}\n")

print("=" * 70)
print("HYBRID ONLY (RRF, no rerank):")
print("=" * 70)
for rank, doc_id in enumerate(hybrid_search(QUERY)[:5], start=1):
    m = chunks[doc_id]
    print(f"#{rank}  source={m['source']}  (id {m['id']})  {m['text'][:90].strip()}...")

print("\n" + "=" * 70)
print("HYBRID + RERANK:")
print("=" * 70)
for rank, (doc_id, score) in enumerate(hybrid_rerank_search(QUERY, k=5), start=1):
    m = chunks[doc_id]
    print(
        f"#{rank}  rerank={score:.3f}  source={m['source']}  (id {m['id']})  {m['text'][:90].strip()}..."
    )
