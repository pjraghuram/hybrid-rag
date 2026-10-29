"""
Step 7 (FINALE): Evaluate every retrieval method with NDCG@10 against the eval set.
For each question, we KNOW the correct chunk id. We check: does each method rank
that correct chunk near the top? NDCG turns that into a score 0-1 (higher = better).
"""

import time
import os
import json
from pathlib import Path
import numpy as np
import bm25s
import cohere
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# ---------- load everything once ----------
with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
with open("eval/eval_set.json", "r", encoding="utf-8") as f:
    eval_set = json.load(f)

bm25 = bm25s.BM25.load("indexes/bm25", load_corpus=False)
model = SentenceTransformer("all-mpnet-base-v2")
doc_embeddings = np.load("indexes/dense_embeddings.npy")


def normalize(vecs):
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


doc_norm = normalize(doc_embeddings)

K = 10  # we evaluate NDCG@10 (look at top 10 results)


# ---------- the four retrieval methods (each returns a ranked list of chunk ids) ----------
def search_bm25(query, k=K):
    q_tokens = bm25s.tokenize(query, stopwords="en")
    results, _ = bm25.retrieve(q_tokens, k=k)
    return list(results[0])


def search_dense(query, k=K):
    q_vec = model.encode([query], convert_to_numpy=True)
    q_norm = normalize(q_vec)[0]
    scores = doc_norm @ q_norm
    return list(np.argsort(-scores)[:k])


def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: -x[1])]


def search_hybrid(query, k=K, candidates=50):
    bm = search_bm25(query, candidates)
    de = search_dense(query, candidates)
    return reciprocal_rank_fusion([bm, de])[:k]


def search_hybrid_rerank(query, k=K, candidates=50):
    candidate_ids = reciprocal_rank_fusion(
        [search_bm25(query, candidates), search_dense(query, candidates)]
    )[:candidates]
    texts = [chunks[i]["text"] for i in candidate_ids]
    reranked = co.rerank(
        model="rerank-english-v3.0", query=query, documents=texts, top_n=k
    )
    time.sleep(7)
    return [candidate_ids[r.index] for r in reranked.results]


# ---------- NDCG@10 ----------
def ndcg_at_k(ranked_ids, correct_id, k=K):
    """
    We have exactly ONE correct chunk per question.
    DCG: if the correct chunk is at position p (1-indexed), gain = 1/log2(p+1).
    If it's not in the top-k at all, score = 0.
    IDCG (ideal): correct chunk at position 1 -> 1/log2(2) = 1.0.
    NDCG = DCG / IDCG.  (Here IDCG is always 1, so NDCG = DCG.)
    """
    for position, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id == correct_id:
            return 1.0 / np.log2(position + 1)
    return 0.0  # correct chunk not found in top-k


# ---------- run every method over the whole eval set ----------
methods = {
    "BM25 only": search_bm25,
    "Dense only": search_dense,
    "Hybrid (RRF)": search_hybrid,
    "Hybrid + Rerank": search_hybrid_rerank,
}

print(f"Evaluating {len(eval_set)} questions with NDCG@{K}...\n")

results_table = {}
for name, search_fn in methods.items():
    scores = []
    for pair in eval_set:
        ranked = search_fn(pair["question"])
        score = ndcg_at_k(ranked, pair["correct_chunk_id"])
        scores.append(score)
    avg = np.mean(scores)
    results_table[name] = avg
    print(f"{name:18s}  NDCG@{K} = {avg:.4f}  ({avg * 100:.1f})")

# ---------- the final scoreboard ----------
print("\n" + "=" * 45)
print("FINAL RESULTS (NDCG@10, higher = better)")
print("=" * 45)
for name, score in sorted(results_table.items(), key=lambda x: -x[1]):
    bar = "█" * int(score * 40)
    print(f"{name:18s} {score * 100:5.1f}  {bar}")
