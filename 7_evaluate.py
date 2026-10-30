import json, time
import numpy as np
from retrieval import (
    chunks,
    search_bm25,
    search_dense,
    search_hybrid,
    search_hybrid_rerank,
)

with open("eval/eval_set.json") as f:
    eval_set = json.load(f)

K = 10


def ndcg_at_k(ranked_ids, correct_id, k=K):
    for position, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id == correct_id:
            return 1.0 / np.log2(position + 1)
    return 0.0


methods = {
    "BM25 only": lambda q: search_bm25(q, k=K),
    "Dense only": lambda q: search_dense(q, k=K),
    "Hybrid (RRF)": lambda q: search_hybrid(q, k=K),
    "Hybrid + Rerank": lambda q: search_hybrid_rerank(q, k=K),
}

for name, fn in methods.items():
    scores = []
    for p in eval_set:
        scores.append(ndcg_at_k(fn(p["question"]), p["correct_chunk_id"]))
        if name == "Hybrid + Rerank":
            time.sleep(7)  # pace only the eval, only for the reranker
    print(f"{name:18s} NDCG = {np.mean(scores):.4f}")
