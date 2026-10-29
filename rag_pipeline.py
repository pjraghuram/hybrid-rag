"""
Step 8: Answer generation (the 'G' in RAG). Take a user question, retrieve the
best chunks with our hybrid+rerank pipeline, and have the LLM answer using ONLY
those chunks -- citing which ICH guideline each fact came from.
"""

import os
import time
import json
from pathlib import Path
import numpy as np
import bm25s
import cohere
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# ---------- load everything once ----------
with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

bm25 = bm25s.BM25.load("indexes/bm25", load_corpus=False)
model = SentenceTransformer("all-mpnet-base-v2")
doc_embeddings = np.load("indexes/dense_embeddings.npy")
llm = ChatGroq(model="openai/gpt-oss-20b")


def normalize(vecs):
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


doc_norm = normalize(doc_embeddings)


# ---------- retrieval (hybrid + rerank, the best method from Step 7) ----------
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
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: -x[1])]


def retrieve(query, k=5, candidates=50):
    """Our best pipeline: hybrid (RRF) -> rerank -> top k chunks."""
    candidate_ids = reciprocal_rank_fusion(
        [search_bm25(query, candidates), search_dense(query, candidates)]
    )[:candidates]
    texts = [chunks[i]["text"] for i in candidate_ids]
    reranked = co.rerank(
        model="rerank-english-v3.0", query=query, documents=texts, top_n=k
    )
    return [candidate_ids[r.index] for r in reranked.results]


# ---------- the NEW part: answer generation with citations ----------
def answer_question(query, k=5):
    # 1. retrieve the best chunks
    chunk_ids = retrieve(query, k=k)
    retrieved = [chunks[i] for i in chunk_ids]

    # 2. build a context block where each chunk is labeled with its source
    context = ""
    for i, c in enumerate(retrieved, start=1):
        context += f"[Source {i}: {c['source']}]\n{c['text']}\n\n"

    # 3. prompt the LLM to answer ONLY from context, and cite sources
    prompt = f"""You are a pharmaceutical regulatory assistant. Answer the user's
question using ONLY the context below, which comes from official ICH guidelines.

Rules:
- Use only information in the context. Do not use outside knowledge.
- Only if the answer is genuinely absent from the context, respond with exactly: "I could not find this in the provided ICH guidelines." Otherwise, do not mention this fallback.
- Cite only the source document name in brackets, like [Q9(R1).pdf].
- Do NOT invent line numbers, section numbers, or page numbers — you do not have that information.

Context:
{context}

Question: {query}

Answer (with citations):"""

    response = llm.invoke(prompt)
    return response.content, retrieved


# ---------- test it ----------
if __name__ == "__main__":
    question = "What is the capital of France?"
    answer, sources = answer_question(question)

    print(f"QUESTION: {question}\n")
    print("ANSWER:")
    print(answer)
    print("\n" + "=" * 60)
    print("SOURCES USED:")
    for i, s in enumerate(sources, start=1):
        print(f"  [{i}] {s['source']} (chunk id {s['id']})")
