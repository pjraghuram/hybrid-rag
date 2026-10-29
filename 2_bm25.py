"""
Step 2: Build a BM25 keyword index from the chunks, save it, then run a test search.
BM25 = keyword/sparse retrieval. No embeddings here -- it works on words.
"""

import json
from pathlib import Path
import bm25s

CHUNKS_FILE = Path("data/processed/chunks.json")
INDEX_DIR = Path("indexes/bm25")

# --- load the chunks we saved in Step 1 ---
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# pull out just the text of each chunk, in order (index position = chunk's position)
corpus_texts = [c["text"] for c in chunks]
print(f"Loaded {len(corpus_texts)} chunks")

# --- STEP 2a: TOKENIZE the corpus (split into words, drop English stopwords) ---
# stopwords ("the", "of", "and"...) are noise for keyword scoring, so we remove them
corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en")

# --- STEP 2b: BUILD the BM25 index ---
retriever = bm25s.BM25()
retriever.index(corpus_tokens)
print("BM25 index built")

# --- STEP 2c: SAVE the index to disk (so later steps don't rebuild it) ---
INDEX_DIR.mkdir(parents=True, exist_ok=True)
retriever.save(str(INDEX_DIR))
print(f"Saved BM25 index to: {INDEX_DIR}")

# --- STEP 2d: TEST SEARCH -- the whole point, SEE it work ---
query = "What does ICH Q9 say about quality risk management?"
print(f"\n----- QUERY: {query} -----\n")

# tokenize the query the SAME way we tokenized the corpus
query_tokens = bm25s.tokenize(query, stopwords="en")

# retrieve top-5. results = chunk positions, scores = how well each matched
results, scores = retriever.retrieve(query_tokens, k=5)

# results[0] is the list of top-5 chunk indices for our single query
for rank, (doc_idx, score) in enumerate(zip(results[0], scores[0]), start=1):
    matched = chunks[doc_idx]  # look up the actual chunk
    print(
        f"#{rank}  score={score:.2f}  source={matched['source']}  (chunk id {matched['id']})"
    )
    print(f"     {matched['text'][:180].strip()}...\n")
