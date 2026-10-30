"""
Step 3: Create dense (semantic) embeddings for every chunk, save as a .npy file,
then run a CONCEPTUAL test query to start seeing where dense beats keyword search.
This is the SLOW one-time step (~5-10 min on CPU for mpnet). Saved after, so it's one-time.
"""

import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path("data/processed/chunks.json")
EMB_FILE = Path("indexes/dense_embeddings.npy")

# --- load the chunks from Step 1 ---
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)
texts = [c["text"] for c in chunks]
print(f"Loaded {len(texts)} chunks")

# --- load the embedding model (downloads ~420MB the first time, then cached) ---
print("Loading model all-mpnet-base-v2 ...")
model = SentenceTransformer("all-mpnet-base-v2")

# --- BUILD or LOAD: don't re-embed if we already did it once ---
if EMB_FILE.exists():
    print("Embeddings already exist, loading from disk.")
    doc_embeddings = np.load(EMB_FILE)

    # --- STALE-EMBEDDING GUARD ---
    # The .npy has one vector per chunk. If the chunk count no longer matches,
    # the saved embeddings are from an OLD chunking and every dense result would
    # silently point at the wrong text. Fail loudly instead.
    if len(doc_embeddings) != len(texts):
        raise ValueError(
            f"\nSTALE EMBEDDINGS DETECTED:\n"
            f"  saved embeddings : {len(doc_embeddings)} vectors\n"
            f"  current chunks   : {len(texts)} chunks\n"
            f"These don't match -- the .npy is from a different chunking.\n"
            f"Fix: delete '{EMB_FILE}' and re-run this script to re-embed.\n"
        )
    print(f"OK: {len(doc_embeddings)} embeddings match {len(texts)} chunks.")

else:
    print("Embedding all chunks... (this is the slow part, be patient)")
    doc_embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    EMB_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMB_FILE, doc_embeddings)
    print(f"Saved embeddings to: {EMB_FILE}")

print(f"Embeddings shape: {doc_embeddings.shape}")  # (num_chunks, 768)


# --- NORMALIZE so cosine similarity == dot product (Dave's trick) ---
def normalize(vecs):
    # divide each vector by its length -> all vectors become length 1
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


doc_norm = normalize(doc_embeddings)

# --- TEST SEARCH with a CONCEPTUAL query (no exact keywords) ---
query = "How do I decide which manufacturing changes need regulatory approval before making them?"
print(f"\n----- QUERY: {query} -----\n")

# embed the query the same way, normalize it too
q_vec = model.encode([query], convert_to_numpy=True)
q_norm = normalize(q_vec)[0]

# cosine similarity = dot product (because everything is normalized)
scores = doc_norm @ q_norm  # one score per chunk
top5 = np.argsort(-scores)[:5]  # indices of the 5 highest scores

for rank, idx in enumerate(top5, start=1):
    matched = chunks[idx]
    print(
        f"#{rank}  score={scores[idx]:.3f}  source={matched['source']}  (chunk id {matched['id']})"
    )
    print(f"     {matched['text'][:180].strip()}...\n")
