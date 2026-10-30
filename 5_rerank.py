from retrieval import chunks, search_hybrid, search_hybrid_rerank

QUERY = "What's the safest way to store medicine so it doesn't go bad over time?"

print("HYBRID ONLY:")
for rank, doc_id in enumerate(search_hybrid(QUERY, k=5), start=1):
    print(
        f"#{rank}  {chunks[doc_id]['source']}  {chunks[doc_id]['text'][:80].strip()}..."
    )

print("\nHYBRID + RERANK:")
for rank, doc_id in enumerate(search_hybrid_rerank(QUERY, k=5), start=1):
    print(
        f"#{rank}  {chunks[doc_id]['source']}  {chunks[doc_id]['text'][:80].strip()}..."
    )
