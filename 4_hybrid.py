from retrieval import chunks, search_hybrid

QUERY = "What's the safest way to store medicine so it doesn't go bad over time?"
for rank, doc_id in enumerate(search_hybrid(QUERY, k=5), start=1):
    m = chunks[doc_id]
    print(
        f"#{rank}  source={m['source']}  (id {m['id']})  {m['text'][:120].strip()}..."
    )
