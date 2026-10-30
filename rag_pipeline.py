from retrieval import chunks, search_hybrid_rerank
from langchain_groq import ChatGroq

llm = ChatGroq(model="openai/gpt-oss-20b")


# ---------- the NEW part: answer generation with citations ----------
def answer_question(query, k=5):
    # 1. retrieve the best chunks
    chunk_ids = search_hybrid_rerank(query, k=k)
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
