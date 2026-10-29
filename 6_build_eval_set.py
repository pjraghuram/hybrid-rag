"""
Step 6: Build an evaluation set. Our ICH docs have NO ground-truth question->doc
pairs (real-world problem). So we GENERATE them: for a sample of chunks, ask an
LLM to write a realistic question that chunk answers. That gives us (question,
correct_chunk_id) pairs = ground truth for NDCG.
"""

import os
import json
import random
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

CHUNKS_FILE = Path("data/processed/chunks.json")
EVAL_FILE = Path("eval/eval_set.json")
EVAL_FILE.parent.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 50  # how many question->chunk pairs to generate
MIN_CHARS = 400  # skip tiny/garbage chunks (tables, fragments)
random.seed(42)  # reproducible sample (same every run)

# --- load chunks ---
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# --- pick a clean sample: skip very short chunks (often table junk) ---
good_chunks = [c for c in chunks if len(c["text"]) >= MIN_CHARS]
sample = random.sample(good_chunks, N_SAMPLES)
print(f"Sampled {len(sample)} chunks (from {len(good_chunks)} usable chunks)")


# --- the LLM that writes questions (plain text, no forced tool-call) ---
llm = ChatGroq(model="openai/gpt-oss-20b")

PROMPT = """You are a pharmaceutical regulatory professional.
Read the following excerpt from an ICH guideline and generate ONE realistic question
that this excerpt answers.

Rules:
- The question must be answerable using ONLY this excerpt.
- Ask it in your OWN words -- do NOT copy phrases directly from the text.
- Make it the kind of question a real person would type, not a textbook heading.
- Respond with ONLY the question text. No preamble, no quotes, no explanation.

Excerpt:
{excerpt}
"""

# --- generate a question for each sampled chunk ---
eval_pairs = []
for i, chunk in enumerate(sample, start=1):
    try:
        response = llm.invoke(PROMPT.format(excerpt=chunk["text"]))
        question = response.content.strip()  # the plain-text question
        eval_pairs.append(
            {
                "question": question,
                "correct_chunk_id": chunk["id"],
                "source": chunk["source"],
            }
        )
        print(f"[{i}/{len(sample)}] {chunk['source']:20s} -> {question[:70]}")
    except Exception as e:
        print(f"[{i}/{len(sample)}] FAILED on chunk {chunk['id']}: {e}")

# --- save the eval set ---
with open(EVAL_FILE, "w", encoding="utf-8") as f:
    json.dump(eval_pairs, f, indent=2)

print(f"\nGenerated {len(eval_pairs)} eval pairs")
print(f"Saved to: {EVAL_FILE}")
