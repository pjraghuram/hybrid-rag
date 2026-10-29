# 💊 ICH Regulatory Assistant — Hybrid RAG

A retrieval-augmented generation (RAG) system for answering questions about the **ICH pharmaceutical quality guidelines (Q1–Q14)**. It combines keyword search (BM25) with semantic search (dense embeddings), fuses them with Reciprocal Rank Fusion, re-ranks with a Cohere cross-encoder, and generates **source-cited** answers with an LLM — all behind a simple Streamlit UI.

Every retrieval method is benchmarked with **NDCG@10** on an LLM-generated evaluation set, so design choices are **measured, not assumed**.

> **Why regulatory documents?** ICH guidelines are dense, terminology-heavy, and identifier-rich (exact codes like `Q9(R1)`, `21 CFR`, section references) *and* conceptual — a corpus where both keyword and semantic retrieval have real, different strengths. In a regulated domain, answers must also be **traceable to their source**, which makes grounded, citable generation a hard requirement rather than a nice-to-have.

---

## Demo

<!-- Replace these with your own screenshots -->
| Cited regulatory answer | Safe refusal of out-of-scope questions |
|---|---|
| ![answer screenshot](docs/answer.png) | ![refusal screenshot](docs/refusal.png) |

*The assistant answers regulatory questions with citations to the source guideline, and declines questions outside the corpus rather than answering from the model's general knowledge.*

---

## How it works

```
            ┌─▶ BM25 (top 50) ──┐
question ───┤                   ├─▶ RRF fusion ─▶ Cohere rerank (top 5) ─▶ LLM (Groq) ─▶ cited answer
            └─▶ Dense (top 50) ─┘
```

1. **Load & chunk** — Every PDF in `data/pdfs/` is parsed with `pypdf` and split into ~1000-character chunks (200-character overlap). Each chunk keeps track of the guideline it came from (used for citations and for scoring retrieval).
2. **Sparse retrieval (BM25)** — A keyword index built with `bm25s` (English stopwords removed). Strong on exact regulatory terms and guideline names.
3. **Dense retrieval** — Chunks embedded with `sentence-transformers/all-mpnet-base-v2` (768-dim) and searched by cosine similarity (normalized vectors → dot product, in NumPy — no vector database needed at this scale). Catches paraphrased and conceptual questions that share few keywords with the text.
4. **Hybrid fusion (RRF)** — Both rankings are combined with Reciprocal Rank Fusion (`score = Σ 1 / (60 + rank)`), which merges by rank position so the two engines' very different score scales don't matter.
5. **Re-ranking** — The fused candidates are re-scored by Cohere's `rerank-english-v3.0` cross-encoder, which reads the query and each passage *together* (a cross-encoder) rather than separately (a bi-encoder), giving far more precise relevance judgments on the shortlist.
6. **Answer generation** — The top 5 passages, each labeled with its source guideline, are passed to `openai/gpt-oss-20b` (via Groq). The model is instructed to answer **only** from that context, **cite the guideline for each fact** (e.g. `[Q9(R1).pdf]`), and **decline out-of-scope questions** instead of using outside knowledge.

---

## Grounded, citable answers

Because this targets a regulated domain, the generation step is built for **traceability and safety**, not just fluency:

- **Source citations.** Every answer cites the guideline each fact came from (e.g. `[Q9(R1).pdf]`), so a user can verify each statement against the official document. Citations are **document-level** — the LLM is explicitly instructed *not* to invent line/section numbers, since that information isn't in the retrieved context. (An early version hallucinated precise line numbers; this was caught and constrained. Real page-level citations are planned — see *Limitations & future work*.)
- **Safe refusal.** The model answers **only** from the retrieved passages. Asked something outside the corpus (e.g. *"What is the capital of France?"*), it responds *"I could not find this in the provided ICH guidelines"* rather than answering from its own knowledge — because in a regulatory context, a confident hallucinated answer is worse than no answer.

---

## Corpus

37 public ICH quality guideline PDFs (Q1A–Q1F, Q2–Q14, plus Q&A and training documents), producing **3,038 chunks**.

The PDFs themselves are **not** included in this repository — download them from [ich.org](https://www.ich.org/page/quality-guidelines) and place them in `data/pdfs/`.

---

## Evaluation

ICH guidelines come with **no ground-truth question → passage pairs** — the real-world problem with any private corpus. So `6_build_eval_set.py` **generates** them: it samples substantial chunks (≥400 characters, fixed random seed for reproducibility) and asks an LLM to write a realistic question each chunk answers, **in its own words** (paraphrased, so the task tests retrieval rather than keyword overlap). That yields `(question, correct_chunk_id)` pairs.

`7_evaluate.py` runs each retrieval method over the eval set and scores it with **NDCG@10** (1.0 = the correct chunk ranked first, 0 = not in the top 10).

| Method | NDCG@10 |
|---|---|
| Dense only | 0.472 |
| BM25 only | 0.665 |
| Hybrid (RRF) | 0.670 |
| **Hybrid + Rerank** | **0.892** |

### Findings

- **BM25 beat dense embeddings by ~19 points (0.665 vs 0.472)** — the *opposite* of the common "semantic search wins" assumption. Regulatory text relies on precise, consistent terminology (e.g. "established conditions," "stability protocol," specific guideline codes), which rewards exact-term matching over semantic similarity, where embeddings tend to blur distinct technical terms.
- **RRF fusion added little on its own (0.665 → 0.670).** With a strong keyword signal and a comparatively weak dense signal on this corpus, fusing them barely moved the score — an honest result, not every pipeline component helps equally.
- **Re-ranking was the single highest-leverage step: a 22-point jump (0.670 → 0.892).** The Cohere cross-encoder, reading query and passage together, dramatically improved ranking quality.
- **Takeaway: the best retrieval method is corpus-dependent.** The value here came almost entirely from reranking, not fusion — which is exactly why the system is *evaluated* rather than assumed to follow textbook patterns.

> **Note on absolute scores.** The eval set is LLM-generated from the chunks themselves, so absolute NDCG is optimistic relative to real user queries. The **relative comparison between methods is still valid**, because all four methods are evaluated against the identical eval set.

---

## Limitations & future work

- **Synthetic eval set.** Questions are generated from the chunks, so absolute scores are optimistic vs. messy real-world queries. A next step is validating against real user questions (and, per LLM-eval best practice, aligning any automated judge to human judgments).
- **Document-level citations only.** The LLM is constrained from inventing precise locations; tracking page metadata through the chunking step would enable **real page-level citations** (`[Q1F.pdf, p.12]`).
- **Table extraction.** Dense, table-heavy passages (e.g. PDE limit tables) extract imperfectly with `pypdf` and add retrieval noise. Structure-aware extraction is a candidate improvement.
- **Chunking strategy.** Fixed-size chunking is the baseline. **Parent-child ("small-to-big") retrieval** — search small chunks, return their larger parent sections — is a planned iteration to measure against the current NDCG.
- **Model comparison.** `all-mpnet-base-v2` was chosen for quality; a measured comparison against a faster model (e.g. `all-MiniLM-L6-v2`) would quantify the quality/latency trade-off.

---

## Project structure

```
.
├── 1_load_and_chunk.py     # PDFs -> data/processed/chunks.json
├── 2_bm25.py               # build + save BM25 index, test query
├── 3_embed.py              # build + save dense embeddings, test query
├── 3b_test_faceoff.py      # same query through BM25 and dense, side by side
├── 4_hybrid.py             # BM25 + dense fused with RRF
├── 5_rerank.py             # hybrid vs. hybrid + Cohere rerank
├── 6_build_eval_set.py     # generate (question, chunk) eval pairs with an LLM
├── 7_evaluate.py           # NDCG@10 for all four retrieval methods
├── rag_pipeline.py         # retrieval + cited answer generation (importable)
├── app.py                  # Streamlit front-end
├── requirements.txt
├── docs/                   # screenshots for this README
├── data/
│   ├── pdfs/               # ICH guideline PDFs (not committed)
│   └── processed/          # chunks.json (generated)
├── indexes/
│   ├── bm25/               # BM25 index (generated)
│   └── dense_embeddings.npy
└── eval/
    └── eval_set.json       # generated eval questions
```

The numbered scripts are meant to be run in order — each builds on the previous one's output and prints results so you can see what that stage does.

---

## Setup

Requires Python 3.10+ (developed on 3.13).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_key
COHERE_API_KEY=your_cohere_key
```

- Groq key: <https://console.groq.com>
- Cohere key: <https://dashboard.cohere.com> (free trial tier is sufficient; it is rate-limited to 10 rerank calls/min, which `7_evaluate.py` paces automatically)

---

## Usage

**1. Build the indexes** (one-time, run from the project root):

```bash
python 1_load_and_chunk.py   # chunk the PDFs
python 2_bm25.py             # BM25 index
python 3_embed.py            # dense embeddings (~5–10 min on CPU; model downloads on first run)
```

`3_embed.py` reuses `indexes/dense_embeddings.npy` if it already exists — delete that file to re-embed after re-chunking.

**2. Explore the retrieval stages** (optional):

```bash
python 3b_test_faceoff.py    # edit QUERY at the top to try your own
python 4_hybrid.py
python 5_rerank.py
```

**3. Evaluate:**

```bash
python 6_build_eval_set.py   # generates eval/eval_set.json
python 7_evaluate.py         # prints the NDCG@10 scoreboard
```

**4. Ask questions:**

```bash
streamlit run app.py
```

Or from Python:

```python
from rag_pipeline import answer_question

answer, sources = answer_question("What should a stability protocol contain?")
```

---

## Tech stack

| Component | Tool |
|---|---|
| PDF parsing | `pypdf` |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Sparse retrieval | `bm25s` |
| Dense retrieval | `sentence-transformers` (`all-mpnet-base-v2`) |
| Fusion | Reciprocal Rank Fusion (custom) |
| Re-ranking | Cohere `rerank-english-v3.0` |
| LLM | `openai/gpt-oss-20b` via Groq (`langchain-groq`) |
| UI | Streamlit |
| Evaluation | NDCG@10 (custom) |

---

## Notes

- This is an independent project built on **public** ICH guidance to explore and evaluate production retrieval techniques. It is not affiliated with ICH and is not a source of regulatory advice.