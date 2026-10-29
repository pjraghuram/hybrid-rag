"""
Streamlit UI for the ICH Regulatory RAG system.
Thin front-end: imports the pipeline, wraps it in a question box + cited answer.
Run with:  streamlit run app.py
"""

import streamlit as st
from rag_pipeline import answer_question

# ---------- page setup ----------
st.set_page_config(
    page_title="ICH Regulatory Assistant", page_icon="💊", layout="centered"
)

st.title("💊 ICH Regulatory Assistant")
st.caption(
    "Ask questions about ICH pharmaceutical quality guidelines (Q1–Q14). "
    "Answers are grounded in the official documents and cite their sources. "
    "Out-of-scope questions are declined rather than answered from general knowledge."
)


# ---------- cache the heavy stuff so it loads ONCE, not every interaction ----------
# (the model + indexes already load at import; this just keeps Streamlit from re-importing)
@st.cache_resource
def _warm():
    return True


_warm()

# ---------- the question box ----------
query = st.text_input(
    "Your question:",
    placeholder="e.g. What should a stability protocol contain?",
)

# ---------- answer on submit ----------
if query:
    with st.spinner("Searching ICH guidelines..."):
        answer, sources = answer_question(query)

    # the answer
    st.markdown("### Answer")
    st.markdown(answer)

    # the sources it used (expandable so it doesn't clutter)
    st.markdown("### Sources")
    unique_sources = sorted({s["source"] for s in sources})
    st.write("**Guidelines referenced:** " + ", ".join(unique_sources))

    with st.expander("See the exact text passages used"):
        for i, s in enumerate(sources, start=1):
            st.markdown(f"**[{i}] {s['source']}**")
            st.markdown(f"> {s['text'][:400].strip()}...")
            st.divider()

# ---------- footer ----------
st.markdown("---")
st.caption(
    "Hybrid retrieval (BM25 + dense embeddings + Reciprocal Rank Fusion + Cohere re-ranking), "
    "evaluated with NDCG@10. Corpus: 37 public ICH quality guidelines."
)
