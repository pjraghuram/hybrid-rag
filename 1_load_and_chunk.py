"""
Step 1 (scaled): Load ALL PDFs in data/pdfs/, chunk them, and keep track of
which PDF each chunk came from. Save the result so later steps can load it.
Start with ~5 PDFs in the folder; the SAME code works for 37 later.
"""

import json
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_DIR = Path("data/pdfs")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)  # make sure the output folder exists

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

all_chunks = []  # every chunk from every PDF ends up here
chunk_id = 0  # a unique number for each chunk across the whole corpus

# --- loop over EVERY pdf in the folder ---
pdf_files = sorted(PDF_DIR.glob("*.pdf"))
print(f"Found {len(pdf_files)} PDFs\n")

for pdf_path in pdf_files:
    # read all the text out of this one PDF
    reader = PdfReader(str(pdf_path))
    full_text = ""
    for page in reader.pages:
        full_text += (page.extract_text() or "") + "\n"

    # split THIS pdf's text into chunks
    chunks = splitter.split_text(full_text)

    # tag each chunk with where it came from
    for chunk_text in chunks:
        all_chunks.append(
            {
                "id": chunk_id,  # unique id for this chunk
                "source": pdf_path.name,  # which PDF it came from  <-- the important part
                "text": chunk_text,  # the actual chunk content
            }
        )
        chunk_id += 1

    print(f"{pdf_path.name:20s} -> {len(chunks)} chunks")

# --- save everything to one file so later steps can just load it ---
out_file = OUT_DIR / "chunks.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2)

print(f"\nTOTAL chunks: {len(all_chunks)}")
print(f"Saved to: {out_file}")

# --- quick sanity peek ---
print("\n----- EXAMPLE CHUNK (with source tracking) -----")
example = all_chunks[len(all_chunks) // 2]
print(f"id:     {example['id']}")
print(f"source: {example['source']}")
print(f"text:   {example['text'][:300]}...")
