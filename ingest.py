"""
Step 1 of the pipeline: PDF -> text -> chunks -> embeddings -> vector store.

Run once before starting the API or the UI:

    python ingest.py
"""

import os
import re
import urllib.request

from dotenv import load_dotenv
from pypdf import PdfReader

import llm
import store

load_dotenv(override=True)

CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 150    # characters shared between hard-split pieces
EMBED_BATCH = 32


# ---------------------------------------------------------------- loading

def download_pdf(url: str, path: str) -> str:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"[ingest] using cached PDF: {path}")
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"[ingest] downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
        f.write(r.read())
    print(f"[ingest] saved {os.path.getsize(path)} bytes -> {path}")
    return path


def clean(text: str) -> str:
    """Tidy PDF extraction artefacts that hurt retrieval quality."""
    text = text.replace("\u00ad", "")                    # soft hyphen
    text = re.sub(r"-\n(?=[a-z])", "", text)             # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if not re.fullmatch(r"\d{1,3}", ln)]          # page numbers
    lines = [ln for ln in lines if ln.upper() != "AGENTIC AI FOR EXECUTIVES"]  # running footer
    text = "\n".join(lines)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def extract_pages(path: str):
    """Return [{page, text}] for every page that has extractable text."""
    pages = []
    for i, page in enumerate(PdfReader(path).pages):
        text = clean(page.extract_text() or "")
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages


# ---------------------------------------------------------------- chunking

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Pack paragraphs into ~size-character chunks, splitting with overlap only
    when a single paragraph is itself longer than `size`."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, buf = [], ""
    for para in paragraphs:
        if len(buf) + len(para) + 2 <= size:
            buf = f"{buf}\n\n{para}".strip()
            continue
        if buf:
            out.append(buf)
            buf = ""
        if len(para) <= size:
            buf = para
        else:
            start = 0
            while start < len(para):
                out.append(para[start : start + size].strip())
                start += size - overlap
    if buf:
        out.append(buf)
    return [c for c in out if len(c) > 40]


def chunk_pages(pages):
    chunks = []
    for page in pages:
        for text in chunk_text(page["text"]):
            chunks.append({"text": text, "page": page["page"]})
    return chunks


# ---------------------------------------------------------------- embedding

def embed_all(documents):
    vectors = []
    for i in range(0, len(documents), EMBED_BATCH):
        part = documents[i : i + EMBED_BATCH]
        vectors.extend(llm.embed_texts(part, input_type="passage"))
        print(f"[ingest] embedded {min(i + EMBED_BATCH, len(documents))}/{len(documents)}")
    return vectors


# ---------------------------------------------------------------- main

def main():
    pdf_path = download_pdf(
        os.getenv("PDF_URL", "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"),
        os.getenv("PDF_PATH", "data/Ebook-Agentic-AI.pdf"),
    )

    pages = extract_pages(pdf_path)
    print(f"[ingest] extracted text from {len(pages)} pages")

    chunks = chunk_pages(pages)
    print(f"[ingest] produced {len(chunks)} chunks")

    source = os.path.basename(pdf_path)
    ids = [f"{source}::p{c['page']}::{i}" for i, c in enumerate(chunks)]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {"page": c["page"], "chunk_id": i, "source": source} for i, c in enumerate(chunks)
    ]

    embeddings = embed_all(documents)
    print(f"[ingest] embedding dimension: {len(embeddings[0])}")

    stored = store.build(os.getenv("CHROMA_PATH", ".chroma"), ids, embeddings, documents, metadatas)
    print(f"[ingest] done - {stored} chunks stored in the Chroma collection")


if __name__ == "__main__":
    main()
