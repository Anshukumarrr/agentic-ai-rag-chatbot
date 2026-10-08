# Agentic AI RAG Chatbot

A RAG (Retrieval-Augmented Generation) chatbot that answers questions using **only**
the *Agentic AI — An Executive's Guide* ebook published by Konverge AI
(<https://konverge.ai/pdf/Ebook-Agentic-AI.pdf>).

Built with **Python + LangGraph + ChromaDB + text embeddings**, and exposed through
both a **FastAPI chat API** and a **minimal Streamlit UI**.

Every response returns the three things the task asked for:

| Field | Meaning |
|---|---|
| `answer` | The generated answer, grounded in the retrieved excerpts |
| `contexts` | The retrieved chunks (text, page number, chunk id, similarity score) |
| `confidence` | Mean similarity score of the retrieved chunks, plus a `grounded` flag |

---

## Architecture

```
                       index time (run once)

  PDF ──► extract text ──► clean ──► chunk (~1000 chars) ──► embed ──► ChromaDB
                                                                      (persistent)

                       query time (per request)

  question
     │
     ▼
  ┌─────────────┐   embed question, top-k similarity search
  │  retrieve   │──────────────────────────────────────────────► ChromaDB
  └─────────────┘
     │
     │ best score >= MIN_SCORE ?
     ├────────────── yes ──────────► ┌────────────┐   prompt = system rules + numbered excerpts
     │                               │  generate  │──────────► LLM (OpenAI-compatible)
     │                               └────────────┘   answer + confidence
     │
     └────────────── no ───────────► ┌─────────────┐
                                     │ no_context  │  fixed "not in the knowledge base" reply
                                     └─────────────┘  (the LLM is not called)
```

The query path is a **LangGraph** `StateGraph` with two nodes and one conditional
edge (`rag.py`):

```
START → retrieve ──(relevant)──► generate ──► END
                └──(not relevant)─► no_context ──► END
```

The conditional edge is what keeps the answers grounded: if nothing in the PDF is
similar enough to the question, generation is skipped entirely, so out-of-scope
questions cannot be answered from the model's own knowledge.

### Files

| File | Purpose |
|---|---|
| `ingest.py` | Downloads the PDF, extracts/cleans text, chunks it, embeds it, loads Chroma |
| `store.py` | Vector store wrapper (ChromaDB) - `build` / `search` / `count` |
| `llm.py` | OpenAI-compatible client for the chat model and the embedding model |
| `rag.py` | The LangGraph RAG pipeline (`retrieve` / `generate` / `no_context`) |
| `api.py` | FastAPI chat API |
| `ui.py` | Minimal Streamlit UI |
| `run_samples.py` | Runs the sample queries and prints the full responses |
| `sample_queries.md` | The sample queries and what each one exercises |

---

## Setup

Requires **Python 3.10+**.

```bash
git clone https://github.com/Anshukumarrr/agentic-ai-rag-chatbot
cd appening-rag-chatbot

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
```

Edit `.env`. Any **OpenAI-compatible** endpoint works; you need one chat model and
one embedding model:

```ini
OPENAI_API_KEY=nvapi-xxxxxxxx
OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
CHAT_MODEL=openai/gpt-oss-20b
EMBED_MODEL=nvidia/nemotron-3-embed-1b
```

<details>
<summary>Other providers (same code, different values)</summary>

| Provider | `OPENAI_BASE_URL` | `CHAT_MODEL` | `EMBED_MODEL` |
|---|---|---|---|
| OpenAI | *(leave empty)* | `gpt-4o-mini` | `text-embedding-3-small` |
| NVIDIA NIM | `https://integrate.api.nvidia.com/v1` | `openai/gpt-oss-20b` | `nvidia/nemotron-3-embed-1b` |
| Ollama (local) | `http://localhost:11434/v1` | `llama3.1` | `nomic-embed-text` |

The embedding model is used for both the stored passages and the query; the code
passes NVIDIA's required `input_type` automatically.

</details>

### Build the index (once)

```bash
python ingest.py
```

This downloads the ebook into `data/`, chunks it, embeds it and writes a persistent
Chroma collection into `.chroma/`. It prints the page/chunk/vector counts as it goes.

---

## Run

### Option A — Chat API (FastAPI)

```bash
uvicorn api:app --reload
```

* `GET /health` → `{"status": "ok", "indexed_chunks": N}`
* `POST /chat` → answer + contexts + confidence
* Interactive docs at <http://127.0.0.1:8000/docs>

```bash
curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Agentic AI?"}'
```

```json
{
  "question": "What is Agentic AI?",
  "answer": "Agentic AI refers to systems that can autonomously decide and act to pursue specific objectives (excerpt 2). These agents are goal-driven, learning continuously and adapting to new situations (excerpt 3). They act independently, anticipate needs, and focus on achieving defined goals rather than merely following static rules (excerpt 4).",
  "contexts": [
    {"text": "...", "page": 7, "chunk_id": 6, "score": 0.6064},
    {"text": "...", "page": 18, "chunk_id": 22, "score": 0.5697},
    {"text": "...", "page": 9, "chunk_id": 10, "score": 0.5482},
    {"text": "...", "page": 11, "chunk_id": 14, "score": 0.5391}
  ],
  "confidence": 0.5658,
  "grounded": true
}
```

### Option B — Streamlit UI

```bash
streamlit run ui.py
```

Plain page, no theme or colour: a question box, the answer, the confidence score and
the retrieved chunks in collapsible sections.

### Sample queries

See [`sample_queries.md`](sample_queries.md), or run all of them at once:

```bash
python run_samples.py
```

The output of a full run (answers + scores + retrieved chunk ids, including a refused
out-of-scope question) is committed as [`sample_results.txt`](sample_results.txt).

---

## Design notes

**Grounding.** The system prompt allows the model to use only the numbered excerpts,
and requires the exact sentence *"I could not find this in the provided knowledge
base."* when they do not contain the answer. Independently of the model, the
`retrieve → route` conditional edge refuses the question before generation if the best
chunk scores below `MIN_SCORE`. Two layers, so an out-of-scope question can neither be
answered from prior knowledge nor hallucinated into an answer.

**Confidence.** `confidence` is the mean cosine similarity of the retrieved chunks
(Chroma is created with `hnsw:space = cosine`, so `score = 1 - distance`). It is a
*retrieval* confidence, not a calibrated probability: a high score means the PDF
contains text close to the question, not that the generated sentence is guaranteed
correct. The per-chunk `score` is returned as well so the retrieval quality is
inspectable.

**Chunking.** Text is cleaned of PDF artefacts (soft hyphens, hyphenated line breaks,
page numbers, the running footer), then packed paragraph-by-paragraph into ~1000
character chunks, with 150 characters of overlap only when a single paragraph has to
be split. Paragraph packing keeps tables and bullet lists intact where possible.

**Vector DB.** The task accepts *Pinecone or any vector DB*. ChromaDB is used because
it is local and needs no account or API key, so the project runs from a fresh clone
with nothing but an LLM API key. `store.py` is the only file that touches the vector
DB - moving to Pinecone means reimplementing `build()`, `search()` and `count()`.

## Limitations

* Confidence is similarity-based, so it is only meaningful relative to other queries.
* Fixed-size chunks are used to keep dependencies small; a token-based recursive
  splitter would give slightly cleaner boundaries on the tables in this PDF.
* The index is rebuilt from scratch on every `ingest.py` run (a 40-page PDF, seconds).
