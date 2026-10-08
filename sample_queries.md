# Sample queries

Six in-scope questions plus one out-of-scope control. Every in-scope answer should
be traceable to the retrieved excerpts (each excerpt carries its page number).

| # | Query | What it exercises |
|---|-------|-------------------|
| 1 | What is Agentic AI? | Definition / chapter 1 |
| 2 | How does Agentic AI differ from traditional AI and from plain LLMs? | Comparison tables in chapter 1 |
| 3 | What are the main capabilities of Agentic AI? | Capability list in chapter 1 |
| 4 | What is a multi-agent system and how does it work? | Chapter 3 |
| 5 | What are the challenges of orchestrating multi-agent systems? | Chapter 4 |
| 6 | What benefits did the retail company report after adopting Agentic AI? | Numbers/benefits section in chapter 1 |
| 7 | *(out of scope)* Who won the 2026 FIFA World Cup? | Must be refused - not in the PDF |

Expected behaviour:

* **1-6** -> an answer drawn only from the retrieved chunks, plus the chunk list,
  page numbers and a confidence score.
* **7** -> no chunk passes `MIN_SCORE`, so the graph takes the `no_context` branch
  and returns `I could not find this in the provided knowledge base.` with
  `grounded: false`. The LLM is never called.

## Running them

```bash
python run_samples.py          # prints every query, its answer, score and chunks
```

or one at a time:

```bash
curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Agentic AI?"}'
```
