"""
Run the sample queries through the RAG graph and print the full response
(answer + confidence + retrieved chunks) for each one.

    python run_samples.py
"""

import json

import rag

QUERIES = [
    "What is Agentic AI?",
    "How does Agentic AI differ from traditional AI and from plain LLMs?",
    "What are the main capabilities of Agentic AI?",
    "What is a multi-agent system and how does it work?",
    "What are the challenges of orchestrating multi-agent systems?",
    "What benefits did the retail company report after adopting Agentic AI?",
    "Who won the 2026 FIFA World Cup?",  # out of scope on purpose
]


def main():
    for i, question in enumerate(QUERIES, 1):
        result = rag.answer_question(question)
        print("=" * 78)
        print(f"Q{i}. {question}")
        print(f"confidence: {result['confidence']}   grounded: {result['grounded']}")
        print("-" * 78)
        print(result["answer"])
        print(f"-- {len(result['contexts'])} chunk(s) retrieved --")
        for j, c in enumerate(result["contexts"], 1):
            print(f"   [{j}] page {c['page']}  score {c['score']}  chunk_id {c['chunk_id']}")
        print()


if __name__ == "__main__":
    main()
