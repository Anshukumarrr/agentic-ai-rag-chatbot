r"""
The RAG pipeline, expressed as a LangGraph state machine.

    START -> retrieve --(relevant context found)--> generate   -> END
                     \--(nothing relevant)-------> no_context -> END

* retrieve    : embeds the question and pulls the top-k chunks from the vector store
* generate    : asks the LLM to answer using ONLY those chunks
* no_context  : skips the LLM when retrieval found nothing useful, so the LLM can
                never invent an answer for an out-of-scope question
"""

import os
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

import llm
import store

load_dotenv(override=True)

NOT_FOUND = "I could not find this in the provided knowledge base."

ANSWER_SYSTEM = (
    "You are a documentation assistant. Answer the user's question using ONLY the "
    "numbered context excerpts from the 'Agentic AI' ebook.\n"
    "Rules:\n"
    "1. Use only facts stated in the excerpts. Never use outside knowledge.\n"
    f'2. If the excerpts do not contain the answer, reply exactly: "{NOT_FOUND}"\n'
    "3. Be concise (at most 5 sentences) and cite the excerpts you used, e.g. (excerpt 2).\n"
)


class RAGState(TypedDict):
    question: str
    contexts: List[Dict[str, Any]]
    answer: str
    confidence: float
    grounded: bool


def _settings():
    return {
        "top_k": int(os.getenv("TOP_K", "4")),
        "min_score": float(os.getenv("MIN_SCORE", "0.25")),
        "path": os.getenv("CHROMA_PATH", ".chroma"),
    }


def retrieve(state: RAGState) -> Dict[str, Any]:
    """Node 1: embed the question, fetch the most similar chunks."""
    s = _settings()
    query_vector = llm.embed_texts([state["question"]], input_type="query")[0]
    return {"contexts": store.search(s["path"], query_vector, s["top_k"])}


def route(state: RAGState) -> str:
    """Conditional edge: is the best chunk similar enough to be worth answering?"""
    hits = state.get("contexts") or []
    if hits and hits[0]["score"] >= _settings()["min_score"]:
        return "generate"
    return "no_context"


def generate(state: RAGState) -> Dict[str, Any]:
    """Node 2: answer strictly from the retrieved chunks."""
    contexts = state["contexts"]
    block = "\n\n".join(
        f"[excerpt {i + 1} | page {c['page']}]\n{c['text']}" for i, c in enumerate(contexts)
    )
    user = (
        f"Context excerpts:\n{block}\n\n"
        f"Question: {state['question']}\n\n"
        "Answer:"
    )
    answer = llm.chat(ANSWER_SYSTEM, user)
    scores = [c["score"] for c in contexts]
    return {
        "answer": answer,
        "confidence": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "grounded": NOT_FOUND.lower() not in answer.lower(),
    }


def no_context(state: RAGState) -> Dict[str, Any]:
    """Node 3: nothing relevant was retrieved."""
    hits = state.get("contexts") or []
    return {"answer": NOT_FOUND, "confidence": round(hits[0]["score"], 4) if hits else 0.0,
            "grounded": False}


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("no_context", no_context)
    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges(
        "retrieve", route, {"generate": "generate", "no_context": "no_context"}
    )
    graph.add_edge("generate", END)
    graph.add_edge("no_context", END)
    return graph.compile()


_graph = None


def answer_question(question: str) -> Dict[str, Any]:
    """Run the graph and return the shape the API/UI needs."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    final = _graph.invoke(
        {"question": question, "contexts": [], "answer": "", "confidence": 0.0, "grounded": False}
    )
    return {
        "question": question,
        "answer": final["answer"],
        "contexts": final["contexts"],
        "confidence": final["confidence"],
        "grounded": final["grounded"],
    }
