"""
Chat API (FastAPI).

    uvicorn api:app --reload

Interactive docs: http://127.0.0.1:8000/docs
"""

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import rag
import store

app = FastAPI(title="Agentic AI RAG Chatbot", version="1.0")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What is Agentic AI?"])


class ContextChunk(BaseModel):
    text: str
    page: Optional[int] = None
    chunk_id: Optional[int] = None
    score: float


class ChatResponse(BaseModel):
    question: str
    answer: str
    contexts: List[ContextChunk]
    confidence: float
    grounded: bool


@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": store.count(os.getenv("CHROMA_PATH", ".chroma"))}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")
    return rag.answer_question(question)
