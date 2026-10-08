"""
Thin client for the LLM and the embedding model.

Both go through the same OpenAI-compatible endpoint (OPENAI_BASE_URL), so the
code runs unchanged against OpenAI, NVIDIA NIM, OpenRouter, Groq or a local
Ollama server - only .env changes.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)


def _client() -> OpenAI:
    return OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=float(os.getenv("OPENAI_TIMEOUT", "120")),
        max_retries=2,
    )


def _is_nvidia() -> bool:
    return "nvidia" in (os.getenv("OPENAI_BASE_URL") or "").lower()


def embed_texts(texts, input_type="passage"):
    """Embed a list of strings and return a list of vectors.

    `input_type` ("passage" for documents, "query" for questions) is required by
    NVIDIA NIM embedding models and simply ignored by other providers.
    """
    kwargs = {"model": os.getenv("EMBED_MODEL", "text-embedding-3-small"), "input": texts}
    if _is_nvidia():
        kwargs["extra_body"] = {"input_type": input_type}
    resp = _client().embeddings.create(**kwargs)
    # the API may return items out of order; sort by the returned index
    return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]


def chat(system: str, user: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
    """Single-turn chat completion, returns the assistant text."""
    resp = _client().chat.completions.create(
        model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    msg = resp.choices[0].message
    text = (msg.content or "").strip()
    if not text:
        # some reasoning models put everything in the reasoning channel when the
        # token budget is too small - fall back to it instead of returning ""
        text = (getattr(msg, "reasoning", None) or "").strip()
    return text
