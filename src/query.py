"""Natural Language Query Processing for TempusRAG.

Flow 2: query rewriting, HyDE, retrieval, answer generation with citations.
Uses google.genai (new client style): no global state, thread-safe per-call client.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from google import genai

from src.config import EXTRACTOR_MODEL, GEMINI_API_KEY
from src.models import Chunk
from src.retriever import hybrid_retrieve

logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google.auth").setLevel(logging.WARNING)


def _gemini_call(prompt: str, api_key: str) -> str:
    """Single Gemini call: fresh client per request, no global state."""
    import src.config as cfg
    model = getattr(cfg, "EXTRACTOR_MODEL", "gemini-3.5-flash")
    fallback = getattr(cfg, "EXTRACTOR_FALLBACK_MODEL", "gemini-3.5-flash-lite")
    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"temperature": 0.3, "max_output_tokens": 2048},
        )
    except Exception as e:
        if fallback and fallback != model:
            logger.warning("Primary query model %r failed (%s); trying %r", model, e, fallback)
            response = client.models.generate_content(
                model=fallback,
                contents=prompt,
                config={"temperature": 0.3, "max_output_tokens": 2048},
            )
        else:
            raise

    if not response.candidates:
        raise RuntimeError("Gemini returned no candidates")
    text = response.text
    if text is None:
        raise RuntimeError("Gemini returned a candidate with no text")
    return text


# ---------------------------------------------------------------------------
# Follow-up detection
# ---------------------------------------------------------------------------

def detect_followup_query(
    query: str,
    conversation_history: list[str] | None = None,
) -> bool:
    """Return True if the query looks like a follow-up needing prior context."""
    q = query.lower().strip()
    followup_indicators = [
        "it", "that", "this", "they", "them", "those", "these",
        "what about", "how about", "and that", "regarding that",
        "about it", "for that", "with that", "from that",
    ]
    has_pronouns = any(ind in q for ind in followup_indicators)
    starts_bare = bool(re.match(r"^(what|how|when|where|why|which)\s*\?*$", q))
    very_short = len(query.split()) <= 3
    lacks_specifics = not re.search(
        r"\b(technology|claims|growth|finance|operations|revenue|profit|investment)\b",
        q,
    )
    return has_pronouns or (very_short and lacks_specifics) or starts_bare


def rewrite_query(
    query: str,
    conversation_history: list[str] | None = None,
    api_key: str | None = None,
) -> str:
    """Expand a follow-up query into a self-contained standalone question.

    Falls back to the original query if no API key is available or the call fails.
    """
    if not detect_followup_query(query, conversation_history):
        return query

    resolved_key = api_key or GEMINI_API_KEY
    if not resolved_key:
        logger.warning("No Gemini key for query rewriting: returning original")
        return query

    context = ""
    if conversation_history:
        context = "\n".join(f"Previous: {item}" for item in conversation_history[-3:])

    prompt = f"""You are helping rewrite a follow-up question to be standalone.

{"CONVERSATION CONTEXT:\n" + context if context else ""}

FOLLOW-UP QUESTION: "{query}"

Rewrite it as a complete, standalone question. Reply with the rewritten question only."""

    try:
        rewritten = _gemini_call(prompt, resolved_key).strip().strip('"').strip("'")
        logger.info("Query rewritten: %r -> %r", query, rewritten)
        return rewritten
    except Exception as exc:
        logger.warning("Query rewriting failed (%s): using original", exc)
        return query


# ---------------------------------------------------------------------------
# HyDE
# ---------------------------------------------------------------------------

def should_use_hyde(query: str) -> bool:
    """Return True when HyDE would improve retrieval for a vague/short query."""
    q = query.lower()
    specific = any(t in q for t in [
        "technology", "claims", "growth", "finance", "operations",
        "revenue", "profit", "investment", "digital", "platform",
        "strategy", "market", "customer", "employee", "regulatory",
    ]) or bool(re.search(r"\d", query))
    vague = len(query.split()) <= 5 or any(
        w in q for w in ["about", "regarding", "concerning", "what", "how", "tell me"]
    )
    return vague and not specific


def generate_hypothetical_answer(query: str, api_key: str | None = None) -> str:
    """Generate a short hypothetical SEC-filing answer for HyDE retrieval expansion.

    Returns empty string if no key is available or the call fails (non-fatal).
    """
    resolved_key = api_key or GEMINI_API_KEY
    if not resolved_key:
        return ""

    prompt = (
        f"Generate a 2-3 sentence hypothetical answer to this question, "
        f"written in the style of a corporate SEC 10-K filing. "
        f"Be concrete about business areas, initiatives, or commitments.\n\n"
        f"Question: {query}"
    )
    try:
        return _gemini_call(prompt, resolved_key).strip()
    except Exception as exc:
        logger.warning("HyDE generation failed (%s): skipping expansion", exc)
        return ""


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def generate_answer_with_citations(
    chunks: list[Chunk],
    query: str,
    api_key: str | None = None,
) -> str:
    """Generate an answer grounded in retrieved chunks, with inline citations."""
    if not chunks:
        return "No relevant information found in the available SEC filings."

    resolved_key = api_key or GEMINI_API_KEY
    if not resolved_key:
        raise ValueError(
            "No Gemini API key available for answer generation. "
            "Set GEMINI_API_KEY in .env or pass api_key explicitly."
        )

    chunk_texts = "\n\n".join(
        f"[Chunk {i} | {c.year} | {c.section}]\n{c.text}"
        for i, c in enumerate(chunks, 1)
    )

    prompt = f"""Based on these SEC filing excerpts, answer the user's question with specific citations.

EXCERPTS:
{chunk_texts}

QUESTION: {query}

Instructions:
- Answer using only the information above.
- Cite sources as (Year YYYY, {chunks[0].section if chunks else 'Section'}).
- If the excerpts are insufficient, say so.

ANSWER:"""

    try:
        answer = _gemini_call(prompt, resolved_key).strip()
        logger.info("Generated answer (%d chars, %d sources)", len(answer), len(chunks))
        return answer
    except Exception as exc:
        logger.error("Answer generation failed: %s", exc)
        raise RuntimeError(f"Failed to generate answer: {exc}") from exc


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def process_query(
    query: str,
    ticker: str,
    conversation_history: list[str] | None = None,
    use_hyde: bool | None = None,
    api_key: str | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    """End-to-end query processing: rewrite → HyDE → retrieve → answer.

    Returns a dict with keys:
        query, rewritten_query, used_hyde, hypothetical_answer,
        answer, source_chunks
    """
    logger.info("Processing query %r for %s", query, ticker)

    is_followup = detect_followup_query(query, conversation_history)
    rewritten = rewrite_query(query, conversation_history, api_key) if is_followup else query

    if use_hyde is None:
        use_hyde = should_use_hyde(rewritten)

    hypothetical = ""
    retrieval_query = rewritten
    if use_hyde:
        hypothetical = generate_hypothetical_answer(rewritten, api_key)
        if hypothetical:
            retrieval_query = f"{rewritten} {hypothetical}"

    try:
        chunks = hybrid_retrieve(retrieval_query, ticker, top_n=top_n)
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        chunks = []

    try:
        answer = generate_answer_with_citations(chunks, rewritten, api_key)
    except Exception as exc:
        logger.error("Answer generation failed: %s", exc)
        answer = f"Could not generate an answer: {exc}"

    return {
        "query": query,
        "rewritten_query": rewritten if is_followup else None,
        "used_hyde": use_hyde,
        "hypothetical_answer": hypothetical if use_hyde else None,
        "answer": answer,
        "source_chunks": chunks,
    }
