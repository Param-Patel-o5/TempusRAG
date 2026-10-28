"""Cross-year promise judging — supports Gemini (default) and Groq providers (LLM Call 2).

Takes extracted Promise objects and searches for delivery evidence across
subsequent fiscal years, then judges each as:
Delivered / Partial / Pending / Silently Abandoned

Provider:  Gemini Flash Lite (default) — 30 RPM, 1M TPM, no regional restrictions.
           Groq gpt-oss-20b (fallback) — 30 RPM, 8K TPM, India-accessible.
Rate limiting: delegated to rate_limit.py per-provider limiter.
Concurrency: up to MAX_JUDGE_WORKERS parallel threads.
Deduplication: identical (year_made, promise_text) pairs judged only once.
Dynamic retrieval: top 2 chunks for numeric promises, top 1 for qualitative.
Year isolation: evidence retrieval is year-filtered — FY2021 chunks can never
    appear in an FY2022 or FY2023 evidence context window.
"""

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import groq
from google import genai
from google.genai import errors as genai_errors

from src.config import (
    GEMINI_JUDGE_API_KEY,
    GEMINI_JUDGE_MODEL,
    GROQ_API_KEY,
    JUDGE_MODEL,
    JUDGE_PROVIDER,
    MAX_JUDGE_CHUNK_WORDS,
    MAX_JUDGE_EVIDENCE_CHUNKS,
    MAX_JUDGE_WORKERS,
)
from src.models import Chunk, DeliveryEvidence, Promise
from src.rate_limit import get_rate_limiter
from src.retriever import hybrid_retrieve
from src.scorer import _METRIC_RE, compute_confidence_score

logger = logging.getLogger(__name__)

# ============================================================================
# JUDGE SYSTEM PROMPT
# ============================================================================

JUDGE_SYSTEM_PROMPT = (
    "Role: financial analyst judging if a corporate promise was kept.\n"
    "Input: promise text + evidence chunks from later SEC filings.\n"
    "Output: JSON only, no preamble, no markdown fences.\n"
    "Keys: status (Delivered|Partial|Pending|Silently Abandoned), "
    "score (0-100), reasoning (max 1 sentence).\n"
    "Rules: judge only from evidence provided, not prior knowledge.\n"
    "Status Definitions:\n"
    "- Delivered: The specific target, number, or milestone was 100% completed or reached (Score: 86-100).\n"
    "- Partial: Substantial progress was made (e.g. partial dollar amounts spent, partial locations opened, target delayed, or launched with reduced scope) but fell short of 100% completion (Score: 31-75).\n"
    "- Pending: Target deadline is in the future relative to the evidence year and initiative is ongoing with no shortfall reported (Score: 60-85).\n"
    "- Silently Abandoned: Target deadline has passed or initiative vanished from subsequent filings with zero confirmation or explanation (Score: 0-30)."
)

VALID_STATUSES = {"Delivered", "Partial", "Pending", "Silently Abandoned"}


# ============================================================================
# CLIENT SINGLETONS — Groq and Gemini (thread-safe lazy initialization)
# ============================================================================

_groq_client = None
_groq_client_lock = threading.Lock()
_groq_client_api_key = None


def _get_groq_client(api_key: str):
    """Return a shared Groq client singleton (thread-safe lazy initialization).

    max_retries=0 disables Groq SDK's internal retry loop to avoid compounding
    with our smarter rate_limit.execute_with_retry backoff.
    """
    global _groq_client, _groq_client_api_key
    if _groq_client is not None and _groq_client_api_key == api_key:
        return _groq_client
    with _groq_client_lock:
        if _groq_client is None or _groq_client_api_key != api_key:
            _groq_client = groq.Groq(api_key=api_key, max_retries=0)
            _groq_client_api_key = api_key
            logger.info("Groq client singleton initialized (max_retries=0)")
    return _groq_client


_gemini_judge_client = None
_gemini_judge_client_lock = threading.Lock()
_gemini_judge_client_api_key = None


def _get_gemini_judge_client(api_key: str):
    """Return a shared Gemini client singleton for judging (thread-safe lazy init).

    Separate singleton from the extractor's client so the two pipelines
    (extraction and judging) can be rate-limited independently.
    """
    global _gemini_judge_client, _gemini_judge_client_api_key
    if _gemini_judge_client is not None and _gemini_judge_client_api_key == api_key:
        return _gemini_judge_client
    with _gemini_judge_client_lock:
        if _gemini_judge_client is None or _gemini_judge_client_api_key != api_key:
            _gemini_judge_client = genai.Client(api_key=api_key)
            _gemini_judge_client_api_key = api_key
            logger.info("Gemini judge client singleton initialized")
    return _gemini_judge_client


# ============================================================================
# HELPERS
# ============================================================================

def _norm_key(year_made: int, text: str) -> str:
    """Normalize (year_made, promise_text) into a deduplication key.  (Change 8)

    year_made is included so that the same promise text appearing in two
    different fiscal years is judged separately with the correct evidence
    windows:
        FY2021 promise -> search evidence in [2022, 2023, 2024]
        FY2022 promise -> search evidence in [2023, 2024]
    Without year_made, FY2022 judgment would incorrectly reuse FY2021s
    (potentially broader) evidence set.
    """
    return f"{year_made}::{re.sub(r'\s+', ' ', text.strip().lower())}"


# ============================================================================
# EVIDENCE RETRIEVAL  (Change 14 - parallel per-year retrieval)
# ============================================================================

def search_evidence(
    promise,
    ticker: str,
    years: list,
    top_n: int = 1,
) -> list:
    """Retrieve evidence chunks from subsequent filings for a promise.

    Year isolation guarantee: hybrid_retrieve is called with an explicit year=
    filter for EACH evidence year independently. Chunks from FY2022 and FY2023
    are retrieved in separate calls and never mixed into the same ChromaDB query.
    The combined list is then deduplicated and capped.

    Concurrency: retrieval calls run in parallel (all local/ChromaDB - no API
    rate limits apply), typically 3x faster than sequential for 3 evidence years.

    Args:
        promise: Promise to find evidence for.
        ticker: Company ticker symbol.
        years: Fiscal years to search (typically N+1, N+2, N+3).
        top_n: Chunks to retrieve per year (1 for qualitative, 2 for numeric).

    Returns:
        Deduplicated chunks capped at MAX_JUDGE_EVIDENCE_CHUNKS.
    """
    if not years:
        return []

    def _retrieve_year(year: int) -> list:
        """Retrieve chunks for a single evidence year with strict year isolation."""
        try:
            # year= filter: only FY{year} chunks are returned by ChromaDB where-clause
            return hybrid_retrieve(promise.promise_text, ticker, top_n=top_n, year=year)
        except Exception as e:
            logger.warning(
                "Evidence retrieval failed for promise %.50s in year %d: %s",
                promise.promise_text, year, e,
            )
            return []

    # Parallelise across evidence years - all local (BM25 + ChromaDB), no rate limits
    all_chunks = []
    with ThreadPoolExecutor(max_workers=min(len(years), 3)) as pool:
        for year_chunks in pool.map(_retrieve_year, years):
            all_chunks.extend(year_chunks)

    # Deduplicate by chunk_id
    seen = set()
    unique = []
    for c in all_chunks:
        if c.chunk_id not in seen:
            unique.append(c)
            seen.add(c.chunk_id)

    capped = unique[:MAX_JUDGE_EVIDENCE_CHUNKS]
    logger.info(
        "Evidence: %d chunks (capped to %d) for promise: %.50s",
        len(unique), len(capped), promise.promise_text,
    )
    return capped


# ============================================================================
# PROMPT BUILDING
# ============================================================================

def build_judge_prompt(promise, chunks: list) -> str:
    """Build judge prompt with promise and evidence chunks."""
    if chunks:
        parts = []
        for i, chunk in enumerate(chunks[:MAX_JUDGE_EVIDENCE_CHUNKS], 1):
            words = chunk.text.split()
            text = (
                " ".join(words[:MAX_JUDGE_CHUNK_WORDS]) + " [...]"
                if len(words) > MAX_JUDGE_CHUNK_WORDS
                else chunk.text
            )
            parts.append(f"Evidence {i} (FY{chunk.year} | {chunk.section}):\n{text}")
        evidence_text = "\n\n".join(parts)
    else:
        evidence_text = "No evidence found in subsequent filings."

    return (
        f"PROMISE (made in FY{promise.year_made}):\n"
        f"\"{promise.promise_text}\"\n\n"
        f"EVIDENCE:\n{evidence_text}"
    )


# ============================================================================
# JUDGE API CALL — dispatches to Gemini (default) or Groq
# ============================================================================

def _call_judge(prompt: str, groq_api_key: str) -> str:
    """Call the configured judge provider with rate limiting and smart retry.

    JUDGE_PROVIDER=gemini (default): uses Gemini Flash Lite via google.genai.
        - 30 RPM, 1M TPM — no regional restrictions, no TPM exhaustion risk.
        - Rate limited by 'gemini_judge' limiter (2.0s interval).

    JUDGE_PROVIDER=groq: uses Groq gpt-oss-20b via groq SDK.
        - 30 RPM, 8K TPM — only use if you have Llama access via standard Groq.
        - Rate limited by 'groq' limiter (7.5s interval).
    """
    if JUDGE_PROVIDER == "gemini":
        return _call_gemini_judge(prompt)
    return _call_groq_judge(prompt, groq_api_key)


def _call_gemini_judge(prompt: str) -> str:
    """Call Gemini Flash Lite for judging via google.genai client."""
    if not GEMINI_JUDGE_API_KEY:
        raise ValueError("No Gemini API key for judge — set GEMINI_JUDGE_API_KEY or GEMINI_API_KEY in .env")
    client = _get_gemini_judge_client(GEMINI_JUDGE_API_KEY)
    limiter = get_rate_limiter("gemini_judge")

    full_prompt = f"{JUDGE_SYSTEM_PROMPT}\n\n{prompt}"

    def _invoke() -> str:
        try:
            response = client.models.generate_content(
                model=GEMINI_JUDGE_MODEL,
                contents=full_prompt,
                config={"temperature": 0.1, "max_output_tokens": 400},
            )
        except genai_errors.APIError as e:
            if getattr(e, "code", None) == 429:
                logger.warning("Gemini judge 429: %s", e)
            raise

        if not response.candidates:
            raise RuntimeError("Gemini judge returned no candidates (prompt blocked)")
        text = response.text
        if not text or not text.strip():
            raise ValueError(f"Empty response from {GEMINI_JUDGE_MODEL}")
        return text.strip()

    return limiter.execute_with_retry(_invoke)


def _call_groq_judge(prompt: str, api_key: str) -> str:
    """Call Groq judge with rate limiting and smart retry."""
    client = _get_groq_client(api_key)
    limiter = get_rate_limiter("groq")

    def _invoke() -> str:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=400,  # score + status + 1 sentence, prevents JSON truncation
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise ValueError(f"Empty response from {JUDGE_MODEL}")
        return content

    return limiter.execute_with_retry(_invoke)


# ============================================================================
# JSON PARSING
# ============================================================================

def _parse_judge_response(raw: str) -> dict:
    """Parse Groq JSON response into status, score, reasoning.

    Handles markdown fences and key name variations.
    Normalizes status to valid label, clamps score to 0-100.
    """
    if not raw:
        raise ValueError("Empty judge response")

    text = raw.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    result = json.loads(text.strip())

    status = result.get("status")
    score = result.get("score", result.get("delivery_score"))
    reasoning = result.get("reasoning", result.get("judge_reasoning"))

    if status is None or score is None or reasoning is None:
        raise ValueError(f"Missing required keys in judge response: {result}")

    # Normalize status
    if status not in VALID_STATUSES:
        matched = next(
            (s for s in VALID_STATUSES if s.lower() in str(status).lower()),
            "Pending"
        )
        status = matched

    # Clamp score
    try:
        score = max(0, min(100, int(round(float(score)))))
    except (TypeError, ValueError):
        score = 50

    return {
        "status": status,
        "delivery_score": score,
        "judge_reasoning": str(reasoning),
    }


# ============================================================================
# SINGLE PROMISE JUDGE
# ============================================================================

def judge_single_promise(
    promise,
    chunks: list,
    api_key = None,
):
    """Judge one promise. Returns None on failure so caller can skip cleanly."""
    resolved_groq_key = api_key or GROQ_API_KEY
    # For Gemini provider, Groq key is unused — GEMINI_JUDGE_API_KEY is read from config directly
    if JUDGE_PROVIDER == "groq" and not resolved_groq_key:
        raise ValueError("No Groq API key - set GROQ_API_KEY or pass api_key")
    if JUDGE_PROVIDER == "gemini" and not GEMINI_JUDGE_API_KEY:
        raise ValueError("No Gemini API key for judge - set GEMINI_JUDGE_API_KEY or GEMINI_API_KEY in .env")

    capped = chunks[:MAX_JUDGE_EVIDENCE_CHUNKS]
    prompt = build_judge_prompt(promise, capped)

    try:
        raw = _call_judge(prompt, resolved_groq_key)
    except Exception as e:
        logger.warning(
            "All retries failed for promise %.50s: %s - skipping",
            promise.promise_text, e,
        )
        return None

    try:
        result = _parse_judge_response(raw)
    except Exception as e:
        logger.warning(
            "Parse failed for promise %.50s: %s - skipping",
            promise.promise_text, e,
        )
        return None

    try:
        evidence = DeliveryEvidence(
            promise=promise,
            evidence_text="\n\n".join(c.text for c in capped) if capped else None,
            year_found=max(c.year for c in capped) if capped else None,
            delivery_score=result["delivery_score"],
            status=result["status"],
            judge_reasoning=result["judge_reasoning"],
        )
        logger.info(
            "Judged (score=%d, status=%s): %.50s",
            evidence.delivery_score, evidence.status, promise.promise_text,
        )
        return evidence

    except Exception as e:
        logger.warning(
            "Failed to build DeliveryEvidence for %.50s: %s - skipping",
            promise.promise_text, e,
        )
        return None


# ============================================================================
# BATCH JUDGE - PARALLEL  (Change 1)
# ============================================================================

def batch_judge_promises(
    promises: list,
    ticker: str,
    available_years: list,
    api_key = None,
) -> list:
    """Judge all promises concurrently with deduplication and dynamic retrieval.

    Algorithm:
    1. Deduplicate by (year_made, normalized promise_text). Promises sharing
       the same text in the same fiscal year are judged once and reused.
       Including year_made prevents cross-year evidence window collisions.
    2. Dynamic top-k: top 2 evidence chunks per year for numeric promises,
       top 1 for qualitative promises.
    3. Parallel execution: up to MAX_JUDGE_WORKERS threads. After change 7 in
       rate_limit.py, enforce_interval reserves time slots without holding the
       lock during sleep - threads can sleep concurrently and each Groq call
       lands at its reserved slot.
    4. Results cached and reused for any duplicate promises.
    """
    start = time.time()
    resolved_key = api_key or GROQ_API_KEY
    if JUDGE_PROVIDER == "groq" and not resolved_key:
        raise ValueError("No Groq API key - set GROQ_API_KEY or pass api_key")
    if JUDGE_PROVIDER == "gemini" and not GEMINI_JUDGE_API_KEY:
        raise ValueError("No Gemini API key for judge - set GEMINI_JUDGE_API_KEY or GEMINI_API_KEY in .env")

    # Deduplicate by (year_made, normalized text)
    unique = []
    seen = set()
    for p in promises:
        k = _norm_key(p.year_made, p.promise_text)
        if k not in seen:
            seen.add(k)
            unique.append(p)

    total_unique = len(unique)
    logger.info(
        "Dedup: %d unique from %d total promises",
        total_unique, len(promises),
    )

    cache = {}
    cache_lock = threading.Lock()

    def _judge_one(indexed_promise) -> None:
        """Judge a single promise and store result in cache (thread-safe)."""
        i, promise = indexed_promise
        search_years = [y for y in available_years if y > promise.year_made][:3]
        top_n = 2 if bool(_METRIC_RE.search(promise.promise_text)) else 1

        logger.info(
            "Judging %d/%d (FY%d, top_n=%d, evidence=%s): %.60s",
            i + 1, total_unique, promise.year_made,
            top_n, search_years, promise.promise_text,
        )

        chunks = search_evidence(promise, ticker, search_years, top_n)
        result = judge_single_promise(promise, chunks, resolved_key)

        key = _norm_key(promise.year_made, promise.promise_text)
        if result is not None:
            with cache_lock:
                cache[key] = result
        else:
            logger.warning("Skipped promise: %.50s", promise.promise_text)

    max_workers = min(MAX_JUDGE_WORKERS, total_unique) if total_unique > 0 else 1
    logger.info(
        "Starting parallel judging: %d workers, %d promises",
        max_workers, total_unique,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_judge_one, enumerate(unique)))

    # Reconstruct full results list in original order, reusing cache
    results = []
    for p in promises:
        k = _norm_key(p.year_made, p.promise_text)
        if k not in cache:
            logger.warning("No judgment for promise: %.50s", p.promise_text)
            continue

        cached = cache[k]
        if p is cached.promise:
            results.append(cached)
        else:
            results.append(DeliveryEvidence(
                promise=p,
                evidence_text=cached.evidence_text,
                year_found=cached.year_found,
                delivery_score=cached.delivery_score,
                status=cached.status,
                judge_reasoning=cached.judge_reasoning,
            ))

    logger.info(
        "Batch complete: %d/%d judged in %.1fs",
        len(results), len(promises), time.time() - start,
    )
    return results


# ============================================================================
# TOP-LEVEL ORCHESTRATOR
# ============================================================================

def reason_across_years(
    promises_by_year: dict,
    ticker: str,
    api_key = None,
) -> list:
    """Orchestrate cross-year judging for all promises.

    Most-recent year promises get Pending automatically (no later filings to
    judge against). All other promises go through batch_judge_promises.
    """
    start = time.time()
    available_years = sorted(promises_by_year.keys())

    if not available_years:
        logger.info("No promises to judge")
        return []

    max_year = max(available_years)

    # Collect promises that have later years to judge against
    to_judge = []
    low_confidence_pending = []
    for year, year_promises in promises_by_year.items():
        if year == max_year:
            continue  # handled as Pending below
        for p in year_promises:
            score = p.confidence_score if p.confidence_score is not None else compute_confidence_score(p)
            if score >= 0.35:
                to_judge.append(p)
            else:
                low_confidence_pending.append(
                    DeliveryEvidence(
                        promise=p,
                        evidence_text=None,
                        year_found=None,
                        delivery_score=50,
                        status="Pending",
                        judge_reasoning=(
                            "Generic statement without specific deadline or quantitative target. "
                            "Assigned Pending automatically."
                        ),
                    )
                )

    # Auto-assign Pending to most-recent year
    pending = [
        DeliveryEvidence(
            promise=p,
            evidence_text=None,
            year_found=None,
            delivery_score=50,
            status="Pending",
            judge_reasoning=(
                f"No later filings available (most recent year is {max_year}). "
                "Assigned Pending automatically."
            ),
        )
        for p in promises_by_year.get(max_year, [])
    ] + low_confidence_pending

    if not to_judge:
        logger.info("No high-conviction promises require LLM judging (all Pending)")
        return pending

    judged = batch_judge_promises(to_judge, ticker, available_years, api_key)

    all_evidence = judged + pending
    logger.info(
        "Cross-year reasoning complete: %d judged + %d pending in %.1fs",
        len(judged), len(pending), time.time() - start,
    )
    return all_evidence