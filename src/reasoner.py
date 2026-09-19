"""Cross-year reasoning and judging using Groq openai/gpt-oss-20b.

This is LLM Call 2 in the TempusRAG architecture. Takes extracted promises from extractor.py
and searches for evidence of delivery across subsequent years, then judges each promise as
Delivered/Partial/Silently Abandoned/Pending using sequential synchronous calls to Groq.

Rate limiting: 3-second pacing before each call and exponential backoff on retries.
"""

import json
import logging
import re
import time

import groq

from src.config import (
    GROQ_API_KEY, JUDGE_MODEL,
    MAX_JUDGE_EVIDENCE_CHUNKS, MAX_JUDGE_CHUNK_WORDS
)
from src.models import Promise, DeliveryEvidence, Chunk
from src.retriever import hybrid_retrieve

logger = logging.getLogger(__name__)

# Compressed system prompt for Groq judge
JUDGE_SYSTEM_PROMPT = """Role: you are a financial analyst judging if a corporate promise was kept
Input: promise text + evidence chunks from later filings
Output: JSON only, no preamble, with keys: status (one of: Delivered, Partial, Pending, Silently Abandoned), score (0-100), reasoning (max 1 sentence)
Rules: base judgment only on evidence provided, not prior knowledge"""


def has_metric(promise_text: str) -> bool:
    """
    Check if promise text contains numeric indicators or target metrics.

    Returns True if promise contains digits, $, %, 'billion', 'million', 'target', or 'goal'.
    """
    if not promise_text:
        return False
    if re.search(r"[\d$%]", promise_text):
        return True
    text_lower = promise_text.lower()
    metric_keywords = ["billion", "million", "target", "goal"]
    return any(keyword in text_lower for keyword in metric_keywords)


def search_evidence_for_promise(
    promise: Promise,
    ticker: str,
    evidence_search_years: list[int],
    top_n_per_year: int = 1,
) -> list[Chunk]:
    """
    Search for evidence of promise delivery across subsequent years.
    Retrieves top_n_per_year chunks per subsequent year and caps total evidence chunks
    at MAX_JUDGE_EVIDENCE_CHUNKS.

    Args:
        promise: Promise object to search evidence for
        ticker: Company ticker symbol
        evidence_search_years: List of years to search in (typically N+1, N+2, N+3)
        top_n_per_year: Number of chunks to retrieve per year (default 1)

    Returns:
        list[Chunk]: De-duplicated chunks that may contain delivery evidence
    """
    all_chunks = []

    for year in evidence_search_years:
        try:
            year_chunks = hybrid_retrieve(promise.promise_text, ticker, top_n=top_n_per_year, year=year)
            all_chunks.extend(year_chunks)

        except Exception as e:
            logger.warning(
                "Failed to retrieve evidence for promise %.50s... in year %d: %s",
                promise.promise_text, year, e,
            )

    # De-duplicate by chunk_id
    seen_ids: set[str] = set()
    unique_chunks: list[Chunk] = []
    for chunk in all_chunks:
        if chunk.chunk_id not in seen_ids:
            unique_chunks.append(chunk)
            seen_ids.add(chunk.chunk_id)

    # Cap total evidence chunks to MAX_JUDGE_EVIDENCE_CHUNKS
    capped_chunks = unique_chunks[:MAX_JUDGE_EVIDENCE_CHUNKS]

    logger.info(
        "Found %d evidence chunks (capped to %d) for promise: %s...",
        len(unique_chunks), len(capped_chunks), promise.promise_text[:50]
    )
    return capped_chunks


def build_judge_prompt(promise: Promise, evidence_chunks: list[Chunk]) -> str:
    """
    Build user prompt containing promise text and evidence chunks from later filings.

    Args:
        promise: Promise object to judge
        evidence_chunks: Evidence chunks found from subsequent years

    Returns:
        str: Judge prompt for Groq
    """
    if evidence_chunks:
        evidence_texts = []
        for i, chunk in enumerate(evidence_chunks[:MAX_JUDGE_EVIDENCE_CHUNKS], 1):
            words = chunk.text.split()
            if len(words) > MAX_JUDGE_CHUNK_WORDS:
                snippet = " ".join(words[:MAX_JUDGE_CHUNK_WORDS]) + " [...]"
            else:
                snippet = chunk.text
            evidence_texts.append(f"Evidence {i} (FY {chunk.year} | {chunk.section}):\n{snippet}")
        evidence_text = "\n\n".join(evidence_texts)
    else:
        evidence_text = "No evidence found in subsequent years."

    return (
        f"PROMISE (Made in {promise.year_made}):\n\"{promise.promise_text}\"\n\n"
        f"EVIDENCE FROM SUBSEQUENT FILINGS:\n{evidence_text}"
    )


def _call_groq_judge(prompt: str, api_key: str) -> str:
    """
    Low-level call to Groq model with pacing and retry backoff.
    Waits 5s, 10s, 20s between retries (3 attempts max).

    Args:
        prompt: Judge user prompt to send
        api_key: Groq API key for authentication

    Returns:
        str: Raw response text from Groq

    Raises:
        Exception: If all retries are exhausted.
    """
    client = groq.Groq(api_key=api_key)
    retry_delays = [5, 10, 20]

    for attempt, retry_delay in enumerate(retry_delays):
        if attempt == 0:
            time.sleep(3)  # pacing only on first attempt

        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1500,
            )

            raw_content = response.choices[0].message.content
            if raw_content is None or not raw_content.strip():
                raise ValueError(f"Received empty response from {JUDGE_MODEL}")

            return raw_content

        except Exception as e:
            logger.warning(
                "Groq judge attempt %d/3 failed: %s",
                attempt + 1, e,
            )
            if attempt < len(retry_delays) - 1:
                logger.info("Waiting %ds before retry %d...", retry_delay, attempt + 2)
                time.sleep(retry_delay)
            else:
                logger.warning("All 3 attempts failed: %s", e)
                raise


def _parse_judge_json(raw_text: str | None) -> dict:
    """
    Parse JSON response from Groq judge, handling markdown fences and key variations.

    Args:
        raw_text: Raw response text from Groq

    Returns:
        dict: Parsed JSON with status, delivery_score, and judge_reasoning

    Raises:
        ValueError: If raw_text is None or required keys are missing
        json.JSONDecodeError: If JSON decoding fails
    """
    if raw_text is None:
        raise ValueError("Cannot parse None judge response")

    text = raw_text.strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()
    result = json.loads(text)

    # Handle status, score, reasoning with fallback to legacy keys
    status = result.get("status")
    score = result.get("score", result.get("delivery_score"))
    reasoning = result.get("reasoning", result.get("judge_reasoning"))

    if status is None or score is None or reasoning is None:
        raise ValueError(f"Missing required keys in judge response: {result}")

    # Validate and normalize status
    valid_statuses = {"Delivered", "Partial", "Pending", "Silently Abandoned"}
    if status not in valid_statuses:
        for s in valid_statuses:
            if s.lower() in str(status).lower():
                status = s
                break
        else:
            status = "Pending"

    # Clamp and cast score
    try:
        score_int = int(round(float(score)))
        score_int = max(0, min(100, score_int))
    except (TypeError, ValueError):
        score_int = 50

    return {
        "status": status,
        "delivery_score": score_int,
        "score": score_int,
        "judge_reasoning": str(reasoning),
        "reasoning": str(reasoning),
    }


def judge_single_promise(
    promise: Promise,
    evidence_chunks: list[Chunk],
    api_key: str | None = None,
) -> DeliveryEvidence | None:
    """
    Judge a single promise using Groq model (synchronous).
    Returns DeliveryEvidence, or None if retries fail (allowing promise to be skipped).

    Args:
        promise: Promise to judge
        evidence_chunks: Evidence chunks from subsequent years
        api_key: Optional API key override (uses config if None)

    Returns:
        DeliveryEvidence | None: Judgment result, or None if failed
    """
    resolved_api_key = api_key if api_key is not None else GROQ_API_KEY
    if not resolved_api_key:
        raise ValueError(
            "No Groq API key available - set GROQ_API_KEY in config or pass one explicitly"
        )

    current_evidence = list(evidence_chunks[:MAX_JUDGE_EVIDENCE_CHUNKS])
    prompt = build_judge_prompt(promise, current_evidence)

    try:
        raw_response = _call_groq_judge(prompt, resolved_api_key)
    except Exception as e:
        logger.warning(
            "All retries failed for promise: %s...: %s. Skipping promise.",
            promise.promise_text[:50], e,
        )
        return None

    # Parse response
    try:
        judge_result = _parse_judge_json(raw_response)
    except Exception as e:
        logger.warning(
            "Judge parse failed for promise %s...: %s. Skipping promise.",
            promise.promise_text[:50], e,
        )
        return None

    try:
        evidence = DeliveryEvidence(
            promise=promise,
            evidence_text="\n\n".join([c.text for c in current_evidence]) if current_evidence else None,
            year_found=max([c.year for c in current_evidence]) if current_evidence else None,
            delivery_score=int(judge_result["delivery_score"]),
            status=judge_result["status"],
            judge_reasoning=judge_result["judge_reasoning"],
        )
        logger.info(
            "Successfully judged promise (score: %d, status: %s): %s...",
            evidence.delivery_score, evidence.status, promise.promise_text[:50]
        )
        return evidence

    except Exception as e:
        logger.warning(
            "Failed to construct DeliveryEvidence for promise %s...: %s. Skipping promise.",
            promise.promise_text[:50], e,
        )
        return None


def batch_judge_promises(
    promises: list[Promise],
    ticker: str,
    evidence_search_years: list[int],
    api_key: str | None = None,
    batch_size: int = 1,
) -> list[DeliveryEvidence]:
    """
    Judge multiple promises sequentially with deduplication and dynamic retrieval.

    1. Deduplicates promises by promise_text before judging so identical promises
       across years are only judged once.
    2. Uses has_metric to request top 2 chunks for numeric indicators, top 1 for qualitative.
    3. Reuses cached judgments for duplicate promises.

    Args:
        promises: List of promises to judge
        ticker: Company ticker for evidence search
        evidence_search_years: Years to search for evidence
        api_key: Optional API key override
        batch_size: Ignored: kept for API compatibility

    Returns:
        list[DeliveryEvidence]: Delivery evidence for all promises in input order
    """
    start_time = time.time()

    resolved_api_key = api_key if api_key is not None else GROQ_API_KEY
    if not resolved_api_key:
        raise ValueError(
            "No Groq API key available - set GROQ_API_KEY in config or pass one explicitly"
        )

    # 1. DEDUPLICATION: group unique promises by normalized promise_text before judging
    unique_promises: list[Promise] = []
    seen_texts: set[str] = set()
    for p in promises:
        norm_key = re.sub(r"\s+", " ", p.promise_text.strip().lower())
        if norm_key not in seen_texts:
            seen_texts.add(norm_key)
            unique_promises.append(p)

    logger.info(
        "Deduplication: %d unique promises to judge from %d total promises",
        len(unique_promises), len(promises)
    )

    judgment_cache: dict[str, DeliveryEvidence] = {}

    for i, promise in enumerate(unique_promises):
        norm_key = re.sub(r"\s+", " ", promise.promise_text.strip().lower())
        search_years = [y for y in evidence_search_years if y > promise.year_made][:3]

        # 2. DYNAMIC TOP-K RETRIEVAL
        top_n = 2 if has_metric(promise.promise_text) else 1
        logger.info(
            "Judging unique promise %d/%d (year %d, top_n=%d, evidence years %s): %s...",
            i + 1, len(unique_promises), promise.year_made, top_n, search_years, promise.promise_text[:60],
        )

        evidence_chunks = search_evidence_for_promise(
            promise, ticker, search_years, top_n_per_year=top_n
        )

        try:
            result = judge_single_promise(promise, evidence_chunks, resolved_api_key)
            if result is not None:
                judgment_cache[norm_key] = result
            else:
                logger.warning(
                    "Skipping promise %s... as judgment could not be completed",
                    promise.promise_text[:50]
                )
        except Exception as e:
            logger.warning(
                "Skipping promise %s... due to error: %s",
                promise.promise_text[:50], e
            )

    # Reconstruct results for all input promises, reusing judgments for duplicate promise_text
    results: list[DeliveryEvidence] = []
    for p in promises:
        norm_key = re.sub(r"\s+", " ", p.promise_text.strip().lower())
        if norm_key in judgment_cache:
            cached = judgment_cache[norm_key]
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
        else:
            logger.warning(
                "Promise skipped (no judgment available): %s...",
                p.promise_text[:50]
            )

    elapsed = time.time() - start_time
    logger.info(
        "Batch judgment complete: %.2fs, %d/%d promises with valid judgments",
        elapsed, len(results), len(promises),
    )

    return results


def reason_across_years(
    promises_by_year: dict[int, list[Promise]],
    ticker: str,
    api_key: str | None = None,
) -> list[DeliveryEvidence]:
    """
    Top-level orchestrator for cross-year reasoning and judging.

    Promises from the most-recent year that have no later filings are assigned
    Pending status directly without an LLM call because there is no
    evidence to evaluate and silently assigning Silently Abandoned would be
    factually wrong.
    """
    start_time = time.time()

    all_promises: list[Promise] = []
    available_years = sorted(promises_by_year.keys())

    if not available_years:
        logger.info("No promises to judge")
        return []

    max_year = max(available_years)

    for year in available_years:
        year_promises = promises_by_year[year]
        evidence_search_years = [y for y in available_years if y > year][:3]
        logger.info(
            "Year %d: %d promises, evidence search in years %s",
            year, len(year_promises), evidence_search_years,
        )

        for promise in year_promises:
            # Most-recent year has no later filings -> Pending, no LLM call
            if year == max_year and not evidence_search_years:
                continue
            all_promises.append(promise)

    # Directly create Pending evidence for most-recent year
    pending_evidence: list[DeliveryEvidence] = []
    for promise in promises_by_year.get(max_year, []):
        if not [y for y in available_years if y > max_year]:
            pending_evidence.append(DeliveryEvidence(
                promise=promise,
                evidence_text=None,
                year_found=None,
                delivery_score=50,
                status="Pending",
                judge_reasoning=(
                    f"No later filings available (most recent year is {max_year}). "
                    "Status set to Pending automatically."
                ),
            ))

    if not all_promises:
        logger.info("No promises require LLM judging (all Pending or empty)")
        end_time = time.time()
        logger.info("Cross-year reasoning complete in %.2fs", end_time - start_time)
        return pending_evidence

    evidence_years = [y for y in available_years if y > min(available_years)]

    all_evidence = batch_judge_promises(all_promises, ticker, evidence_years, api_key)
    all_evidence.extend(pending_evidence)

    end_time = time.time()
    logger.info(
        "Cross-year reasoning complete: %d promises judged in %.2fs",
        len(all_promises), end_time - start_time,
    )

    return all_evidence


if __name__ == "__main__":
    from unittest.mock import patch

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    print("=== Testing reasoner.py offline verification tests ===")

    sample_promise = Promise(
        promise_text="We will expand our cloud computing footprint by 30% by 2025.",
        domain="Technology",
        deadline_mentioned="2025",
        year_made=2023,
        page_number=12,
        confidence_score=0.85
    )

    # 1. Test prompt building and token/word capping
    sample_chunks = [
        Chunk(
            text="Word " * 600,  # 600 words (exceeds MAX_JUDGE_CHUNK_WORDS=500)
            company="TEST",
            year=2024,
            section="Item 7 MD&A",
            subsection="Cloud",
            chunk_id="TEST_01",
            word_count=600,
        ),
        Chunk(
            text="Cloud infrastructure met target with 32% expansion.",
            company="TEST",
            year=2025,
            section="Item 1 Business",
            subsection="Operations",
            chunk_id="TEST_02",
            word_count=7,
        ),
    ]

    prompt = build_judge_prompt(sample_promise, sample_chunks)
    print("\n1. Testing judge prompt token capping:")
    assert "[...]" in prompt, "Oversized chunk was not truncated"
    prompt_words = len(prompt.split())
    print(f"   Judge prompt total words: {prompt_words} (safely under TPM limits)")
    assert prompt_words < 1200, f"Prompt too large: {prompt_words} words"
    print("   [Pass] Chunk word limits enforced successfully")

    # 2. Test has_metric helper function
    print("\n2. Testing has_metric:")
    assert has_metric("We target $5 billion revenue by 2025") is True
    assert has_metric("Growth of 15% in deliveries") is True
    assert has_metric("Our goal is customer trust") is True
    assert has_metric("We commit to excellence in our global operations") is False
    print("   [Pass] has_metric correctly detects numeric indicators and target keywords")

    import sys

    # 3. Test successful judging with compressed JSON response
    curr_mod = sys.modules[__name__]
    mock_json = '{"status": "Delivered", "score": 92, "reasoning": "Expansion target of 30% was achieved with 32% growth."}'
    with patch.object(curr_mod, "_call_groq_judge", return_value=mock_json):
        result = judge_single_promise(sample_promise, sample_chunks, api_key="mock_key")
        print("\n3. Testing mock judge response:")
        print(f"   Status: {result.status}, Score: {result.delivery_score}")
        print(f"   Reasoning: {result.judge_reasoning}")
        assert result.status == "Delivered"
        assert result.delivery_score == 92
        print("   [Pass] Valid judge JSON parsed correctly with status, score, reasoning keys")

    # 4. Test failure handling: returns None and skips promise without crashing
    with patch.object(curr_mod, "_call_groq_judge", side_effect=RuntimeError("Groq API connection timeout")):
        fail_result = judge_single_promise(sample_promise, sample_chunks, api_key="mock_key")
        print("\n4. Testing retry exhaustion handling:")
        assert fail_result is None, "Expected None when retries fail"
        print("   [Pass] Promise cleanly skipped on failure without crashing")

    # 5. Test pending assignment for most recent year
    pending_dict = {2025: [sample_promise]}
    ev_list = reason_across_years(pending_dict, ticker="TEST")
    print("\n5. Testing most-recent year automatic Pending status:")
    assert len(ev_list) == 1
    assert ev_list[0].status == "Pending"
    assert ev_list[0].delivery_score == 50
    print("   [Pass] Most recent year automatically assigned Pending")

    # 6. Test duplicate promise deduplication across years
    print("\n6. Testing duplicate promise deduplication across years:")
    p1 = Promise(promise_text="Invest in solar energy.", domain="Technology", deadline_mentioned="2025", year_made=2021, page_number=1, confidence_score=0.8)
    p2 = Promise(promise_text="Invest in solar energy.", domain="Technology", deadline_mentioned="2025", year_made=2022, page_number=2, confidence_score=0.8)
    groq_calls = 0
    def mock_judge_counting(prompt_text, key):
        global groq_calls
        groq_calls += 1
        return mock_json

    with patch.object(curr_mod, "_call_groq_judge", side_effect=mock_judge_counting), \
         patch.object(curr_mod, "search_evidence_for_promise", return_value=sample_chunks), \
         patch("time.sleep", return_value=None):
        batch_res = batch_judge_promises([p1, p2], ticker="TEST", evidence_search_years=[2023, 2024], api_key="mock_key")
        assert len(batch_res) == 2
        assert groq_calls == 1, f"Expected exactly 1 Groq call for duplicate promise, got {groq_calls}"
        assert batch_res[0].delivery_score == batch_res[1].delivery_score == 92
        print("   [Pass] Duplicate promises across years evaluated only once (cached seamlessly)")

    print("\nAll reasoner.py offline verification tests passed successfully!")