"""Promise extraction from MD&A and Future Outlook sections using Gemini.

This is LLM Call 1 in the TempusRAG architecture. Takes filtered chunks from
promise-extraction sections (MD&A, Future Outlook, Growth Strategy) and extracts
structured Promise objects.

API Key handling: Reads GEMINI_API_KEY from config.py at call time. The Streamlit
UI may set API keys at runtime, so extract_promises_for_year() accepts an optional
api_key parameter that overrides the config value if provided.

Rate limiting: Uses the shared provider rate limiter and retry coordinator
(gemini_generate) in src/rate_limit.py. Daily quota errors are treated as terminal
and not retried.

Failure visibility: a model answer of [] is a real result ("no promises found").
Anything else (API errors, invalid JSON, missing key, exhausted quota) is a failure
and is raised, so a broken run can never look like a company with zero promises.
"""

import json
import logging
import os
import re
from collections import Counter
from typing import Any

from google import genai
from google.genai import errors as genai_errors

from src.config import (
    CHUNK_MAX_TOKENS,
    EXTRACTOR_MODEL,
    EXTRACTOR_FALLBACK_MODEL,
    GEMINI_API_KEY,
    PROMISE_EXTRACTION_SECTIONS,
)
from src.models import Promise
from src.rate_limit import DailyQuotaExhaustedError, get_rate_limiter

logger = logging.getLogger(__name__)

# Suppress noisy google/httpx loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google.auth").setLevel(logging.WARNING)

# Chunk selection settings derived from CHUNK_MAX_TOKENS so ~5,000 words are sent
TARGET_PROMISE_WORDS = 5000
MAX_PROMISE_CHUNKS = max(5, TARGET_PROMISE_WORDS // max(1, CHUNK_MAX_TOKENS))
MIN_SECTION_WORDS = TARGET_PROMISE_WORDS // 2
MIN_CUE_HITS = 3
ENABLE_CUE_FALLBACK = True


def is_dry_run() -> bool:
    """Return True if TEMPUS_DRY_RUN environment variable is set to 1, true, or yes."""
    val = os.getenv("TEMPUS_DRY_RUN", "").strip().lower()
    return val in ("1", "true", "yes")

# Sections that never hold management commitments (matched as lowercase substrings).
# Applied only to gate cue-phrase fallback candidates.
# Note: "exhibits and financial" is excluded from this list because ingestion mislabeling
# would otherwise drop large portions of real filing content.
_EXCLUDED_SECTION_KEYWORDS = (
    "risk factor",
    "legal proceeding",
    "controls and procedures",
    "cybersecurity",
    "signature",
    "properties",
    "mine safety",
    "unresolved staff",
    "financial statements",
    "notes to",
    "security ownership",
    "directors, executive officers",
    "exhibit index",
)

# Phrases that signal a forward-looking commitment
_CUE_RE = re.compile(
    r"\bwe (?:plan|intend|expect|anticipate|aim|seek|target|will|are committed|are working|continue to invest)\b"
    r"|\bour (?:goal|goals|objective|objectives|target|targets|priorit(?:y|ies)|commitment)\b"
    r"|\bcommitted to\b"
    r"|\bby (?:the end of )?(?:fiscal )?(?:year )?20\d\d\b"
    r"|\bover the next (?:\w+ ){0,2}(?:years?|months?|quarters?)\b",
    re.IGNORECASE,
)

# Standard legal disclaimer paragraph, not a commitment
_SAFE_HARBOR_MARKER = "private securities litigation reform act"


def _cue_count(text: str) -> int:
    """Count forward-looking cue phrases in a text."""
    return len(_CUE_RE.findall(text))


def _is_excluded_chunk(chunk) -> bool:
    """True for chunks from sections that never hold commitments, or safe-harbor boilerplate."""
    section = chunk.section.lower()
    if any(keyword in section for keyword in _EXCLUDED_SECTION_KEYWORDS):
        return True
    return _SAFE_HARBOR_MARKER in chunk.text.lower()


def _augment_and_cap_chunks(year: int, all_chunks: list, selected: list) -> list:
    """Ensure Gemini receives sufficient relevant context.

    If the section filter matched very little text (for example when section detection
    only caught a table of contents), add the chunks with the most forward-looking cue
    phrases from non-excluded sections. Then cap the total at MAX_PROMISE_CHUNKS, keeping
    the highest cue scores, and restore original document order.
    """
    selected = list(selected)
    selected_ids = {c.chunk_id for c in selected}
    selected_words = sum(c.word_count for c in selected)

    if selected_words < MIN_SECTION_WORDS:
        candidates = [c for c in all_chunks if c.chunk_id not in selected_ids]

        # Log warning if section exclusions drop more than 50% of candidates
        excluded_count = sum(1 for c in candidates if _is_excluded_chunk(c))
        if candidates and (excluded_count / len(candidates)) > 0.5:
            logger.warning(
                f"Fiscal year {year}: section exclusion dropped {excluded_count}/{len(candidates)} "
                f"({excluded_count / len(candidates):.1%}) of fallback candidate chunks."
            )

        valid_candidates = [c for c in candidates if not _is_excluded_chunk(c)]
        ranked = sorted(valid_candidates, key=lambda c: _cue_count(c.text), reverse=True)
        extras = [c for c in ranked if _cue_count(c.text) >= MIN_CUE_HITS]
        room = max(MAX_PROMISE_CHUNKS - len(selected), 0)
        added = extras[:room]
        if added:
            logger.info(
                f"Fiscal year {year}: section filter matched only {selected_words} words, "
                f"added {len(added)} chunks by forward-looking cue phrases"
            )
        selected.extend(added)

    if len(selected) > MAX_PROMISE_CHUNKS:
        selected = sorted(selected, key=lambda c: _cue_count(c.text), reverse=True)[:MAX_PROMISE_CHUNKS]

    order = {c.chunk_id: i for i, c in enumerate(all_chunks)}
    selected.sort(key=lambda c: order.get(c.chunk_id, 0))
    return selected


def _log_selected_chunks(year: int, chunks: list) -> None:
    """Log what is about to be sent to Gemini, so an empty result can be explained."""
    total_words = sum(c.word_count for c in chunks)
    logger.info(f"Fiscal year {year}: sending {len(chunks)} chunks ({total_words} words) to Gemini")
    for c in chunks:
        preview = c.text[:110].replace("\n", " ")
        logger.info(
            f"  [{c.section} > {c.subsection}] {c.word_count}w, "
            f"{_cue_count(c.text)} cues | {preview}"
        )


def filter_promise_chunks(
    chunks_by_year: dict[int, list],
    sections: list[str] = PROMISE_EXTRACTION_SECTIONS,
) -> dict[int, list]:
    """Filter chunks to those in promise-extraction sections (case-insensitive substring match).

    The exclusion list (_EXCLUDED_SECTION_KEYWORDS) is applied only to fallback
    candidates (cue-phrase widening), not to chunks that already matched an
    inclusion keyword.

    Args:
        chunks_by_year: Dictionary mapping fiscal year to list of chunks.
        sections: List of keyword patterns to match against section names.

    Returns:
        dict[int, list]: Filtered chunks from matching sections.
    """
    filtered = {}
    for year, chunks in chunks_by_year.items():
        section_counts = Counter(c.section for c in chunks)
        logger.info("Fiscal year %d: section names found: %s", year, dict(section_counts))

        selected = [
            c for c in chunks
            if any(keyword.lower() in c.section.lower() for keyword in sections)
        ]

        logger.info(
            "Fiscal year %d: filtered %d chunks down to %d promise-extraction chunks",
            year, len(chunks), len(selected),
        )

        if ENABLE_CUE_FALLBACK:
            selected = _augment_and_cap_chunks(year, chunks, selected)

        filtered[year] = selected

    return filtered


def build_extraction_prompt(year: int, chunks: list) -> str:
    """Build prompt for extracting forward-looking commitments from chunks.

    Args:
        year: Fiscal year context for the extraction.
        chunks: List of Chunk objects to extract promises from.

    Returns:
        str: Complete prompt for the LLM.
    """
    chunk_texts = []
    for i, chunk in enumerate(chunks, 1):
        chunk_texts.append(f"--- Chunk {i} ---\n{chunk.text}")

    combined_text = "\n\n".join(chunk_texts)

    prompt = f"""You are analyzing SEC filing content from fiscal year {year} to extract forward-looking statements, commitments, strategic plans, and management intentions.

TASK: Extract any forward-looking statement that indicates management's intentions, plans, expectations, or commitments about future business activities. This includes:

- Strategic plans and business initiatives
- Technology investments or development plans
- Growth, expansion, or market plans
- Operational improvements or changes
- Financial targets, guidance, or expectations
- Commitments to stakeholders (employees, customers, shareholders)
- Business model changes or adaptations
- Regulatory compliance or adaptation plans

IMPORTANT: Include statements even with cautious language like "may", "could", "intend to", "plan to", "expect to", "committed to", "will continue", "are working to", etc. These indicate management intentions and strategic direction.

INSTRUCTIONS:
1. Extract each distinct forward-looking statement as a separate promise
2. For each promise, determine the domain (must be exactly one of: Technology, Claims, Growth, Finance, Operations)
3. If a specific deadline is mentioned in the text, capture it verbatim; otherwise use null
4. The year_made is {year} (the fiscal year of this filing)
5. Set page_number to 0 (page numbers are not available from chunk text)

RESPONSE FORMAT: Respond with ONLY a JSON array matching this exact structure, no markdown fences, no preamble:
[{{"promise_text": "exact text of the commitment or strategic statement", "domain": "Technology|Claims|Growth|Finance|Operations", "deadline_mentioned": "exact deadline text or null", "year_made": {year}, "page_number": 0}}]

If no forward-looking statements are found, respond with an empty array: []

CONTENT TO ANALYZE:
{combined_text}"""

    return prompt


def _validate_and_extract_response_text(response: Any) -> str:
    """Extract candidate text from Gemini response, checking finish_reason."""
    if not response.candidates:
        raise RuntimeError("Gemini returned no candidates (prompt may have been blocked)")

    first_candidate = response.candidates[0]
    finish_reason = getattr(first_candidate, "finish_reason", None)
    finish_str = str(finish_reason).upper() if finish_reason else ""

    if "MAX_TOKENS" in finish_str:
        raise RuntimeError(
            f"Gemini response truncated: finish_reason is MAX_TOKENS (output exceeded max_output_tokens limit). "
            f"Extraction JSON is incomplete."
        )

    if "SAFETY" in finish_str or "BLOCK" in finish_str:
        raise RuntimeError(
            f"Gemini generation was blocked due to safety policy (finish_reason={finish_reason})."
        )

    text = response.text
    if text is None and first_candidate.content and first_candidate.content.parts:
        parts_text = "".join(
            p.text for p in first_candidate.content.parts
            if hasattr(p, "text") and p.text
        )
        if parts_text:
            text = parts_text

    if text is None:
        raise RuntimeError("Gemini returned a candidate with no text content")

    return text


def _call_gemini(prompt: str, api_key: str) -> str:
    """Low-level Gemini API call with coordinated rate limiting, retry logic, and model fallback.

    Args:
        prompt: The prompt to send to Gemini.
        api_key: API key for authentication.

    Returns:
        str: Raw response text from Gemini.
    """
    client = genai.Client(api_key=api_key)
    limiter = get_rate_limiter("gemini_generate")

    def _invoke(model_name: str):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"temperature": 0.1, "max_output_tokens": 8192},
            )
        except genai_errors.APIError as api_err:
            if getattr(api_err, "code", None) == 429:
                logger.warning(f"Gemini API returned 429 error code: {api_err}")
            raise

        return _validate_and_extract_response_text(response)

    import src.config as cfg
    primary_model = getattr(cfg, "EXTRACTOR_MODEL", EXTRACTOR_MODEL)
    fallback_model = getattr(cfg, "EXTRACTOR_FALLBACK_MODEL", EXTRACTOR_FALLBACK_MODEL)

    try:
        return limiter.execute_with_retry(lambda: _invoke(primary_model))
    except Exception as e:
        status_code = getattr(e, "code", None)
        err_msg = str(e).lower()
        is_fallback_eligible = (
            status_code in (404, 503)
            or "503" in err_msg
            or "404" in err_msg
            or "unavailable" in err_msg
            or "overloaded" in err_msg
            or "high demand" in err_msg
            or "not found" in err_msg
            or "no longer available" in err_msg
        )
        if is_fallback_eligible and fallback_model and fallback_model != primary_model:
            logger.warning(
                f"Primary extractor model '{primary_model}' failed ({e}). "
                f"Falling back to '{fallback_model}'."
            )
            return limiter.execute_with_retry(lambda: _invoke(fallback_model))
        raise


def _parse_promise_json(raw_text: str) -> list[dict]:
    """Parse JSON response from Gemini, handling markdown code fences.

    Args:
        raw_text: Raw response text from Gemini.

    Returns:
        list[dict]: Parsed JSON data.

    Raises:
        json.JSONDecodeError: If JSON parsing fails.
    """
    text = raw_text.strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()
    parsed = json.loads(text)

    if isinstance(parsed, dict):
        if isinstance(parsed.get("promises"), list):
            return parsed["promises"]
        return [parsed] if "promise_text" in parsed else []

    return parsed


def extract_promises_for_year(
    year: int,
    chunks: list,
    api_key: str | None = None,
) -> list[Promise]:
    """Extract promises from chunks for a single fiscal year using Gemini.

    Supports TEMPUS_DRY_RUN=1 to test chunk selection without creating fake Promise objects.

    Args:
        year: Fiscal year for context.
        chunks: List of Chunk objects to extract from.
        api_key: Optional API key override (uses config if None).

    Returns:
        list[Promise]: Extracted and validated Promise objects (or [] if dry run).
    """
    resolved_api_key = api_key if api_key is not None else GEMINI_API_KEY
    if not resolved_api_key and not is_dry_run():
        raise ValueError(
            "No Gemini API key available: set GEMINI_API_KEY in config or pass one explicitly"
        )

    if not chunks:
        logger.warning(f"No chunks provided for fiscal year {year}, skipping promise extraction")
        return []

    logger.info(f"Extracting promises from {len(chunks)} chunks for fiscal year {year}")

    # Dry run support to test chunk selection without calling live Gemini
    if is_dry_run():
        logger.info(
            f"TEMPUS_DRY_RUN=1 active: logging {len(chunks)} selected chunks for fiscal year {year} "
            f"and skipping Gemini API call (returning [] promises)"
        )
        return []

    prompt = build_extraction_prompt(year, chunks)

    try:
        raw_response = _call_gemini(prompt, resolved_api_key)
        promise_dicts = _parse_promise_json(raw_response)

    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed for fiscal year {year} on first attempt: {e}")
        retry_prompt = (
            prompt
            + "\n\nYour previous response was not valid JSON. Respond with ONLY the JSON array, nothing else."
        )

        try:
            raw_response = _call_gemini(retry_prompt, resolved_api_key)
            promise_dicts = _parse_promise_json(raw_response)
            logger.info(f"Retry successful for fiscal year {year}")

        except json.JSONDecodeError as retry_error:
            logger.error(f"JSON parse failed for fiscal year {year} after retry: {retry_error}")
            logger.error(f"Raw response: {raw_response[:500]}...")
            raise ValueError(f"Gemini returned invalid JSON for fiscal year {year} twice") from retry_error

    if not promise_dicts:
        total_words = sum(c.word_count for c in chunks)
        logger.warning(
            f"Gemini returned an empty array for fiscal year {year} "
            f"({len(chunks)} chunks, {total_words} words). Raw response: {raw_response[:200]!r}"
        )

    promises = []
    for i, item in enumerate(promise_dicts):
        try:
            item["year_made"] = year

            dl = item.get("deadline_mentioned")
            if dl is not None and str(dl).strip().lower() in ("null", "none", ""):
                item["deadline_mentioned"] = None

            domain_raw = item.get("domain", "")
            domain_map = {d.lower(): d for d in ("Technology", "Claims", "Growth", "Finance", "Operations")}
            if domain_raw and domain_raw.lower() in domain_map:
                item["domain"] = domain_map[domain_raw.lower()]

            promise = Promise(**item)
            promises.append(promise)

        except Exception as e:
            logger.warning("Failed to create Promise object %d for fiscal year %d: %s", i, year, e)

    logger.info(f"Successfully extracted {len(promises)} promises for fiscal year {year}")
    return promises


def extract_all_promises(
    chunks_by_year: dict[int, list],
    api_key: str | None = None,
) -> dict[int, list[Promise]]:
    """Extract promises from all fiscal years of chunks.

    Args:
        chunks_by_year: Dictionary mapping fiscal year to list of Chunk objects.
        api_key: Optional API key override (uses config if None).

    Returns:
        dict[int, list[Promise]]: Dictionary mapping fiscal year to extracted Promise objects.
    """
    resolved_api_key = api_key if api_key is not None else GEMINI_API_KEY
    if not resolved_api_key and not is_dry_run():
        raise ValueError(
            "No Gemini API key available: set GEMINI_API_KEY in config or pass one explicitly"
        )

    logger.info(f"Starting promise extraction for {len(chunks_by_year)} years")

    filtered_chunks = filter_promise_chunks(chunks_by_year)

    results = {}
    total_promises = 0
    attempted_years = []
    failures = {}

    for year in sorted(filtered_chunks.keys()):
        chunks = filtered_chunks[year]

        if not chunks:
            logger.warning(f"Fiscal year {year}: no chunks selected, nothing to extract")
            results[year] = []
            continue

        _log_selected_chunks(year, chunks)
        attempted_years.append(year)

        try:
            promises = extract_promises_for_year(year, chunks, resolved_api_key)
            results[year] = promises
            total_promises += len(promises)

        except DailyQuotaExhaustedError as e:
            logger.error(f"Gemini daily quota exhausted at fiscal year {year}: {e}")
            raise

        except Exception as e:
            logger.error(f"Promise extraction failed for fiscal year {year}: {e}")
            failures[year] = str(e)
            results[year] = []

    if attempted_years and len(failures) == len(attempted_years):
        raise RuntimeError(f"Promise extraction failed for every year: {failures}")

    if failures:
        logger.warning(f"Promise extraction failed for years {sorted(failures)}: {failures}")

    logger.info(f"Promise extraction complete: {total_promises} total promises across all years")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    from src.models import Chunk

    def make_chunk(year: int, section: str, subsection: str, text: str, idx: int) -> Chunk:
        return Chunk(
            text=text,
            company="HIG",
            year=year,
            section=section,
            subsection=subsection,
            chunk_id=f"HIG_{year}_{idx:02d}",
            word_count=len(text.split()),
        )

    sample_chunks = {
        2021: [
            make_chunk(
                2021, "Item 7 MD&A", "Technology",
                "[Hartford | 2021 | Item 7 MD&A > Technology] We are committed to reducing claims processing time "
                "by 40% by the end of 2023 through our new digital platform.", 0
            ),
            make_chunk(
                2021, "Item 1. Business", "full_text",
                "[Hartford | 2021 | Item 1. Business] We plan to expand our market share in the small "
                "business segment by 15% over the next two years.", 1
            ),
        ]
    }

    print("Testing promise extraction (offline checks)...")
    print(f"Sample chunks: {sum(len(chunks) for chunks in sample_chunks.values())} total")

    # Test filtering
    filtered = filter_promise_chunks(sample_chunks)
    print(f"Filtered chunks: {sum(len(chunks) for chunks in filtered.values())} in promise-extraction sections")
    assert len(filtered[2021]) == 2

    # Test prompt building
    prompt = build_extraction_prompt(2021, filtered[2021])
    print(f"Sample prompt length: {len(prompt)} characters")
    assert f"fiscal year 2021" in prompt

    # Test cue fallback and exclusion filtering
    print("\n--- Testing cue fallback and section exclusion ---")
    cue_text = (
        "[Hartford | 2026 | Item 1] We plan to invest in claims automation. We expect to launch a new "
        "platform by 2027. Our goal is to cut costs. We are committed to growth over the next three years. "
    ) * 3
    boiler_text = "[Hartford | 2026 | Item 1] The Private Securities Litigation Reform Act of 1995 provides a safe harbor. " * 5
    risk_text = "[Hartford | 2026 | Risk Factors] We plan to face many risks by 2027. We expect to see losses. " * 3
    fallback_input = {
        2026: [
            make_chunk(2026, "Item 7 MD&A", "full_text", "[Hartford | 2026 | Item 7] Table of contents page 12.", 0),
            make_chunk(2026, "Item 1", "full_text", cue_text, 1),
            make_chunk(2026, "Item 1", "full_text", boiler_text, 2),
            make_chunk(2026, "Risk Factors", "full_text", risk_text, 3),
        ]
    }
    picked = filter_promise_chunks(fallback_input)[2026]
    picked_ids = [c.chunk_id for c in picked]
    print(f"Picked chunk ids: {picked_ids}")
    assert "HIG_2026_01" in picked_ids, "Cue chunk should be included by fallback"
    assert "HIG_2026_02" not in picked_ids, "Safe-harbor chunk should be excluded"
    assert "HIG_2026_03" not in picked_ids, "Risk-factor chunk should be excluded"

    # Test MAX_TOKENS finish_reason raises clear error
    print("\n--- Testing MAX_TOKENS finish_reason handling ---")
    class MockCandidate:
        def __init__(self, finish_reason: str):
            self.finish_reason = finish_reason
            self.content = None

    class MockResponse:
        def __init__(self, finish_reason: str):
            self.candidates = [MockCandidate(finish_reason)]
            self.text = None

    try:
        _validate_and_extract_response_text(MockResponse("MAX_TOKENS"))
        assert False, "Failed: Did not raise on MAX_TOKENS"
    except RuntimeError as re_err:
        assert "finish_reason is MAX_TOKENS" in str(re_err)
        print("OK: MAX_TOKENS correctly raised clear RuntimeError")

    # Test TEMPUS_DRY_RUN=1 behavior
    print("\n--- Testing TEMPUS_DRY_RUN=1 ---")
    os.environ["TEMPUS_DRY_RUN"] = "1"
    try:
        assert is_dry_run() is True
        dry_results = extract_all_promises(sample_chunks)
        assert 2021 in dry_results
        assert dry_results[2021] == [], f"Expected [] in dry run, got {dry_results[2021]}"
        print("OK: TEMPUS_DRY_RUN returned [] and logged selected chunks without fake promises")
    finally:
        os.environ.pop("TEMPUS_DRY_RUN", None)

    # Test fallback model invocation on 503 error
    print("\n--- Testing fallback model on 503/overload ---")
    from unittest.mock import MagicMock, patch
    call_log = []
    def mock_gen_content(model, contents, config):
        call_log.append(model)
        if model == EXTRACTOR_MODEL:
            err = genai_errors.APIError(503, {"error": {"message": "Model is overloaded. Please try again later."}})
            raise err
        mock_cand = MagicMock()
        mock_cand.finish_reason = "STOP"
        mock_cand.content.parts = []
        mock_res = MagicMock()
        mock_res.candidates = [mock_cand]
        mock_res.text = "[]"
        return mock_res

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = mock_gen_content
    with patch("google.genai.Client", return_value=mock_client), patch("time.sleep", return_value=None):
        res = _call_gemini("test prompt", "fake-key")
        assert res == "[]"
        assert EXTRACTOR_MODEL in call_log
        assert EXTRACTOR_FALLBACK_MODEL in call_log
        print(f"OK: Successfully fell back from {EXTRACTOR_MODEL} to {EXTRACTOR_FALLBACK_MODEL} on 503")

    print("\nExtractor testing complete and verified offline!")