"""Promise extraction from MD&A sections using Gemini (LLM Call 1).

Takes filtered chunks from promise-extraction sections and extracts
structured Promise objects using Gemini 3.5 Flash Lite.

Rate limiting: delegated to rate_limit.py gemini_generate limiter (4s interval, 15 RPM).
Daily quota errors are terminal and trip the circuit breaker.
A model answer of [] is a valid result (no promises found).
Any API/parse error is raised so a broken run never looks like zero promises.
"""

import json
import logging
import os
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import errors as genai_errors

from src.config import (
    CHUNK_MAX_TOKENS,
    EXTRACTOR_FALLBACK_MODEL,
    EXTRACTOR_MODEL,
    GEMINI_API_KEY,
    MAX_EXTRACTOR_WORKERS,
    PROMISE_EXTRACTION_SECTIONS,
)
from src.models import Promise
from src.rate_limit import DailyQuotaExhaustedError, get_rate_limiter

logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google.auth").setLevel(logging.WARNING)

# ============================================================================
# GEMINI CLIENT SINGLETON  (Change 12)
# ============================================================================

_gemini_client = None
_gemini_client_lock = threading.Lock()
_gemini_client_api_key = None


def _get_gemini_client(api_key: str):
    """Return a shared Gemini client singleton (thread-safe lazy initialization).

    A single client reuses the underlying HTTP session across all extraction
    calls, avoiding per-call TLS handshake overhead. Safe for concurrent use
    since google.genai.Client is stateless after construction.
    """
    global _gemini_client, _gemini_client_api_key
    if _gemini_client is not None and _gemini_client_api_key == api_key:
        return _gemini_client
    with _gemini_client_lock:
        if _gemini_client is None or _gemini_client_api_key != api_key:
            _gemini_client = genai.Client(api_key=api_key)
            _gemini_client_api_key = api_key
            logger.info("Gemini client singleton initialized")
    return _gemini_client

# ============================================================================
# CHUNK SELECTION CONSTANTS
# ============================================================================

TARGET_PROMISE_WORDS = 10000
MAX_PROMISE_CHUNKS = max(10, TARGET_PROMISE_WORDS // max(1, CHUNK_MAX_TOKENS))
MIN_SECTION_WORDS = TARGET_PROMISE_WORDS // 2
MIN_CUE_HITS = 2

# Sections that never hold commitments — applied only to cue-phrase fallback candidates
_EXCLUDED_SECTION_KEYWORDS = (
    "risk factor",
    "legal proceeding",
    "controls and procedures",
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

# Forward-looking commitment cue phrases
_CUE_RE = re.compile(
    r"\bwe (?:plan|intend|expect|anticipate|aim|seek|target|will|are committed|are working|continue to invest)\b"
    r"|\bour (?:goal|goals|objective|objectives|target|targets|priorit(?:y|ies)|commitment)\b"
    r"|\bcommitted to\b"
    r"|\bby (?:the end of )?(?:fiscal )?(?:year )?20\d\d\b"
    r"|\bover the next (?:\w+ ){0,2}(?:years?|months?|quarters?)\b",
    re.IGNORECASE,
)

def _cue_count(text: str) -> int:
    return len(_CUE_RE.findall(text))


def _is_excluded_chunk(chunk) -> bool:
    section = chunk.section.lower()
    if any(kw in section for kw in _EXCLUDED_SECTION_KEYWORDS):
        return True
    return _SAFE_HARBOR_MARKER in chunk.text.lower()


def _is_dry_run() -> bool:
    return os.getenv("TEMPUS_DRY_RUN", "").strip().lower() in ("1", "true", "yes")


# ============================================================================
# CHUNK SELECTION
# ============================================================================

def _augment_and_cap_chunks(year: int, all_chunks: list, selected: list) -> list:
    """Add high-cue chunks when section filter matched too little text, then cap."""
    selected = list(selected)
    selected_ids = {c.chunk_id for c in selected}
    selected_words = sum(c.word_count for c in selected)

    if selected_words < MIN_SECTION_WORDS:
        candidates = [c for c in all_chunks if c.chunk_id not in selected_ids]
        valid = [c for c in candidates if not _is_excluded_chunk(c)]
        cue_map = {c.chunk_id: _cue_count(c.text) for c in valid}
        ranked = sorted(valid, key=lambda c: cue_map[c.chunk_id], reverse=True)
        extras = [c for c in ranked if cue_map[c.chunk_id] >= MIN_CUE_HITS]
        room = max(MAX_PROMISE_CHUNKS - len(selected), 0)
        added = extras[:room]
        if added:
            logger.info(
                "Year %d: section filter matched only %d words, "
                "added %d chunks via cue-phrase fallback",
                year, selected_words, len(added),
            )
        selected.extend(added)

    if len(selected) > MAX_PROMISE_CHUNKS:
        sel_cues = {c.chunk_id: _cue_count(c.text) for c in selected}
        selected = sorted(
            selected, key=lambda c: sel_cues[c.chunk_id], reverse=True
        )[:MAX_PROMISE_CHUNKS]

    order = {c.chunk_id: i for i, c in enumerate(all_chunks)}
    selected.sort(key=lambda c: order.get(c.chunk_id, 0))
    return selected


def filter_promise_chunks(
    chunks_by_year: dict[int, list],
    sections: list[str] = PROMISE_EXTRACTION_SECTIONS,
) -> dict[int, list]:
    """Filter chunks to promise-extraction sections, with cue-phrase fallback."""
    filtered = {}
    for year, chunks in chunks_by_year.items():
        section_counts = Counter(c.section for c in chunks)
        logger.info("Year %d: sections found: %s", year, dict(section_counts))

        selected = [
            c for c in chunks
            if any(kw.lower() in c.section.lower() for kw in sections)
        ]
        logger.info(
            "Year %d: %d → %d chunks after section filter",
            year, len(chunks), len(selected),
        )

        selected = _augment_and_cap_chunks(year, chunks, selected)
        filtered[year] = selected

    return filtered


# ============================================================================
# PROMPT BUILDING
# ============================================================================

def build_extraction_prompt(year: int, chunks: list) -> str:
    """Build Gemini extraction prompt for a single fiscal year."""
    chunk_texts = "\n\n".join(
        f"--- Chunk {i} ---\n{c.text}" for i, c in enumerate(chunks, 1)
    )

    return f"""You are analyzing SEC Form 10-K filing content from fiscal year {year}.

TASK: Extract every forward-looking commitment, strategic plan, operational target, or expectation made by company management about future activities.

INCLUDE STATEMENTS REGARDING:
- Technology, Software & AI: Autonomous driving, AI/robotics, computing architectures, platform rollouts, cybersecurity upgrades.
- Growth, Products & Production: Vehicle/hardware production ramps, new model launches, capacity expansions, customer demand goals.
- Finance & Capital Allocation: Capital expenditures, share repurchases, dividend expectations, debt retirement.
- Operations: M&A integration milestones, facility/branch buildouts, supply chain commitments, workforce initiatives.

GUIDELINES:
1. Extract each distinct forward-looking statement as a separate promise.
2. Domain must be exactly one of: Technology, Claims, Growth, Finance, Operations.
3. Capture deadline verbatim if mentioned in text (e.g. "2025", "fourth quarter of 2026", "by year-end 2024"), otherwise null.
4. Exclude purely historical descriptions of past results or accounting policies.
5. Exclude external third-party actions (e.g. government regulator decisions or lawsuits).
6. year_made is {year}. page_number is 0.

RESPONSE FORMAT: JSON array only, no markdown fences, no preamble:
[{{"promise_text": "...", "domain": "...", "deadline_mentioned": "..." or null, "year_made": {year}, "page_number": 0}}]

If no forward-looking statements found, respond with: []

CONTENT:
{chunk_texts}"""


# ============================================================================
# GEMINI API CALL
# ============================================================================

def _call_gemini(prompt: str, api_key: str) -> str:
    """Call Gemini with rate limiting, retry, and model fallback.

    Primary model: EXTRACTOR_MODEL (gemini-3.5-flash-lite)
    Fallback: EXTRACTOR_FALLBACK_MODEL (gemini-3.5-flash) on 404/503 only.
    Rate limiting and retry handled by gemini_generate limiter (4s interval).
    """
    client = _get_gemini_client(api_key)  # singleton: reuses HTTP session, avoids TLS overhead
    limiter = get_rate_limiter("gemini_generate")

    def _invoke(model_name: str) -> str:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"temperature": 0.1, "max_output_tokens": 8192},
            )
        except genai_errors.APIError as e:
            if getattr(e, "code", None) == 429:
                logger.warning("Gemini 429 on model %s: %s", model_name, e)
            raise

        return _extract_response_text(response)

    try:
        return limiter.execute_with_retry(lambda: _invoke(EXTRACTOR_MODEL))

    except Exception as e:
        # Only fall back on model-unavailable errors, not rate limits
        code = getattr(e, "code", None)
        err_lower = str(e).lower()
        is_model_error = (
            code in (404, 503)
            or any(k in err_lower for k in (
                "404", "503", "not found", "unavailable",
                "overloaded", "high demand", "no longer available",
            ))
        )
        if is_model_error and EXTRACTOR_FALLBACK_MODEL != EXTRACTOR_MODEL:
            logger.warning(
                "Primary model '%s' unavailable (%s), falling back to '%s'",
                EXTRACTOR_MODEL, e, EXTRACTOR_FALLBACK_MODEL,
            )
            return limiter.execute_with_retry(lambda: _invoke(EXTRACTOR_FALLBACK_MODEL))
        raise


def _extract_response_text(response) -> str:
    """Extract text from Gemini response, raising on truncation or safety blocks."""
    if not response.candidates:
        raise RuntimeError("Gemini returned no candidates (prompt may be blocked)")

    candidate = response.candidates[0]
    finish = str(getattr(candidate, "finish_reason", "")).upper()

    if "MAX_TOKENS" in finish:
        raise RuntimeError(
            "Gemini response truncated (finish_reason=MAX_TOKENS). "
            "Extraction JSON is incomplete."
        )
    if "SAFETY" in finish or "BLOCK" in finish:
        raise RuntimeError(
            f"Gemini generation blocked by safety policy (finish_reason={finish})"
        )

    text = response.text
    if text is None and candidate.content and candidate.content.parts:
        text = "".join(
            p.text for p in candidate.content.parts
            if hasattr(p, "text") and p.text
        )

    if not text:
        raise RuntimeError("Gemini returned a candidate with no text content")

    return text


# ============================================================================
# JSON PARSING
# ============================================================================

def _parse_promise_json(raw_text: str) -> list[dict]:
    """Parse Gemini JSON response, stripping markdown fences if present."""
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


# ============================================================================
# EXTRACTION — SINGLE YEAR
# ============================================================================

def extract_promises_for_year(
    year: int,
    chunks: list,
    api_key: str | None = None,
) -> list[Promise]:
    """Extract Promise objects from chunks for one fiscal year.

    Returns [] on dry run or when Gemini finds no promises.
    Raises on API errors, invalid JSON, or quota exhaustion.
    """
    resolved_key = api_key or GEMINI_API_KEY
    if not resolved_key and not _is_dry_run():
        raise ValueError("No Gemini API key — set GEMINI_API_KEY or pass api_key")

    if not chunks:
        logger.warning("Year %d: no chunks, skipping extraction", year)
        return []

    total_words = sum(c.word_count for c in chunks)
    logger.info("Year %d: extracting from %d chunks (%d words)", year, len(chunks), total_words)

    if _is_dry_run():
        logger.info("TEMPUS_DRY_RUN=1: skipping Gemini call for year %d", year)
        return []

    prompt = build_extraction_prompt(year, chunks)

    # First attempt
    try:
        raw = _call_gemini(prompt, resolved_key)
        promise_dicts = _parse_promise_json(raw)

    except json.JSONDecodeError as e:
        logger.warning("Year %d: JSON parse failed, retrying with correction prompt: %s", year, e)
        retry_prompt = (
            prompt
            + "\n\nYour previous response was not valid JSON. "
            "Respond with ONLY the JSON array, nothing else."
        )
        try:
            raw = _call_gemini(retry_prompt, resolved_key)
            promise_dicts = _parse_promise_json(raw)
            logger.info("Year %d: retry successful", year)
        except json.JSONDecodeError as e2:
            raise ValueError(
                f"Gemini returned invalid JSON for year {year} twice"
            ) from e2

    if not promise_dicts:
        logger.warning(
            "Year %d: Gemini returned empty array (%d chunks, %d words). Raw: %r",
            year, len(chunks), total_words, raw[:200],
        )

    # Build Promise objects
    promises = []
    domain_map = {d.lower(): d for d in ("Technology", "Claims", "Growth", "Finance", "Operations")}

    for i, item in enumerate(promise_dicts):
        try:
            item["year_made"] = year

            dl = item.get("deadline_mentioned")
            if dl is not None and str(dl).strip().lower() in ("null", "none", ""):
                item["deadline_mentioned"] = None

            domain_raw = str(item.get("domain", "")).strip()
            if domain_raw and domain_raw.lower() in domain_map:
                item["domain"] = domain_map[domain_raw.lower()]
            else:
                matched = next((d for d in ("Operations", "Growth", "Finance", "Technology", "Claims") if d.lower() in domain_raw.lower()), "Operations")
                item["domain"] = matched

            promises.append(Promise(**item))

        except Exception as e:
            logger.warning("Year %d: failed to build Promise object %d: %s", year, i, e)

    logger.info("Year %d: extracted %d promises", year, len(promises))
    return promises


# ============================================================================
# EXTRACTION — ALL YEARS
# ============================================================================

def extract_all_promises(
    chunks_by_year: dict[int, list],
    api_key: str | None = None,
) -> dict[int, list[Promise]]:
    """Extract promises across all fiscal years in parallel.  (Change 6)

    Year isolation: each worker thread receives ONLY its own fiscal year's
    filtered chunks. No cross-year chunk mixing is possible since each
    extract_promises_for_year call builds a prompt exclusively from that
    year's data. FY2021 chunks never appear in an FY2022 or FY2023 context.

    Stops immediately on daily quota exhaustion (after running tasks wind down).
    Continues on per-year failures, raises only if every year fails.
    """
    resolved_key = api_key or GEMINI_API_KEY
    if not resolved_key and not _is_dry_run():
        raise ValueError("No Gemini API key — set GEMINI_API_KEY or pass api_key")

    logger.info("Starting extraction for %d years", len(chunks_by_year))
    filtered = filter_promise_chunks(chunks_by_year)

    results: dict[int, list[Promise]] = {}
    failures: dict[int, str] = {}
    total = 0
    total_lock = threading.Lock()
    quota_exhausted = threading.Event()  # signals all pending threads to abort
    quota_error_holder: list = []  # mutable container for inter-thread error capture

    def _extract_year(year: int) -> tuple[int, list[Promise]]:
        """Year-isolated extraction worker.

        Only sees filtered[year] chunks — strictly no cross-year leakage.
        Short-circuits if quota was already exhausted by another thread.
        """
        if quota_exhausted.is_set():
            logger.info("Year %d: skipping (quota already exhausted)", year)
            return year, []
        chunks = filtered.get(year, [])
        if not chunks:
            logger.warning("Year %d: no chunks after filtering, skipping", year)
            return year, []
        try:
            promises = extract_promises_for_year(year, chunks, resolved_key)
            return year, promises
        except DailyQuotaExhaustedError:
            quota_exhausted.set()
            raise

    # Sort valid vs. empty years up front
    valid_years = [y for y in sorted(filtered.keys()) if filtered.get(y)]
    for year in filtered:
        if not filtered.get(year):
            results[year] = []

    if not valid_years:
        logger.warning("No years with chunks after filtering")
        return results

    max_workers = min(MAX_EXTRACTOR_WORKERS, len(valid_years))
    logger.info(
        "Starting parallel extraction: %d workers for years %s",
        max_workers, valid_years,
    )

    attempted: list[int] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_extract_year, y): y for y in valid_years}

        for future in as_completed(future_map):
            year = future_map[future]
            attempted.append(year)
            try:
                _, promises = future.result()
                results[year] = promises
                with total_lock:
                    total += len(promises)

            except DailyQuotaExhaustedError as e:
                logger.error("Daily quota exhausted at year %d: %s", year, e)
                results[year] = []
                if not quota_error_holder:
                    quota_error_holder.append(e)

            except Exception as e:
                logger.error("Extraction failed for year %d: %s", year, e)
                failures[year] = str(e)
                results[year] = []

    # Re-raise quota error after executor finishes (allows all threads to wind down)
    if quota_error_holder:
        raise quota_error_holder[0]

    if attempted and len(failures) == len(attempted):
        raise RuntimeError(f"Extraction failed for every year: {failures}")

    if failures:
        logger.warning("Extraction failed for years %s: %s", sorted(failures), failures)

    logger.info("Extraction complete: %d total promises across all years", total)
    return results