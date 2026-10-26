"""Environment configuration and shared constants for TempusRAG.

Single source of truth for all tunable parameters.
No logic, no I/O beyond loading .env.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMADB_PATH = Path(os.getenv("CHROMADB_PATH", str(PROJECT_ROOT / "chroma_store"))).resolve()
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data" / "filings"))).resolve()
LOG_DIR = Path(os.getenv("LOG_DIR", str(PROJECT_ROOT / "logs"))).resolve()

CHROMADB_PATH.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# YEAR CONVENTION
# ============================================================================

YEAR_CONVENTION = "fiscal_year"
"""Fiscal year of the reported period end date (not the filing calendar date).
e.g. 10-K filed Feb 2026 covering period ending Dec 31 2025 → year=2025.
"""

# ============================================================================
# API KEYS
# ============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_JUDGE_API_KEY = os.getenv("GEMINI_JUDGE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
"""Separate Gemini API key for cross-year judging (LLM Call 2).
Using a different key from extraction isolates rate-limit quotas so judging
calls never consume the extraction key's 15 RPM / 500 RPD allowance.
Falls back to GEMINI_API_KEY if GEMINI_JUDGE_API_KEY is not set."""
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "")

# ============================================================================
# CACHE
# ============================================================================

CACHE_STALENESS_DAYS = int(os.getenv("CACHE_STALENESS_DAYS", "180"))

# ============================================================================
# CHUNKING
# ============================================================================

CHUNK_MAX_TOKENS = 500
"""Max words per chunk. 500 words ≈ 650 subword tokens, safely under embedding limits."""

CHUNK_OVERLAP_TOKENS = 50

# ============================================================================
# EMBEDDINGS
# ============================================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
"""Local sentence-transformers model. Zero rate limits, runs on CPU."""

EMBEDDING_DIMENSIONS = 384
EMBEDDING_DISTANCE_SPACE = "cosine"

# ============================================================================
# RETRIEVAL
# ============================================================================

DENSE_TOP_K = 20
SPARSE_TOP_K = 20
RRF_K = 60
FINAL_TOP_N = 5

# ============================================================================
# SECTION BOOST WEIGHTS
# ============================================================================

SECTION_BOOST_WEIGHTS = {
    "Item 7":  1.3,   # MD&A: most promise-rich
    "MD&A":    1.3,
    "outlook": 1.4,
    "Item 1.": 1.2,   # Business: strategy
    "Item 1A": 0.7,   # Risk Factors: not promise-bearing
    "Risk Factors": 0.7,
    "Item 8":  0.7,   # Financial Statements: numbers not commitments
}

DEFAULT_SECTION_BOOST = 1.0

# ============================================================================
# SIMILARITY THRESHOLDS
# ============================================================================

SIMILARITY_STRONG_EVIDENCE = 0.7
SIMILARITY_WEAK_EVIDENCE = 0.5

# ============================================================================
# SECTION ROLES
# ============================================================================

PROMISE_EXTRACTION_SECTIONS = [
    "Item 7",
    "Item 1.",
    "MD&A",
    "management's discussion",
    "outlook",
    "strategy",
    "growth",
]

EVIDENCE_SEARCH_SECTIONS = ["Item 7", "Item 1.", "Item 8"]

# ============================================================================
# RECENCY WEIGHTS
# ============================================================================

def recency_weight(year: int, most_recent_year: int) -> float:
    """Recency weight: 1.0 for most recent year, -0.1 per year back, floor 0.6."""
    offset = most_recent_year - year
    return max(1.0 - (offset * 0.1), 0.6)

# ============================================================================
# CONFIDENCE SCORE WEIGHTS
# ============================================================================

CONFIDENCE_SIGNAL_WEIGHTS = {
    "has_deadline": 0.3,
    "has_metric": 0.3,
    "has_commitment_verb": 0.2,
    "domain_detectable": 0.2,
}

# ============================================================================
# JUDGE SCORING RUBRIC
# ============================================================================

JUDGE_ANCHORS = """90-100: Explicit delivery with matching metrics
70-89:  Confirmed progress, metrics incomplete
50-69:  Topic mentioned, no clear confirmation
30-49:  Contradicted or shortfall shown
0-29:   No evidence or explicit failure""".strip()

# ============================================================================
# LLM MODELS
# ============================================================================

EXTRACTOR_MODEL = os.getenv("EXTRACTOR_MODEL", "gemini-3.5-flash-lite")
"""Gemini model for promise extraction.
gemini-3.5-flash-lite: 15 RPM, 500 RPD, 250K TPM — highest combined rate limits on free tier.
"""

EXTRACTOR_FALLBACK_MODEL = os.getenv("EXTRACTOR_FALLBACK_MODEL", "gemini-3.1-flash-lite")
"""Fallback if primary Gemini model returns 503/404. gemini-3.1-flash-lite: 15 RPM, 500 RPD, 250K TPM."""

# Judge provider: "gemini" (default, recommended) or "groq"
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "gemini")
"""Which provider to use for cross-year judging.
'gemini': uses Gemini Flash Lite (250K TPM, 15 RPM) — available everywhere, no regional restrictions.
'groq':  uses Groq (faster per call, but Llama models are geo-restricted; only gpt-oss-20b works in India).
"""

GEMINI_JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-3.5-flash-lite")
"""Gemini model for judging. gemini-3.5-flash-lite: 15 RPM, 500 RPD, 250K TPM on free tier.
Much higher limits than Groq gpt-oss-20b (8K TPM). Supports concurrent judging.
"""

# Legacy Groq judge model (only used when JUDGE_PROVIDER=groq)
JUDGE_MODEL = os.getenv("GROQ_MODEL", os.getenv("JUDGE_MODEL", "openai/gpt-oss-20b"))
"""Groq model for cross-year judging. Free tier: 30 RPM, 1K RPD, 8K TPM.
Only used when JUDGE_PROVIDER=groq.
"""

MAX_JUDGE_EVIDENCE_CHUNKS = int(os.getenv("MAX_JUDGE_EVIDENCE_CHUNKS", "3"))
"""Max evidence chunks per judge call. 3 chunks × 300 words ≈ 2,500 tokens input.
With Gemini (1M TPM) this is comfortably within limits even at 3 concurrent workers."""

MAX_JUDGE_CHUNK_WORDS = int(os.getenv("MAX_JUDGE_CHUNK_WORDS", "300"))
"""Max words per evidence chunk sent to judge. 300 words ≈ 400 tokens per chunk."""

# ============================================================================
# RATE LIMITS — verified 2026-09-19
# ============================================================================

SEC_MIN_REQUEST_INTERVAL = 0.11
"""Min seconds between EDGAR requests (stay under 10 req/sec)."""

# Gemini 3.5 Flash Lite: 15 RPM, 500 RPD
GEMINI_GENERATE_RPM = int(os.getenv("GEMINI_GENERATE_RPM", "15"))
GEMINI_GENERATE_RPD = int(os.getenv("GEMINI_GENERATE_RPD", "500"))

# Groq gpt-oss-20b: 30 RPM, 1K RPD, 8K TPM (only used when JUDGE_PROVIDER=groq)
GROQ_RPM = int(os.getenv("GROQ_RPM", "30"))
GROQ_RPD = int(os.getenv("GROQ_RPD", "1000"))

# Pacing interval for Groq judge calls (only used when JUDGE_PROVIDER=groq)
GROQ_REQUEST_INTERVAL = float(os.getenv("GROQ_REQUEST_INTERVAL", "7.5"))
"""Seconds between Groq judge calls when JUDGE_PROVIDER=groq.
8K TPM ÷ ~900 tokens/call ≈ 8 safe calls/min → 7.5s interval.
"""

# Gemini judge rate limits: gemini-3.5-flash-lite: 15 RPM, 500 RPD, 250K TPM
GEMINI_JUDGE_RPM = int(os.getenv("GEMINI_JUDGE_RPM", "15"))
GEMINI_JUDGE_REQUEST_INTERVAL = float(os.getenv("GEMINI_JUDGE_REQUEST_INTERVAL", "4.0"))
"""Seconds between Gemini judge calls. 4.0s interval = 15 RPM limit on gemini-3.5-flash-lite.
Pacing requests 4.0s apart prevents 429 RESOURCE_EXHAUSTED server retries.
"""

GEMINI_REQUEST_INTERVAL = 60.0 / GEMINI_GENERATE_RPM
"""Seconds between Gemini extractor calls (= 4.0s at 15 RPM)."""

# ============================================================================
# RETRY CONFIG
# ============================================================================

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0
RETRY_MAX_DELAY = 60.0

CIRCUIT_BREAKER_COOLDOWN_SECONDS = float(
    os.getenv("CIRCUIT_BREAKER_COOLDOWN_SECONDS", "300")
)
"""Cooldown before auto-reset after a fatal error. 300s (5 min) is a safe default that
does not freeze dev iterations the way 1800s (30 min) does."""


# ============================================================================
# CONCURRENCY
# ============================================================================

MAX_JUDGE_WORKERS = int(os.getenv("MAX_JUDGE_WORKERS", "3"))
"""Max concurrent judge threads."""

MAX_EXTRACTOR_WORKERS = int(os.getenv("MAX_EXTRACTOR_WORKERS", "3"))
"""Max concurrent Gemini extractor threads.
At 4s interval × 3 workers ≈ 9 RPM sustained — within 15 RPM free-tier limit.
"""


if __name__ == "__main__":
    print("=== TempusRAG Configuration ===")
    print(f"PROJECT_ROOT:             {PROJECT_ROOT}")
    print(f"EXTRACTOR_MODEL:          {EXTRACTOR_MODEL}")
    print(f"EXTRACTOR_FALLBACK_MODEL: {EXTRACTOR_FALLBACK_MODEL}")
    print(f"JUDGE_MODEL:              {JUDGE_MODEL}")
    print(f"MAX_JUDGE_EVIDENCE_CHUNKS:{MAX_JUDGE_EVIDENCE_CHUNKS}")
    print(f"MAX_JUDGE_CHUNK_WORDS:    {MAX_JUDGE_CHUNK_WORDS}")
    print(f"GEMINI_GENERATE_RPM:      {GEMINI_GENERATE_RPM}")
    print(f"GEMINI_REQUEST_INTERVAL:  {GEMINI_REQUEST_INTERVAL:.1f}s")
    print(f"JUDGE_PROVIDER:           {JUDGE_PROVIDER}")
    print(f"GEMINI_JUDGE_MODEL:       {GEMINI_JUDGE_MODEL}")
    print(f"GEMINI_JUDGE_INTERVAL:    {GEMINI_JUDGE_REQUEST_INTERVAL}s")
    print(f"GROQ_RPM:                 {GROQ_RPM}")
    print(f"GROQ_REQUEST_INTERVAL:    {GROQ_REQUEST_INTERVAL}s")
    print(f"MAX_JUDGE_WORKERS:        {MAX_JUDGE_WORKERS}")
    print(f"EMBEDDING_MODEL:          {EMBEDDING_MODEL_NAME}")
    print(f"recency_weight(2022,2024):{recency_weight(2022, 2024):.1f}")
    print("Config loaded successfully!")