"""Environment configuration and shared constants: single source of truth for all tunable parameters.

Every other module in src/ imports from this file to ensure consistency across the pipeline.
No logic, no I/O beyond loading .env: pure constants and a single utility function.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# ============================================================================
# PROJECT ROOT AND DIRECTORY PATHS
# ============================================================================

# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMADB_PATH = Path(os.getenv("CHROMADB_PATH", str(PROJECT_ROOT / "chroma_store"))).resolve()
"""Persistent ChromaDB storage directory (one collection per ticker), absolute path."""

DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data" / "filings"))).resolve()
"""Cached SEC 10-K filings stored as plaintext (ticker_year.txt format), absolute path."""

LOG_DIR = Path(os.getenv("LOG_DIR", str(PROJECT_ROOT / "logs"))).resolve()
"""Application log files directory, absolute path."""

# Ensure all directories exist
CHROMADB_PATH.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# YEAR CONVENTION
# ============================================================================

YEAR_CONVENTION = "fiscal_year"
"""Single year convention across TempusRAG: fiscal year (period-of-report end year).

All SEC 10-K filings are identified by the fiscal year of the reported period,
not the calendar filing date. For example, a 10-K filed in February 2026 covering
the period ending December 31, 2025 is tagged with year=2025 and saved as
HIG_2025.txt. This convention is consistently used in chunker prefixes, extractor
prompts, collection metadata, and recency weight calculations.
"""

# ============================================================================
# API KEYS AND EXTERNAL SERVICE CONFIG
# ============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
"""Google Generative AI API key for promise extraction and embeddings."""

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
"""Groq API key for judge model."""

SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "")
"""User-Agent header for SEC EDGAR requests. Required by SEC policy."""

# ============================================================================
# CACHE AND STALENESS CONFIG
# ============================================================================

CACHE_STALENESS_DAYS = int(os.getenv("CACHE_STALENESS_DAYS", "180"))
"""Filing cache is considered stale if older than this many days; triggers re-download."""

# ============================================================================
# CHUNKING PARAMETERS (ingestion + chunking pipeline)
# ============================================================================

CHUNK_MAX_TOKENS = 500
"""Maximum words per chunk after splitting; enforced by chunker.py.

Set to 500 words (~650 to 750 subword tokens in standard English).
Table token risk: SEC 10-K filings contain extensive unpunctuated financial tables,
XBRL snippets, and numerical schedules. Table text has a much higher token-to-word
ratio (often 1.5 to 2.5x) due to isolated digits, delimiters, and symbols.
Setting the maximum chunk size to 500 words provides a robust safety margin,
preventing table chunks from overflowing the Gemini embedding 2,048-token limit
while leaving ample room for prepended contextual prefixes.
Treated as words (len(text.split())), not raw subword tokens.
"""

CHUNK_OVERLAP_TOKENS = 50
"""Overlap in words between consecutive chunks to preserve context bridges."""

# ============================================================================
# EMBEDDING PARAMETERS
# ============================================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
"""Local lightweight sentence-transformers model for dense embeddings (384-dimensional).

Runs offline on CPU in seconds with zero API rate limits or network latency.
"""

EMBEDDING_DIMENSIONS = 384
"""Output dimensionality of the all-MiniLM-L6-v2 embedding model (384)."""

EMBEDDING_DISTANCE_SPACE = "cosine"
"""Distance metric for ChromaDB collections: 'cosine', 'l2', or 'ip'.

When set to 'cosine', Chroma computes cosine distance: d = 1 - cosine_similarity.
Vectors produced are L2-normalized prior to storage.
"""

# ============================================================================
# RETRIEVAL PARAMETERS (hybrid search + RRF fusion)
# ============================================================================

DENSE_TOP_K = 20
"""Top-K results from dense (semantic) search before RRF fusion."""

SPARSE_TOP_K = 20
"""Top-K results from sparse (BM25) search before RRF fusion."""

RRF_K = 60
"""RRF parameter; controls rank weighting in reciprocal rank fusion."""

FINAL_TOP_N = 5
"""Final number of ranked results returned to extraction/reasoning pipeline."""

# ============================================================================
# SECTION BOOST WEIGHTS (retrieval result re-ranking)
# ============================================================================

SECTION_BOOST_WEIGHTS = {
    # Keys are matched as case-insensitive substrings against chunk.section.
    # Real section labels from EDGAR look like: "Item 7. Management's Discussion..."
    # so we use the Item-number prefix as the key.
    "Item 7":  1.3,   # MD&A: most promise-rich section
    "Item 1.": 1.2,   # Business section: strategy / segment descriptions
    "Item 1A": 0.7,   # Risk Factors: informative but not promise-bearing
    "Item 8":  0.7,   # Financial Statements: numbers, not forward commitments
}
"""Multipliers applied to relevance scores based on section source.

Sections in Future Outlook / MD&A are more likely to contain commitments.
Sections outside this dict fall back to DEFAULT_SECTION_BOOST.
"""

DEFAULT_SECTION_BOOST = 1.0
"""Default boost weight for sections not explicitly listed above."""

# ============================================================================
# SIMILARITY THRESHOLDS (evidence classification)
# ============================================================================

SIMILARITY_STRONG_EVIDENCE = 0.7
"""Cosine similarity >= this threshold indicates strong evidence for delivery."""

SIMILARITY_WEAK_EVIDENCE = 0.5
"""Cosine similarity in range [WEAK, STRONG) indicates weak or partial evidence.

Cosine similarity < this threshold indicates absent evidence, candidate for Silently Abandoned.
In cosine distance space, similarity = 1.0 - distance for L2-normalized vectors.
"""

# ============================================================================
# SECTION ROLES (where promises are extracted vs where evidence is searched)
# ============================================================================

PROMISE_EXTRACTION_SECTIONS = [
    "Item 7",          # MD&A body (matches "Item 7. Management's Discussion...")
    "Item 1.",         # Business section: strategy / segment descriptions
    "MD&A",            # direct label fallback
    "management's discussion",  # plain-text variant
    "outlook",         # future-outlook subsections
    "strategy",        # strategy subsections
    "growth",          # growth/expansion subsections
]
"""Keyword patterns for substring matching in section names (case-insensitive).

Deliberately excludes 'management' as a bare word (which matches Item 12
'Security Ownership of Certain Beneficial Owners and Management...').
Also excludes 'discussion' alone (too broad).

Chunks that match these inclusion filters are kept. The separate exclusion list
in extractor._EXCLUDED_SECTION_KEYWORDS is applied only to cue-phrase fallback
candidates (chunks outside the primary sections) to prevent non-commitment sections
like exhibits or legal proceedings from entering the fallback set.
"""

EVIDENCE_SEARCH_SECTIONS = ["Item 7", "Item 1.", "Item 8"]
"""Canonical sections examined for evidence of promise delivery.

Consistent with real EDGAR Item headings: Item 7 (MD&A), Item 1 (Business),
and Item 8 (Financial Statements).
"""

# ============================================================================
# RECENCY WEIGHTS FOR CREDIBILITY SCORING
# ============================================================================


def recency_weight(year: int, most_recent_year: int) -> float:
    """Returns recency weight for a given fiscal year relative to the most recent year analyzed.

    Most recent year = 1.0, each year back decreases by 0.1, floor at 0.6.
    Matches documented mapping:
        - most_recent = 1.0
        - -1 year = 0.9
        - -2 years = 0.8
        - -3 years = 0.7
        - -4+ years = 0.6 (clamped)

    Args:
        year: Fiscal year of the promise or evidence.
        most_recent_year: The most recent year in the analysis set.

    Returns:
        float: Recency weight in range [0.6, 1.0].
    """
    offset = most_recent_year - year
    weight = 1.0 - (offset * 0.1)
    return max(weight, 0.6)


# ============================================================================
# CONFIDENCE SCORE SIGNAL WEIGHTS (rule-based promise extraction)
# ============================================================================

CONFIDENCE_SIGNAL_WEIGHTS = {
    "has_deadline": 0.3,
    "has_metric": 0.3,
    "has_commitment_verb": 0.2,
    "domain_detectable": 0.2,
}
"""Weights for signals used to compute confidence_score on extracted promises.

Sum of all weights equals 1.0.
Used by scorer.py to assign confidence_score (0.0 to 1.0) before LLM judge.
"""

# ============================================================================
# JUDGE PROMPT ANCHORS (LLM Call 2 scoring scale)
# ============================================================================

JUDGE_ANCHORS = """90-100: Explicit delivery with matching metrics
70-89:  Confirmed progress, metrics incomplete
50-69:  Topic mentioned, no clear confirmation
30-49:  Contradicted or shortfall shown
0-29:   No evidence or explicit failure""".strip()
"""Scoring rubric for Groq judge (reasoner.py, LLM Call 2).

Embedded directly into judge prompts to ground delivery_score (0-100).
"""

# ============================================================================
# LLM MODEL IDENTIFIERS
# ============================================================================

EXTRACTOR_MODEL = os.getenv("EXTRACTOR_MODEL", "gemini-3.5-flash-lite")
"""Google Generative AI model for LLM Call 1 (promise extraction). 15 RPM, 500 RPD."""

EXTRACTOR_FALLBACK_MODEL = os.getenv("EXTRACTOR_FALLBACK_MODEL", "gemini-3.5")
"""Fallback Gemini model if primary model experiences 503 high-demand spikes or deprecation."""

JUDGE_MODEL = os.getenv("GROQ_MODEL", os.getenv("JUDGE_MODEL", "openai/gpt-oss-20b"))
"""Groq model identifier for LLM Call 2 (cross-year reasoning and scoring)."""

GROQ_MODEL = JUDGE_MODEL
"""Alias for JUDGE_MODEL to support GROQ_MODEL environment variable or direct config imports."""

MAX_JUDGE_EVIDENCE_CHUNKS = int(os.getenv("MAX_JUDGE_EVIDENCE_CHUNKS", "3"))
"""Maximum total evidence chunks sent to Groq judge."""

MAX_JUDGE_CHUNK_WORDS = int(os.getenv("MAX_JUDGE_CHUNK_WORDS", "300"))
"""Maximum words per evidence chunk passed to Groq judge prompt."""

# ============================================================================
# RATE LIMITING AND PROVIDER QUOTAS
# ============================================================================

SEC_MIN_REQUEST_INTERVAL = 0.11
"""Minimum seconds between SEC EDGAR requests (stay well under 10 req/sec limit)."""

# Provider quotas: verified on 2026-09-19 from Google AI Studio and Groq Console docs.
# Configurable via environment variables.

# 1. Google Gemini Generation (Promise Extractor)
# Verified from user Google AI Studio table: Gemini 3.5 Flash Lite is 15 RPM, 500 RPD.
GEMINI_GENERATE_RPM = int(os.getenv("GEMINI_GENERATE_RPM", "15"))
GEMINI_GENERATE_RPD = int(os.getenv("GEMINI_GENERATE_RPD", "500"))


# 3. Groq (Cross-Year Judge: openai/gpt-oss-20b: 30 RPM, 1000 RPD, 8K TPM)
GROQ_RPM = int(os.getenv("GROQ_RPM", "30"))
GROQ_RPD = int(os.getenv("GROQ_RPD", "1000"))

GROQ_REQUEST_INTERVAL = 3.0
"""Minimum interval in seconds between Groq requests for pacing."""

# Circuit breaker auto-reset cooldown
CIRCUIT_BREAKER_COOLDOWN_SECONDS = float(os.getenv("CIRCUIT_BREAKER_COOLDOWN_SECONDS", "1800"))
"""Circuit breaker auto-reset cooldown in seconds (default: 1800s / 30 minutes)."""

# Retry configuration for APIs
MAX_RETRIES = 3
"""Maximum number of retries for transient API errors."""

RETRY_BASE_DELAY = 1.0
"""Base delay in seconds for exponential backoff retry."""

RETRY_MAX_DELAY = 60.0
"""Maximum delay in seconds for exponential backoff retry."""

# ============================================================================
# VALIDATION AT IMPORT TIME
# ============================================================================

# Note: We do NOT raise errors for missing API keys or SEC_USER_AGENT here.
# - LLM keys may be provided directly via Streamlit UI at runtime (app.py handles it).
# - SEC_USER_AGENT fallback is logged in ingestion.py if empty.
# - This keeps config.py pure and prevents app startup failures for missing credentials.


if __name__ == "__main__":
    print("=== TempusRAG Configuration ===\n")

    print(f"Year Convention: {YEAR_CONVENTION}")
    print(f"Project Root: {PROJECT_ROOT}")

    print("\nDirectories:")
    print(f"  CHROMADB_PATH: {CHROMADB_PATH}")
    print(f"  DATA_DIR: {DATA_DIR}")
    print(f"  LOG_DIR: {LOG_DIR}")

    print("\nChunking:")
    print(f"  CHUNK_MAX_TOKENS: {CHUNK_MAX_TOKENS}")
    print(f"  CHUNK_OVERLAP_TOKENS: {CHUNK_OVERLAP_TOKENS}")

    print("\nEmbeddings:")
    print(f"  EMBEDDING_MODEL_NAME: {EMBEDDING_MODEL_NAME}")
    print(f"  EMBEDDING_DIMENSIONS: {EMBEDDING_DIMENSIONS}")
    print(f"  EMBEDDING_DISTANCE_SPACE: {EMBEDDING_DISTANCE_SPACE}")

    print("\nRetrieval:")
    print(f"  DENSE_TOP_K: {DENSE_TOP_K}")
    print(f"  SPARSE_TOP_K: {SPARSE_TOP_K}")
    print(f"  RRF_K: {RRF_K}")
    print(f"  FINAL_TOP_N: {FINAL_TOP_N}")

    print("\nSection Boost Weights:")
    for section, weight in SECTION_BOOST_WEIGHTS.items():
        print(f"  {section}: {weight}")
    print(f"  DEFAULT: {DEFAULT_SECTION_BOOST}")

    print("\nSimilarity Thresholds:")
    print(f"  STRONG_EVIDENCE: {SIMILARITY_STRONG_EVIDENCE}")
    print(f"  WEAK_EVIDENCE: {SIMILARITY_WEAK_EVIDENCE}")

    print("\nSections:")
    print(f"  PROMISE_EXTRACTION: {PROMISE_EXTRACTION_SECTIONS}")
    print(f"  EVIDENCE_SEARCH: {EVIDENCE_SEARCH_SECTIONS}")

    print("\nRecency Weights (example):")
    print(f"  recency_weight(2024, 2024): {recency_weight(2024, 2024):.1f}")
    print(f"  recency_weight(2023, 2024): {recency_weight(2023, 2024):.1f}")
    print(f"  recency_weight(2022, 2024): {recency_weight(2022, 2024):.1f}")
    print(f"  recency_weight(2021, 2024): {recency_weight(2021, 2024):.1f}")
    print(f"  recency_weight(2020, 2024): {recency_weight(2020, 2024):.1f}")

    print("\nConfidence Weights:")
    for signal, weight in CONFIDENCE_SIGNAL_WEIGHTS.items():
        print(f"  {signal}: {weight}")

    print("\nLLM Models:")
    print(f"  EXTRACTOR_MODEL:          {EXTRACTOR_MODEL}")
    print(f"  EXTRACTOR_FALLBACK_MODEL: {EXTRACTOR_FALLBACK_MODEL}")
    print(f"  JUDGE_MODEL:              {JUDGE_MODEL}")
    print(f"  MAX_JUDGE_CHUNKS:         {MAX_JUDGE_EVIDENCE_CHUNKS}")
    print(f"  MAX_JUDGE_CHUNK_WORDS:    {MAX_JUDGE_CHUNK_WORDS}")

    print("\nProvider Rate Limits:")
    print(f"  Gemini Generate: {GEMINI_GENERATE_RPM} RPM, {GEMINI_GENERATE_RPD} RPD")

    print(f"  Groq:            {GROQ_RPM} RPM, {GROQ_RPD} RPD")
    print(f"  Breaker Cooldown:{CIRCUIT_BREAKER_COOLDOWN_SECONDS}s")
    print(f"  SEC Interval:    {SEC_MIN_REQUEST_INTERVAL}s")

    print("\nConfig loaded successfully!")
