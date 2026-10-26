"""TempusRAG pipeline orchestrator.

Chains all components: ingest → chunk → embed → extract → judge → score.
Produces a CompanyCredibilityReport from SEC 10-K filings.
"""

import logging
import time
from typing import Callable, Optional

from src.config import DATA_DIR, GEMINI_API_KEY, GROQ_API_KEY
from src.chunker import chunk_year_sections
from src.embedder import embed_and_store_chunks, get_chroma_collection
from src.extractor import extract_all_promises
from src.ingestion import ingest_company, load_ticker_cik_map
from src.models import CompanyCredibilityReport, Chunk, DeliveryEvidence, Promise
from src.reasoner import reason_across_years
from src.scorer import generate_credibility_report

logger = logging.getLogger(__name__)

# Per-ticker artifacts from the most recent run (delivery evidence + timing)
_LAST_ARTIFACTS: dict[str, dict] = {}


def get_last_artifacts(ticker: str) -> dict | None:
    return _LAST_ARTIFACTS.get(ticker.upper())


def validate_ticker(ticker: str, cik_map: dict) -> str:
    """Return normalized ticker or raise ValueError if not in CIK map."""
    upper = ticker.upper()
    if upper not in cik_map:
        raise ValueError(f"Ticker '{ticker}' not found in SEC CIK map.")
    return upper


def run_tempusrag_pipeline(
    ticker: str,
    company_display_name: str | None = None,
    gemini_api_key: str | None = None,
    groq_api_key: str | None = None,
    force_reingest: bool = False,
    on_stage: Callable[[str], None] | None = None,
    cik_map: dict | None = None,
) -> CompanyCredibilityReport:
    """Run the full TempusRAG pipeline for a ticker.

    Args:
        ticker: Company ticker e.g. 'HIG'.
        company_display_name: Human-readable name (defaults to ticker).
        gemini_api_key: Override for Gemini key (uses config if None).
        groq_api_key: Override for Groq key (uses config if None).
        force_reingest: Re-download and re-embed even if cached.
        on_stage: Optional callback called with current stage label.
        cik_map: Pre-loaded CIK map (skips network call if provided).

    Returns:
        CompanyCredibilityReport

    Raises:
        ValueError: Ticker not found or validation fails.
        RuntimeError: Critical stage failure.
    """
    start = time.time()
    display_name = company_display_name or ticker.upper()
    gemini_key = gemini_api_key or GEMINI_API_KEY
    groq_key = groq_api_key or GROQ_API_KEY

    def stage(label: str) -> None:
        logger.info("Stage: %s", label)
        if on_stage:
            on_stage(label)

    # ── Validate ────────────────────────────────────────────────────────────
    stage("Validate")
    resolved_cik_map = cik_map if cik_map is not None else load_ticker_cik_map()
    validated = validate_ticker(ticker, resolved_cik_map)
    logger.info("Validated ticker: %s", validated)

    # ── Ingest ──────────────────────────────────────────────────────────────
    stage("Ingest")
    year_sections = ingest_company(validated, resolved_cik_map, DATA_DIR)
    if not year_sections:
        raise ValueError(f"No filings found for {validated}")
    logger.info("Ingested %d years", len(year_sections))

    # ── Chunk ───────────────────────────────────────────────────────────────
    stage("Chunk")
    all_chunks: dict[int, list[Chunk]] = {}
    total_chunks = 0
    for year, sections in year_sections.items():
        chunks = chunk_year_sections(
            sections,
            company_ticker=validated,
            company_display_name=display_name,
            year=year,
        )
        all_chunks[year] = chunks
        total_chunks += len(chunks)
        logger.info("Year %d: %d chunks", year, len(chunks))
    logger.info("Total chunks: %d", total_chunks)

    # ── Embed ───────────────────────────────────────────────────────────────
    stage("Embed")
    embed_error: Exception | None = None
    for year, chunks in all_chunks.items():
        if not chunks:
            continue
        try:
            # _rebuild_bm25=False: defer BM25 invalidation until all years are done
            embed_and_store_chunks(chunks, validated, [year], api_key=gemini_key, _rebuild_bm25=False)
            logger.info("Embedded %d chunks for %d", len(chunks), year)
        except Exception as e:
            logger.error("Embedding failed for %s %d: %s", validated, year, e)
            embed_error = e

    # Invalidate BM25 once after all years are embedded, then pre-warm
    from src.embedder import _bm25_indices, get_bm25_index
    _bm25_indices.pop(validated.upper(), None)
    get_bm25_index(validated)
    logger.info("BM25 index pre-warmed for %s", validated)

    try:
        collection_count = get_chroma_collection(validated).count()
    except Exception as e:
        collection_count = 0
        embed_error = embed_error or e

    if collection_count == 0:
        msg = (
            f"No chunks in ChromaDB for {validated}. "
            "Embedding failed and no cached vectors exist."
        )
        raise RuntimeError(f"{msg} Root cause: {embed_error}") from embed_error

    logger.info("ChromaDB collection %s: %d chunks", validated, collection_count)

    # ── Extract ─────────────────────────────────────────────────────────────
    stage("Extract")
    all_promises = extract_all_promises(all_chunks, api_key=gemini_key)
    total_promises = sum(len(p) for p in all_promises.values())
    logger.info("Total promises extracted: %d", total_promises)

    # ── Judge ───────────────────────────────────────────────────────────────
    stage("Judge")
    delivery_evidence: list[DeliveryEvidence] = []
    if all_promises:
        try:
            delivery_evidence = reason_across_years(
                all_promises, validated, api_key=groq_key
            )
            logger.info("Generated %d delivery judgments", len(delivery_evidence))
        except Exception as e:
            logger.error("Judging failed: %s — continuing with empty evidence", e)
    else:
        logger.info("No promises to judge")

    # ── Score ───────────────────────────────────────────────────────────────
    stage("Score")
    report = generate_credibility_report(
        company_ticker=validated,
        company_display_name=display_name,
        promises_by_year=all_promises,
        delivery_evidence=delivery_evidence,
    )

    # ── Done ────────────────────────────────────────────────────────────────
    stage("Done")
    elapsed = time.time() - start
    logger.info(
        "Pipeline complete: %s, %.1fs, %d promises, score %.1f",
        validated, elapsed, report.total_promises, report.overall_score,
    )

    _LAST_ARTIFACTS[validated] = {
        "delivery_evidence": delivery_evidence,
        "promises_by_year": all_promises,
        "elapsed_seconds": elapsed,
    }

    return report