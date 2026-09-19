"""
TempusRAG Pipeline Orchestrator

End-to-end pipeline that chains all components together:
ingestion -> chunking -> embedding -> extraction -> reasoning -> scoring

Given a ticker, produces a CompanyCredibilityReport from SEC filings.
"""

import logging
import time
from typing import Callable, Dict, List, Optional

from src.ingestion import load_ticker_cik_map, ingest_company
from src.chunker import chunk_year_sections
from src.embedder import embed_and_store_chunks, get_chroma_collection
from src.extractor import extract_all_promises
from src.reasoner import reason_across_years
from src.scorer import generate_credibility_report
from src.config import DATA_DIR, GEMINI_API_KEY, GROQ_API_KEY
from src.models import CompanyCredibilityReport, Chunk, Promise, DeliveryEvidence

logger = logging.getLogger(__name__)

# Last-run artifacts for the dashboard (not part of the report schema).
_LAST_ARTIFACTS: Dict[str, dict] = {}


def get_last_artifacts(ticker: str) -> Optional[dict]:
    """Return delivery evidence and promises from the most recent pipeline run."""
    return _LAST_ARTIFACTS.get(ticker.upper())


def validate_ticker(ticker: str, cik_map: dict) -> str:
    """
    Validate ticker exists in CIK map and return normalized ticker.
    
    Args:
        ticker: Company ticker symbol
        cik_map: Mapping from ticker to CIK
        
    Returns:
        Normalized ticker (uppercase)
        
    Raises:
        ValueError: If ticker not found in CIK map
    """
    ticker_upper = ticker.upper()
    if ticker_upper not in cik_map:
        available_tickers = sorted(list(cik_map.keys()))[:10]  # Show first 10
        raise ValueError(
            f"Ticker '{ticker}' not found in CIK map. "
            f"Available tickers include: {available_tickers}..."
        )
    return ticker_upper


def run_tempusrag_pipeline(
    ticker: str,
    company_display_name: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    groq_api_key: Optional[str] = None,
    force_reingest: bool = False,
    on_stage: Optional[Callable[[str], None]] = None,
) -> CompanyCredibilityReport:
    """
    Run complete TempusRAG pipeline for a company ticker.
    
    Args:
        ticker: Company ticker (e.g. "HIG")
        company_display_name: Human-readable name (defaults to ticker if None)
        gemini_api_key: Optional override for Gemini key (uses config if None)
        groq_api_key: Optional override for Groq key (uses config if None)
        force_reingest: If True, re-download and re-embed filings even if cached
        on_stage: Optional callback invoked with the current stage label
        
    Returns:
        CompanyCredibilityReport with analysis results
        
    Raises:
        ValueError: If ticker not found or validation fails
        Exception: If critical pipeline steps fail
    """
    start_time = time.time()
    
    # Use provided display name or default to ticker
    if company_display_name is None:
        company_display_name = ticker.upper()
    
    # Use provided API keys or fall back to config
    gemini_key = gemini_api_key or GEMINI_API_KEY
    groq_key = groq_api_key or GROQ_API_KEY
    
    logger.info(f"Starting TempusRAG pipeline for {ticker} ({company_display_name})")

    def stage(label: str) -> None:
        logger.info(label)
        if on_stage:
            on_stage(label)

    try:
        stage("Validate")
        cik_map = load_ticker_cik_map()
        validated_ticker = validate_ticker(ticker, cik_map)
        logger.info(f"Validated ticker: {validated_ticker}")

        stage("Ingest")
        try:
            year_sections = ingest_company(validated_ticker, cik_map, DATA_DIR)
            if not year_sections:
                raise ValueError(f"No filings found for ticker {validated_ticker}")
            logger.info(f"Ingested {len(year_sections)} years of filings")
        except Exception as e:
            logger.error(f"Ingestion failed for {validated_ticker}: {e}")
            raise Exception(f"Failed to ingest filings for {validated_ticker}: {e}")
        
        stage("Chunk")
        all_chunks: dict[int, list] = {}
        total_chunks = 0
        try:
            for year, sections_dict in year_sections.items():
                chunks = chunk_year_sections(
                    sections_dict,
                    company_ticker=validated_ticker,
                    company_display_name=company_display_name,
                    year=year
                )
                all_chunks[year] = chunks
                total_chunks += len(chunks)
                logger.info(f"Chunked {year}: {len(chunks)} chunks")
            logger.info(f"Total chunks across all years: {total_chunks}")
        except Exception as e:
            logger.error(f"Chunking failed: {e}")
            raise Exception(f"[Chunk] Chunking failed: cannot proceed without chunks: {e}") from e
        
        stage("Embed")
        embed_err: Optional[Exception] = None
        for year, chunks in all_chunks.items():
            if chunks:
                years_covered = [year]
                try:
                    embed_and_store_chunks(chunks, validated_ticker, years_covered, api_key=gemini_key)
                    logger.info(f"Embedded and stored {len(chunks)} chunks for {year}")
                except Exception as e:
                    logger.error(f"Embedding failed for {validated_ticker} {year}: {e}")
                    embed_err = e

        # Ensure collection exists and has chunks before proceeding to downstream extraction and judging
        try:
            coll = get_chroma_collection(validated_ticker)
            collection_count = coll.count()
        except Exception as ce:
            collection_count = 0
            if embed_err is None:
                embed_err = ce

        if collection_count == 0:
            err_msg = (
                f"[Embed] No chunks stored in ChromaDB for {validated_ticker}. "
                f"Embedding failed and no cached vectors exist. "
                f"Cannot proceed with downstream evaluation on an empty knowledge base."
            )
            if embed_err:
                raise RuntimeError(f"{err_msg} Root error: {embed_err}") from embed_err
            raise RuntimeError(err_msg)

        logger.info(f"ChromaDB collection {validated_ticker} verified with {collection_count} chunks")
        
        stage("Extract")
        all_promises: dict[int, list] = {}
        total_promises = 0
        try:
            all_promises = extract_all_promises(
                all_chunks,
                api_key=gemini_key
            )
            for year, promises in all_promises.items():
                total_promises += len(promises)
                logger.info(f"Extracted {len(promises)} promises from {year}")
            logger.info(f"Total promises extracted: {total_promises}")
        except Exception as e:
            logger.error(f"Promise extraction failed: {e}")
            raise Exception(f"[Extract] Promise extraction failed: {e}") from e
        
        stage("Judge")
        delivery_evidence = []
        try:
            if all_promises:
                delivery_evidence = reason_across_years(
                    all_promises, 
                    validated_ticker, 
                    api_key=groq_key
                )
                logger.info(f"Generated {len(delivery_evidence)} delivery judgments")
            else:
                logger.info("No promises to judge - skipping reasoning step")
        except Exception as e:
            logger.error(f"Cross-year reasoning failed: {e}")
            logger.warning("Continuing without delivery evidence - will use confidence-only scoring")
            delivery_evidence = []
        
        stage("Score")
        try:
            report = generate_credibility_report(
                company_ticker=validated_ticker,
                company_display_name=company_display_name,
                promises_by_year=all_promises,
                delivery_evidence=delivery_evidence
            )
            logger.info("Successfully generated credibility report")
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            raise Exception(f"Failed to generate credibility report: {e}")
        
        stage("Done")
        elapsed_time = time.time() - start_time
        logger.info(f"Pipeline completed successfully in {elapsed_time:.2f} seconds")
        logger.info(f"Final report: {report.total_promises} promises, "
                   f"overall score {report.overall_score:.1f}/100.0")

        _LAST_ARTIFACTS[validated_ticker] = {
            "delivery_evidence": delivery_evidence,
            "promises_by_year": all_promises,
            "elapsed_seconds": elapsed_time,
        }

        return report
        
    except ValueError:
        # Re-raise validation errors as-is
        raise
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Pipeline failed after {elapsed_time:.2f} seconds: {e}")
        raise


if __name__ == "__main__":
    import os
    from unittest.mock import patch, MagicMock

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    print("Running pipeline.py offline verification tests...")

    # Test 1: validate_ticker
    mock_cik = {"HIG": "0000874766", "NVDA": "0001045810"}
    assert validate_ticker("hig", mock_cik) == "HIG"
    try:
        validate_ticker("UNKNOWN_TICKER", mock_cik)
        assert False, "Expected ValueError on unknown ticker"
    except ValueError as e:
        assert "not found in CIK map" in str(e)
    print("  [Pass] validate_ticker handles valid and invalid tickers correctly")

    import sys

    # Test 2: Pipeline stops if ChromaDB collection is empty after embedding failure
    curr_mod = sys.modules[__name__]
    empty_coll = MagicMock()
    empty_coll.count.return_value = 0

    with patch.object(curr_mod, "load_ticker_cik_map", return_value={"HIG": "0000874766"}), \
         patch.object(curr_mod, "ingest_company", return_value={2021: {"Item 7 MD&A": "text"}}), \
         patch.object(curr_mod, "chunk_year_sections", return_value=[MagicMock(spec=Chunk)]), \
         patch.object(curr_mod, "embed_and_store_chunks", side_effect=RuntimeError("Simulated 429 Quota Exceeded")), \
         patch.object(curr_mod, "get_chroma_collection", return_value=empty_coll):

        try:
            run_tempusrag_pipeline(ticker="HIG")
            assert False, "Pipeline should have raised RuntimeError on empty collection"
        except RuntimeError as err:
            assert "No chunks stored in ChromaDB" in str(err)
            print("  [Pass] Pipeline aborts gracefully when embedding fails with empty ChromaDB")

    # Test 3: Pipeline completes end-to-end when stages succeed
    mock_report = MagicMock(spec=CompanyCredibilityReport)
    mock_report.total_promises = 5
    mock_report.overall_score = 88.0

    valid_coll = MagicMock()
    valid_coll.count.return_value = 10

    with patch.object(curr_mod, "load_ticker_cik_map", return_value={"HIG": "0000874766"}), \
         patch.object(curr_mod, "ingest_company", return_value={2021: {"Item 7 MD&A": "text"}}), \
         patch.object(curr_mod, "chunk_year_sections", return_value=[MagicMock(spec=Chunk)]), \
         patch.object(curr_mod, "embed_and_store_chunks", return_value=None), \
         patch.object(curr_mod, "get_chroma_collection", return_value=valid_coll), \
         patch.object(curr_mod, "extract_all_promises", return_value={2021: [MagicMock(spec=Promise)]}), \
         patch.object(curr_mod, "reason_across_years", return_value=[MagicMock(spec=DeliveryEvidence)]), \
         patch.object(curr_mod, "generate_credibility_report", return_value=mock_report):

        rep = run_tempusrag_pipeline(ticker="HIG")
        assert rep.total_promises == 5
        assert rep.overall_score == 88.0
        assert get_last_artifacts("HIG") is not None
        print("  [Pass] Pipeline runs end-to-end and stores last artifacts")

    print("All pipeline.py verification tests passed!")