"""Hybrid search + RRF fusion + section boosting + lost-in-middle reordering.

Combines dense (ChromaDB) and sparse (BM25) retrieval into a single ranked list of chunks.
Applies section-based score boosting and lost-in-middle reordering for optimal LLM performance.
"""

import logging
import numpy as np

from src.config import (
    DENSE_TOP_K,
    SPARSE_TOP_K,
    RRF_K,
    FINAL_TOP_N,
    SECTION_BOOST_WEIGHTS,
    DEFAULT_SECTION_BOOST
)
from src.embedder import get_chroma_collection, get_bm25_index, tokenize_for_bm25, _get_embedding_model
from src.models import Chunk

logger = logging.getLogger(__name__)


def dense_search(query: str, ticker: str, top_k: int = DENSE_TOP_K, year: int | None = None) -> list[tuple[str, float]]:
    """
    Perform dense semantic search using ChromaDB.
    
    Args:
        query: Search query string.
        ticker: Company ticker symbol.
        top_k: Number of results to return.
        year: Optional year filter: only return chunks from this fiscal year.
        
    Returns:
        list[tuple[str, float]]: List of (chunk_id, score) tuples, ranked best-first.
    """
    try:
        # Get ChromaDB collection
        collection = get_chroma_collection(ticker)
        
        # Embed the query using local model
        model = _get_embedding_model()
        query_embedding = model.encode_query(query)

        # Build optional year filter (Chroma 1.x where-clause style)
        where_filter = {"year": {"$eq": int(year)}} if year is not None else None
        
        # Query ChromaDB
        query_kwargs: dict = dict(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["distances", "metadatas", "documents"],
        )
        if where_filter:
            query_kwargs["where"] = where_filter

        results = collection.query(**query_kwargs)
        
        # Extract results - ChromaDB returns cosine distance (dist = 1 - cosine_similarity)
        if not results["ids"] or not results["ids"][0]:
            return []

        chunk_ids = results["ids"][0]
        distances = results["distances"][0]
        
        # For unit-normalized vectors with cosine distance, similarity = 1.0 - dist
        similarities = [max(-1.0, min(1.0, 1.0 - float(dist))) for dist in distances]
        
        ranked_results = list(zip(chunk_ids, similarities))
        ranked_results.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Dense search for '{query}' in {ticker}: {len(ranked_results)} results")
        return ranked_results
        
    except Exception as e:
        logger.error(f"Dense search failed for {ticker}: {e}")
        return []


def sparse_search(query: str, ticker: str, top_k: int = SPARSE_TOP_K, year: int | None = None) -> list[tuple[str, float]]:
    """
    Perform sparse keyword search using BM25.
    
    Args:
        query: Search query string.
        ticker: Company ticker symbol.
        top_k: Number of results to return.
        year: Optional year filter: only return chunks from this fiscal year.
        
    Returns:
        list[tuple[str, float]]: List of (chunk_id, score) tuples, ranked best-first.
    """
    # Get BM25 index
    bm25_entry = get_bm25_index(ticker)
    if bm25_entry is None:
        logger.error(f"No BM25 index found for {ticker}, falling back to dense-only search")
        return []
    
    bm25_index = bm25_entry["bm25"]
    chunks = bm25_entry["chunks"]
    
    # Tokenize query
    query_tokens = tokenize_for_bm25(query)
    if not query_tokens:
        logger.warning(f"Query '{query}' tokenized to empty list")
        return []
    
    # Get BM25 scores
    scores = bm25_index.get_scores(query_tokens)
    
    # Get top candidate indices
    top_indices = np.argsort(scores)[::-1]
    
    # Map indices to chunk_ids and scores with optional year filter
    results = []
    for idx in top_indices:
        if idx < len(chunks) and scores[idx] > 0:
            chunk = chunks[idx]
            if year is not None and chunk.year != int(year):
                continue
            results.append((chunk.chunk_id, float(scores[idx])))
            if len(results) >= top_k:
                break
    
    logger.debug(f"Sparse search for '{query}' in {ticker}: {len(results)} results")
    return results


def reciprocal_rank_fusion(
    dense_results: list[tuple[str, float]], 
    sparse_results: list[tuple[str, float]], 
    k: int = RRF_K,
    sparse_weight: float = 1.0
) -> list[tuple[str, float]]:
    """
    Fuse dense and sparse results using Weighted Reciprocal Rank Fusion.
    
    Args:
        dense_results: Results from dense search (chunk_id, score).
        sparse_results: Results from sparse search (chunk_id, score).
        k: RRF parameter (tuned to 30 for financial 10-Ks).
        sparse_weight: Multiplier for sparse BM25 hits (1.5).
        
    Returns:
        list[tuple[str, float]]: Fused results (chunk_id, fused_score), ranked best-first.
    """
    # Build rank mappings (1-indexed)
    dense_ranks = {chunk_id: rank + 1 for rank, (chunk_id, _) in enumerate(dense_results)}
    sparse_ranks = {chunk_id: rank + 1 for rank, (chunk_id, _) in enumerate(sparse_results)}
    
    # Get all unique chunk_ids
    all_chunk_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
    
    # Compute fused scores
    fused_results = []
    for chunk_id in all_chunk_ids:
        dense_rank = dense_ranks.get(chunk_id, float('inf'))
        sparse_rank = sparse_ranks.get(chunk_id, float('inf'))
        
        # Weighted RRF formula: dense_score + sparse_weight * sparse_score
        dense_score = 1.0 / (k + dense_rank) if dense_rank != float('inf') else 0.0
        sparse_score = (1.0 / (k + sparse_rank)) * sparse_weight if sparse_rank != float('inf') else 0.0
        
        fused_score = dense_score + sparse_score
        fused_results.append((chunk_id, fused_score))
    
    # Sort by fused score (descending)
    fused_results.sort(key=lambda x: x[1], reverse=True)
    
    logger.debug(f"RRF fusion: {len(dense_results)} dense + {len(sparse_results)} sparse -> {len(fused_results)} fused")
    return fused_results


def apply_section_boost(
    ranked_chunk_ids: list[tuple[str, float]], 
    chunk_lookup: dict[str, Chunk]
) -> list[tuple[str, float]]:
    """
    Apply section-based score boosting.
    
    Keys in SECTION_BOOST_WEIGHTS are matched as case-insensitive substrings
    against chunk.section. The highest-matching boost is used if multiple keys
    match the same section label.
    
    Args:
        ranked_chunk_ids: List of (chunk_id, score) tuples.
        chunk_lookup: Mapping from chunk_id to Chunk objects.
        
    Returns:
        list[tuple[str, float]]: Boosted results (chunk_id, boosted_score), ranked best-first.
    """
    boosted_results = []
    
    for chunk_id, score in ranked_chunk_ids:
        if chunk_id not in chunk_lookup:
            logger.warning(f"Chunk ID {chunk_id} not found in chunk lookup, skipping")
            continue
        
        chunk = chunk_lookup[chunk_id]
        section_lower = chunk.section.lower()
        
        # Substring match: find the highest-priority boost key that appears in the section label
        boost_weight = DEFAULT_SECTION_BOOST
        for key, weight in SECTION_BOOST_WEIGHTS.items():
            if key.lower() in section_lower:
                if weight > boost_weight or boost_weight == DEFAULT_SECTION_BOOST:
                    boost_weight = weight
        
        boosted_score = score * boost_weight
        boosted_results.append((chunk_id, boosted_score))
        logger.debug(
            f"Boosted {chunk_id} (section: {chunk.section}): "
            f"{score:.4f} -> {boosted_score:.4f} (boost: {boost_weight})"
        )
    
    # Re-sort by boosted score
    boosted_results.sort(key=lambda x: x[1], reverse=True)
    
    return boosted_results


def lost_in_middle_reorder(ranked_chunk_ids: list[str]) -> list[str]:
    """
    Reorder chunks using lost-in-middle strategy.
    
    Places most relevant chunks at beginning and end, with less relevant in the middle.
    Algorithm: rank 1 -> position 1, rank 2 -> last position, rank 3+ fill middle sequentially.
    
    Args:
        ranked_chunk_ids: List of chunk IDs in rank order.
        
    Returns:
        list[str]: Reordered chunk IDs optimized for LLM performance.
    """
    n = len(ranked_chunk_ids)
    
    if n == 0:
        return []
    elif n == 1:
        return ranked_chunk_ids.copy()
    elif n == 2:
        return ranked_chunk_ids.copy()  # rank 1 at pos 0, rank 2 at pos 1
    
    # Canonical lost-in-the-middle: alternate placing highest remaining items at the ends,
    # leaving the least relevant items in the center.
    reordered = [None] * n
    left, right = 0, n - 1
    for i, chunk_id in enumerate(ranked_chunk_ids):
        if i % 2 == 0:
            reordered[left] = chunk_id
            left += 1
        else:
            reordered[right] = chunk_id
            right -= 1
    
    logger.debug(f"Lost-in-middle reordering: {n} chunks reordered")
    return reordered


def build_chunk_lookup(ticker: str) -> dict[str, Chunk]:
    """
    Build mapping from chunk_id to Chunk objects for a ticker.
    
    Args:
        ticker: Company ticker symbol.
        
    Returns:
        dict[str, Chunk]: Mapping from chunk_id to Chunk objects.
    """
    bm25_entry = get_bm25_index(ticker)
    if bm25_entry is None:
        logger.error(f"No BM25 index found for {ticker}, cannot build chunk lookup")
        return {}
    
    if "chunk_lookup" not in bm25_entry:
        bm25_entry["chunk_lookup"] = {chunk.chunk_id: chunk for chunk in bm25_entry["chunks"]}
        logger.debug(f"Built and cached chunk lookup for {ticker}: {len(bm25_entry['chunk_lookup'])} chunks")
    
    return bm25_entry["chunk_lookup"]


def hybrid_retrieve(
    query: str, 
    ticker: str, 
    top_n: int = FINAL_TOP_N, 
    year: int | None = None,
    reorder: bool = True,
) -> list[Chunk]:
    """
    Perform hybrid retrieval with RRF fusion, boosting, and reordering.
    
    Args:
        query: Search query string.
        ticker: Company ticker symbol.
        top_n: Number of final results to return.
        year: Optional year filter: restricts retrieval to chunks from this fiscal year.
        reorder: If True, apply lost-in-the-middle reordering for LLM context windows.
                 If False, preserve strict rank order (best for IR evaluation).
        
    Returns:
        list[Chunk]: Final ranked chunks, length <= top_n.
    """
    logger.info(f"Hybrid retrieval for '{query}' in {ticker}, requesting top {top_n}"
                + (f", year={year}" if year else ""))
    
    # Build chunk lookup first (needed for boosting and final mapping)
    chunk_lookup = build_chunk_lookup(ticker)
    if not chunk_lookup:
        logger.warning(f"No chunks available for {ticker}")
        return []
    
    # Dense and sparse search (both with optional year filter)
    candidate_k = max(top_n * 4, 10)
    dense_k = min(candidate_k, DENSE_TOP_K)
    sparse_k = min(candidate_k, SPARSE_TOP_K)
    dense_results = dense_search(query, ticker, top_k=dense_k, year=year)
    sparse_results = sparse_search(query, ticker, top_k=sparse_k, year=year)
    
    if not dense_results and not sparse_results:
        logger.warning(f"Both dense and sparse search returned no results for '{query}' in {ticker}")
        return []
    
    # Reciprocal Rank Fusion (tuned with k=30 and sparse_weight=1.5 for financial filings)
    fused_results = reciprocal_rank_fusion(dense_results, sparse_results, k=30, sparse_weight=1.5)
    
    if not fused_results:
        logger.warning(f"RRF fusion resulted in no results for '{query}' in {ticker}")
        return []
    
    # Apply section boosting
    boosted_results = apply_section_boost(fused_results, chunk_lookup)
    
    # Take top N by boosted score
    top_boosted = boosted_results[:top_n]
    top_chunk_ids = [chunk_id for chunk_id, _ in top_boosted]
    
    # Apply lost-in-middle reordering if requested
    reordered_chunk_ids = lost_in_middle_reorder(top_chunk_ids) if reorder else top_chunk_ids
    
    # Map back to Chunk objects
    final_chunks = []
    for chunk_id in reordered_chunk_ids:
        if chunk_id in chunk_lookup:
            final_chunks.append(chunk_lookup[chunk_id])
        else:
            logger.warning(f"Chunk ID {chunk_id} missing from lookup during final mapping")
    
    final_chunk_ids = [chunk.chunk_id for chunk in final_chunks]
    logger.info(f"Hybrid retrieval complete for '{query}' in {ticker}: {final_chunk_ids}")
    
    return final_chunks


if __name__ == "__main__":
    from src.embedder import _get_chroma_client, embed_and_store_chunks, _bm25_indices
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    
    print("=== Testing retriever.py offline verification tests ===")
    
    TEST_TICKER = "RTRV_TEST"
    client = _get_chroma_client()
    try:
        client.delete_collection(TEST_TICKER)
    except Exception:
        pass
    _bm25_indices.pop(TEST_TICKER, None)
    
    # Create sample chunks across two years
    test_chunks = [
        Chunk(
            text="We plan to expand our cloud computing footprint by 30% next year.",
            company=TEST_TICKER,
            year=2024,
            section="Item 7 MD&A",
            subsection="Cloud Strategy",
            chunk_id=f"{TEST_TICKER}_2024_01",
            word_count=12,
        ),
        Chunk(
            text="Risk factors include macroeconomic headwinds and semiconductor shortages.",
            company=TEST_TICKER,
            year=2024,
            section="Item 1A Risk Factors",
            subsection="full_text",
            chunk_id=f"{TEST_TICKER}_2024_02",
            word_count=9,
        ),
        Chunk(
            text="In 2025 our cloud computing footprint expanded by 32%, meeting our targets.",
            company=TEST_TICKER,
            year=2025,
            section="Item 7 MD&A",
            subsection="Operations Review",
            chunk_id=f"{TEST_TICKER}_2025_01",
            word_count=12,
        ),
    ]
    
    # Embed and index test chunks with all-MiniLM-L6-v2
    embed_and_store_chunks(test_chunks, ticker=TEST_TICKER, years_covered=[2024, 2025])
    
    # 1. Test dense search with cosine similarity (1.0 - dist)
    dense_res = dense_search("cloud computing expansion", ticker=TEST_TICKER, top_k=2)
    print(f"\n1. Dense search results: {dense_res}")
    assert len(dense_res) > 0
    best_id, best_score = dense_res[0]
    print(f"   Best match: {best_id} with score {best_score:.4f}")
    assert best_score > 0.4, f"Expected positive similarity score, got {best_score}"
    
    # 2. Test sparse search with year filter
    sparse_all = sparse_search("cloud computing", ticker=TEST_TICKER, top_k=5)
    sparse_2025 = sparse_search("cloud computing", ticker=TEST_TICKER, top_k=5, year=2025)
    print(f"\n2. Sparse search all years: {[cid for cid, _ in sparse_all]}")
    print(f"   Sparse search year 2025: {[cid for cid, _ in sparse_2025]}")
    assert all("_2025_" in cid for cid, _ in sparse_2025), "Year filter failed in sparse_search"
    
    # 3. Test RRF fusion
    fused = reciprocal_rank_fusion(dense_res, sparse_all)
    print(f"\n3. RRF fused results: {fused}")
    assert len(fused) > 0
    
    # 4. Test lost-in-middle reordering
    test_ids = ["A", "B", "C", "D", "E"]
    reordered = lost_in_middle_reorder(test_ids)
    print(f"\n4. Lost-in-middle: {test_ids} -> {reordered}")
    assert reordered[0] == "A" and reordered[-1] == "B", "Lost-in-middle ordering incorrect"
    
    # 5. Test full hybrid_retrieve
    retrieved = hybrid_retrieve("cloud computing expansion targets", ticker=TEST_TICKER, top_n=2, year=2025)
    print(f"\n5. Hybrid retrieved chunks (year=2025): {[c.chunk_id for c in retrieved]}")
    assert len(retrieved) > 0
    assert retrieved[0].year == 2025
    
    # Cleanup
    client.delete_collection(TEST_TICKER)
    _bm25_indices.pop(TEST_TICKER, None)
    print("\nAll retriever.py verification tests passed successfully!")