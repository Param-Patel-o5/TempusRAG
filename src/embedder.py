"""Dense and sparse embeddings and ChromaDB storage for TempusRAG chunks.

Handles local sentence-transformers (all-MiniLM-L6-v2) encoding to dense vectors
stored in ChromaDB (persistent), and in-memory BM25 index construction for sparse
retrieval. Implements caching and staleness checks so each (ticker, year) is
embedded once and reused while fresh.

The pipeline calls embed_and_store_chunks once per fiscal year. Freshness is tracked
per fiscal year by inspecting ChromaDB metadata and existing chunks, and the BM25 index
is rebuilt lazily from the full collection so it always covers every embedded year.
"""

import hashlib
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import (
    CACHE_STALENESS_DAYS,
    CHROMADB_PATH,
    CHUNK_MAX_TOKENS,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_DISTANCE_SPACE,
    EMBEDDING_MODEL_NAME,
)
from src.models import Chunk

logger = logging.getLogger(__name__)

# Module-level state
_chroma_client: "chromadb.api.ClientAPI | None" = None  # type: ignore[assignment]
_sentence_transformer_model: SentenceTransformer | None = None
_bm25_indices: dict[str, dict] = {}

# Upsert batch size for ChromaDB writes
_UPSERT_BATCH_SIZE = 500

# Default batch size for local CPU sentence-transformers encoding
DEFAULT_EMBED_BATCH_SIZE = 64


def _get_chroma_client() -> "chromadb.api.ClientAPI":
    """Return the ChromaDB persistent client, initializing it on first call."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(CHROMADB_PATH))
        logger.info("ChromaDB client opened: %s", CHROMADB_PATH)
    return _chroma_client


def _expected_collection_metadata() -> dict[str, Any]:
    """Return canonical collection metadata configuration stamps."""
    return {
        "distance_space": EMBEDDING_DISTANCE_SPACE,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "chunk_max_tokens": CHUNK_MAX_TOKENS,
    }


def _get_sentence_transformer() -> SentenceTransformer:
    """Return cached SentenceTransformer model singleton."""
    global _sentence_transformer_model
    if _sentence_transformer_model is None:
        logger.info("Loading local embedding model: %s", EMBEDDING_MODEL_NAME)
        try:
            _sentence_transformer_model = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
        except Exception:
            _sentence_transformer_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _sentence_transformer_model


class SentenceTransformerEmbeddingWrapper:
    """Wrapper providing encode_documents, encode_query, and encode methods."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name

    def encode_documents(
        self,
        texts: list[str],
        batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    ) -> np.ndarray:
        """Encode document chunks locally with unit normalization."""
        if not texts:
            return np.empty((0, EMBEDDING_DIMENSIONS), dtype=np.float32)
        model = _get_sentence_transformer()
        vecs = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single search query locally with unit normalization."""
        model = _get_sentence_transformer()
        vec = model.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.astype(np.float32)

    def encode(
        self,
        texts: list[str] | str,
        show_progress_bar: bool = False,
        batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    ) -> np.ndarray:
        """SentenceTransformer-compatible encode method."""
        if isinstance(texts, str):
            return self.encode_query(texts)
        return self.encode_documents(texts, batch_size=batch_size)


_embedding_model: SentenceTransformerEmbeddingWrapper | None = None


def _get_embedding_model() -> SentenceTransformerEmbeddingWrapper:
    """Get the local embedding model wrapper singleton."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformerEmbeddingWrapper()
    return _embedding_model


def collection_exists_and_fresh(ticker: str) -> bool:
    """Check if a ChromaDB collection exists, is fresh, and matches current embedding config.

    Args:
        ticker: Company ticker symbol (e.g. "HIG").

    Returns:
        bool: True if collection exists, is fresh, and metadata matches config.
    """
    collection_name = ticker.upper()

    try:
        collection = _get_chroma_client().get_collection(name=collection_name)
        metadata = collection.metadata or {}

        # Validate configuration stamps: any missing stamp counts as a mismatch
        expected = _expected_collection_metadata()
        for key, expected_val in expected.items():
            if metadata.get(key) != expected_val:
                logger.info(
                    f"Collection {collection_name} configuration stamp mismatch on '{key}': "
                    f"found {metadata.get(key)!r}, expected {expected_val!r}. Will re-ingest."
                )
                return False

        if "ingested_at" not in metadata:
            logger.info(f"Collection {collection_name} exists but has no ingestion timestamp, will re-ingest")
            return False

        ingested_at_str = metadata["ingested_at"]
        try:
            ingested_at = datetime.fromisoformat(ingested_at_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            logger.warning(f"Invalid ingested_at timestamp for {collection_name}, will re-ingest")
            return False

        now = datetime.now(timezone.utc)
        age_days = (now - ingested_at).days

        if age_days <= CACHE_STALENESS_DAYS:
            logger.info(f"Collection {collection_name} is fresh (ingested {age_days} days ago)")
            return True
        else:
            logger.info(f"Collection {collection_name} is stale (ingested {age_days} days ago), will re-ingest")
            return False

    except Exception:
        logger.info(f"No collection found for {collection_name}")
        return False


def tokenize_for_bm25(text: str) -> list[str]:
    """Simple tokenizer for BM25 indexing.

    Strips punctuation, converts to lowercase, and splits on whitespace.

    Args:
        text: Input text to tokenize.

    Returns:
        list[str]: List of tokens.
    """
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = [token for token in cleaned.split() if token.strip()]
    return tokens


def _get_or_create_collection(collection_name: str):
    """Return the named collection, creating or rebuilding it if settings mismatch.

    Any missing stamp or mismatched configuration triggers an automatic rebuild.
    """
    client = _get_chroma_client()
    expected_meta = _expected_collection_metadata()
    create_meta = dict(expected_meta)
    create_meta["hnsw:space"] = EMBEDDING_DISTANCE_SPACE

    try:
        collection = client.get_collection(name=collection_name)
        meta = collection.metadata or {}

        # If any expected stamp is missing or different, treat as a mismatch
        mismatch = any(meta.get(k) != v for k, v in expected_meta.items())
        if "hnsw:space" in meta and meta["hnsw:space"] != EMBEDDING_DISTANCE_SPACE:
            mismatch = True

        if mismatch:
            logger.warning(
                f"Collection {collection_name} settings mismatch (existing: {meta}). "
                f"Dropping and rebuilding collection."
            )
            client.delete_collection(name=collection_name)
            collection = client.create_collection(name=collection_name, metadata=create_meta)
            logger.info(f"Recreated collection {collection_name} with configuration: {create_meta}")
        else:
            logger.info(f"Using existing collection {collection_name}")
    except Exception:
        collection = client.create_collection(name=collection_name, metadata=create_meta)
        logger.info(f"Created new collection {collection_name} with configuration: {create_meta}")

    return collection


def _year_present(collection, year: int) -> bool:
    """Return True if the collection already holds at least one chunk for this year."""
    try:
        result = collection.get(where={"year": year}, limit=1, include=[])
        return bool(result["ids"])
    except Exception as e:
        logger.warning(f"Could not check whether year {year} is present: {e}")
        return False


def _parse_years(raw) -> set[int]:
    """Parse a comma separated years string from collection metadata."""
    if not raw:
        return set()
    return {int(part) for part in str(raw).split(",") if part.strip().isdigit()}


def embed_and_store_chunks(
    chunks: list[Chunk],
    ticker: str,
    years_covered: list[int] | None = None,
    api_key: str | None = None,
) -> None:
    """Embed chunks and store in ChromaDB, one fiscal year at a time.

    Encodes chunks locally with all-MiniLM-L6-v2 without any external API calls,
    rate limits, or network overhead.

    Args:
        chunks: List of Chunk objects to embed and store.
        ticker: Company ticker symbol.
        years_covered: Kept for backward compatibility. Actual years are derived
            directly from chunks.
        api_key: Kept for signature compatibility (not needed for local model).
    """
    if not chunks:
        logger.warning(f"Empty chunks list provided for {ticker}, skipping embedding")
        return

    ticker_upper = ticker.upper()
    collection_name = ticker_upper
    years_in_batch = sorted({chunk.year for chunk in chunks})

    is_fresh = collection_exists_and_fresh(ticker)
    collection = _get_or_create_collection(collection_name)

    # A stale (or brand new) collection is treated as if no year is stored yet
    if is_fresh:
        missing_years = [y for y in years_in_batch if not _year_present(collection, y)]
    else:
        missing_years = list(years_in_batch)

    if not missing_years:
        logger.info(f"{collection_name} years {years_in_batch} already stored and fresh, skipping")
        return

    to_embed = [chunk for chunk in chunks if chunk.year in missing_years]
    logger.info(
        f"Embedding {len(to_embed)} chunks for {collection_name} years {missing_years} via {EMBEDDING_MODEL_NAME}"
    )
    start_time = time.time()

    # Step 1: Batch encode all chunk texts locally
    model = _get_embedding_model()
    chunk_texts = [chunk.text for chunk in to_embed]
    embeddings = model.encode_documents(chunk_texts, batch_size=DEFAULT_EMBED_BATCH_SIZE)

    assert len(embeddings) == len(to_embed), (
        f"Generated embedding count {len(embeddings)} does not match chunk count {len(to_embed)}"
    )

    # Step 2: Remove old chunks for these years
    for year in missing_years:
        collection.delete(where={"year": year})

    # Step 3: Upsert to ChromaDB in batches
    for start in range(0, len(to_embed), _UPSERT_BATCH_SIZE):
        batch = to_embed[start : start + _UPSERT_BATCH_SIZE]
        batch_embeddings = embeddings[start : start + _UPSERT_BATCH_SIZE]

        collection.upsert(
            ids=[chunk.chunk_id for chunk in batch],
            embeddings=batch_embeddings.tolist(),
            documents=[chunk.text for chunk in batch],
            metadatas=[
                {
                    "company": chunk.company,
                    "year": chunk.year,
                    "section": chunk.section,
                    "subsection": chunk.subsection,
                    "word_count": chunk.word_count,
                }
                for chunk in batch
            ],
        )

    # Step 4: Update collection metadata
    previous_years = _parse_years((collection.metadata or {}).get("years_covered")) if is_fresh else set()
    all_years = sorted(previous_years | set(missing_years))
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    updated_meta = _expected_collection_metadata()
    updated_meta["ingested_at"] = now_iso
    updated_meta["years_covered"] = ",".join(map(str, all_years))

    collection.modify(metadata=updated_meta)

    # Invalidate BM25 cache so it is rebuilt lazily from the full collection
    _bm25_indices.pop(ticker_upper, None)

    elapsed_time = time.time() - start_time
    logger.info(
        f"Embedded and stored {len(to_embed)} chunks for {ticker_upper} in {elapsed_time:.2f}s "
        f"(ChromaDB count: {collection.count()})"
    )


def get_bm25_index(ticker: str) -> dict | None:
    """Get the BM25 index for a ticker, rebuilding lazily from ChromaDB if needed.

    Args:
        ticker: Company ticker symbol.

    Returns:
        dict: BM25 index entry with 'bm25' and 'chunks' keys, or None if not found.
    """
    ticker_upper = ticker.upper()
    entry = _bm25_indices.get(ticker_upper)

    if entry is not None:
        return entry

    logger.info(f"BM25 index missing for {ticker_upper}, rebuilding from ChromaDB collection")

    try:
        collection = _get_chroma_client().get_collection(name=ticker_upper)
        results = collection.get(include=["documents", "metadatas"])

        if not results["ids"]:
            logger.warning(f"ChromaDB collection {ticker_upper} is empty")
            return None

        chunks = []
        for i, chunk_id in enumerate(results["ids"]):
            document = results["documents"][i]
            metadata = results["metadatas"][i]

            chunk = Chunk(
                chunk_id=chunk_id,
                company=metadata["company"],
                year=int(metadata["year"]),
                section=metadata["section"],
                subsection=metadata["subsection"],
                text=document,
                word_count=int(metadata["word_count"]),
            )
            chunks.append(chunk)

        tokenized_chunks = [tokenize_for_bm25(chunk.text) for chunk in chunks]
        bm25_index = BM25Okapi(tokenized_chunks)

        _bm25_indices[ticker_upper] = {
            "bm25": bm25_index,
            "chunks": chunks,
        }

        years = sorted({chunk.year for chunk in chunks})
        logger.info(
            f"Successfully rebuilt BM25 index for {ticker_upper} with {len(chunks)} chunks "
            f"across years {years}"
        )
        return _bm25_indices[ticker_upper]

    except Exception as e:
        logger.error(f"Failed to rebuild BM25 index for {ticker_upper}: {e}")
        return None


def get_chroma_collection(ticker: str):
    """Get the ChromaDB collection for a ticker.

    Args:
        ticker: Company ticker symbol.

    Returns:
        chromadb.Collection: The ChromaDB collection.

    Raises:
        Exception: If collection does not exist.
    """
    collection_name = ticker.upper()
    try:
        return _get_chroma_client().get_collection(name=collection_name)
    except Exception as e:
        logger.error(f"Failed to get ChromaDB collection {collection_name}: {e}")
        raise


class FakeEmbeddingWrapper:
    """Deterministic, offline fake embedder producing unit-normalized vectors for tests."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    def _embed_single(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(h[:4], "big")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(EMBEDDING_DIMENSIONS).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm if norm > 0 else 1.0)

    def encode_documents(self, texts: list[str], batch_size: int = DEFAULT_EMBED_BATCH_SIZE) -> np.ndarray:
        return np.array([self._embed_single(t) for t in texts], dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        return self._embed_single(query)

    def encode(
        self,
        texts: list[str] | str,
        show_progress_bar: bool = False,
        batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    ) -> np.ndarray:
        if isinstance(texts, str):
            return self.encode_query(texts)
        return self.encode_documents(texts, batch_size=batch_size)


def _run_tests() -> None:
    global _embedding_model, _chroma_client
    print("=== Testing embedder.py with all-MiniLM-L6-v2 ===")

    TEST_TICKER = "ZZTEST"
    client = _get_chroma_client()

    try:
        client.delete_collection(TEST_TICKER)
    except Exception:
        pass
    _bm25_indices.pop(TEST_TICKER, None)

    # -------------------------------------------------------------------------
    # Test 1: Missing stamp in legacy collection triggers rebuild
    # -------------------------------------------------------------------------
    print("\n--- 1. Testing legacy collection rebuild on missing stamps ---")
    legacy_ticker = "LEGACY_TEST"
    try:
        client.delete_collection(legacy_ticker)
    except Exception:
        pass

    legacy_coll = client.create_collection(
        name=legacy_ticker,
        metadata={"ingested_at": "2024-01-01T00:00:00Z", "years_covered": "2021"}
    )
    assert "distance_space" not in (legacy_coll.metadata or {})

    rebuilt_coll = _get_or_create_collection(legacy_ticker)
    meta = rebuilt_coll.metadata or {}
    print(f"Rebuilt collection stamps: {meta}")
    assert meta.get("distance_space") == EMBEDDING_DISTANCE_SPACE
    assert meta.get("embedding_model") == EMBEDDING_MODEL_NAME
    assert meta.get("embedding_dimensions") == EMBEDDING_DIMENSIONS
    assert meta.get("chunk_max_tokens") == CHUNK_MAX_TOKENS
    client.delete_collection(legacy_ticker)
    print("OK: Missing stamps correctly detected and triggered collection rebuild")

    # -------------------------------------------------------------------------
    # Test 2: Distance space verification (cosine must give 0.5, not 1.0)
    # -------------------------------------------------------------------------
    print("\n--- 2. Testing distance space is cosine across create, modify, and reopen ---")
    dist_ticker = "DIST_TEST"
    try:
        client.delete_collection(dist_ticker)
    except Exception:
        pass

    dist_coll = client.create_collection(
        name=dist_ticker,
        metadata={"hnsw:space": "cosine", "distance_space": "cosine"}
    )

    u = np.zeros(EMBEDDING_DIMENSIONS, dtype=np.float32)
    u[0] = 1.0

    v = np.zeros(EMBEDDING_DIMENSIONS, dtype=np.float32)
    v[0] = 0.5
    v[1] = float(np.sqrt(0.75))

    dist_coll.add(ids=["v1"], embeddings=[v.tolist()], documents=["test doc"])

    res_before = dist_coll.query(query_embeddings=[u.tolist()], n_results=1)
    dist_before = round(res_before["distances"][0][0], 4)
    print(f"Distance before modify: {dist_before} (expected 0.5)")
    assert dist_before == 0.5, f"Expected cosine distance 0.5, got {dist_before}"

    dist_coll.modify(metadata={"distance_space": "cosine", "ingested_at": "now"})
    res_after = dist_coll.query(query_embeddings=[u.tolist()], n_results=1)
    dist_after = round(res_after["distances"][0][0], 4)
    print(f"Distance after modify: {dist_after} (expected 0.5)")
    assert dist_after == 0.5, f"Expected cosine distance 0.5, got {dist_after}"

    _chroma_client = None
    client_reopened = _get_chroma_client()
    dist_coll_reopened = client_reopened.get_collection(dist_ticker)
    res_reopened = dist_coll_reopened.query(query_embeddings=[u.tolist()], n_results=1)
    dist_reopened = round(res_reopened["distances"][0][0], 4)
    print(f"Distance after reopening client: {dist_reopened} (expected 0.5)")
    assert dist_reopened == 0.5, f"Expected cosine distance 0.5, got {dist_reopened}"

    client_reopened.delete_collection(dist_ticker)
    print("OK: Cosine distance space verified before modify, after modify, and after client reopen")

    # -------------------------------------------------------------------------
    # Test 3: Local SentenceTransformer model encoding test
    # -------------------------------------------------------------------------
    print("\n--- 3. Testing local all-MiniLM-L6-v2 embedding ---")
    wrapper = _get_embedding_model()
    test_vec = wrapper.encode_query("Capital expenditures in digital infrastructure")
    assert test_vec.shape == (EMBEDDING_DIMENSIONS,), f"Expected shape ({EMBEDDING_DIMENSIONS},), got {test_vec.shape}"
    norm = float(np.linalg.norm(test_vec))
    print(f"Vector shape: {test_vec.shape}, L2 norm: {norm:.4f}")
    assert abs(norm - 1.0) < 1e-4, f"Expected unit norm, got {norm}"
    print("OK: Local embedding model encodes query to unit-normalized 384-dim vector")

    # -------------------------------------------------------------------------
    # Test 4: End-to-end embed and store with local embedder
    # -------------------------------------------------------------------------
    print("\n--- 4. Testing embed_and_store_chunks with multiple years ---")
    def make_chunk(year: int, section: str, subsection: str, text: str, idx: int) -> Chunk:
        return Chunk(
            text=text,
            company=TEST_TICKER,
            year=year,
            section=section,
            subsection=subsection,
            chunk_id=f"{TEST_TICKER}_{year}_{idx:02d}",
            word_count=len(text.split()),
        )

    chunks_2021 = [
        make_chunk(2021, "MD&A", "Technology",
                   "[Test | 2021 | MD&A > Technology] We are investing heavily in our digital claims platform.", 0),
        make_chunk(2021, "Risk Factors", "full_text",
                   "[Test | 2021 | Risk Factors] Our business faces cybersecurity risks.", 1),
    ]
    chunks_2022 = [
        make_chunk(2022, "MD&A", "Claims",
                   "[Test | 2022 | MD&A > Claims] Claims processing efficiency improved significantly this year.", 0),
    ]

    embed_and_store_chunks(chunks_2021, ticker=TEST_TICKER, years_covered=[2021])
    embed_and_store_chunks(chunks_2022, ticker=TEST_TICKER, years_covered=[2022])
    embed_and_store_chunks(chunks_2021, ticker=TEST_TICKER, years_covered=[2021])  # Skipped

    coll = get_chroma_collection(TEST_TICKER)
    count = coll.count()
    print(f"ChromaDB count: {count} (expected 3)")
    assert count == 3, f"Expected 3 chunks, got {count}"

    bm25_entry = get_bm25_index(TEST_TICKER)
    assert bm25_entry is not None
    indexed_years = sorted({c.year for c in bm25_entry["chunks"]})
    print(f"BM25 chunks indexed: {len(bm25_entry['chunks'])}, years: {indexed_years}")
    assert indexed_years == [2021, 2022]

    # Clean up test collection
    _get_chroma_client().delete_collection(TEST_TICKER)
    _bm25_indices.pop(TEST_TICKER, None)

    print("\nEmbedder testing complete and verified with local all-MiniLM-L6-v2!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    _run_tests()
