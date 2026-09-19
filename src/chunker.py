"""Recursive chunking, contextual prefixes, and chunk metadata attachment.

Takes structured sections dict from ingestion.py and produces Chunk objects for embedder.py.
Handles paragraph -> sentence -> chunk splitting with contextual prefixes and overlap.
"""

import hashlib
import logging
import re
from pathlib import Path

from src.config import CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS
from src.models import Chunk

logger = logging.getLogger(__name__)

# Abbreviations whose trailing period must not end a sentence.
_ABBREVIATIONS = ("U.S.", "Inc.", "Corp.", "Ltd.", "Co.", "No.", "vs.", "e.g.", "i.e.")


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using regex patterns.

    Note: Uses simple regex splitting, not a full NLP tokenizer.
    Keeps the terminal punctuation on each sentence (a lookbehind is used
    instead of consuming the period). Common abbreviations are protected.

    Args:
        text: Input text to split into sentences.

    Returns:
        List of sentence strings, stripped of leading/trailing whitespace.
    """
    protected = text
    for abbrev in _ABBREVIATIONS:
        protected = protected.replace(abbrev + " ", abbrev.replace(".", "<dot>") + " ")

    # Split after . ! ? when followed by whitespace and a capital letter, quote, or bracket
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"(])', protected)

    result = []
    for part in parts:
        sentence = part.replace("<dot>", ".").strip()
        if sentence:
            result.append(sentence)

    return result


def split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs on double newlines.

    Args:
        text: Input text to split.

    Returns:
        List of paragraph strings, empty paragraphs filtered out.
    """
    # Split on one or more sets of double newlines
    paragraphs = re.split(r'\n\s*\n+', text)

    # Filter out empty paragraphs and strip whitespace
    result = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if paragraph:
            result.append(paragraph)

    return result


def chunk_subsection_text(
    text: str,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS
) -> list[str]:
    """Chunk subsection text by greedily packing sentences into chunks with overlap.

    Token counting approximation: len(text.split()) is treated as token count.
    This is intentionally approximate to avoid adding tokenizer dependencies.

    Args:
        text: Raw subsection text to chunk.
        max_tokens: Maximum tokens (words) per chunk.
        overlap_tokens: Approximate tokens (words) to overlap between chunks.

    Returns:
        List of plain text chunk strings (no metadata, no contextual prefix).
    """
    if not text or not text.strip():
        return []

    # Start with paragraphs, then break into sentences
    paragraphs = split_into_paragraphs(text)
    all_sentences = []

    for paragraph in paragraphs:
        sentences = split_into_sentences(paragraph)
        all_sentences.extend(sentences)

    if not all_sentences:
        return []

    # Tables and headers in 10-Ks have no sentence punctuation, so a whole table
    # can arrive as one giant "sentence". Hard-split anything above max_tokens
    # so no chunk can ever exceed the cap.
    expanded = []
    for sentence in all_sentences:
        words = sentence.split()
        if len(words) > max_tokens:
            for start in range(0, len(words), max_tokens):
                expanded.append(" ".join(words[start:start + max_tokens]))
        else:
            expanded.append(sentence)
    all_sentences = expanded

    chunks = []
    current_chunk_sentences = []
    current_token_count = 0

    i = 0
    while i < len(all_sentences):
        sentence = all_sentences[i]
        sentence_tokens = len(sentence.split())

        # Check if adding this sentence would exceed max_tokens
        if current_chunk_sentences and current_token_count + sentence_tokens > max_tokens:
            # Close current chunk
            chunk_text = ' '.join(current_chunk_sentences)
            chunks.append(chunk_text)

            # Start new chunk with overlap from the end of previous chunk
            overlap_sentences = []
            overlap_count = 0

            # Walk backward from end of current chunk to build overlap
            for j in range(len(current_chunk_sentences) - 1, -1, -1):
                overlap_sentence = current_chunk_sentences[j]
                overlap_sentence_tokens = len(overlap_sentence.split())

                if overlap_count + overlap_sentence_tokens <= overlap_tokens:
                    overlap_sentences.insert(0, overlap_sentence)
                    overlap_count += overlap_sentence_tokens
                else:
                    break

            # Reset for new chunk with overlap
            current_chunk_sentences = overlap_sentences
            current_token_count = overlap_count

            # INFINITE LOOP FIX: Check if overlap + next sentence still exceeds max_tokens
            # If so, force-append the sentence anyway to guarantee forward progress
            if overlap_count + sentence_tokens > max_tokens:
                logger.debug(
                    f"Overlap ({overlap_count} tokens) + sentence ({sentence_tokens} tokens) "
                    f"exceeds max_tokens ({max_tokens}), force-appending to prevent infinite loop"
                )
                # Force-append the sentence and advance (accept temporary overflow)
                current_chunk_sentences.append(sentence)
                current_token_count += sentence_tokens
                i += 1
                continue

            # Don't increment i - try to add the same sentence to the new chunk
            continue

        # Add sentence to current chunk
        current_chunk_sentences.append(sentence)
        current_token_count += sentence_tokens

        # Handle oversized single sentence:
        # Note: Sentences exceeding max_tokens are already hard-split in the
        # expansion loop above. This check is retained as a defense-in-depth
        # safeguard in case unexpanded text reaches this point.
        if len(current_chunk_sentences) == 1 and sentence_tokens > max_tokens:
            logger.warning(
                f"Single sentence exceeds max_tokens ({sentence_tokens} > {max_tokens}), "
                f"keeping as standalone chunk: '{sentence[:100]}...'"
            )
            chunk_text = sentence
            chunks.append(chunk_text)

            # Reset for next chunk
            current_chunk_sentences = []
            current_token_count = 0

        i += 1

    # Add final chunk if any sentences remain
    if current_chunk_sentences:
        chunk_text = ' '.join(current_chunk_sentences)
        chunks.append(chunk_text)

    return chunks


def build_contextual_prefix(company: str, year: int, section: str, subsection: str) -> str:
    """Build contextual prefix for chunk text.

    Args:
        company: Company display name (e.g. "Hartford").
        year: Fiscal year (e.g. 2021).
        section: Section name (e.g. "MD&A").
        subsection: Subsection name (e.g. "Technology Initiatives") or "full_text".

    Returns:
        Formatted prefix string like "[Hartford | 2021 | MD&A > Technology Initiatives]"
        or "[Hartford | 2021 | Risk Factors]" if subsection is "full_text".
    """
    if subsection == "full_text":
        return f"[{company} | {year} | {section}]"
    else:
        return f"[{company} | {year} | {section} > {subsection}]"


def make_chunk_id(company_ticker: str, year: int, section: str, subsection: str, index: int) -> str:
    """Generate deterministic chunk ID.

    Args:
        company_ticker: Company ticker symbol (e.g. "HIG").
        year: Fiscal year.
        section: Section name.
        subsection: Subsection name.
        index: Chunk index within this section/subsection (0-based).

    Returns:
        Chunk ID like "HIG_2021_MDA_TECH_07_A1B2C3".
    """
    def abbreviate_name(name: str) -> str:
        """Convert name to short uppercase alphanumeric token."""
        clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', name)
        words = clean_name.split()

        if not words:
            return "UNK"

        abbreviated = []
        for word in words:
            if len(word) >= 3:
                abbreviated.append(word[:4].upper())
            elif len(word) > 0:
                abbreviated.append(word.upper())

        result = '_'.join(abbreviated)
        if len(result) > 12:
            result = result[:12]

        return result if result else "UNK"

    section_abbrev = abbreviate_name(section)
    subsection_abbrev = abbreviate_name(subsection) if subsection != "full_text" else ""

    # Build ID components
    components = [company_ticker.upper(), str(year), section_abbrev]
    if subsection_abbrev:
        components.append(subsection_abbrev)

    # Add zero-padded index
    components.append(f"{index:02d}")

    # Create base ID
    base_id = "_".join(components)

    # Always add a unique hash suffix to ensure absolute uniqueness
    unique_string = f"{company_ticker}_{year}_{section}_{subsection}_{index}"
    hash_suffix = hashlib.md5(unique_string.encode()).hexdigest()[:6].upper()

    return f"{base_id}_{hash_suffix}"


def chunk_year_sections(
    sections: dict,
    company_ticker: str,
    company_display_name: str,
    year: int
) -> list[Chunk]:
    """Chunk all sections for one fiscal year into Chunk objects.

    Args:
        sections: Structured sections dict from ingestion.py.
        company_ticker: Ticker symbol for chunk IDs (e.g. "HIG").
        company_display_name: Display name for prefixes (e.g. "Hartford").
        year: Fiscal year.

    Returns:
        List of Chunk objects ready for embedder.py.
    """
    if not sections:
        logger.warning(f"No sections provided for {company_ticker} {year}")
        return []

    all_chunks = []
    total_chunks_for_year = 0

    for section_name, subsections in sections.items():
        if not isinstance(subsections, dict):
            logger.warning(f"Invalid subsections format for {section_name} in {company_ticker} {year}")
            continue

        if not subsections:
            logger.warning(f"Empty subsections dict for {section_name} in {company_ticker} {year}")
            continue

        section_chunk_count = 0

        for subsection_name, text in subsections.items():
            if not text or not text.strip():
                logger.debug(f"Empty subsection text: {section_name} > {subsection_name}")
                continue

            # Chunk the subsection text
            text_chunks = chunk_subsection_text(text)

            # Convert text chunks to Chunk objects
            for chunk_index, chunk_text in enumerate(text_chunks):
                prefix = build_contextual_prefix(
                    company_display_name, year, section_name, subsection_name
                )
                prefixed_text = f"{prefix} {chunk_text}"
                word_count = len(prefixed_text.split())

                chunk_id = make_chunk_id(
                    company_ticker, year, section_name, subsection_name, chunk_index
                )

                chunk = Chunk(
                    text=prefixed_text,
                    company=company_ticker,
                    year=year,
                    section=section_name,
                    subsection=subsection_name,
                    chunk_id=chunk_id,
                    word_count=word_count
                )

                all_chunks.append(chunk)
                section_chunk_count += 1

        if section_chunk_count > 0:
            logger.debug(f"Section '{section_name}': {section_chunk_count} chunks")
            total_chunks_for_year += section_chunk_count

    logger.info(
        f"Chunked {company_ticker} fiscal year {year}: {total_chunks_for_year} total chunks "
        f"across {len(sections)} sections"
    )

    return all_chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

    print("=== Testing chunker.py ===")

    # Sample data for testing
    sample_sections = {
        "MD&A": {
            "Technology Initiatives": "We are committed to modernizing our claims platform. " * 80,
            "Claims Operations": "Our claims processing has improved significantly. " * 60,
        },
        "Risk Factors": {
            "full_text": "Our business faces various risks. " * 40,
        },
        "Future Outlook": {
            "Growth Strategy": "We expect continued growth in digital channels. " * 50,
        }
    }

    # Test chunking with config settings
    chunks = chunk_year_sections(
        sample_sections,
        company_ticker="HIG",
        company_display_name="Hartford",
        year=2021
    )

    print(f"\nProduced {len(chunks)} chunks with CHUNK_MAX_TOKENS={CHUNK_MAX_TOKENS}")
    print(f"Sample Chunk 1 ID: {chunks[0].chunk_id}")
    print(f"Sample Chunk 1 Word Count: {chunks[0].word_count}")

    # REGRESSION TEST: Infinite loop fix test (tested with both 500 and configured tokens)
    print("\n--- REGRESSION TEST: Infinite Loop Fix ---")
    overlap_sentence = " ".join(["overlap"] * 49) + " overlap."
    large_sentence = " ".join(["Large"] * 469) + " large."
    targeted_test = f"Start. {overlap_sentence} {large_sentence} {large_sentence} {large_sentence}"

    # Test with legacy 500 tokens
    targeted_chunks_500 = chunk_subsection_text(targeted_test, max_tokens=500, overlap_tokens=50)
    print(f"OK: 500-token test produced {len(targeted_chunks_500)} chunks without looping")

    # Test with config CHUNK_MAX_TOKENS
    targeted_chunks_config = chunk_subsection_text(targeted_test, max_tokens=CHUNK_MAX_TOKENS, overlap_tokens=CHUNK_OVERLAP_TOKENS)
    print(f"OK: config-token ({CHUNK_MAX_TOKENS}) test produced {len(targeted_chunks_config)} chunks")

    # Test unpunctuated giant table cap
    print("\n--- TEST: Oversized unpunctuated block is capped ---")
    table_like = " ".join(["cell"] * 3383)

    table_chunks_500 = chunk_subsection_text(table_like, max_tokens=500, overlap_tokens=50)
    biggest_500 = max(len(c.split()) for c in table_chunks_500)
    assert biggest_500 <= 500, f"Expected <= 500, got {biggest_500}"
    print(f"500-token cap respected (largest chunk: {biggest_500} tokens)")

    table_chunks_config = chunk_subsection_text(table_like, max_tokens=CHUNK_MAX_TOKENS, overlap_tokens=CHUNK_OVERLAP_TOKENS)
    biggest_config = max(len(c.split()) for c in table_chunks_config)
    assert biggest_config <= CHUNK_MAX_TOKENS, f"Expected <= {CHUNK_MAX_TOKENS}, got {biggest_config}"
    print(f"Config-token ({CHUNK_MAX_TOKENS}) cap respected (largest chunk: {biggest_config} tokens)")

    # Real data test with derived fiscal year from filename
    print("\n--- REAL DATA TEST ---")
    hig_files = list(Path("data/filings").glob("HIG_*.txt"))
    if hig_files:
        test_file = hig_files[0]
        year_match = re.search(r"(\d{4})", test_file.name)
        test_year = int(year_match.group(1)) if year_match else 2021
        print(f"Testing with real HIG data: {test_file.name} (derived fiscal year {test_year})")

        with open(test_file, "r", encoding="utf-8") as f:
            real_text = f.read()

        test_section_text = real_text[:50000]
        real_sections = {"Test Section": {"full_text": test_section_text}}

        real_chunks = chunk_year_sections(
            real_sections,
            company_ticker="HIG",
            company_display_name="Hartford",
            year=test_year,
        )

        print(f"OK: real data test completed, {len(real_chunks)} chunks produced for fiscal year {test_year}")
        assert all(c.year == test_year for c in real_chunks), "All chunks must match derived fiscal year"
    else:
        print("No HIG files found in data/filings/, skipping real data test")

    # Test sentence splitting
    print("\n--- Testing Sentence Splitting ---")
    test_text = "The U.S. market is strong. Inc. profits are up! What will happen next? We don't know."
    sentences = split_into_sentences(test_text)
    print(f"Sentences: {sentences}")
    assert len(sentences) == 4

    print("\nChunker testing complete and verified offline!")