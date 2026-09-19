"""SEC EDGAR fetch, cache-first loading, and section parsing for TempusRAG.

Year convention: fiscal year (period-of-report end year). hig-20251231.htm
(filed 2026-02-20) is stored as HIG_2025.

Flow: EDGAR -> data/raw/{T}_{FY}.html (saved once) -> parse -> data/filings/{T}_{FY}.txt
The cache file starts with a parser version marker. A missing or old marker is a
cache miss, so parser fixes never reuse text written by an older parser. Fallback
parses (no real sections found) are never cached.

Section detection:
  1. Extract text block by block (inline spans stay on one line, hidden XBRL header removed).
  2. Find heading candidates: a line that STARTS with "Item N". Cross-references inside
     sentences are not headings. Rows ending in a page number are table of contents rows.
  3. Per Item number keep the occurrence with the most text after it.
  4. Hartford-style layout: when Item 7 is a stub, find the block titled
     "Management's Discussion and Analysis" and assign it to Item 7, removing it from
     whichever section it was sitting in.
"""

from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from src.config import DATA_DIR, PROJECT_ROOT, SEC_MIN_REQUEST_INTERVAL, SEC_USER_AGENT

try:
    import pymupdf
    PYMUPDF_AVAILABLE = True
except ImportError:
    try:
        import fitz as pymupdf  # legacy alias
        PYMUPDF_AVAILABLE = True
    except ImportError:
        PYMUPDF_AVAILABLE = False

logger = logging.getLogger(__name__)

PARSER_VERSION = 3
CACHE_MARKER = f"# tempusrag-parser-version: {PARSER_VERSION}"

RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEBUG_DIR = PROJECT_ROOT / "data" / "debug"

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL_TEMPLATE = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

_ticker_cik_cache: dict[str, str] | None = None
_company_directory: list[dict[str, str]] | None = None
_last_request_time: float = 0.0

# Years that failed in the most recent ingest_company call: {fiscal_year: reason}
LAST_FAILURES: dict[int, str] = {}


class EdgarFetchError(Exception):
    """Raised when SEC EDGAR returns a non-200 response or is unreachable."""


# ---------------------------------------------------------------------------
# EDGAR HTTP helpers
# ---------------------------------------------------------------------------

def _make_request(url: str) -> requests.Response:
    """Rate-limited GET to SEC EDGAR with explicit error surfacing."""
    global _last_request_time

    elapsed = time.time() - _last_request_time
    if elapsed < SEC_MIN_REQUEST_INTERVAL:
        time.sleep(SEC_MIN_REQUEST_INTERVAL - elapsed)

    if not SEC_USER_AGENT:
        logger.warning("SEC_USER_AGENT is empty; EDGAR requires a contact e-mail. Set it in .env.")

    headers = {
        "User-Agent": SEC_USER_AGENT or "TempusRAG/1.0 user@example.com",
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json, text/html, */*",
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        _last_request_time = time.time()
        if response.status_code != 200:
            logger.error("HTTP %s for %s", response.status_code, url)
            raise EdgarFetchError(f"HTTP {response.status_code}: {url}")
        return response
    except requests.exceptions.RequestException as exc:
        logger.error("Request failed for %s: %s", url, exc)
        raise EdgarFetchError(f"Request failed for {url}: {exc}") from exc


def _ensure_ticker_cache() -> None:
    global _ticker_cik_cache, _company_directory
    if _ticker_cik_cache is not None and _company_directory is not None:
        return

    import json
    cache_path = Path(DATA_DIR).parent / "tickers_cache.json"

    # Read from local disk cache if available and under 7 days old
    if cache_path.exists():
        try:
            mtime = cache_path.stat().st_mtime
            if (time.time() - mtime) < (7 * 86400):
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                cached_map = cached.get("ticker_map", {})
                cached_dir = cached.get("directory", [])
                if cached_map and cached_dir:
                    _ticker_cik_cache = cached_map
                    _company_directory = cached_dir
                    logger.info("Loaded %d ticker-to-CIK mappings from local disk cache", len(cached_map))
                    return
        except Exception as e:
            logger.warning("Could not read local tickers cache: %s", e)

    logger.info("Fetching SEC ticker-to-CIK mapping from EDGAR")
    data = _make_request(COMPANY_TICKERS_URL).json()

    ticker_map: dict[str, str] = {}
    directory: list[dict[str, str]] = []
    for entry in data.values():
        ticker = entry["ticker"].upper()
        cik = str(entry["cik_str"]).zfill(10)
        title = (entry.get("title") or ticker).strip()
        ticker_map[ticker] = cik
        directory.append({"ticker": ticker, "title": title, "cik": cik})

    logger.info("Loaded %d ticker-to-CIK mappings", len(ticker_map))
    _ticker_cik_cache = ticker_map
    _company_directory = directory

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"ticker_map": ticker_map, "directory": directory}, f)
        logger.info("Saved SEC ticker-to-CIK mapping to %s", cache_path)
    except Exception as e:
        logger.warning("Could not persist tickers cache to disk: %s", e)


def load_ticker_cik_map() -> dict[str, str]:
    """Return ticker -> zero-padded CIK mapping from SEC EDGAR (cached)."""
    _ensure_ticker_cache()
    return _ticker_cik_cache or {}


def load_company_directory() -> list[dict[str, str]]:
    """Return [{ticker, title, cik}, ...] for every listed company (cached)."""
    _ensure_ticker_cache()
    return list(_company_directory or [])


def get_recent_10k_filings(ticker: str, cik_map: dict[str, str], count: int = 5) -> list[dict]:
    """Return metadata for the `count` most recent 10-K filings."""
    ticker = ticker.upper()
    cik = cik_map.get(ticker)
    if not cik:
        raise ValueError(f"Ticker not found in CIK map: {ticker}")

    logger.info("Fetching recent 10-K filings for %s (CIK %s)", ticker, cik)
    submissions = _make_request(SUBMISSIONS_URL_TEMPLATE.format(cik=cik)).json()

    recent = submissions["filings"]["recent"]
    report_dates = recent.get("reportDate", [None] * len(recent["form"]))
    filings: list[dict] = []

    for i, form in enumerate(recent["form"]):
        if form != "10-K":
            continue
        accession = recent["accessionNumber"][i]
        primary_document = recent["primaryDocument"][i]
        document_url = ARCHIVES_URL_TEMPLATE.format(
            cik=str(int(cik)),
            accession=accession.replace("-", ""),
            document=primary_document,
        )
        filings.append({
            "accessionNumber": accession,
            "filingDate": recent["filingDate"][i],
            "reportDate": report_dates[i],
            "primaryDocument": primary_document,
            "document_url": document_url,
        })

    filings.sort(key=lambda x: x["filingDate"], reverse=True)
    result = filings[:count]
    logger.info("Found %d 10-K filings for %s, returning %d", len(filings), ticker, len(result))
    return result


def _fiscal_year_from_filing(filing: dict) -> int:
    """Fiscal year from reportDate; else filing year minus 1 if filed Jan to Apr; else filing year."""
    report_date = filing.get("reportDate")
    if report_date:
        try:
            return int(str(report_date)[:4])
        except (ValueError, TypeError):
            pass

    filing_date = filing.get("filingDate", "")
    try:
        fy = int(filing_date[:4])
        month = int(filing_date[5:7])
        return fy - 1 if month <= 4 else fy
    except (ValueError, IndexError):
        return 0


def download_filing(filing: dict) -> bytes:
    """Download raw filing bytes from EDGAR."""
    logger.info("Downloading 10-K filed %s (%s)", filing.get("filingDate", "?"), filing.get("primaryDocument", "?"))
    return _make_request(filing["document_url"]).content


def detect_format(content: bytes, filename: str = "") -> str:
    """Return 'html', 'pdf', or 'text'."""
    fn = filename.lower()
    if fn.endswith(".pdf"):
        return "pdf"
    if fn.endswith((".htm", ".html")):
        return "html"
    if content.startswith(b"%PDF"):
        return "pdf"
    sample = content[:1024].lower()
    if b"<html" in sample or b"<!doctype" in sample or b"<div" in sample:
        return "html"
    return "text"


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

_BLOCK_TAGS = ["div", "p", "tr", "li", "table", "h1", "h2", "h3", "h4", "h5", "h6"]


def _html_to_lines(content: bytes) -> list[str]:
    """Convert filing HTML to clean lines: one line per block, inline spans joined."""
    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    # Inline XBRL hidden header holds IDs and numbers, not filing text
    for tag in soup.find_all("ix:header"):
        tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for tag in soup.find_all(_BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    for tag in soup.find_all(["td", "th"]):
        tag.insert_after(" ")

    text = soup.get_text().replace("\u00a0", " ")
    lines = []
    for raw in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw).strip()
        if line:
            lines.append(line)
    return lines


def _text_to_lines(text: str) -> list[str]:
    text = text.replace("\u00a0", " ")
    return [re.sub(r"[ \t]+", " ", l).strip() for l in text.split("\n") if l.strip()]


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------

_CANONICAL_TITLES: dict[str, str] = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity", "2": "Properties", "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures", "5": "Market for Common Equity", "6": "Reserved",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures about Market Risk",
    "8": "Financial Statements", "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures", "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions",
    "10": "Directors and Executive Officers", "11": "Executive Compensation",
    "12": "Security Ownership", "13": "Certain Relationships",
    "14": "Principal Accountant Fees", "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}

_ITEM_LINE = re.compile(r"^item\s+(\d{1,2}[a-c]?)\s*[.:\-\u2013\u2014]?\s*(.*)$", re.IGNORECASE)
_TRAILING_PAGE = re.compile(r"(?:\s|\.{2,})\d{1,3}(?:\s*[-\u2013,]\s*\d{1,3})?$")
_MDA_TITLE = re.compile(r"^(?:[a-z ]{0,25})?management.s discussion and analysis\b")
_STATEMENT_END = re.compile(
    r"^(?:report of independent registered|consolidated statements? of|"
    r"financial statements and supplementary|notes to consolidated)",
    re.IGNORECASE,
)

_MIN_BODY_WORDS = 150   # below this an Item is treated as a stub
_STUB_WORDS = 1500      # Item 7 under this triggers the MD&A block search
_MAX_HEADING_WORDS = 30


def _find_candidates(lines: list[str]) -> list[dict]:
    """Lines that START with 'Item N' and look like headings."""
    cands = []
    for i, line in enumerate(lines):
        if len(line.split()) > _MAX_HEADING_WORDS:
            continue
        m = _ITEM_LINE.match(line)
        if not m:
            continue
        num = m.group(1).upper()
        if num not in _CANONICAL_TITLES:
            continue
        title = m.group(2).strip()
        # "Item 1A of this report..." is a sentence, not a heading
        if title and not (title[0].isupper() or title[0].isdigit() or title[0] in "\"'(\u201c\u2018"):
            continue
        if not title and i + 1 < len(lines):
            nxt = lines[i + 1]
            if len(nxt.split()) <= 15 and not _ITEM_LINE.match(nxt):
                title = nxt
        cands.append({
            "idx": i,
            "num": num,
            "title": title,
            "toc": bool(_TRAILING_PAGE.search(line)),
        })
    return cands


def _clean_title(num: str, title: str) -> str:
    title = _TRAILING_PAGE.sub("", title).strip(" .:-\u2013\u2014")
    if len(title) < 4 or len(title) > 90:
        title = _CANONICAL_TITLES.get(num, f"Item {num}")
    return title


def _cumulative_words(lines: list[str]) -> list[int]:
    cum = [0]
    for line in lines:
        cum.append(cum[-1] + len(line.split()))
    return cum


def _select_body_candidates(cands: list[dict], cum: list[int], n_lines: int) -> list[dict]:
    """Per Item number keep the non-TOC occurrence with the most text after it."""
    for c in cands:
        end = n_lines
        for other in cands:
            if other["idx"] > c["idx"] and other["num"] != c["num"] and not other["toc"]:
                end = other["idx"]
                break
        c["words_after"] = cum[end] - cum[c["idx"] + 1]

    by_num: dict[str, list[dict]] = {}
    for c in cands:
        by_num.setdefault(c["num"], []).append(c)

    selected = []
    for occurrences in by_num.values():
        pool = [c for c in occurrences if not c["toc"]] or occurrences
        selected.append(max(pool, key=lambda c: (c["words_after"], c["idx"])))
    selected.sort(key=lambda c: c["idx"])
    return selected


# ---------------------------------------------------------------------------
# Segments and the Hartford-style MD&A carve
# ---------------------------------------------------------------------------

def _build_segments(selected: list[dict], n_lines: int) -> list[list]:
    """[[label, start_line, end_line], ...]; body starts after the heading line."""
    segments = []
    for i, c in enumerate(selected):
        end = selected[i + 1]["idx"] if i + 1 < len(selected) else n_lines
        label = f"Item {c['num']}. {_clean_title(c['num'], c['title'])}"
        segments.append([label, c["idx"] + 1, end])
    return segments


def _carve_mda_block(segments: list[list], lines: list[str], cum: list[int], cands: list[dict]) -> bool:
    """If Item 7 is a stub, move a block titled 'Management's Discussion and Analysis' into it."""
    item7_words = sum(cum[b] - cum[a] for lab, a, b in segments if lab.startswith("Item 7."))
    if item7_words >= _STUB_WORDS:
        return False

    end_marks = sorted(c["idx"] for c in cands if not c["toc"])
    best = None  # (words, start, end)
    for i, line in enumerate(lines):
        if len(line.split()) > 25 or _ITEM_LINE.match(line) or _TRAILING_PAGE.search(line):
            continue
        if not _MDA_TITLE.match(line.lower()):
            continue
        end = len(lines)
        for mark in end_marks:
            if mark > i:
                end = mark
                break
        for j in range(i + 1, end):
            if _STATEMENT_END.match(lines[j]):
                end = j
                break
        words = cum[end] - cum[i + 1]
        if best is None or words > best[0]:
            best = (words, i, end)

    if best is None or best[0] < _STUB_WORDS or best[0] < 1.5 * item7_words:
        logger.warning(
            "Item 7 has %d words and no separate 'Management's Discussion and Analysis' block "
            "was found. Run --dump-headings to inspect this filing.", item7_words,
        )
        return False

    words, start, end = best
    label = next((lab for lab, _, _ in segments if lab.startswith("Item 7.")),
                 f"Item 7. {_CANONICAL_TITLES['7']}")

    carved = []
    for lab, a, b in segments:
        if b <= start or a >= end:
            carved.append([lab, a, b])
            continue
        if a < start:
            carved.append([lab, a, start])
        if b > end:
            carved.append([lab, end, b])
    carved.append([label, start, end])
    carved.sort(key=lambda s: s[1])
    segments[:] = carved
    logger.info("Moved a %d word 'Management's Discussion and Analysis' block into '%s'", words, label)
    return True


def _segments_to_sections(segments: list[list], lines: list[str]) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {}
    for label, a, b in sorted(segments, key=lambda s: s[1]):
        body = "\n\n".join(lines[a:b]).strip()
        if not body:
            continue
        existing = sections.get(label, {}).get("full_text", "")
        sections[label] = {"full_text": (existing + "\n\n" + body).strip() if existing else body}
    return sections


def _log_section_stats(sections: dict[str, dict[str, str]]) -> None:
    counts = {name: sum(len(b.split()) for b in subs.values()) for name, subs in sections.items()}
    total = sum(counts.values()) or 1
    for name, words in counts.items():
        logger.info("  section %r: %d words", name, words)
        if words / total > 0.6:
            logger.warning("Section %r holds %.0f%% of the filing; section detection may be wrong",
                           name, 100 * words / total)
    item7 = sum(w for n, w in counts.items() if n.startswith("Item 7."))
    if item7 < _STUB_WORDS:
        logger.warning("Item 7 has only %d words (stub or missed)", item7)


def _is_fallback(sections: dict) -> bool:
    return set(sections.keys()) == {"full_text"}


def _parse_lines(lines: list[str]) -> dict[str, dict[str, str]]:
    """Shared parser for html, pdf and text filings."""
    fallback = {"full_text": {"full_text": "\n\n".join(lines)}}
    if len(" ".join(lines)) < 500:
        logger.warning("Very little text extracted, returning full_text fallback")
        return fallback

    cands = _find_candidates(lines)
    if not cands:
        logger.warning("No Item headings found, returning full_text fallback")
        return fallback

    cum = _cumulative_words(lines)
    selected = _select_body_candidates(cands, cum, len(lines))
    segments = _build_segments(selected, len(lines))
    _carve_mda_block(segments, lines, cum, cands)

    sections = _segments_to_sections(segments, lines)
    if len(sections) < 3:
        logger.warning("Only %d sections detected, returning full_text fallback", len(sections))
        return fallback

    _log_section_stats(sections)
    return sections


def parse_html_filing(content: bytes) -> dict[str, dict[str, str]]:
    """Parse an SEC 10-K HTML filing into {section: {"full_text": text}}."""
    return _parse_lines(_html_to_lines(content))


def parse_pdf_filing(content: bytes) -> dict[str, dict[str, str]]:
    """Extract text from a PDF filing and parse it into sections."""
    if not PYMUPDF_AVAILABLE:
        raise RuntimeError("PyMuPDF is not installed (pip install pymupdf). Cannot parse PDF filings.")
    doc = pymupdf.open(stream=content, filetype="pdf")
    try:
        pages = [doc[n].get_text() for n in range(doc.page_count)]
    finally:
        doc.close()
    return _parse_lines(_text_to_lines("\n".join(p for p in pages if p.strip())))


def parse_text_filing(content: bytes) -> dict[str, dict[str, str]]:
    """Parse a plain-text filing into sections."""
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1", errors="replace")
    return _parse_lines(_text_to_lines(text))


# ---------------------------------------------------------------------------
# Cache read and write
# ---------------------------------------------------------------------------

_CACHED_SECTION_RE = re.compile(r"^=== (.+?) ===\s*$")
_CACHED_SUBSECTION_RE = re.compile(r"^--- (.+?) ---\s*$")


def parse_cached_filing(text: str) -> dict[str, dict[str, str]]:
    """Parse a .txt cache file written by ingest_company (=== section ===, --- subsection ---)."""
    sections: dict[str, dict[str, str]] = {}
    section: str | None = None
    subsection = "full_text"
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        if section is None:
            buf = []
            return
        body = "\n".join(buf).strip()
        sections.setdefault(section, {})
        existing = sections[section].get(subsection, "")
        sections[section][subsection] = (existing + "\n\n" + body).strip() if existing else body
        buf = []

    for line in text.splitlines():
        m_sec = _CACHED_SECTION_RE.match(line.strip())
        if m_sec:
            flush()
            section = m_sec.group(1).strip()
            subsection = "full_text"
            continue
        m_sub = _CACHED_SUBSECTION_RE.match(line.strip())
        if m_sub and section is not None:
            flush()
            subsection = m_sub.group(1).strip()
            continue
        buf.append(line)

    flush()
    return sections


def _write_cache(cache_file: Path, sections: dict[str, dict[str, str]]) -> None:
    lines: list[str] = [CACHE_MARKER]
    for sec_name, subsections in sections.items():
        lines.append(f"=== {sec_name} ===")
        for sub_name, body in subsections.items():
            if sub_name != "full_text":
                lines.append(f"--- {sub_name} ---")
            lines.append(body)
    cache_file.write_text("\n\n".join(lines), encoding="utf-8")


def _read_cache(cache_file: Path) -> dict[str, dict[str, str]] | None:
    """Return parsed sections, or None on any miss (absent, empty, old parser version)."""
    if not cache_file.exists():
        return None
    try:
        text = cache_file.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Cannot read cache %s: %s", cache_file.name, exc)
        return None
    if not text.startswith(CACHE_MARKER):
        logger.info("Cache %s has no current parser version marker, will rebuild", cache_file.name)
        return None
    sections = parse_cached_filing(text)
    return sections or None


def _has_mda_section(sections: dict) -> bool:
    return any(name.upper().startswith("ITEM 7.") for name in sections)


# ---------------------------------------------------------------------------
# Raw filing storage
# ---------------------------------------------------------------------------

_RAW_EXTS = ("html", "pdf", "txt")


def _load_raw(filing: dict, ticker: str, fiscal_year: int) -> tuple[bytes, str]:
    """Return (raw bytes, format). Reads data/raw when present, else downloads and saves."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for ext in _RAW_EXTS:
        path = RAW_DIR / f"{ticker}_{fiscal_year}.{ext}"
        if path.exists():
            return path.read_bytes(), {"html": "html", "pdf": "pdf", "txt": "text"}[ext]

    raw = download_filing(filing)
    fmt = detect_format(raw, filing["primaryDocument"])
    ext = {"html": "html", "pdf": "pdf"}.get(fmt, "txt")
    (RAW_DIR / f"{ticker}_{fiscal_year}.{ext}").write_bytes(raw)
    return raw, fmt


def _parse_raw(raw: bytes, fmt: str) -> dict[str, dict[str, str]]:
    if fmt == "html":
        return parse_html_filing(raw)
    if fmt == "pdf":
        return parse_pdf_filing(raw)
    return parse_text_filing(raw)


# ---------------------------------------------------------------------------
# Main ingestion entry point
# ---------------------------------------------------------------------------

def ingest_company(ticker: str, cik_map: dict[str, str], data_dir: str | Path = DATA_DIR) -> dict[int, dict]:
    """Load sections for the most recent 5 10-K filings. Returns {fiscal_year: sections}.

    Years that fail are left out of the result and recorded in LAST_FAILURES.
    """
    ticker = ticker.upper()
    output_dir = Path(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LAST_FAILURES.clear()

    try:
        filings = get_recent_10k_filings(ticker, cik_map, count=5)
    except (EdgarFetchError, ValueError) as exc:
        logger.warning("Cannot reach EDGAR (%s). Using valid cache files only.", exc)
        results: dict[int, dict] = {}
        for path in sorted(output_dir.glob(f"{ticker}_*.txt")):
            m = re.search(r"_(\d{4})\.txt$", path.name)
            sections = _read_cache(path) if m else None
            if sections:
                results[int(m.group(1))] = sections
        return results

    if not filings:
        logger.warning("No 10-K filings found for %s", ticker)
        return {}

    results = {}
    for filing in filings:
        fiscal_year = _fiscal_year_from_filing(filing)
        if fiscal_year == 0:
            logger.warning("Could not determine fiscal year for %s, skipping", filing.get("accessionNumber"))
            continue

        cache_file = output_dir / f"{ticker}_{fiscal_year}.txt"
        cached = _read_cache(cache_file)
        if cached:
            logger.info("Cache hit: %s (%d sections)", cache_file.name, len(cached))
            results[fiscal_year] = cached
            continue

        try:
            raw, fmt = _load_raw(filing, ticker, fiscal_year)
            logger.info("Parsing %s FY%s as %s", ticker, fiscal_year, fmt)
            sections = _parse_raw(raw, fmt)

            if _is_fallback(sections):
                LAST_FAILURES[fiscal_year] = "section detection failed (full_text fallback, not cached)"
                logger.error("%s FY%s: %s", ticker, fiscal_year, LAST_FAILURES[fiscal_year])
                continue
            if not _has_mda_section(sections):
                logger.warning("No Item 7 found for %s FY%s", ticker, fiscal_year)

            _write_cache(cache_file, sections)
            logger.info("Cached %s", cache_file.name)
            results[fiscal_year] = sections

        except EdgarFetchError as exc:
            LAST_FAILURES[fiscal_year] = f"EDGAR fetch failed: {exc}"
            logger.error("%s FY%s: %s", ticker, fiscal_year, LAST_FAILURES[fiscal_year])
        except Exception as exc:
            LAST_FAILURES[fiscal_year] = f"parse failed: {exc}"
            logger.exception("%s FY%s: parse failed", ticker, fiscal_year)

    if not results:
        logger.error("ingest_company: no years loaded for %s", ticker)
    return results


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def dump_headings(ticker: str, cik_map: dict[str, str]) -> None:
    """Write data/debug/{T}_{FY}_headings.txt: every Item-like line and MD&A title line."""
    ticker = ticker.upper()
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    for filing in get_recent_10k_filings(ticker, cik_map, count=5):
        fy = _fiscal_year_from_filing(filing)
        raw, fmt = _load_raw(filing, ticker, fy)
        lines = _html_to_lines(raw) if fmt == "html" else _text_to_lines(raw.decode("utf-8", "replace"))
        cum = _cumulative_words(lines)
        marks = [
            i for i, l in enumerate(lines)
            if len(l.split()) <= _MAX_HEADING_WORDS
            and (re.match(r"^item\s+\d", l, re.IGNORECASE) or "management" in l.lower() and "discussion and analysis" in l.lower())
        ]
        out = [f"{ticker} FY{fy}: {len(lines)} lines, {cum[-1]} words, {len(marks)} marks"]
        for k, i in enumerate(marks):
            nxt = marks[k + 1] if k + 1 < len(marks) else len(lines)
            out.append(f"line {i:6d} | {cum[nxt] - cum[i + 1]:7d} words after | {lines[i][:110]}")
        path = DEBUG_DIR / f"{ticker}_{fy}_headings.txt"
        path.write_text("\n".join(out), encoding="utf-8")
        print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Offline tests
# ---------------------------------------------------------------------------

def _fixture_html(hartford: bool) -> bytes:
    def body(n: int) -> str:
        return "<p>" + "We plan to invest in claims. " * n + "</p>"

    toc_rows = [("1", "Business", 3), ("1A", "Risk Factors", 9), ("7", "Management's Discussion and Analysis", 20),
                ("8", "Financial Statements", 60), ("15", "Exhibits and Financial Statement Schedules", 90)]
    toc = "<table>" + "".join(
        f"<tr><td>Item {n}.</td><td>{t}</td><td>{p}</td></tr>" for n, t, p in toc_rows) + "</table>"

    parts = ['<html><body><ix:header><div>HIDDENXBRLJUNK 12345</div></ix:header>', toc,
             "<div>Item 1. Business</div>", body(80),
             "<div>Item 1A. Risk Factors</div>", body(80),
             "<p>Item 1A of this report describes risks. See Item 7. Management's Discussion in this report.</p>"]
    if hartford:
        parts += ["<div>Item 7. Management's Discussion and Analysis</div><p>See the MD&amp;A section of this report.</p>",
                  "<div>Item 8. Financial Statements and Supplementary Data</div><p>See the financial statements.</p>",
                  "<div>Management's Discussion and Analysis of Financial Condition and Results of Operations</div>",
                  body(400),
                  "<div>Item 15. Exhibits and Financial Statement Schedules</div><p>Exhibits are listed below.</p>"]
    else:
        parts += ["<div>Item 7. Management's Discussion and Analysis</div>", body(400),
                  "<div>Item 8. Financial Statements and Supplementary Data</div>", body(80),
                  "<div>Item 15. Exhibits and Financial Statement Schedules</div><p>Exhibits are listed below.</p>"]
    parts.append("</body></html>")
    return "".join(parts).encode("utf-8")


def _words(sections: dict, prefix: str) -> int:
    return sum(len(b.split()) for n, s in sections.items() if n.startswith(prefix) for b in s.values())


def _run_tests() -> None:
    print("--- 1. Standard layout with TOC, cross-references and hidden XBRL ---")
    s = parse_html_filing(_fixture_html(hartford=False))
    print({n: _words(s, n) for n in s})
    assert not _is_fallback(s)
    for prefix in ("Item 1.", "Item 1A.", "Item 7.", "Item 8.", "Item 15."):
        assert any(n.startswith(prefix) for n in s), f"missing {prefix}"
    assert _words(s, "Item 7.") >= 2000, "Item 7 body lost"
    assert _words(s, "Item 1A.") < 700, "cross-reference sentence must not become a heading"
    assert "HIDDENXBRLJUNK" not in " ".join(b for subs in s.values() for b in subs.values())
    print("OK")

    print("--- 2. Hartford-style layout (Item 7 stub, MD&A block elsewhere) ---")
    s = parse_html_filing(_fixture_html(hartford=True))
    print({n: _words(s, n) for n in s})
    assert _words(s, "Item 7.") >= 2000, "MD&A block was not moved into Item 7"
    assert _words(s, "Item 15.") < 100, "Item 15 must not hold the MD&A text"
    print("OK")

    print("--- 3. Cache version marker and fallback are not cached ---")
    tmp = Path(DEBUG_DIR) / "_test_cache.txt"
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    tmp.write_text("=== Item 7. MD&A ===\n\nold parser text", encoding="utf-8")
    assert _read_cache(tmp) is None, "unmarked cache must be a miss"
    _write_cache(tmp, s)
    assert _read_cache(tmp) is not None and _has_mda_section(_read_cache(tmp))
    tmp.unlink()
    assert _is_fallback(parse_html_filing(b"<html><body><p>tiny</p></body></html>"))
    print("OK")

    print("\nIngestion offline tests passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = sys.argv[1:]

    if not args or args[0] == "--test":
        _run_tests()
    elif args[0] == "--dump-headings":
        dump_headings(args[1] if len(args) > 1 else "HIG", load_ticker_cik_map())
    elif args[0] == "--ingest":
        tk = args[1] if len(args) > 1 else "HIG"
        res = ingest_company(tk, load_ticker_cik_map(), DATA_DIR)
        print(f"\n=== {tk}: {len(res)} years, failures: {LAST_FAILURES or 'none'} ===")
        for year in sorted(res):
            print(f"  FY{year}: {len(res[year])} sections")
            for name, subs in res[year].items():
                print(f"    {name!r}: {sum(len(b.split()) for b in subs.values())} words")
    else:
        print("Usage: python -m src.ingestion [--test | --dump-headings TICKER | --ingest TICKER]")