"""TempusRAG dashboard: corporate commitment intelligence over SEC 10-Ks."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import streamlit as st

from src.config import GEMINI_API_KEY, GROQ_API_KEY
from src.ingestion import load_company_directory
from src.models import CompanyCredibilityReport, DeliveryEvidence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Quiet noisy third-party loggers
for _noisy in ("httpx", "huggingface_hub", "sentence_transformers", "chromadb"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

st.set_page_config(
    page_title="TempusRAG",
    page_icon="■",
    layout="wide",
    initial_sidebar_state="collapsed",
)

STATUS_ORDER = ("Delivered", "Partial", "Silently Abandoned", "Pending")


def _start_background_warmup() -> None:
    """Pre-warm SentenceTransformer and PyTorch in background thread so Analyze starts instantly."""
    import threading

    def _warm() -> None:
        try:
            from src.embedder import _get_sentence_transformer
            _get_sentence_transformer()
        except Exception:
            pass

    threading.Thread(target=_warm, daemon=True).start()


def init_session_state() -> None:
    defaults: dict[str, Any] = {
        "ticker": "HIG",
        "company_name": "The Hartford",
        "issuer_query": "HIG",
        "report": None,
        "evidence": [],
        "conversation": [],
        "gemini_override_key": "",
        "groq_override_key": "",
        "use_custom_gemini": False,
        "use_custom_groq": False,
        "analysis_running": False,
        "selected_row": 0,
        "status_filter": "All",
        "elapsed_seconds": None,
        "_model_warmed": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if not st.session_state.get("_model_warmed"):
        st.session_state["_model_warmed"] = True
        _start_background_warmup()


def inject_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Instrument+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap');

html, body, [class*="css"], [data-testid="stAppViewContainer"], .stMarkdown, p, label {
    font-family: "Instrument Sans", "Segoe UI", sans-serif;
}

.stApp {
    background: #26251F;
    color: #F4F1EA;
}

#MainMenu, header[data-testid="stHeader"], footer, .stDeployButton {
    visibility: hidden;
    height: 0;
}

.block-container {
    padding-top: 1.25rem;
    padding-bottom: 2rem;
    max-width: 1280px;
}

.tr-masthead {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    border-bottom: 1px solid #4A463C;
    padding-bottom: 0.95rem;
    margin-bottom: 1.5rem;
}
.tr-wordmark {
    font-family: "Fraunces", "Times New Roman", serif;
    font-optical-sizing: auto;
    font-size: 1.55rem;
    font-weight: 550;
    letter-spacing: -0.02em;
    color: #D97757;
    margin: 0;
    line-height: 1.1;
}
.tr-sub {
    font-size: 0.88rem;
    color: #B5AFA3;
    margin: 0.35rem 0 0 0;
}
.tr-meta {
    font-size: 0.75rem;
    color: #B5AFA3;
    text-align: right;
}

.tr-score-wrap {
    border: 1px solid #4A463C;
    background: #32302A;
    padding: 1.35rem 1.5rem;
    margin-bottom: 1rem;
    border-radius: 10px;
}
.tr-score-label {
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #B5AFA3;
}
.tr-score {
    font-family: "Fraunces", "Times New Roman", serif;
    font-size: 3.4rem;
    font-weight: 550;
    line-height: 1;
    color: #F4F1EA;
    margin: 0.35rem 0 0.55rem 0;
}
.tr-score span {
    font-family: "Instrument Sans", sans-serif;
    font-size: 1rem;
    color: #B5AFA3;
}
.tr-thesis {
    font-size: 0.98rem;
    line-height: 1.55;
    color: #E4DFD4;
    margin: 0;
    max-width: 52rem;
}

.tr-kpis {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: #4A463C;
    border: 1px solid #4A463C;
    margin-bottom: 1.25rem;
    border-radius: 10px;
    overflow: hidden;
}
.tr-kpi {
    background: #32302A;
    padding: 0.95rem 1rem;
}
.tr-kpi .n {
    font-family: "Fraunces", serif;
    font-size: 1.55rem;
    color: #F4F1EA;
}
.tr-kpi .l {
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #B5AFA3;
    margin-top: 0.15rem;
}

.tr-flag {
    border-left: 2px solid #D97757;
    padding: 0.45rem 0.75rem;
    color: #E4DFD4;
    font-size: 0.88rem;
    margin-bottom: 0.4rem;
    background: #32302A;
    border-radius: 0 6px 6px 0;
}

.tr-cite {
    font-size: 0.75rem;
    color: #B5AFA3;
}

.tr-picked {
    font-size: 0.8rem;
    color: #B5AFA3;
    margin: 0.1rem 0 0.35rem 0;
}
.tr-picked strong {
    color: #F4F1EA;
    font-weight: 600;
}

.tr-stages {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem 0.55rem;
    padding: 0.75rem 0.9rem;
    background: #32302A;
    border: 1px solid #4A463C;
    border-radius: 10px;
    margin: 0.5rem 0 0.75rem 0;
}
.tr-stage {
    font-size: 0.78rem;
    color: #7A756C;
    letter-spacing: 0.04em;
}
.tr-stage.is-done {
    color: #B5AFA3;
}
.tr-stage.is-active {
    color: #D97757;
    font-weight: 600;
}

div[data-testid="stSidebar"] {
    background: #32302A;
}

.stButton > button[kind="primary"] {
    background: #D97757;
    color: #26251F;
    border: none;
    border-radius: 8px;
    font-family: "Instrument Sans", sans-serif;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.stButton > button[kind="secondary"] {
    background: transparent;
    color: #F4F1EA;
    border: 1px solid #4A463C;
    border-radius: 8px;
}

[data-testid="stMetricValue"] {
    font-family: "Fraunces", serif;
}

.tr-badge {
    display: inline-block;
    padding: 0.22rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.tr-badge-delivered {
    background: #064E3B;
    color: #34D399;
    border: 1px solid #059669;
}
.tr-badge-partial {
    background: #451A03;
    color: #FBBF24;
    border: 1px solid #D97706;
}
.tr-badge-abandoned {
    background: #450A0A;
    color: #F87171;
    border: 1px solid #DC2626;
}
.tr-badge-pending {
    background: #1E293B;
    color: #93C5FD;
    border: 1px solid #3B82F6;
}
.tr-badge-broken {
    background: #450A0A;
    color: #EF4444;
    border: 1px solid #DC2626;
}

.tr-detail-card {
    background: #32302A;
    border: 1px solid #4A463C;
    border-radius: 10px;
    padding: 1.15rem;
    margin-top: 0.85rem;
    margin-bottom: 0.85rem;
}
.tr-detail-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.65rem;
}
.tr-detail-score {
    font-size: 0.85rem;
    color: #B5AFA3;
}
.tr-detail-score strong {
    color: #F4F1EA;
    font-size: 1.05rem;
}
.tr-detail-quote {
    font-family: "Fraunces", Georgia, serif;
    font-size: 1.05rem;
    line-height: 1.45;
    color: #F4F1EA;
    margin-bottom: 0.5rem;
}
.tr-detail-meta {
    font-size: 0.78rem;
    color: #B5AFA3;
}

.tr-reasoning-card {
    background: #2D2B24;
    border-left: 3px solid #D97757;
    border-radius: 0 8px 8px 0;
    padding: 0.85rem 1rem;
    margin-bottom: 1rem;
}
.tr-card-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #D97757;
    margin-bottom: 0.35rem;
}
.tr-reasoning-text {
    font-size: 0.88rem;
    line-height: 1.5;
    color: #E4DFD4;
}

.tr-evidence-card {
    background: #2B2923;
    border: 1px solid #454137;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.65rem;
}
.tr-evidence-badge {
    font-size: 0.7rem;
    font-weight: 600;
    color: #C2BAAC;
    letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
}
.tr-evidence-text {
    font-size: 0.84rem;
    line-height: 1.5;
    color: #E0DDD5;
    white-space: pre-wrap;
    word-break: break-word;
}

/* Filing Q&A Chat Message Styling */
[data-testid="stChatMessage"] {
    background: #32302A !important;
    border: 1px solid #4A463C !important;
    border-radius: 8px !important;
    padding: 0.75rem 1rem !important;
    margin-bottom: 0.75rem !important;
}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    color: #F4F1EA !important;
    font-size: 0.92rem !important;
    line-height: 1.5 !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False, ttl=3600)
def _cached_issuer_directory() -> list[dict[str, str]]:
    data = load_company_directory()
    if not data:
        raise RuntimeError("Empty issuer directory returned from EDGAR")
    return data


def issuer_directory() -> list[dict[str, str]]:
    """
    Load the SEC issuer list once per hour and cache it.
    Does not cache failed empty responses so subsequent retries succeed.
    """
    try:
        return _cached_issuer_directory()
    except Exception as exc:
        logger.warning("Could not load SEC issuer directory: %s", exc)
        return []


def search_issuers(query: str, directory: list[dict[str, str]], limit: int = 8) -> list[dict[str, str]]:
    q = (query or "").strip().lower()
    if not q:
        return []

    exact_ticker: list[dict[str, str]] = []
    ticker_prefix: list[dict[str, str]] = []
    name_prefix: list[dict[str, str]] = []
    name_contains: list[dict[str, str]] = []
    ticker_contains: list[dict[str, str]] = []

    for row in directory:
        ticker = row["ticker"]
        title = row["title"]
        t_low = ticker.lower()
        n_low = title.lower()
        if t_low == q:
            exact_ticker.append(row)
        elif t_low.startswith(q):
            ticker_prefix.append(row)
        elif n_low.startswith(q):
            name_prefix.append(row)
        elif q in n_low:
            name_contains.append(row)
        elif q in t_low:
            ticker_contains.append(row)

    ranked = exact_ticker + ticker_prefix + name_prefix + name_contains + ticker_contains
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for row in ranked:
        if row["ticker"] in seen:
            continue
        seen.add(row["ticker"])
        unique.append(row)
        if len(unique) >= limit:
            break
    return unique


def resolve_issuer(query: str, directory: list[dict[str, str]]) -> dict[str, str] | None:
    matches = search_issuers(query, directory, limit=8)
    if not matches:
        return None
    q = (query or "").strip().upper()
    for row in matches:
        if row["ticker"] == q:
            return row
    return matches[0]


def get_active_api_keys() -> tuple[str, str]:
    gemini = (
        st.session_state.gemini_override_key
        if st.session_state.use_custom_gemini and st.session_state.gemini_override_key
        else GEMINI_API_KEY
    )
    groq = (
        st.session_state.groq_override_key
        if st.session_state.use_custom_groq and st.session_state.groq_override_key
        else GROQ_API_KEY
    )
    return gemini or "", groq or ""


def key_label(present: bool, custom: bool) -> str:
    if not present:
        return "missing"
    return "session" if custom else "configured"


def render_sidebar() -> None:
    st.sidebar.markdown("**Credentials**")
    st.sidebar.caption("Keys stay in this browser session. They are not written to disk.")

    gemini_key, groq_key = get_active_api_keys()
    st.sidebar.text(
        f"Gemini  {key_label(bool(gemini_key), st.session_state.use_custom_gemini)}\n"
        f"Groq    {key_label(bool(groq_key), st.session_state.use_custom_groq)}"
    )

    st.session_state.use_custom_gemini = st.sidebar.checkbox(
        "Override Gemini key", value=st.session_state.use_custom_gemini
    )
    if st.session_state.use_custom_gemini:
        st.session_state.gemini_override_key = st.sidebar.text_input(
            "Gemini API key",
            type="password",
            value=st.session_state.gemini_override_key,
        )

    st.session_state.use_custom_groq = st.sidebar.checkbox(
        "Override Groq key", value=st.session_state.use_custom_groq
    )
    if st.session_state.use_custom_groq:
        st.session_state.groq_override_key = st.sidebar.text_input(
            "Groq API key",
            type="password",
            value=st.session_state.groq_override_key,
        )

    st.sidebar.divider()
    st.sidebar.caption(
        "Extraction uses Gemini. Cross-year judgment uses Groq. "
        "Free tiers are rate-limited; cached filings skip a full re-ingest."
    )


PIPELINE_STAGES = ("Validate", "Ingest", "Chunk", "Embed", "Extract", "Judge", "Score", "Done")


def stages_html(current: str) -> str:
    reached = True
    parts = ['<div class="tr-stages">']
    for name in PIPELINE_STAGES:
        cls = "tr-stage"
        if name == current:
            cls += " is-active"
            reached = False
        elif reached:
            cls += " is-done"
        parts.append(f'<span class="{cls}">{name}</span>')
    parts.append("</div>")
    return "".join(parts)


def classify_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg or "quota" in msg:
        return (
            "Provider rate limit reached. Wait and retry, or paste your own keys "
            "in the sidebar. Cached filings are reused when available."
        )
    if "401" in msg or "unauthorized" in msg or "api key" in msg:
        return "API key rejected. Check GEMINI_API_KEY / GROQ_API_KEY or the sidebar overrides."
    if "not found" in msg and "ticker" in msg:
        return str(exc)
    if "network" in msg or "connection" in msg or "edgar" in msg:
        return f"Upstream request failed: {exc}"
    return f"Analysis failed: {exc}"


def run_analysis(ticker: str) -> None:
    gemini_key, groq_key = get_active_api_keys()
    if not gemini_key:
        st.error("Gemini key required for promise extraction. Set GEMINI_API_KEY or use the sidebar.")
        return
    if not groq_key:
        st.error("Groq key required for delivery judgment. Set GROQ_API_KEY or use the sidebar.")
        return

    st.session_state.analysis_running = True
    rerun_needed = False
    try:
        stage_box = st.empty()
        stage_box.markdown(stages_html("Validate"), unsafe_allow_html=True)

        def on_stage(label: str) -> None:
            stage_box.markdown(stages_html(label), unsafe_allow_html=True)

        from src.pipeline import get_last_artifacts, run_tempusrag_pipeline

        report = run_tempusrag_pipeline(
            ticker=ticker,
            company_display_name=st.session_state.get("company_name") or ticker,
            gemini_api_key=gemini_key,
            groq_api_key=groq_key,
            force_reingest=False,
            on_stage=on_stage,
        )
        artifacts = get_last_artifacts(ticker) or {}
        stage_box.markdown(stages_html("Done"), unsafe_allow_html=True)

        st.session_state.report = report
        st.session_state.evidence = artifacts.get("delivery_evidence") or []
        st.session_state.elapsed_seconds = artifacts.get("elapsed_seconds")
        st.session_state.conversation = []
        st.session_state.selected_row = 0
        rerun_needed = True
    except Exception as exc:
        logger.exception("Pipeline failed for %s", ticker)
        st.error(classify_error(exc))
    finally:
        st.session_state.analysis_running = False

    # st.rerun() must be called AFTER the finally block so analysis_running
    # is always reset before the rerun is triggered.
    if rerun_needed:
        st.rerun()


def evidence_rows(evidence: list[DeliveryEvidence]) -> list[dict[str, Any]]:
    rows = []
    for item in evidence:
        rows.append(
            {
                "Year": item.promise.year_made,
                "Domain": item.promise.domain,
                "Status": item.status,
                "Score": item.delivery_score,
                "Deadline": item.promise.deadline_mentioned or "N/A",
                "Found": item.year_found if item.year_found is not None else "N/A",
                "Commitment": item.promise.promise_text,
                "Evidence": item.evidence_text or "N/A",
                "Reasoning": item.judge_reasoning,
                "Confidence": item.promise.confidence_score if item.promise.confidence_score is not None else "N/A",
                "Page": item.promise.page_number,
            }
        )
    return rows


def render_masthead(report: CompanyCredibilityReport | None) -> None:
    years = ""
    elapsed = ""
    if report:
        years = " · ".join(str(y) for y in report.years_analyzed) or "N/A"
        if st.session_state.elapsed_seconds is not None:
            elapsed = f"{st.session_state.elapsed_seconds:.0f}s last run"
    st.markdown(
        f"""
<div class="tr-masthead">
  <div>
    <p class="tr-wordmark">TempusRAG</p>
    <p class="tr-sub">Executive commitment tracking from temporally indexed 10-K filings</p>
  </div>
  <div class="tr-meta">{years}<br>{elapsed}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_toolbar() -> None:
    # -- issuer directory ---------------------------------------------------
    # Load lazily so the page renders immediately on first visit.
    # On the very first run session_state["_dir_loaded"] is False; we kick off
    # the (cached) directory load and rerun so the selectbox appears on the
    # second render cycle, which is instant because the resource is now cached.
    directory: list[dict[str, str]] = []
    directory = issuer_directory()
    if directory:
        st.session_state["_dir_loaded"] = True
        st.session_state.pop("_directory_error", None)
    else:
        st.session_state["_directory_error"] = "SEC issuer list unavailable"

    if "issuer_search_box" not in st.session_state:
        st.session_state.issuer_search_box = st.session_state.issuer_query

    search_col, btn_col = st.columns([4.2, 1.0])
    with search_col:
        st.text_input(
            "Issuer",
            placeholder="Company name or ticker: Apple, Hartford, AAPL, HIG",
            label_visibility="collapsed",
            key="issuer_search_box",
        )
        st.session_state.issuer_query = st.session_state.issuer_search_box or ""

    matches = search_issuers(st.session_state.issuer_query, directory, limit=8)

    if directory and st.session_state.issuer_query.strip() and matches:
        labels = [f"{m['ticker']} : {m['title']}" for m in matches]
        current_label = None
        for label, row in zip(labels, matches):
            if row["ticker"] == st.session_state.ticker:
                current_label = label
                break
        index = labels.index(current_label) if current_label in labels else 0
        picked = st.selectbox(
            "Matching issuers",
            labels,
            index=index,
            help="Resolved from the SEC issuer list.",
        )
        chosen = matches[labels.index(picked)]
        st.session_state.ticker = chosen["ticker"]
        st.session_state.company_name = chosen["title"]
    elif st.session_state.issuer_query.strip() and not matches:
        st.caption("No listed match." if directory else "Issuer directory not available: enter a ticker directly.")
    else:
        resolved = resolve_issuer(st.session_state.ticker, directory)
        if resolved:
            st.session_state.company_name = resolved["title"]

    st.markdown(
        f'<p class="tr-picked"><strong>{st.session_state.ticker}</strong>'
        f" · {st.session_state.company_name}</p>",
        unsafe_allow_html=True,
    )

    with btn_col:
        analyze = st.button(
            "Analyze",
            type="primary",
            disabled=st.session_state.analysis_running,
            width="stretch",
        )

    if st.session_state.get("_directory_error") and not directory:
        st.warning(
            "Could not load the SEC issuer directory. You can still enter a ticker directly. "
            f"({st.session_state['_directory_error']})"
        )

    if analyze:
        ticker = st.session_state.ticker
        if not re.match(r"^[A-Z][A-Z0-9.-]{0,9}$", ticker):
            fallback = resolve_issuer(st.session_state.issuer_query, directory)
            if fallback:
                ticker = fallback["ticker"]
                st.session_state.ticker = ticker
                st.session_state.company_name = fallback["title"]
            else:
                st.error("Select a matching issuer from the list, or enter a listed ticker.")
                return
        run_analysis(ticker)


def render_hero(report: CompanyCredibilityReport) -> None:
    total = report.total_promises or 0
    delivered_pct = (report.delivered / total * 100) if total else 0.0
    years_n = len(report.years_analyzed)

    if report.insufficient_data:
        thesis = (
            f"No forward-looking commitments were extracted from {report.company}'s "
            f"filings across {years_n} fiscal year{'s' if years_n != 1 else ''}. "
            "The pipeline ran successfully but found no scorable promises; "
            "this may indicate a section-detection issue or that the filings are "
            "primarily backward-looking. Check the logs for details."
        )
        score_display = "N/A"
    else:
        thesis = (
            f"{report.company} delivered on {report.delivered} of {total} extracted forward-looking "
            f"commitments across {years_n} fiscal year{'s' if years_n != 1 else ''} "
            f"({delivered_pct:.0f}% delivered). "
            f"{len(report.red_flags)} risk flag{'s' if len(report.red_flags) != 1 else ''}."
        )
        score_display = f"{report.overall_score:.0f}"

    st.markdown(
        f"""
<div class="tr-score-wrap">
  <div class="tr-score-label">Credibility score</div>
  <div class="tr-score">{score_display}<span> / 100</span></div>
  <p class="tr-thesis">{thesis}</p>
</div>
<div class="tr-kpis">
  <div class="tr-kpi"><div class="n">{report.delivered}</div><div class="l">Delivered</div></div>
  <div class="tr-kpi"><div class="n">{report.partial}</div><div class="l">Partial</div></div>
  <div class="tr-kpi"><div class="n">{report.abandoned}</div><div class="l">Abandoned</div></div>
  <div class="tr-kpi"><div class="n">{total}</div><div class="l">Commitments</div></div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_domains_and_flags(report: CompanyCredibilityReport) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown("**Domain scores**")
        domains = {k: v for k, v in report.domain_scores.items() if v is not None}
        if not domains:
            st.caption("No domain scores yet.")
        else:
            for name, score in sorted(domains.items(), key=lambda x: x[1], reverse=True):
                st.progress(min(max(score / 100.0, 0.0), 1.0), text=f"{name}  {score:.0f}")
    with right:
        st.markdown("**Flags**")
        if not report.red_flags:
            st.caption("None raised by the rule layer.")
        else:
            for flag in report.red_flags:
                st.markdown(f'<div class="tr-flag">{flag}</div>', unsafe_allow_html=True)


def parse_evidence_chunks(raw_evidence: str, status: str = "", year_made: int = 0) -> list[tuple[str, str]]:
    """Parse combined evidence string into list of (header, text) tuples."""
    cleaned = (raw_evidence or "").strip()
    if not cleaned or cleaned in ("N/A", "-", "None"):
        if status == "Pending":
            return [("Status Context", f"Pending evaluation: Commitment was made in FY {year_made} (the most recent analyzed fiscal year). Delivery will be assessed against future 10-K filings.")]
        if "abandoned" in status.lower():
            return [("Status Context", "No confirming evidence found in subsequent 10-K filings across MD&A, Business, or Outlook sections.")]
        return []

    pattern = re.compile(
        r"(\[(?:FY\s*)?\d{4}\s*\|[^\]]+\]|\[[^\]]+\|\s*\d{4}\s*\|[^\]]+\]|--- Chunk \d+ ---)",
        re.IGNORECASE,
    )
    splits = pattern.split(cleaned)
    if len(splits) > 1:
        chunks = []
        i = 1
        while i < len(splits):
            hdr = splits[i].strip().strip("[]- ")
            body = splits[i + 1].strip() if (i + 1) < len(splits) else ""
            if body:
                chunks.append((hdr, body))
            i += 2
        if chunks:
            return chunks

    if "\n\n---\n\n" in cleaned or "\n---\n" in cleaned:
        raw_parts = [p.strip() for p in re.split(r"\n+---+\n+", cleaned) if p.strip()]
        if len(raw_parts) > 1:
            return [(f"Excerpt {idx + 1}", part) for idx, part in enumerate(raw_parts)]

    return [("10-K Evidence Context", cleaned)]


def render_commitments() -> None:
    evidence: list[DeliveryEvidence] = st.session_state.evidence or []
    st.markdown("**Commitments**")
    if not evidence:
        st.caption(
            "Row-level judgments were not returned for this run. "
            "Headline counts above still come from the scored report."
        )
        return

    rows = evidence_rows(evidence)
    statuses = ["All", *STATUS_ORDER]
    filter_col, _ = st.columns([1.2, 3])
    with filter_col:
        choice = st.selectbox("Status", statuses, label_visibility="collapsed")
        st.session_state.status_filter = choice

    if choice != "All":
        rows = [r for r in rows if r["Status"] == choice]

    display = [
        {
            "Year": r["Year"],
            "Domain": r["Domain"],
            "Status": r["Status"],
            "Score": r["Score"],
            "Deadline": r["Deadline"],
            "Commitment": r["Commitment"],
        }
        for r in rows
    ]
    event = st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        height=360,
        column_config={
            "Year": st.column_config.NumberColumn("Year", format="%d", width="small"),
            "Domain": st.column_config.TextColumn("Domain", width="small"),
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
            "Deadline": st.column_config.TextColumn("Deadline", width="small"),
            "Commitment": st.column_config.TextColumn("Commitment", width="large"),
        },
    )

    selected = 0
    try:
        selected_rows = event.selection.rows  # type: ignore[attr-defined]
        if selected_rows:
            selected = int(selected_rows[0])
    except Exception:
        selected = 0

    if not rows:
        st.caption("No commitments in this filter.")
        return

    selected = min(selected, len(rows) - 1)
    detail = rows[selected]

    status = detail["Status"]
    score = detail["Score"]
    status_key = status.lower()
    if "delivered" in status_key:
        badge_cls = "tr-badge-delivered"
    elif "partial" in status_key:
        badge_cls = "tr-badge-partial"
    elif "abandoned" in status_key:
        badge_cls = "tr-badge-abandoned"
    elif "pending" in status_key:
        badge_cls = "tr-badge-pending"
    else:
        badge_cls = "tr-badge-pending"

    st.markdown(
        f"""
<div class="tr-detail-card">
  <div class="tr-detail-header">
    <span class="tr-badge {badge_cls}">{status}</span>
    <span class="tr-detail-score">Delivery Score: <strong>{score}</strong> / 100</span>
  </div>
  <div class="tr-detail-quote">"{detail['Commitment']}"</div>
  <div class="tr-detail-meta">
    {detail['Domain']} · Made FY {detail['Year']} · Target Deadline: {detail['Deadline']} · Found FY: {detail['Found']} · Page: {detail['Page']}
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="tr-reasoning-card">
  <div class="tr-card-title">LLM Judge Reasoning</div>
  <div class="tr-reasoning-text">{detail['Reasoning']}</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**10-K Evidence Context**")
    evidence_items = parse_evidence_chunks(
        detail["Evidence"],
        status=detail["Status"],
        year_made=detail["Year"],
    )
    if not evidence_items:
        st.caption("No evidence chunks available.")
    else:
        for header, body in evidence_items:
            st.markdown(
                f"""
<div class="tr-evidence-card">
  <div class="tr-evidence-badge">{header}</div>
  <div class="tr-evidence-text">{body}</div>
</div>
                """,
                unsafe_allow_html=True,
            )


def conversation_as_strings() -> list[str]:
    out: list[str] = []
    for message in st.session_state.conversation:
        role = message.get("role", "")
        content = message.get("content", "")
        out.append(f"{role}: {content}")
    return out


def format_chunk_cite(chunk: Any) -> str:
    year = getattr(chunk, "year", "?")
    section = getattr(chunk, "section", "")
    subsection = getattr(chunk, "subsection", "")
    path = " > ".join(p for p in (section, subsection) if p)
    return f"{year} {path}".strip()


def render_ask() -> None:
    st.markdown("**Filing Q&A**")
    st.caption("Ask after the report: answers cite retrieved 10-K chunks.")

    if not st.session_state.conversation:
        st.caption("No questions yet.")
        return

    for message in st.session_state.conversation:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            cites = message.get("cites") or []
            if cites:
                st.markdown(
                    '<p class="tr-cite">Sources: ' + " · ".join(cites) + "</p>",
                    unsafe_allow_html=True,
                )


def handle_chat_prompt(prompt: str) -> None:
    st.session_state.conversation.append({"role": "user", "content": prompt})
    gemini_key, _ = get_active_api_keys()
    try:
        from src.query import process_query
        with st.spinner("Analyzing filings and generating answer..."):
            result = process_query(
                query=prompt,
                ticker=st.session_state.ticker,
                conversation_history=conversation_as_strings()[:-1],
                api_key=gemini_key,
            )
        chunks = result.get("source_chunks") or []
        cites = [format_chunk_cite(c) for c in chunks[:5]]
        st.session_state.conversation.append(
            {
                "role": "assistant",
                "content": result.get("answer") or "No answer generated.",
                "cites": cites,
            }
        )
    except Exception as exc:
        logger.exception("Query failed")
        st.session_state.conversation.append(
            {"role": "assistant", "content": classify_error(exc)}
        )
    st.rerun()


def render_empty() -> None:
    st.caption("Demo: HIG. Score is an LLM judge, not an audit.")


def main() -> None:
    init_session_state()
    inject_css()
    render_sidebar()
    render_masthead(st.session_state.report)
    render_toolbar()

    report = st.session_state.report
    if report is None:
        render_empty()
        return

    render_hero(report)
    render_domains_and_flags(report)
    st.divider()
    table_col, ask_col = st.columns([1.55, 1])
    with table_col:
        render_commitments()
    with ask_col:
        render_ask()

    prompt = st.chat_input("Ask about this issuer's filings")
    if prompt:
        handle_chat_prompt(prompt)


if __name__ == "__main__":
    main()
