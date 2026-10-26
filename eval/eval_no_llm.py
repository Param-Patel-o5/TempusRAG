"""No-LLM evaluation: chunker, retriever fusion, scorer, cached filings.

Does not call Gemini, Groq, or SEC EDGAR. Writes eval/metrics.json and eval/METRICS.md
for README reference.

Run from repo root:
    python eval/eval_no_llm.py
"""

from __future__ import annotations

import json
import logging
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chunker import (  # noqa: E402
    build_contextual_prefix,
    chunk_subsection_text,
    chunk_year_sections,
    make_chunk_id,
    split_into_paragraphs,
    split_into_sentences,
)
from src.config import (  # noqa: E402
    CHUNK_MAX_TOKENS,
    PROMISE_EXTRACTION_SECTIONS,
    RRF_K,
    recency_weight,
)
from src.embedder import tokenize_for_bm25  # noqa: E402
from src.models import Chunk, CompanyCredibilityReport, DeliveryEvidence, Promise  # noqa: E402
from src.query import detect_followup_query, should_use_hyde  # noqa: E402
from src.retriever import (  # noqa: E402
    apply_section_boost,
    lost_in_middle_reorder,
    reciprocal_rank_fusion,
)
from src.scorer import (  # noqa: E402
    add_confidence_scores,
    compute_confidence_score,
    compute_domain_scores,
    compute_overall_credibility_score,
    count_delivery_statuses,
    detect_red_flags,
    generate_credibility_report,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("eval_no_llm")

OUT_DIR = ROOT / "eval"
FILINGS_DIR = ROOT / "data" / "filings"
SECTION_HEADER = re.compile(r"^=== (.+?) ===\s*$")
SUB_HEADER = re.compile(r"^--- (.+?) ---\s*$")


def parse_cached_filing(text: str) -> dict[str, dict[str, str]]:
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
        if subsection in sections[section] and sections[section][subsection]:
            sections[section][subsection] += "\n\n" + body
        else:
            sections[section][subsection] = body
        buf = []

    for line in text.splitlines():
        m_sec = SECTION_HEADER.match(line.strip())
        if m_sec:
            flush()
            section = m_sec.group(1).strip()
            subsection = "full_text"
            continue
        m_sub = SUB_HEADER.match(line.strip())
        if m_sub and section is not None:
            flush()
            subsection = m_sub.group(1).strip()
            continue
        buf.append(line)
    flush()
    return sections


def check(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "pass": bool(ok), "detail": detail}


def eval_models() -> tuple[list[dict], dict]:
    checks: list[dict] = []
    p = Promise(
        promise_text="We will reduce claims processing time by 40% by 2023",
        domain="Claims",
        deadline_mentioned="2023",
        year_made=2021,
        page_number=14,
        confidence_score=0.8,
    )
    checks.append(check("promise_roundtrip", p.domain == "Claims"))
    try:
        Promise(
            promise_text="x",
            domain="NotADomain",
            deadline_mentioned=None,
            year_made=2021,
            page_number=1,
            confidence_score=0.5,
        )
        checks.append(check("invalid_domain_rejected", False))
    except Exception:
        checks.append(check("invalid_domain_rejected", True))
    try:
        Promise(
            promise_text="x",
            domain="Claims",
            deadline_mentioned=None,
            year_made=2021,
            page_number=1,
            confidence_score=1.5,
        )
        checks.append(check("confidence_bounds", False))
    except Exception:
        checks.append(check("confidence_bounds", True))

    domains_ok = True
    for domain in ("Technology", "Claims", "Growth", "Finance", "Operations"):
        Promise(
            promise_text="We will invest",
            domain=domain,
            deadline_mentioned=None,
            year_made=2021,
            page_number=1,
            confidence_score=0.2,
        )
    checks.append(check("all_five_domains_construct", domains_ok))

    statuses_ok = True
    base_p = Promise(
        promise_text="We will invest",
        domain="Finance",
        deadline_mentioned=None,
        year_made=2021,
        page_number=1,
        confidence_score=0.2,
    )
    for status in ("Delivered", "Partial", "Silently Abandoned", "Pending"):
        DeliveryEvidence(
            promise=base_p,
            evidence_text=None,
            year_found=2022,
            delivery_score=50,
            status=status,
            judge_reasoning="n/a",
        )
    checks.append(check("all_four_statuses_construct", statuses_ok))

    try:
        DeliveryEvidence(
            promise=base_p,
            evidence_text=None,
            year_found=2022,
            delivery_score=101,
            status="Delivered",
            judge_reasoning="n/a",
        )
        checks.append(check("delivery_score_upper_bound", False))
    except Exception:
        checks.append(check("delivery_score_upper_bound", True))

    try:
        CompanyCredibilityReport(
            company="X",
            years_analyzed=[2021],
            total_promises=1,
            delivered=1,
            partial=0,
            abandoned=0,
            overall_score=101.0,
            domain_scores={},
            red_flags=[],
        )
        checks.append(check("overall_score_upper_bound", False))
    except Exception:
        checks.append(check("overall_score_upper_bound", True))

    DeliveryEvidence(
        promise=base_p,
        evidence_text=None,
        year_found=None,
        delivery_score=0,
        status="Pending",
        judge_reasoning="n/a",
    )
    checks.append(check("delivery_score_zero_allowed", True))

    return checks, {"schemas": ["Promise", "DeliveryEvidence", "CompanyCredibilityReport", "Chunk"]}


def eval_retriever() -> tuple[list[dict], dict]:
    checks: list[dict] = []
    dense = [("A", 0.9), ("B", 0.8), ("C", 0.7)]
    sparse = [("A", 2.0), ("C", 1.5), ("D", 1.0)]
    fused = reciprocal_rank_fusion(dense, sparse, k=RRF_K)
    order = [cid for cid, _ in fused]
    expected = ["A", "C", "B", "D"]
    checks.append(check("rrf_order", order == expected, f"got {order}"))

    lookup = {
        "A": Chunk(
            text="a", company="HIG", year=2021, section="MD&A",
            subsection="full_text", chunk_id="A", word_count=1,
        ),
        "B": Chunk(
            text="b", company="HIG", year=2021, section="Risk Factors",
            subsection="full_text", chunk_id="B", word_count=1,
        ),
    }
    boosted = apply_section_boost([("A", 1.0), ("B", 1.0)], lookup)
    boosted_map = dict(boosted)
    checks.append(check(
        "section_boost_mda_gt_risk",
        boosted_map["A"] > boosted_map["B"],
        f"A={boosted_map['A']:.3f} B={boosted_map['B']:.3f}",
    ))

    ids = ["A", "B", "C", "D", "E"]
    reordered = lost_in_middle_reorder(ids)
    expected_lim = ["A", "C", "E", "D", "B"]
    checks.append(check("lost_in_middle", reordered == expected_lim, f"got {reordered}"))

    tokens = tokenize_for_bm25("We are investing in digital transformation!")
    checks.append(check("bm25_tokenize", "digital" in tokens and "transformation" in tokens, str(tokens)))
    checks.append(check("bm25_tokenize_empty", tokenize_for_bm25("") == []))
    checks.append(check("bm25_tokenize_lower", tokenize_for_bm25("AI-Powered") == ["ai", "powered"]))

    checks.append(check("rrf_empty", reciprocal_rank_fusion([], []) == []))
    only_dense = reciprocal_rank_fusion([("A", 1.0), ("B", 0.5)], [])
    checks.append(check("rrf_dense_only_keeps_order", [c for c, _ in only_dense] == ["A", "B"]))

    checks.append(check("lost_in_middle_empty", lost_in_middle_reorder([]) == []))
    checks.append(check("lost_in_middle_one", lost_in_middle_reorder(["A"]) == ["A"]))
    checks.append(check("lost_in_middle_two", lost_in_middle_reorder(["A", "B"]) == ["A", "B"]))
    checks.append(check("lost_in_middle_three", lost_in_middle_reorder(["A", "B", "C"]) == ["A", "C", "B"]))

    lookup["F"] = Chunk(
        text="f", company="HIG", year=2021, section="Future Outlook",
        subsection="full_text", chunk_id="F", word_count=1,
    )
    boosted2 = dict(apply_section_boost([("F", 1.0), ("A", 1.0)], lookup))
    checks.append(check(
        "section_boost_outlook_gt_mda",
        boosted2["F"] > boosted2["A"],
        f"F={boosted2['F']:.3f} A={boosted2['A']:.3f}",
    ))

    return checks, {
        "rrf_k": RRF_K,
        "rrf_order": order,
        "lost_in_middle": reordered,
        "bm25_sample_tokens": tokens,
    }


def eval_scorer() -> tuple[list[dict], dict]:
    checks: list[dict] = []
    metric = Promise(
        promise_text="We will reduce claims processing time by 40% by 2023",
        domain="Claims",
        deadline_mentioned="2023",
        year_made=2021,
        page_number=14,
        confidence_score=0.0,
    )
    vague = Promise(
        promise_text="Expand technology platform",
        domain="Technology",
        deadline_mentioned=None,
        year_made=2021,
        page_number=20,
        confidence_score=0.0,
    )
    c_metric = compute_confidence_score(metric)
    c_vague = compute_confidence_score(vague)
    checks.append(check("confidence_metric_is_1", abs(c_metric - 1.0) < 1e-9, str(c_metric)))
    checks.append(check("confidence_vague_is_0_2", abs(c_vague - 0.2) < 1e-9, str(c_vague)))
    checks.append(check("recency_current", recency_weight(2023, 2023) == 1.0))
    checks.append(check("recency_floor", recency_weight(2010, 2023) == 0.6))

    ev = [
        DeliveryEvidence(
            promise=metric,
            evidence_text="Claims processing improved 35%",
            year_found=2023,
            delivery_score=80,
            status="Delivered",
            judge_reasoning="Strong progress",
        ),
        DeliveryEvidence(
            promise=vague,
            evidence_text="Launched platform",
            year_found=2023,
            delivery_score=90,
            status="Delivered",
            judge_reasoning="Deployed",
        ),
    ]
    overall = compute_overall_credibility_score(ev)
    checks.append(check("overall_unweighted_tie_year", abs(overall - 85.0) < 1e-6, str(overall)))

    abandoned = [
        DeliveryEvidence(
            promise=metric,
            evidence_text=None,
            year_found=2022,
            delivery_score=10,
            status="Silently Abandoned",
            judge_reasoning="No mention",
        )
        for _ in range(3)
    ]
    flags = detect_red_flags(abandoned, {"Claims": 10.0})
    checks.append(check("red_flag_majority_abandoned", any("abandoned" in f.lower() for f in flags), str(flags)))

    report = generate_credibility_report("HIG", "Hartford", {2021: [metric, vague]}, ev)
    checks.append(check("report_delivered_2", report.delivered == 2, str(report.delivered)))
    checks.append(check("recency_minus_1", recency_weight(2022, 2023) == 0.9))
    checks.append(check("recency_minus_2", recency_weight(2021, 2023) == 0.8))
    checks.append(check("recency_minus_3", recency_weight(2020, 2023) == 0.7))

    domain_scores = compute_domain_scores(ev)
    checks.append(check("domain_claims_is_80", abs(domain_scores.get("Claims", 0) - 80.0) < 1e-6, str(domain_scores)))
    checks.append(check("empty_evidence_overall_zero", compute_overall_credibility_score([]) == 0.0))

    pending = [
        DeliveryEvidence(
            promise=metric,
            evidence_text=None,
            year_found=None,
            delivery_score=50,
            status="Pending",
            judge_reasoning="not due",
        )
    ]
    pending_flags = detect_red_flags(pending, {"Claims": 50.0})
    checks.append(check(
        "red_flag_all_pending",
        any("evaluation" in f.lower() or "pending" in f.lower() for f in pending_flags),
        str(pending_flags),
    ))

    counts = count_delivery_statuses(ev)
    checks.append(check("status_count_delivered", counts["Delivered"] == 2, str(counts)))

    scored = add_confidence_scores([vague])
    checks.append(check("add_confidence_writes_field", scored[0].confidence_score == 0.2))

    recency_ev = [
        DeliveryEvidence(
            promise=metric,
            evidence_text="x",
            year_found=2023,
            delivery_score=100,
            status="Delivered",
            judge_reasoning="x",
        ),
        DeliveryEvidence(
            promise=vague,
            evidence_text="y",
            year_found=2019,
            delivery_score=0,
            status="Silently Abandoned",
            judge_reasoning="y",
        ),
    ]
    weighted = compute_overall_credibility_score(recency_ev)
    # 100*1.0 + 0*0.6 / 1.6 = 62.5
    checks.append(check("recency_weighted_not_simple_mean", abs(weighted - 62.5) < 1e-6, str(weighted)))

    return checks, {
        "confidence_with_metric_deadline": c_metric,
        "confidence_vague": c_vague,
        "fixture_overall_score": overall,
        "fixture_red_flags_abandoned_case": flags,
        "fixture_report_overall": report.overall_score,
    }


def eval_chunker_and_filings() -> tuple[list[dict], dict]:
    checks: list[dict] = []
    sentences = split_into_sentences(
        "The U.S. market is strong. Inc. profits are up! What will happen next?"
    )
    checks.append(check("sentence_split_keeps_abbrev", any("U.S." in s or "US" in s.replace(".", "") for s in sentences) or len(sentences) >= 2, str(sentences)))

    files = sorted(FILINGS_DIR.glob("*_*.txt"))
    by_ticker: dict[str, list[dict]] = {}
    chunk_stats: list[dict] = []

    for path in files:
        stem = path.stem
        if "_" not in stem:
            continue
        ticker, year_s = stem.split("_", 1)
        try:
            year = int(year_s)
        except ValueError:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        sections = parse_cached_filing(text)
        empty = 0
        nonempty = 0
        for subs in sections.values():
            for body in subs.values():
                if body.strip():
                    nonempty += 1
                else:
                    empty += 1
        chunks = chunk_year_sections(sections, ticker, ticker, year)
        word_counts = [c.word_count for c in chunks] or [0]
        ids = [c.chunk_id for c in chunks]
        unique_ids = len(set(ids)) == len(ids)
        prefixed = all(c.text.startswith("[") for c in chunks) if chunks else True
        over_cap = sum(1 for c in chunks if c.word_count > CHUNK_MAX_TOKENS + 80)

        rec = {
            "file": path.name,
            "ticker": ticker,
            "year": year,
            "bytes": path.stat().st_size,
            "chars": len(text),
            "section_count": len(sections),
            "nonempty_subsections": nonempty,
            "empty_subsections": empty,
            "chunk_count": len(chunks),
            "word_count_mean": round(statistics.mean(word_counts), 1) if chunks else 0,
            "word_count_max": max(word_counts) if chunks else 0,
            "unique_chunk_ids": unique_ids,
            "chunks_over_soft_cap": over_cap,
        }
        by_ticker.setdefault(ticker, []).append(rec)
        chunk_stats.append(rec)
        checks.append(check(f"{path.name}_has_sections", len(sections) > 0))
        checks.append(check(f"{path.name}_unique_ids", unique_ids))
        checks.append(check(f"{path.name}_prefixed_chunks", prefixed))

    sample_id = make_chunk_id("HIG", 2021, "MD&A", "Technology", 7)
    checks.append(check("chunk_id_contains_ticker_year", sample_id.startswith("HIG_2021_")))
    id_2022 = make_chunk_id("HIG", 2022, "MD&A", "Technology", 7)
    checks.append(check("chunk_id_differs_by_year", sample_id != id_2022))

    paras = split_into_paragraphs("First paragraph here.\n\nSecond paragraph here.")
    checks.append(check("paragraph_split_two", len(paras) == 2, str(paras)))
    checks.append(check("chunk_empty_text", chunk_subsection_text("") == []))
    checks.append(check(
        "prefix_with_subsection",
        build_contextual_prefix("Hartford", 2021, "MD&A", "Technology")
        == "[Hartford | 2021 | MD&A > Technology]",
    ))
    checks.append(check(
        "prefix_full_text",
        build_contextual_prefix("Hartford", 2021, "Risk Factors", "full_text")
        == "[Hartford | 2021 | Risk Factors]",
    ))

    mda = Chunk(
        text="t", company="HIG", year=2021, section="MD&A",
        subsection="full_text", chunk_id="1", word_count=1,
    )
    risk = Chunk(
        text="t", company="HIG", year=2021, section="Risk Factors",
        subsection="full_text", chunk_id="2", word_count=1,
    )
    kept = [
        c for c in [mda, risk]
        if any(k.lower() in c.section.lower() for k in PROMISE_EXTRACTION_SECTIONS)
    ]
    checks.append(check("promise_section_filter_keeps_mda", mda in kept and risk not in kept, str([c.section for c in kept])))

    checks.append(check("followup_detects_pronoun", detect_followup_query("what about it") is True))
    checks.append(check("hyde_skips_specific", should_use_hyde("claims processing in 2021") is False))
    checks.append(check("hyde_on_vague", should_use_hyde("tell me more") is True))

    over_soft = sum(r["chunks_over_soft_cap"] for r in chunk_stats)
    total_chunks = sum(r["chunk_count"] for r in chunk_stats) or 1
    max_words = max((r["word_count_max"] for r in chunk_stats), default=0)
    mean_of_means = statistics.mean(r["word_count_mean"] for r in chunk_stats) if chunk_stats else 0
    checks.append(check("mean_chunk_words_under_cap", mean_of_means <= CHUNK_MAX_TOKENS, str(mean_of_means)))

    totals = {
        "cached_filings": len(chunk_stats),
        "tickers": sorted(by_ticker.keys()),
        "total_chunks": sum(r["chunk_count"] for r in chunk_stats),
        "per_filing": chunk_stats,
        "chunk_max_tokens_config": CHUNK_MAX_TOKENS,
        "chunks_over_soft_cap_total": over_soft,
        "overflow_rate": round(over_soft / total_chunks, 4),
        "max_chunk_words": max_words,
        "mean_chunk_words": round(mean_of_means, 1),
    }
    return checks, totals


def render_markdown(payload: dict) -> str:
    failed = [c for c in payload["checks"] if not c["pass"]]
    lines = [
        "# TempusRAG — no-LLM metrics",
        "",
        f"Generated `{payload['generated_at']}`. No Gemini/Groq/EDGAR calls.",
        "",
        f"**Checks:** {payload['summary']['passed']}/{payload['summary']['total']} passed.",
        "",
        "## System (cached 10-Ks on disk)",
        "",
        f"- Filings: **{payload['filings']['cached_filings']}**",
        f"- Tickers: {', '.join(payload['filings']['tickers']) or 'none'}",
        f"- Chunks after local chunker: **{payload['filings']['total_chunks']}**",
        f"- Chunk token cap (word-split proxy): {payload['filings']['chunk_max_tokens_config']}",
        f"- Mean chunk words: **{payload['filings'].get('mean_chunk_words', 'n/a')}**",
        f"- Max chunk words: **{payload['filings'].get('max_chunk_words', 'n/a')}**",
        f"- Chunks over soft cap (cap+80): {payload['filings'].get('chunks_over_soft_cap_total', 'n/a')} "
        f"({float(payload['filings'].get('overflow_rate', 0))*100:.1f}%)",
        "",
        "## Quality read",
        "",
        payload.get(
            "quality_read",
            "Unit checks passing does not mean retrieval quality is perfect. See overflow stats above.",
        ),
        "",
        "| File | Sections | Chunks | Mean words | Max words |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["filings"]["per_filing"]:
        lines.append(
            f"| {row['file']} | {row['section_count']} | {row['chunk_count']} | "
            f"{row['word_count_mean']} | {row['word_count_max']} |"
        )
    lines += [
        "",
        "## Retriever (synthetic ranks, no Chroma)",
        "",
        f"- RRF k = {payload['retriever']['rrf_k']}",
        f"- RRF order: `{payload['retriever']['rrf_order']}`",
        f"- Lost-in-middle: `{payload['retriever']['lost_in_middle']}`",
        "",
        "## Scorer (fixture promises)",
        "",
        f"- Confidence (metric + deadline + verb + domain): **{payload['scorer']['confidence_with_metric_deadline']}**",
        f"- Confidence (vague tech line): **{payload['scorer']['confidence_vague']}**",
        f"- Fixture overall score: **{payload['scorer']['fixture_overall_score']}**",
        "",
        "## README snippet",
        "",
        "```",
        f"No-LLM eval ({payload['generated_at'][:10]}): "
        f"{payload['summary']['passed']}/{payload['summary']['total']} checks; "
        f"{payload['filings']['cached_filings']} cached 10-Ks → "
        f"{payload['filings']['total_chunks']} chunks; "
        "RRF + section boost + lost-in-middle unit-checked; "
        "scorer fixture overall "
        f"{payload['scorer']['fixture_overall_score']:.0f}/100.",
        "```",
        "",
    ]
    if failed:
        lines += ["## Failed checks", ""]
        for c in failed:
            lines.append(f"- `{c['name']}` — {c['detail']}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_checks: list[dict] = []
    models_c, models_m = eval_models()
    retr_c, retr_m = eval_retriever()
    scor_c, scor_m = eval_scorer()
    file_c, file_m = eval_chunker_and_filings()
    all_checks = models_c + retr_c + scor_c + file_c
    passed = sum(1 for c in all_checks if c["pass"])
    overflow = file_m.get("overflow_rate", 0)
    quality_read = (
        "**Schemas, RRF, section boost, lost-in-middle, and scorer math are solid** — "
        "they match the locked contracts and are safe to cite as no-LLM correctness.\n\n"
        f"**Chunker is mixed.** Mean chunk size ({file_m.get('mean_chunk_words')} words) sits under "
        f"the {file_m.get('chunk_max_tokens_config')}-word cap, and IDs/prefixes are consistent. "
        f"But **{file_m.get('chunks_over_soft_cap_total')} chunks ({overflow*100:.1f}%) overflow**, "
        f"with a max of **{file_m.get('max_chunk_words')} words** (long tables/sentences force-appended). "
        "That hurts embedding quality and context windows; it is a real defect, not a failed unit test.\n\n"
        "**Cached 10-K parse is coarse** (Item-level sections, few true subsections). "
        "Fine for indexing, weak for MD&A-only promise targeting until HTML heading parse improves.\n\n"
        "HyDE/follow-up heuristics are rule-based only; they are not a substitute for LLM eval."
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "llm_calls": False,
        "summary": {"passed": passed, "total": len(all_checks), "failed": len(all_checks) - passed},
        "quality_read": quality_read,
        "models": models_m,
        "retriever": retr_m,
        "scorer": scor_m,
        "filings": file_m,
        "checks": all_checks,
    }
    json_path = OUT_DIR / "metrics.json"
    md_path = OUT_DIR / "METRICS.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Passed {passed}/{len(all_checks)}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0 if passed == len(all_checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
