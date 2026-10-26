"""LLM eval vs eval/llm_gold.json. Default is offline (no Gemini/Groq).

Run from repo root:
    python eval/eval_llm.py              # validate gold + optional replay of last run
    python eval/eval_llm.py --live       # Gemini extract + Groq judge; archive JSON
    python eval/eval_llm.py --replay PATH

Do not put --live in CI. Rate-limit check is mocked (no quality claim).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import GEMINI_API_KEY, GROQ_API_KEY, MAX_RETRIES  # noqa: E402
from src.extractor import (  # noqa: E402
    _call_gemini,
    extract_promises_for_year,
)
from src.rate_limit import get_rate_limiter  # noqa: E402
from src.models import Chunk, Promise  # noqa: E402
from src.reasoner import judge_single_promise  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("eval_llm")

EXPANDED_GOLD_PATH = ROOT / "eval" / "llm_gold_expanded.json"
DEFAULT_GOLD_PATH = ROOT / "eval" / "llm_gold.json"
GOLD_PATH = EXPANDED_GOLD_PATH if EXPANDED_GOLD_PATH.exists() else DEFAULT_GOLD_PATH
RUNS_DIR = ROOT / "eval" / "llm_runs"
METRICS_PATH = ROOT / "eval" / "llm_metrics.json"
DOMAINS = {"Technology", "Claims", "Growth", "Finance", "Operations"}
STATUSES = {"Delivered", "Partial", "Silently Abandoned", "Pending"}
BANDS = {"0-30", "31-60", "61-85", "86-100"}
ADJACENT = {
    frozenset({"Delivered", "Partial"}),
    frozenset({"Partial", "Pending"}),
    frozenset({"Partial", "Silently Abandoned"}),
}


def load_gold(path: Path | None = None) -> dict:
    target = path or GOLD_PATH
    return json.loads(target.read_text(encoding="utf-8"))


def band_for_score(score: int) -> str:
    if score <= 30:
        return "0-30"
    if score <= 60:
        return "31-60"
    if score <= 85:
        return "61-85"
    return "86-100"


def make_chunk(excerpt: str, year: int, item_id: str) -> Chunk:
    return Chunk(
        text=excerpt,
        company="HIG",
        year=year,
        section="MD&A",
        subsection="eval_gold",
        chunk_id=f"gold_{item_id}",
        word_count=len(excerpt.split()),
    )


def validate_gold(gold: dict) -> list[dict]:
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    extract = gold.get("extract_gold") or []
    judge = gold.get("judge_gold") or []
    qa = gold.get("qa_gold") or []
    add("extract_count_min_20", len(extract) >= 20, str(len(extract)))
    add("judge_count_min_15", len(judge) >= 15, str(len(judge)))
    add("qa_count_10", len(qa) == 10, str(len(qa)))
    none_n = sum(1 for item in extract if not item.get("gold"))
    add("at_least_3_no_promise", none_n >= 3, str(none_n))
    dl_n = sum(1 for item in extract for g in item.get("gold") or [] if g.get("has_deadline"))
    add("at_least_3_with_deadline", dl_n >= 3, str(dl_n))
    add("labeling_rules_present", bool(gold.get("labeling_rules")))

    for item in extract:
        eid = item.get("id", "?")
        words = len((item.get("excerpt") or "").split())
        add(f"{eid}_excerpt_nonempty", words >= 20, f"words={words}")
        add(f"{eid}_excerpt_le_400_words", words <= 400, f"words={words}")
        add(f"{eid}_year_int", isinstance(item.get("year"), int))
        for g in item.get("gold") or []:
            add(f"{eid}_domain_ok", g.get("domain") in DOMAINS, str(g.get("domain")))
            add(f"{eid}_deadline_bool", isinstance(g.get("has_deadline"), bool))

    for item in judge:
        jid = item.get("id", "?")
        add(f"{jid}_status_ok", item.get("gold_status") in STATUSES, str(item.get("gold_status")))
        add(f"{jid}_band_ok", item.get("gold_score_band") in BANDS, str(item.get("gold_score_band")))
        add(f"{jid}_promise_text", bool(item.get("promise_text")))
        add(f"{jid}_evidence", bool(item.get("evidence_excerpt")))

    for item in qa:
        qid = item.get("id", "?")
        add(f"{qid}_question", bool(item.get("question")))
        add(f"{qid}_year", isinstance(item.get("relevant_year"), int))

    return checks


def gold_has_promise(item: dict) -> bool:
    return bool(item.get("gold"))


def gold_primary_domain(item: dict) -> str | None:
    gold = item.get("gold") or []
    return gold[0]["domain"] if gold else None


def gold_has_deadline(item: dict) -> bool:
    return any(g.get("has_deadline") for g in item.get("gold") or [])


def pred_has_deadline(promises: list[dict]) -> bool:
    for p in promises:
        val = p.get("deadline_mentioned")
        if val is not None and str(val).strip() and str(val).strip().lower() not in {"null", "none"}:
            return True
    return False


def is_transport_error(pred: dict) -> bool:
    err = (pred.get("error") or "").lower()
    return "429" in err or "quota" in err or "rate limit" in err


def score_extract(gold_items: list[dict], predictions: dict[str, dict]) -> dict:
    tp = fp = fn = tn = 0
    domain_ok = domain_n = 0
    deadline_ok = deadline_n = 0
    parse_ok = 0
    parse_n = 0
    n = 0
    skipped_quota = 0
    rows = []
    for item in gold_items:
        n += 1
        pred = predictions.get(item["id"]) or {}
        if is_transport_error(pred):
            skipped_quota += 1
            rows.append(
                {
                    "id": item["id"],
                    "gold_has_promise": gold_has_promise(item),
                    "pred_has_promise": None,
                    "json_parse_ok": None,
                    "n_pred": 0,
                    "skipped": "quota",
                }
            )
            continue
        parsed = bool(pred.get("json_parse_ok"))
        parse_n += 1
        parse_ok += int(parsed)
        promises = pred.get("promises") or []
        pred_pos = bool(promises)
        gold_pos = gold_has_promise(item)
        if gold_pos and pred_pos:
            tp += 1
        elif gold_pos and not pred_pos:
            fn += 1
        elif not gold_pos and pred_pos:
            fp += 1
        else:
            tn += 1
        if gold_pos and pred_pos:
            domain_n += 1
            gdom = gold_primary_domain(item)
            pdoms = {p.get("domain") for p in promises}
            if gdom in pdoms:
                domain_ok += 1
            deadline_n += 1
            if gold_has_deadline(item) == pred_has_deadline(promises):
                deadline_ok += 1
        rows.append(
            {
                "id": item["id"],
                "gold_has_promise": gold_pos,
                "pred_has_promise": pred_pos,
                "json_parse_ok": parsed,
                "n_pred": len(promises),
            }
        )
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "n": n,
        "n_scored": parse_n,
        "skipped_quota": skipped_quota,
        "json_parse_success_rate": parse_ok / parse_n if parse_n else 0.0,
        "promise_precision": round(precision, 4),
        "promise_recall": round(recall, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "domain_accuracy": round(domain_ok / domain_n, 4) if domain_n else None,
        "deadline_accuracy": round(deadline_ok / deadline_n, 4) if deadline_n else None,
        "rows": rows,
    }


def score_judge(gold_items: list[dict], predictions: dict[str, dict]) -> dict:
    n = 0
    exact = 0
    adjacent = 0
    far = 0
    parse_ok = 0
    band_ok = 0
    band_n = 0
    rows = []
    
    # Ordered labels for 4-way classification
    labels = sorted(STATUSES)  # ['Delivered', 'Partial', 'Pending', 'Silently Abandoned']
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    cm = [[0 for _ in labels] for _ in labels]  # cm[gold_idx][pred_idx]

    gold_counts: dict[str, int] = {lbl: 0 for lbl in labels}
    pred_counts: dict[str, int] = {lbl: 0 for lbl in labels}

    for item in gold_items:
        n += 1
        pred = predictions.get(item["id"]) or {}
        parsed = bool(pred.get("json_parse_ok"))
        parse_ok += int(parsed)
        gold_s = item["gold_status"]
        pred_s = pred.get("status")

        if gold_s in label_to_idx:
            gold_counts[gold_s] += 1

        if pred_s in label_to_idx:
            pred_counts[pred_s] += 1
            if gold_s in label_to_idx:
                cm[label_to_idx[gold_s]][label_to_idx[pred_s]] += 1

        if pred_s == gold_s:
            exact += 1
            kind = "exact"
        elif pred_s in STATUSES and frozenset({pred_s, gold_s}) in ADJACENT:
            adjacent += 1
            kind = "adjacent"
        else:
            far += 1
            kind = "far"

        score = pred.get("delivery_score")
        band_match = None
        if isinstance(score, int):
            band_n += 1
            band_match = band_for_score(score) == item.get("gold_score_band")
            band_ok += int(band_match)

        rows.append(
            {
                "id": item["id"],
                "gold_status": gold_s,
                "pred_status": pred_s,
                "match": kind,
                "delivery_score": score,
                "gold_score_band": item.get("gold_score_band"),
                "band_match": band_match,
                "json_parse_ok": parsed,
            }
        )

    # Calculate per-class metrics
    per_class: dict[str, dict[str, float | int]] = {}
    f1_list: list[float] = []
    total_valid = sum(gold_counts.values())

    for idx, lbl in enumerate(labels):
        tp = cm[idx][idx]
        fn = sum(cm[idx][j] for j in range(len(labels)) if j != idx)
        fp = sum(cm[i][idx] for i in range(len(labels)) if i != idx)
        support = gold_counts[lbl]

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[lbl] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        f1_list.append(f1)

    macro_f1 = round(sum(f1_list) / len(labels), 4) if labels else 0.0
    weighted_f1 = (
        round(sum(per_class[lbl]["f1"] * gold_counts[lbl] for lbl in labels) / total_valid, 4)
        if total_valid > 0
        else 0.0
    )

    # Cohen's Kappa: (Po - Pe) / (1 - Pe)
    po = exact / n if n else 0.0
    pe = sum((gold_counts[lbl] / n) * (pred_counts[lbl] / n) for lbl in labels) if n else 0.0
    kappa = (po - pe) / (1.0 - pe) if (1.0 - pe) > 0 else 0.0
    kappa = round(max(-1.0, min(1.0, kappa)), 4)

    confusion_matrix_dict = {
        "labels": labels,
        "matrix": cm,  # row = gold, col = pred
    }

    return {
        "n": n,
        "json_parse_success_rate": parse_ok / n if n else 0.0,
        "exact_status_match_pct": round(100.0 * exact / n, 1) if n else 0.0,
        "near_agreement_pct": round(100.0 * (exact + adjacent) / n, 1) if n else 0.0,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "cohens_kappa": kappa,
        "adjacent_errors": adjacent,
        "far_errors": far,
        "score_band_accuracy": round(band_ok / band_n, 4) if band_n else None,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix_dict,
        "rows": rows,
        "headline": (
            f"Judge status agreement {exact}/{n} ({100.0 * exact / n:.1f}%), "
            f"Macro-F1={macro_f1:.2f}, Kappa={kappa:.2f}; LLM-as-judge, N={n}."
            if n
            else "No judge items"
        ),
    }


from src.rate_limit import DailyQuotaExhaustedError, get_rate_limiter, is_daily_quota_error  # noqa: E402

def run_live_extract(gold_items: list[dict], existing_results: dict | None = None) -> dict[str, dict]:
    out: dict[str, dict] = dict(existing_results or {})
    quota_exhausted = False
    for item in gold_items:
        if item["id"] in out and out[item["id"]].get("json_parse_ok") and not out[item["id"]].get("skipped_quota"):
            logger.info("Extract %s already completed (checkpoint)", item["id"])
            continue
        if quota_exhausted:
            out[item["id"]] = {
                "json_parse_ok": False,
                "error": "Quota limit reached",
                "promises": [],
                "skipped_quota": True,
            }
            continue
        chunk = make_chunk(item["excerpt"], item["year"], item["id"])
        parse_ok = True
        err = None
        promises: list[dict] = []
        try:
            objs = extract_promises_for_year(item["year"], [chunk])
            promises = [p.model_dump() for p in objs]
        except DailyQuotaExhaustedError as exc:
            quota_exhausted = True
            parse_ok = False
            err = str(exc)
            logger.warning("Daily quota exhausted during extract for %s: %s", item["id"], exc)
        except Exception as exc:
            if is_daily_quota_error(exc):
                quota_exhausted = True
                logger.warning("Daily quota exhausted during extract for %s: %s", item["id"], exc)
            parse_ok = False
            err = str(exc)
            logger.exception("Extract failed for %s", item["id"])
        out[item["id"]] = {
            "json_parse_ok": parse_ok,
            "error": err,
            "promises": promises,
            "skipped_quota": quota_exhausted,
        }
        logger.info("Extract %s -> %d promises", item["id"], len(promises))
    return out


def run_live_judge(gold_items: list[dict], existing_results: dict | None = None) -> dict[str, dict]:
    out: dict[str, dict] = dict(existing_results or {})
    quota_exhausted = False
    for item in gold_items:
        if item["id"] in out and out[item["id"]].get("json_parse_ok") and not out[item["id"]].get("skipped_quota"):
            logger.info("Judge %s already completed (checkpoint)", item["id"])
            continue
        if quota_exhausted:
            out[item["id"]] = {
                "json_parse_ok": False,
                "error": "Quota limit reached",
                "status": None,
                "delivery_score": None,
                "judge_reasoning": None,
                "skipped_quota": True,
            }
            continue
        promise = Promise(
            promise_text=item["promise_text"],
            domain=item.get("domain") or "Finance",
            deadline_mentioned=item.get("deadline_mentioned"),
            year_made=item["year_made"],
            page_number=0,
        )
        ev_year = int(item.get("evidence_year") or item["year_made"])
        evidence = [make_chunk(item["evidence_excerpt"], ev_year, item["id"])]
        parse_ok = True
        err = None
        payload: dict[str, Any] = {}
        try:
            ev = judge_single_promise(promise, evidence)
            if ev is not None:
                payload = ev.model_dump()
                if ev.judge_reasoning.startswith("Judge call failed"):
                    parse_ok = False
            else:
                parse_ok = False
                err = "judge_single_promise returned None"
        except DailyQuotaExhaustedError as exc:
            quota_exhausted = True
            parse_ok = False
            err = str(exc)
            logger.warning("Daily quota exhausted during judge for %s: %s", item["id"], exc)
        except Exception as exc:
            if is_daily_quota_error(exc):
                quota_exhausted = True
                logger.warning("Daily quota exhausted during judge for %s: %s", item["id"], exc)
            parse_ok = False
            err = str(exc)
            logger.exception("Judge failed for %s", item["id"])
        out[item["id"]] = {
            "json_parse_ok": parse_ok,
            "error": err,
            "status": payload.get("status"),
            "delivery_score": payload.get("delivery_score"),
            "judge_reasoning": payload.get("judge_reasoning"),
            "skipped_quota": quota_exhausted,
        }
        logger.info("Judge %s -> %s %s", item["id"], payload.get("status"), payload.get("delivery_score"))
    return out


class RateLimitHarness(unittest.TestCase):
    """Inject 429; assert retry/backoff. Not a quality claim."""

    def test_gemini_backoff_positive(self) -> None:
        limiter = get_rate_limiter("gemini_generate")
        self.assertIsNotNone(limiter)

    def test_gemini_retries_on_429(self) -> None:
        calls = {"n": 0}

        def fake_fn():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("429 rate limit")
            return "[]"

        limiter = get_rate_limiter("gemini_generate")
        with patch("src.rate_limit.time.sleep"):
            text = limiter.execute_with_retry(fake_fn)
        self.assertEqual(text, "[]")
        self.assertEqual(calls["n"], 2)


def run_rate_limit_harness() -> dict:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RateLimitHarness)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures) + len(result.errors),
        "ok": result.wasSuccessful(),
    }


def latest_run_path() -> Path | None:
    if not RUNS_DIR.exists():
        return None
    files = sorted(RUNS_DIR.glob("run_*.json"))
    return files[-1] if files else None


def metrics_from_run(gold: dict, run: dict) -> dict:
    extract_m = score_extract(gold["extract_gold"], run.get("extract") or {})
    judge_m = score_judge(gold["judge_gold"], run.get("judge") or {})
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gold_meta": gold.get("meta"),
        "extract": {k: v for k, v in extract_m.items() if k != "rows"} | {"rows": extract_m["rows"]},
        "judge": {k: v for k, v in judge_m.items() if k != "rows"} | {"rows": judge_m["rows"]},
        "headline": judge_m["headline"],
        "caveat": "LLM-as-judge, N=15, HIG only. Do not treat overall 0–100 credibility as ground truth.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="TempusRAG LLM eval (offline by default)")
    parser.add_argument("--gold", type=Path, default=None, help="Path to benchmark JSON (e.g. eval/llm_gold_expanded.json)")
    parser.add_argument("--live", action="store_true", help="Call Gemini/Groq once and archive the run")
    parser.add_argument("--replay", type=Path, default=None, help="Score an archived run JSON")
    args = parser.parse_args()

    gold_path = args.gold or GOLD_PATH
    gold = load_gold(gold_path)
    checks = validate_gold(gold)
    failed = [c for c in checks if not c["pass"]]
    print(f"Gold dataset [{gold_path.name}] schema checks: {len(checks) - len(failed)}/{len(checks)} passed")
    for c in failed:
        print(f"  FAIL {c['name']}: {c['detail']}")

    rl = run_rate_limit_harness()
    print(f"Rate-limit harness (mocked 429): ok={rl['ok']} tests={rl['tests_run']} failures={rl['failures']}")

    if failed:
        return 1

    run: dict | None = None
    if args.live:
        if not GEMINI_API_KEY or not GROQ_API_KEY:
            print("Missing GEMINI_API_KEY or GROQ_API_KEY; refuse --live")
            return 2
        checkpoint_file = RUNS_DIR / "checkpoint_latest.json"
        existing_run = json.loads(checkpoint_file.read_text(encoding="utf-8")) if checkpoint_file.exists() else {}

        print(f"Live extract (Gemini)... [{gold_path.name}]")
        extract_pred = run_live_extract(gold["extract_gold"], existing_run.get("extract"))
        print(f"Live judge (Gemini/Groq)... [{gold_path.name}]")
        judge_pred = run_live_judge(gold["judge_gold"], existing_run.get("judge"))
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run = {
            "created_at": stamp,
            "mode": "live",
            "gold_file": gold_path.name,
            "extract": extract_pred,
            "judge": judge_pred,
            "qa": {"skipped": True, "reason": "Hit@5 needs a Chroma index; not run in this harness"},
        }
        out = RUNS_DIR / f"run_{stamp}.json"
        out.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        checkpoint_file.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Archived {out} and updated {checkpoint_file}")
    elif args.replay:
        run = json.loads(args.replay.read_text(encoding="utf-8"))
        print(f"Replay {args.replay}")
    else:
        path = latest_run_path()
        if path:
            run = json.loads(path.read_text(encoding="utf-8"))
            print(f"Replay latest {path}")
        else:
            print("No archived LLM run. Gold is valid. Use --live to call APIs (not CI).")
            return 0

    metrics = metrics_from_run(gold, run)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(metrics["headline"])
    ex = metrics["extract"]
    print(
        "Extract: scored={}/{} (quota skip {}) parse={:.0%} P={:.2f} R={:.2f} domain={} deadline={}".format(
            ex.get("n_scored", ex["n"]),
            ex["n"],
            ex.get("skipped_quota", 0),
            ex["json_parse_success_rate"],
            ex["promise_precision"],
            ex["promise_recall"],
            ex["domain_accuracy"],
            ex["deadline_accuracy"],
        )
    )
    print(f"Wrote {METRICS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
