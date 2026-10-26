"""Rule-based confidence scoring and credibility report generation.

This module computes confidence_score on promises using rule-based signals, aggregates
delivery scores across domains and years, and produces a final CompanyCredibilityReport.
Pure Python: no LLM calls, no I/O beyond logging.
"""

import logging
import re

from src.config import CONFIDENCE_SIGNAL_WEIGHTS, recency_weight
from src.models import Promise, DeliveryEvidence, CompanyCredibilityReport

logger = logging.getLogger(__name__)

# Valid domains for domain_detectable signal
VALID_DOMAINS = {"Technology", "Claims", "Growth", "Finance", "Operations"}


# Signal 2 metric pattern: detect real quantitative targets.
_METRIC_RE = re.compile(
    r'\d+\s*%'                          # percentage: 40%, 12.5 %
    r'|\$\s*\d'                          # dollar amount: $2B, $100M
    r'|\d+\s*x\b'                        # multiplier: 3x, 2x
    r'|\d+\s*(?:million|billion|bn|M\b|B\b)'  # large units
    r'|\d+\s*(?:basis points?|bps?)'     # financial: 50 bps
    r'|\d+\s*(?:days?|months?|quarters?)' # time-bound metric: 30 days
    r'|\b\d{2,}\s*(?:employees?|customers?|stores?|locations?)',  # scale: 500 employees
    re.IGNORECASE,
)


def compute_confidence_score(promise: Promise) -> float:
    """
    Compute confidence score for a promise using rule-based signals.
    
    Args:
        promise: Promise object to score
        
    Returns:
        float: Confidence score in range [0.0, 1.0]
    """
    score = 0.0
    signals_matched = []
    
    # Signal 1: has_deadline (0.3 weight)
    if promise.deadline_mentioned is not None and promise.deadline_mentioned.strip():
        score += CONFIDENCE_SIGNAL_WEIGHTS["has_deadline"]
        signals_matched.append("has_deadline")
    
    # Signal 2: has_metric (0.3 weight): detect real quantitative targets.
    if _METRIC_RE.search(promise.promise_text):
        score += CONFIDENCE_SIGNAL_WEIGHTS["has_metric"]
        signals_matched.append("has_metric")
    
    # Signal 3: has_commitment_verb (0.2 weight)
    commitment_verbs = ["will", "committed", "commit", "shall", "plan", "planned"]
    promise_lower = promise.promise_text.lower()
    if any(verb in promise_lower for verb in commitment_verbs):
        score += CONFIDENCE_SIGNAL_WEIGHTS["has_commitment_verb"]
        signals_matched.append("has_commitment_verb")
    
    # Signal 4: domain_detectable (0.2 weight)
    if promise.domain is not None and promise.domain in VALID_DOMAINS:
        score += CONFIDENCE_SIGNAL_WEIGHTS["domain_detectable"]
        signals_matched.append("domain_detectable")
    
    # Clamp to [0.0, 1.0]
    final_score = max(0.0, min(1.0, score))
    
    logger.debug(f"Promise confidence score: {final_score:.2f} (signals: {signals_matched}) - {promise.promise_text[:50]}...")
    
    return final_score


def add_confidence_scores(promises: list[Promise]) -> list[Promise]:
    """
    Add confidence scores to all promises in the list.
    
    Args:
        promises: List of promises to score
        
    Returns:
        list[Promise]: Same promises with confidence_score field set
    """
    if not promises:
        logger.warning("Empty promises list provided to add_confidence_scores")
        return promises
    
    total_score = 0.0
    
    for promise in promises:
        confidence_score = compute_confidence_score(promise)
        promise.confidence_score = confidence_score
        total_score += confidence_score
    
    average_confidence = total_score / len(promises)
    logger.info(f"Added confidence scores to {len(promises)} promises (average: {average_confidence:.2f})")
    
    return promises


def aggregate_by_domain(delivery_evidence: list[DeliveryEvidence]) -> dict[str, list[DeliveryEvidence]]:
    """
    Group delivery evidence by promise domain.
    
    Args:
        delivery_evidence: List of delivery evidence objects
        
    Returns:
        dict[str, list[DeliveryEvidence]]: Evidence grouped by domain
    """
    domain_groups = {}
    
    for evidence in delivery_evidence:
        domain = evidence.promise.domain
        
        if domain is None:
            logger.warning(f"Promise has no domain: {evidence.promise.promise_text[:50]}...")
            continue
            
        if domain not in domain_groups:
            domain_groups[domain] = []
        
        domain_groups[domain].append(evidence)
    
    return domain_groups


def compute_domain_scores(delivery_evidence: list[DeliveryEvidence]) -> dict[str, float]:
    """
    Compute average delivery scores by domain.
    
    Args:
        delivery_evidence: List of delivery evidence objects
        
    Returns:
        dict[str, float]: Average delivery score per domain
    """
    domain_groups = aggregate_by_domain(delivery_evidence)
    domain_scores = {}
    
    for domain, evidence_list in domain_groups.items():
        if evidence_list:
            scores = [evidence.delivery_score for evidence in evidence_list]
            average_score = sum(scores) / len(scores)
            domain_scores[domain] = average_score
        else:
            domain_scores[domain] = 0.0
    
    logger.info(f"Domain score breakdown: {domain_scores}")
    return domain_scores


def count_delivery_statuses(delivery_evidence: list[DeliveryEvidence]) -> dict[str, int]:
    """
    Count delivery evidence by status.
    
    Args:
        delivery_evidence: List of delivery evidence objects
        
    Returns:
        dict[str, int]: Count of each delivery status
    """
    # Initialize all possible statuses to 0
    status_counts = {
        "Delivered": 0,
        "Partial": 0, 
        "Silently Abandoned": 0,
        "Pending": 0
    }
    
    for evidence in delivery_evidence:
        status = evidence.status
        if status in status_counts:
            status_counts[status] += 1
        else:
            logger.warning(f"Unknown delivery status: {status}")
    
    logger.info(f"Status counts: {status_counts}")
    return status_counts


def compute_overall_credibility_score(delivery_evidence: list[DeliveryEvidence]) -> float:
    """
    Compute recency-weighted average of delivery scores.
    
    Args:
        delivery_evidence: List of delivery evidence objects
        
    Returns:
        float: Overall credibility score in range [0.0, 100.0]
    """
    if not delivery_evidence:
        logger.warning("No delivery evidence provided for overall credibility score")
        return 0.0
    
    # Find most recent year for recency weighting
    years_found = [evidence.year_found for evidence in delivery_evidence if evidence.year_found is not None]
    
    if not years_found:
        # Fallback to unweighted average if no year_found data
        logger.warning("No year_found data available, using unweighted average")
        scores = [evidence.delivery_score for evidence in delivery_evidence]
        return max(0.0, min(100.0, sum(scores) / len(scores)))
    
    most_recent_year = max(years_found)
    
    # Compute weighted average
    weighted_sum = 0.0
    weight_sum = 0.0
    
    for evidence in delivery_evidence:
        if evidence.year_found is not None:
            weight = recency_weight(evidence.year_found, most_recent_year)
            weighted_sum += evidence.delivery_score * weight
            weight_sum += weight
        else:
            # Include evidence without year_found with minimum weight
            weight = 0.6  # Minimum recency weight from config
            weighted_sum += evidence.delivery_score * weight
            weight_sum += weight
    
    if weight_sum == 0:
        logger.warning("Zero weight sum in credibility score computation")
        return 0.0
    
    overall_score = weighted_sum / weight_sum
    final_score = max(0.0, min(100.0, overall_score))
    
    logger.info(f"Overall credibility score: {final_score:.1f} (recency-weighted, most recent year: {most_recent_year})")
    
    return final_score


def detect_red_flags(
    delivery_evidence: list[DeliveryEvidence],
    domain_scores: dict[str, float],
    status_counts: dict[str, int] | None = None,
) -> list[str]:
    """
    Detect red flags in company promise delivery performance.
    
    Args:
        delivery_evidence: List of delivery evidence objects
        domain_scores: Average scores by domain
        status_counts: Precomputed status counts (optional)
        
    Returns:
        list[str]: List of red flag messages
    """
    red_flags = []
    
    if not delivery_evidence:
        return red_flags
    
    # Count statuses for percentage calculations
    if status_counts is None:
        status_counts = count_delivery_statuses(delivery_evidence)
    total_evidence = len(delivery_evidence)
    
    # Red flag 1: More than 50% "Silently Abandoned"
    abandoned_percentage = status_counts["Silently Abandoned"] / total_evidence
    if abandoned_percentage > 0.5:
        red_flags.append("Majority of promises abandoned")
    
    # Red flag 2: Any domain with average score < 30
    for domain, score in domain_scores.items():
        if score < 30:
            red_flags.append(f"Domain {domain} has low credibility (avg score: {score:.1f})")
    
    # Red flag 3: All promises are "Pending"
    if status_counts["Pending"] == total_evidence and total_evidence > 0:
        red_flags.append("No promises have reached evaluation point yet")
    
    # Red flag 4: Average delivery score < 40 across all evidence
    if delivery_evidence:
        avg_delivery_score = sum(evidence.delivery_score for evidence in delivery_evidence) / len(delivery_evidence)
        if avg_delivery_score < 40:
            red_flags.append("Overall credibility is low")
    
    if red_flags:
        logger.warning(f"Red flags detected: {red_flags}")
    
    return red_flags


def generate_credibility_report(company_ticker: str, company_display_name: str, 
                              promises_by_year: dict[int, list[Promise]], 
                              delivery_evidence: list[DeliveryEvidence]) -> CompanyCredibilityReport:
    """
    Generate comprehensive credibility report for a company.
    
    Args:
        company_ticker: Company ticker symbol (e.g., "HIG")
        company_display_name: Human-readable company name (e.g., "Hartford")
        promises_by_year: Promises grouped by year they were made
        delivery_evidence: Delivery evidence with judgments from reasoner
        
    Returns:
        CompanyCredibilityReport: Complete credibility assessment
    """
    logger.info(f"Generating credibility report for {company_display_name} ({company_ticker})")
    
    # Flatten promises from all years
    all_promises = []
    for year_promises in promises_by_year.values():
        all_promises.extend(year_promises)
    
    if not all_promises:
        logger.warning("No promises provided for credibility report")
    
    # Add confidence scores to all promises
    all_promises = add_confidence_scores(all_promises)
    
    # Handle empty delivery evidence
    if not delivery_evidence:
        logger.warning("No delivery evidence provided for credibility report")
        return CompanyCredibilityReport(
            company=company_display_name,
            years_analyzed=sorted(promises_by_year.keys()) if promises_by_year else [],
            total_promises=len(all_promises),
            delivered=0,
            partial=0,
            abandoned=0,
            overall_score=0.0,
            domain_scores={},
            red_flags=["No delivery evidence available"],
            insufficient_data=(len(all_promises) == 0),
        )
    
    # Compute aggregations
    status_counts = count_delivery_statuses(delivery_evidence)
    domain_scores = compute_domain_scores(delivery_evidence)
    overall_score = compute_overall_credibility_score(delivery_evidence)
    red_flags = detect_red_flags(delivery_evidence, domain_scores, status_counts=status_counts)
    
    # Construct final report
    report = CompanyCredibilityReport(
        company=company_display_name,
        years_analyzed=sorted(promises_by_year.keys()),
        total_promises=len(all_promises),
        delivered=status_counts["Delivered"],
        partial=status_counts["Partial"],
        abandoned=status_counts["Silently Abandoned"],
        pending=status_counts["Pending"],
        overall_score=overall_score,
        domain_scores=domain_scores,
        red_flags=red_flags
    )
    
    logger.info(f"Credibility report complete: {report.total_promises} promises, "
               f"overall score {report.overall_score:.1f}, {len(report.red_flags)} red flags")
    
    return report


if __name__ == "__main__":
    import logging
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    from src.models import Promise, DeliveryEvidence
    
    # Sample promises
    sample_promises = {
        2021: [
            Promise(
                promise_text="We will reduce claims processing time by 40% by 2023",
                domain="Claims",
                deadline_mentioned="2023",
                year_made=2021,
                page_number=0,
                confidence_score=0.0  # Will be computed
            ),
            Promise(
                promise_text="Expand technology platform",
                domain="Technology", 
                deadline_mentioned=None,
                year_made=2021,
                page_number=0,
                confidence_score=0.0
            )
        ]
    }
    
    # Sample delivery evidence
    sample_evidence = [
        DeliveryEvidence(
            promise=sample_promises[2021][0],
            evidence_text="Claims processing improved 35% year-over-year",
            year_found=2023,
            delivery_score=80,
            status="Delivered",
            judge_reasoning="Strong progress toward target"
        ),
        DeliveryEvidence(
            promise=sample_promises[2021][1],
            evidence_text="Launched new claims platform", 
            year_found=2023,
            delivery_score=90,
            status="Delivered",
            judge_reasoning="Platform fully deployed"
        )
    ]
    
    print("Testing scorer.py...")
    print(f"Sample promises: {sum(len(promises) for promises in sample_promises.values())}")
    print(f"Sample evidence: {len(sample_evidence)}")
    
    # Test confidence scoring
    print(f"\n--- TESTING CONFIDENCE SCORING ---")
    for year, promises in sample_promises.items():
        for promise in promises:
            original_score = promise.confidence_score
            computed_score = compute_confidence_score(promise)
            print(f"Promise: {promise.promise_text[:60]}...")
            print(f"  Original confidence: {original_score}")
            print(f"  Computed confidence: {computed_score:.2f}")
            print(f"  Has deadline: {promise.deadline_mentioned is not None}")
            print(f"  Has numbers: {bool(re.search(r'\\d', promise.promise_text))}")
    
    # Test domain aggregation
    print(f"\n--- TESTING DOMAIN AGGREGATION ---")
    domain_groups = aggregate_by_domain(sample_evidence)
    for domain, evidence_list in domain_groups.items():
        print(f"Domain {domain}: {len(evidence_list)} evidence objects")
    
    # Test full report generation
    print(f"\n--- GENERATING FULL REPORT ---")
    report = generate_credibility_report(
        company_ticker="HIG",
        company_display_name="Hartford",
        promises_by_year=sample_promises,
        delivery_evidence=sample_evidence
    )
    
    print(f"\nFinal Report:")
    print(f"  Company: {report.company}")
    print(f"  Years Analyzed: {report.years_analyzed}")
    print(f"  Total Promises: {report.total_promises}")
    print(f"  Overall Score: {report.overall_score:.1f}")
    print(f"  Delivered: {report.delivered}, Partial: {report.partial}, Abandoned: {report.abandoned}")
    print(f"  Domain Scores: {report.domain_scores}")
    print(f"  Red Flags: {report.red_flags}")
    
    print(f"\n🎯 Scorer testing complete!")