"""Pydantic schemas for TempusRAG data structures. No internal dependencies."""

from typing import Literal

from pydantic import BaseModel, Field

Domain = Literal["Technology", "Claims", "Growth", "Finance", "Operations"]
DeliveryStatus = Literal["Delivered", "Partial", "Silently Abandoned", "Pending"]


class Promise(BaseModel):
    promise_text: str
    domain: Domain
    deadline_mentioned: str | None
    year_made: int
    page_number: int
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)


class DeliveryEvidence(BaseModel):
    promise: Promise
    evidence_text: str | None
    year_found: int | None
    delivery_score: int = Field(ge=0, le=100)
    status: DeliveryStatus
    judge_reasoning: str


class CompanyCredibilityReport(BaseModel):
    company: str
    years_analyzed: list[int]
    total_promises: int
    delivered: int
    partial: int
    abandoned: int
    overall_score: float = Field(ge=0.0, le=100.0)
    domain_scores: dict[str, float]
    red_flags: list[str]
    insufficient_data: bool = False
    """True when no promises were extracted (score of 0.0 means no data, not low credibility)."""


class Chunk(BaseModel):
    text: str
    company: str
    year: int
    section: str
    subsection: str
    chunk_id: str
    word_count: int


if __name__ == "__main__":
    p = Promise(
        promise_text="We will reduce claims processing time by 40% by 2023",
        domain="Claims",
        deadline_mentioned="2023",
        year_made=2021,
        page_number=14,
        confidence_score=0.8,
    )
    print(p.model_dump_json(indent=2))
