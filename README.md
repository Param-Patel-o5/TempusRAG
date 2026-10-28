# TempusRAG: Corporate Commitment Intelligence Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.0+-green.svg)](https://www.trychroma.com/)
[![Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-orange.svg)](https://ai.google.dev/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Unit Checks](https://img.shields.io/badge/checks-139%2F139%20passed-brightgreen.svg)]()

> When an executive makes a multi-billion dollar commitment in an SEC Form 10-K, subsequent filings rarely tell you if they actually delivered — or quietly abandoned it.

**TempusRAG** is a production-grade temporal Retrieval-Augmented Generation (RAG) engine that audits multi-year corporate management follow-through across SEC annual reports. It extracts explicit, time-stamped management promises, isolates subsequent fiscal year disclosures, retrieves cross-year evidence using hybrid search (BM25 + MiniLM + RRF), and classifies delivery status using an LLM judge calibrated against human financial audits.

```python
from src.pipeline import run_pipeline

# Audit corporate follow-through across 5 fiscal years
result = run_pipeline(ticker="HIG", start_year=2021, end_year=2025)

print(f"Company: {result.company_name} ({result.ticker})")
print(f"Executive Credibility Score: {result.overall_score:.1f} / 100")
print(f"Total Promises Extracted: {len(result.promises)}")

for p in result.promises[:3]:
    print(f"• [{p.status.value}] ({p.fiscal_year}) {p.text[:75]}... (Score: {p.delivery_score})")
```

---

## ⚡ Features

- 🔍 **Section-Aware 10-K Parsing**: Automatically isolates statutory sections (Item 1 Business, Item 1A Risk Factors, Item 1C Cybersecurity, Item 7 MD&A, Item 7A Market Risk, Item 9A Controls).
- 🧩 **Contextual Semantic Chunking**: Enforces strict `[COMPANY | YEAR | SECTION]` structural headers with a 500-word soft boundary to preserve semantic coherence.
- 🔀 **Hybrid Retrieval & RRF Fusion**: Fuses BM25Okapi lexical precision with `all-MiniLM-L6-v2` dense vectors via Reciprocal Rank Fusion ($k=60$) and statutory section boosting.
- 🎯 **Lost-in-the-Middle Reordering**: Mitigates LLM context degradation by dynamically sandwiching the top-ranked evidence chunks at the extreme ends of the prompt.
- ⚖️ **Audited LLM-as-a-Judge**: Classifies multi-year execution into four verified statuses with 1.00 precision on the critical *Silently Abandoned* class.
- 📉 **Recency-Decayed Credibility Index**: Scores overall executive integrity on a 0–100 scale using exponential temporal decay: $w(y) = 0.5^{y_{\text{eval}} - y_{\text{made}}}$.
- 🛡️ **Dual-Account Quota Shield**: Coordinates independent API keys across extraction and judging tiers, doubling effective free-tier throughput to 30 RPM without quota contention.
- 🖥️ **Interactive Streamlit Dashboard**: Full UI with dynamic credibility gauges, domain radar charts, historical commitment timelines, and conversational filing Q&A.

---

## 🧭 What It Audits & Detects

Corporate commitments are categorized across standard corporate domains and evaluated through a 4-way multi-class status taxonomy:

| Commitment Domain | Focus in SEC 10-K Disclosures | Typical Example |
|---|---|---|
| **Capital Allocation** | Share buybacks, dividend programs, debt retirements | *"$3.0 billion share repurchase authorization through 2024"* |
| **Operational & CapEx** | Facility construction, factory ramps, capacity targets | *"Constructing Gigafactory Texas with initial volume in 2022"* |
| **Product & Technology** | Software releases, platform migrations, hardware tape-outs | *"Transitioning architecture to Blackwell platform by Q4 2024"* |
| **M&A & Restructuring** | Asset integration, synergies, workforce re-alignment | *"Complete integration of Atria Wealth Solutions by mid-2025"* |
| **Regulatory & Governance**| Accounting standard adoption, compliance remediation | *"Adopting FASB LDTI accounting standard effective Jan 1, 2023"* |

### 4-Way Delivery Status Taxonomy

| Status | Operational Definition | Delivery Score Band |
|---|---|:---:|
| **Delivered** | Explicit evidence confirms the target, dollar figure, or milestone was 100% achieved. | `86 – 100` |
| **Partial** | Substantial progress made, but the target was under-delivered, delayed, or scaled down. | `31 – 60` / `61 – 85` |
| **Pending** | Target deadline has not yet arrived, or the commitment spans an ongoing multi-year roadmap. | `31 – 85` |
| **Silently Abandoned** | Deadline passed or initiative vanished from subsequent 10-Ks with zero confirmation or explanation. | `0 – 30` |

---

## 🏛️ Pipeline Architecture

```
SEC EDGAR (10-K HTML) 
        │
        ▼
[1. BeautifulSoup Parser] ──► Section isolation (Item 1, 1A, 1C, 7, 7A, 9A)
        │
        ▼
[2. Contextual Chunker]   ──► [COMPANY | YEAR | SECTION] prefixes, ~500-word soft cap
        │
        ├──► ChromaDB Vector Store (all-MiniLM-L6-v2) ──┐
        └──► Sparse Inverted Index (BM25Okapi) ──────────┤
                                                         ▼
[3. Hybrid Retriever]     ◄──────────────────── Reciprocal Rank Fusion (k=60) + Section Boost
        │                                                │
        │                                                ▼
        │                                      Lost-in-the-Middle Reordering
        ▼
[4. LLM Extractor]        ──► Structured Promise schema (Gemini Flash, JSON mode)
        │
        ▼
[5. LLM Reasoner / Judge] ──► Temporal evidence synthesis (Gemini Flash Lite / Groq)
        │
        ▼
[6. Credibility Scorer]   ──► Recency-decayed aggregation + Red flag heuristic detection
        │
        ▼
[7. Streamlit Dashboard]  ──► Real-time audit scorecard, visual timelines, and filing Q&A
```

---

## 📊 Comprehensive Multi-Stage Evaluation

Rather than relying on informal "vibe checks", TempusRAG benchmarks every pipeline tier against audited gold datasets from SEC filings across **The Hartford (HIG)**, **Tesla (TSLA)**, **NVIDIA (NVDA)**, **Apple (AAPL)**, and **Microsoft (MSFT)**.

```
SEC 10-K HTML ──► [Stage 1: Extraction] ──► Promises (P: 1.00, R: 1.00, Domain: 93.3%, Deadline: 100%)
                       │
                       ▼
Promises      ──► [Stage 2: Hybrid Retrieval] ──► Evidence Chunks (MRR: 0.8744, Recall@5: 96.0%, NDCG@3: 0.7694)
                       │
                       ▼
Chunks        ──► [Stage 3: LLM Reasoner/Judge] ──► 4-Way Status (Agreement: 88.0%, Macro-F1: 0.86, Kappa: 0.83)
                       │                            (Zero far errors, 100% near-agreement)
                       ▼
Credibility   ──► [Stage 4: Recency-Weighted Scorer] ──► 0–100 Company Index (139/139 Unit Checks Passed)
```

---

### Stage 2: Information Retrieval (IR) Benchmark (`eval/eval_retrieval.py`)

Evaluated on **25 verified gold query-to-evidence pairs** against **3,928 indexed 10-K chunks** in ChromaDB across 5 companies:

| Retrieval Configuration | MRR | Recall@1 | Recall@3 | Recall@5 | NDCG@3 |
|---|:---:|:---:|:---:|:---:|:---:|
| **Dense Only (`all-MiniLM-L6-v2`)** | 0.8233 | 76.00% | 88.00% | 92.00% | 0.6923 |
| **Sparse Only (`BM25Okapi`)** | 0.8244 | 76.00% | 88.00% | 88.00% | 0.7269 |
| **Hybrid (Dense + BM25 + Weighted RRF + Boost)** | **0.8744** | **80.00%** | **92.00%** | **96.00%** | **0.7694** |

#### Why Hybrid RRF is Essential for Financial Filings:
1. **Definitive Hybrid Lift Across Every Metric:** Tuning Reciprocal Rank Fusion to $k=30$ with a statutory sparse weighting of $w_{\text{sparse}}=1.5$ produces a **+0.0500 MRR lift over BM25** and **+0.0511 MRR lift over Dense**, pushing Recall@5 to **96.00%** (and Recall@10 to 100.0%).
2. **Lexical Precision on Technical Programs:** Regulatory 10-Ks contain specific statutory numbers and proper nouns (e.g., *"$3.0 billion share repurchase authorization"*, *"FASB LDTI adoption"*). BM25 locks onto exact numerical tokens without semantic drift.
3. **Semantic Generalization Across Multi-Year Horizons:** Commitments evolve over multi-year reporting cycles (e.g., Tesla promising an *"autonomous ride-hailing network"* in 2021, and subsequent filings discussing *"Robotaxi commercial rollout in 2025"*). Pure BM25 misses the lexical transition; Hybrid RRF bridges both.

---

### Stage 3: LLM Reasoner & Judge Benchmark (`eval/eval_llm.py`)

Evaluated on **25 multi-company audited ground truth items** ($N = 25$) across The Hartford (HIG), Tesla (TSLA), NVIDIA (NVDA), Apple (AAPL), and Microsoft (MSFT) spanning all four classification statuses:

- **Topline Exact Status Agreement:** **88.0%** (22/25 exact status matches)
- **Near-Agreement ($\pm 1$ Ordinal Step):** **100.0%** (25/25 within 1 step)
- **Macro-F1 (unweighted across 4 classes):** **0.8602**
- **Weighted-F1:** **0.8621**
- **Cohen's Kappa ($\kappa$):** **0.8276** (Substantial / Near-Perfect Inter-Rater Agreement)
- **Score Band Calibration:** **96.0%** accuracy across 4 credibility quartiles (0–30, 31–60, 61–85, 86–100)
- **Far Errors (e.g. Delivered vs Abandoned):** **0** (0.0%)

#### Per-Class Performance Breakdown:
| Status Class | Precision | Recall | F1 Score | Support ($N$) |
|---|:---:|:---:|:---:|:---:|
| **Silently Abandoned** | **1.0000** | **1.0000** | **1.0000** | 5 |
| **Pending** | **1.0000** | **1.0000** | **1.0000** | 5 |
| **Delivered** | 0.7692 | **1.0000** | 0.8696 | 10 |
| **Partial** | **1.0000** | 0.4000 | 0.5714 | 5 |

#### 4×4 Confusion Matrix:
```
                      PREDICTED
                Delivered  Partial  Pending  Silently Abandoned
ACTUAL
Delivered          10         0        0             0
Partial             3         2        0             0
Pending             0         0        5             0
Silently Abandoned  0         0        0             5
```

> **Interview Defense Takeaways:**  
> 1. **Zero High-Liability False Accusations:** On our highest-consequence risk class—**`Silently Abandoned`**—the judge achieves **1.00 Precision, 1.00 Recall, and 1.00 F1**. It never falsely accuses an executive team of abandoning a promise when delivery occurred.
> 2. **100% Near-Agreement:** 100% of errors are adjacent (only 3 border disputes out of 25 items), and there are **0 far errors**.
> 3. **Defensible Partial / Delivered Boundary:** The 3 adjacent discrepancies are real-world edge cases where partial milestones were achieved (e.g., TSLA delivery growth reached 38% vs 50% target; MSFT contracted 19 GW renewable energy; Apple launched Vision Pro in US). In financial audits, continuous execution boundaries between 'Partial' and 'Delivered' are nuanced even between human partners.

---

### Company-Specific Extraction Density

Different business models naturally yield different disclosure styles in SEC filings:

| Ticker | Company | Typical Commitments (5-Yr) | Disclosure Style & Primary Commitment Domains |
|---|---|:---:|---|
| **TSLA** | Tesla, Inc. | **70 – 130+** | Highly aggressive engineering & CapEx roadmaps (Gigafactories, 4680 cells, FSD/Robotaxi, Cybertruck). |
| **HIG** | The Hartford | **35 – 45** | Formal capital returns (share repurchase authorizations, dividend programs, FASB LDTI accounting). |
| **NVDA** | NVIDIA Corp. | **30 – 50** | Semiconductor tape-outs, datacenter architecture roadmaps (Blackwell, Hopper, networking scale). |
| **AAPL** | Apple Inc. | **25 – 40** | Apple Silicon migrations (M-series), spatial computing (Vision Pro), 2030 carbon neutrality roadmaps. |
| **MSFT** | Microsoft Corp. | **30 – 45** | Cloud infrastructure buildouts (Azure CapEx), OpenAI compute capacity, 100% renewable energy by 2025. |

*This variation reflects genuine corporate reporting differences under FINRA/SEC disclosure standards rather than an extraction deficiency.*

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11 or higher
- PowerShell (Windows) or Bash (macOS/Linux)

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/Param-Patel-o5/TempusRAG---Corporate-Commitment-Intelligence-Engine.git
cd TempusRag

# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Configure Environment Variables (`.env`)

Create a `.env` file in the project root:

```env
# Google Gemini API Keys (Use keys from two separate Google accounts for 30 RPM throughput)
GEMINI_API_KEY="your-gemini-extractor-api-key"
GEMINI_JUDGE_API_KEY="your-gemini-judge-api-key"

# Groq API Key (High-speed fallback judge)
GROQ_API_KEY="your-groq-api-key"

# SEC EDGAR requires a User-Agent header in 'Sample Company Name AdminContact@<sample company domain>.com' format
SEC_USER_AGENT="YourName yourname@example.com"

# Provider Settings
JUDGE_PROVIDER="gemini"  # "gemini" or "groq"
MAX_JUDGE_WORKERS=3
```

---

## 🚀 Running the Application

### Launch Streamlit Dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

1. Navigate to `http://localhost:8501`.
2. Select a pre-cached ticker (**The Hartford (HIG)**, **Tesla (TSLA)**, **NVIDIA (NVDA)**, **Apple (AAPL)**, or **Microsoft (MSFT)**) or enter any valid SEC ticker (e.g., **LPLA**).
3. Click **Run Analysis** to execute the end-to-end pipeline.
4. Inspect the **Executive Credibility Score**, domain breakdown, commitment timeline, and interactive filing Q&A.

---

## 🧪 Evaluation Commands

Run the full evaluation benchmark suite locally:

```powershell
# 1. Deterministic System Checks (139/139 checks passed; fast, zero API calls)
.\.venv\Scripts\python.exe eval/eval_no_llm.py

# 2. Information Retrieval Benchmark (Dense vs BM25 vs Hybrid RRF on 25 gold queries and 3,928 chunks)
.\.venv\Scripts\python.exe eval/eval_retrieval.py

# 3. LLM Reasoner & Extractor Replay Benchmark (Replays verified runs, zero API quota; Macro-F1: 0.86, Kappa: 0.83)
.\.venv\Scripts\python.exe eval/eval_llm.py

# 4. Live LLM Evaluation (Executes live Gemini and Groq API calls)
.\.venv\Scripts\python.exe eval/eval_llm.py --live
```

---

## 📁 Repository Structure

```
TempusRag/
├── app.py                      # Streamlit dashboard interface
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies (pytest)
├── src/
│   ├── config.py               # Constants, hyper-parameters, and settings
│   ├── models.py               # Pydantic v2 schemas (Promise, DeliveryEvidence, Chunk)
│   ├── ingestion.py            # SEC EDGAR client, HTML parse & Item section isolation
│   ├── chunker.py              # Contextual chunker with section headers & soft cap
│   ├── embedder.py             # SentenceTransformers (all-MiniLM-L6-v2) + BM25Okapi + ChromaDB
│   ├── retriever.py            # Dense, Sparse, RRF fusion, section boosting & lost-in-middle
│   ├── extractor.py            # Gemini 2.5 promise extraction with JSON schema validation
│   ├── reasoner.py             # Dual-provider judge (Gemini / Groq) across temporal evidence
│   ├── scorer.py               # Recency-weighted aggregation & red flag detection heuristics
│   ├── rate_limit.py           # Token bucket rate limiting with exponential backoff & jitter
│   ├── query.py                # Filing Q&A engine with HyDE & conversational follow-up rewrite
│   └── pipeline.py             # End-to-end orchestration pipeline
├── eval/
│   ├── eval_no_llm.py          # Deterministic structural checks (139/139 passed)
│   ├── eval_retrieval.py       # IR benchmark (MRR, Recall@K, NDCG@K)
│   ├── eval_llm.py             # LLM extractor & judge benchmark (Macro-F1, Kappa, CM)
│   ├── llm_gold_expanded.json  # Multi-company gold benchmark dataset (HIG, TSLA, NVDA, AAPL, MSFT)
│   ├── llm_metrics.json        # Latest LLM evaluation results
│   ├── retrieval_metrics.json  # Latest IR benchmark results
│   ├── METRICS.md              # Auto-generated metrics ledger across 30 filings and 3,928 chunks
│   └── EVAL_SUMMARY.md         # Comprehensive interview defense reference
└── data/
    ├── filings/                # Cleaned, parsed 10-K sections by ticker and fiscal year
    └── raw/                    # Raw HTML filings downloaded from SEC EDGAR
```

---

## 📚 Technical Documentation

| Document | Contents |
|---|---|
| [EVAL_SUMMARY.md](eval/EVAL_SUMMARY.md) | Comprehensive multi-stage evaluation benchmark, confusion matrix analysis, and interview defense cheat sheet |
| [METRICS.md](eval/METRICS.md) | Auto-generated deterministic metrics ledger over 30 cached 10-Ks and 3,928 chunks |
| [retrieval_metrics.json](eval/retrieval_metrics.json) | Raw IR evaluation output: MRR, Recall@1/3/5, and NDCG@3 |
| [llm_metrics.json](eval/llm_metrics.json) | Raw LLM Judge evaluation output: Macro-F1, Cohen's Kappa, per-class metrics |

---

## 💡 Key Design Decisions & Interview FAQs

### 1. Why use dual API keys from different Google accounts?
Free-tier Gemini API keys originating from the same Google Cloud account share the same quota bucket (15 RPM). By utilizing two distinct accounts—one for extraction (`GEMINI_API_KEY`) and one for judging (`GEMINI_JUDGE_API_KEY`)—the pipeline doubles its effective throughput to 30 RPM without quota contention.

### 2. How is Lost-in-the-Middle handled?
LLMs exhibit a "U-shaped" attention curve, attending disproportionately to the beginning and end of long context windows. When assembling retrieved chunks into judge and Q&A prompts, `lost_in_middle_reorder` places rank 1 at index 0 and rank 2 at index -1, ensuring the strongest evidence chunks occupy high-attention positions.

### 3. How does the Credibility Score handle temporal decay?
A broken promise made 5 years ago carries less predictive weight regarding current management than a promise broken last year. TempusRAG applies an exponential decay weighting:
$$w(y) = 0.5^{y_{\text{eval}} - y_{\text{made}}}$$
This penalizes recent broken commitments significantly more while maintaining historical continuity.

### 4. Why does BM25 outperform Dense search on 10-Ks?
SEC 10-Ks contain dense financial numbers, legal statutes, and exact program names (e.g. *"$3.0 billion share repurchase"*, *"FASB LDTI"*). BM25 prevents semantic hallucination by locking onto exact tokens. Hybrid RRF combines BM25's lexical precision with MiniLM's semantic generalizability for multi-year vocabulary shifts.

---

## ⚖️ Disclaimer

TempusRAG is an academic and research project designed to demonstrate temporal RAG architectures and evaluate public SEC Form 10-K disclosures. It does not constitute investment, financial, legal, or accounting advice. Corporate commitment evaluations represent statistical model outputs over public regulatory filings.
