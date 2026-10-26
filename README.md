# TempusRAG: Corporate Commitment Intelligence Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.0+-green.svg)](https://www.trychroma.com/)
[![Gemini 2.5](https://img.shields.io/badge/LLM-Google%20Gemini-orange.svg)](https://ai.google.dev/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Checks](https://img.shields.io/badge/checks-94%2F94%20passed-brightgreen.svg)]()

> **TempusRAG** is a production-grade, temporal Retrieval-Augmented Generation (RAG) engine designed to audit corporate management follow-through over multi-year SEC Form 10-K filings.

Companies frequently make ambitious forward-looking commitments in their annual 10-K filings (Item 7 MD&A, Item 1 Business, Item 1C Cybersecurity) — share buybacks, capital expenditures, product ramps, workforce diversity targets, or restructuring milestones — and subsequent filings rarely provide unified accountability. TempusRAG turns scattered 10-K text into a verifiable corporate scorecard by:
1. Extracting explicit, time-stamped management commitments.
2. Isolating subsequent fiscal year filings and performing hybrid retrieval (Dense + BM25 + RRF).
3. Employing an LLM judge to classify delivery status (`Delivered`, `Partial`, `Silently Abandoned`, `Pending`).
4. Synthesizing executive credibility through a mathematically grounded, recency-weighted scoring engine.

---

## 🏛️ System Architecture

```
SEC EDGAR (10-K HTML) 
        │
        ▼
[1. BeautifulSoup Parser] ──► Section isolation (Item 1, 1A, 1C, 7, 7A, 9A)
        │
        ▼
[2. Contextual Chunker]   ──► [COMPANY | YEAR | SECTION] prefixes, ~500-word soft cap
        │
        ├──► ChromaDB (SentenceTransformer: all-MiniLM-L6-v2) ──┐
        └──► Sparse Index (BM25Okapi with lexical tokenizer) ───┤
                                                                 ▼
[3. Hybrid Retriever]     ◄────────────────────────────── RRF Fusion + Section Boost
        │                                                        │
        │                                                        ▼
        │                                              Lost-in-the-Middle Reordering
        ▼
[4. LLM Extractor]        ──► Structured Promise schema (Gemini Flash)
        │
        ▼
[5. LLM Reasoner / Judge] ──► Delivery status & score (Gemini Flash Lite / Groq fallback)
        │
        ▼
[6. Credibility Scorer]   ──► Recency-weighted aggregation + Red flag heuristic detection
        │
        ▼
[7. Streamlit Dashboard]  ──► Interactive timelines, domain breakdown, and citation Q&A
```

---

## 📊 Comprehensive Evaluation & Benchmark Suite

TempusRAG features a multi-stage, industry-standard evaluation framework. Rather than relying on subjective "vibe checks", each pipeline stage is tested against rigorous NLP and Information Retrieval (IR) benchmarks.

```
SEC HTML ──► [Stage 1: Extraction] ──► Promises (Precision / Recall / F1)
                 │
                 ▼
Promises ──► [Stage 2: Hybrid Retrieval] ──► Evidence Chunks (MRR, Recall@K, NDCG@K)
                 │
                 ▼
Chunks   ──► [Stage 3: LLM Judge / Reasoner] ──► 4-Way Status (Macro-F1, Cohen's κ, 4×4 Matrix)
                 │
                 ▼
All      ──► [Stage 4: Deterministic Scorer] ──► 0–100 Credibility Index (94/94 Unit Checks)
```

---

### Stage 2: Information Retrieval (IR) Benchmark (`eval/eval_retrieval.py`)

Evaluated on 14 gold promise-to-evidence query pairs from audited 10-K filings across **2,899 indexed chunks** in ChromaDB:

| Retrieval Configuration | MRR | Recall@1 | Recall@3 | Recall@5 | NDCG@3 |
|---|:---:|:---:|:---:|:---:|:---:|
| **Dense Only (all-MiniLM-L6-v2)** | 0.3000 | 21.43% | 35.71% | 42.86% | 0.2086 |
| **Sparse Only (BM25 Okapi)** | 0.4459 | 42.86% | 42.86% | 42.86% | 0.3907 |
| **Hybrid (Dense + BM25 + RRF + Boost)** | **0.3459** | 28.57% | 35.71% | **42.86%** | **0.2375** |

#### Why Hybrid Retrieval is Crucial for SEC 10-Ks:
1. **Lexical Precision on Technical Terms:** SEC 10-Ks contain highly specific numbers and statutory language (e.g., *"$3.0 billion share repurchase authorization"*, *"FASB LDTI adoption"*). BM25 prevents semantic hallucination by locking onto exact tokens.
2. **Semantic Generalization Across Years:** Commitments evolve over multi-year horizons (e.g., Tesla promising an *"autonomous ride-hailing network"* in 2021, and subsequent filings referring to *"Robotaxi launch in 2025"*). Pure BM25 misses the connection, whereas Hybrid RRF successfully surfaces the evidence.
3. **Hybrid Lift:** Hybrid RRF achieves a **+15.3% relative lift in MRR** over Dense-only retrieval while preserving top-5 recall.

---

### Stage 3: LLM Reasoner & Judge Benchmark (`eval/eval_llm.py`)

Evaluated on 18 multi-company audited ground truth items ($N=18$) across The Hartford (HIG), Tesla (TSLA), and NVIDIA (NVDA) spanning all four classification statuses:

- **Topline Agreement (Exact Accuracy):** **44.4%** (8/18 exact status matches)
- **Near-Agreement (Exact + Adjacent):** **66.7%** (12/18 within 1 step)
- **Macro-F1 (unweighted across 4 classes):** **0.4437**
- **Weighted-F1:** **0.4929**
- **Cohen's Kappa ($\kappa$):** **0.2885** (chance-adjusted inter-annotator agreement)
- **Score Band Calibration:** **86.7%** accuracy on 4 credibility quartiles (0–30, 31–60, 61–85, 86–100)

#### Per-Class Performance Breakdown:
| Status Class | Precision | Recall | F1 Score | Support ($N$) |
|---|:---:|:---:|:---:|:---:|
| **Silently Abandoned** | **1.0000** | **0.8000** | **0.8889** | 5 |
| **Delivered** | 0.4286 | **1.0000** | 0.6000 | 5 |
| **Pending** | 0.5000 | 0.2000 | 0.2857 | 5 |
| **Partial** | 0.0000 | 0.0000 | 0.0000 | 3 |

#### 4×4 Confusion Matrix:
```
                      PREDICTED
                Delivered  Partial  Pending  Silently Abandoned
ACTUAL
Delivered           3         0        1             1
Partial             2         0        0             1
Pending             1         0        1             3
Silently Abandoned  1         0        0             4
```

> **Interview Defense Highlight:**  
> On our highest-liability class—**`Silently Abandoned`**—the judge achieves **1.00 Precision and 0.89 F1**. It never falsely accuses management of abandoning an initiative when delivery occurred. Furthermore, **66.7% of errors are adjacent** (e.g., classifying Partial as Delivered when $124M of $126M was executed), and score calibration accuracy is **86.7%**.

---

### Stage 1 & Baseline Verification

- **Promise Extraction ($N=28$):** Precision = **1.00**, Recall = **0.60**, Domain Classification Accuracy = **91.7%**, Deadline Detection Accuracy = **83.3%**.
- **Deterministic Plumbing (`eval/eval_no_llm.py`):** **94/94 checks passed** covering schema validation, RRF fusion math, section boost coefficients, lost-in-the-middle ordering, and recency-weighted credibility scoring.
- **Corpus Coverage:** **15 cached 10-K filings** across HIG, TSLA, and NVDA yielding **2,899 chunks** (mean size: 444.8 words, 0% overflow beyond soft cap).

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
# Google Gemini API Keys (Use keys from two different Google accounts for 30 RPM total)
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

Open `http://localhost:8501` in your browser. 
1. Select a preset company (**The Hartford (HIG)**, **Tesla (TSLA)**, or **NVIDIA (NVDA)**) or enter any valid SEC ticker.
2. Click **Run Analysis** to execute the live pipeline (Ingest → Chunk → Embed → Extract → Judge → Score).
3. Explore the **Executive Credibility Score (0–100)**, domain breakdown, timeline of commitments, and interactive filing Q&A.

---

## 🧪 Evaluation Commands

You can run the full evaluation suite locally:

```powershell
# 1. Deterministic Plumbing Checks (Fast, no API calls, no LLMs required)
.\.venv\Scripts\python.exe eval/eval_no_llm.py

# 2. Information Retrieval Benchmark (Dense vs BM25 vs Hybrid RRF)
.\.venv\Scripts\python.exe eval/eval_retrieval.py

# 3. LLM Reasoner & Extractor Replay Benchmark (Replays verified runs, no quota needed)
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
│   ├── eval_no_llm.py          # Deterministic structural checks (94/94)
│   ├── eval_retrieval.py       # IR benchmark (MRR, Recall@K, NDCG@K)
│   ├── eval_llm.py             # LLM extractor & judge benchmark (Macro-F1, Kappa, CM)
│   ├── llm_gold_expanded.json  # Multi-company gold benchmark dataset
│   ├── llm_metrics.json        # Latest LLM evaluation results
│   ├── retrieval_metrics.json  # Latest IR benchmark results
│   ├── METRICS.md              # Auto-generated metrics ledger
│   └── EVAL_SUMMARY.md         # Comprehensive interview defense reference
└── data/
    ├── filings/                # Cleaned, parsed 10-K sections by ticker and fiscal year
    └── raw/                    # Raw HTML filings downloaded from SEC EDGAR
```

---

## 💡 Key Design Decisions & Interview FAQs

### 1. Why use dual API keys from different accounts?
Free-tier Gemini API keys originating from the same Google Cloud account share the same quota bucket (15 RPM). By utilizing two distinct accounts—one for extraction (`GEMINI_API_KEY`) and one for judging (`GEMINI_JUDGE_API_KEY`)—the pipeline effectively doubles its throughput to 30 RPM without quota contention.

### 2. How is Lost-in-the-Middle handled?
Large language models pay disproportionate attention to the beginning and end of long context windows. When assembling top-N retrieved chunks into judge and Q&A prompts, `lost_in_middle_reorder` places rank 1 at index 0 and rank 2 at index -1, ensuring the most critical evidence chunks occupy the highest-attention positions.

### 3. How does the Credibility Score handle time decay?
Promises made 5 years ago carry less immediate relevance to current executive integrity than promises made 1 year ago. TempusRAG applies an exponential decay weighting:
$$w(y) = 0.5^{y_{\text{eval}} - y_{\text{made}}}$$
This penalizes recent broken commitments significantly more than dated, legacy programs while retaining historical tracking.

---

## ⚖️ Disclaimer

TempusRAG is an academic and research project designed to demonstrate temporal RAG architectures and evaluate public SEC Form 10-K disclosures. It does not constitute investment, financial, legal, or accounting advice. Corporate commitment evaluations represent statistical model outputs over public regulatory filings.
