# TempusRAG — no-LLM metrics

Generated `2026-10-27T08:44:36Z`. No Gemini/Groq/EDGAR calls.

**Checks:** 139/139 passed.

## System (cached 10-Ks on disk)

- Filings: **30**
- Tickers: AAPL, HIG, LPLA, MSFT, NVDA, TSLA
- Chunks after local chunker: **4718**
- Chunk token cap (word-split proxy): 500
- Mean chunk words: **433.2**
- Max chunk words: **500**
- Chunks over soft cap (cap+80): 0 (0.0%)

## Quality read

**Schemas, RRF, section boost, lost-in-middle, and scorer math are solid** — they match the locked contracts and are safe to cite as no-LLM correctness.

**Chunker is mixed.** Mean chunk size (433.2 words) sits under the 500-word cap, and IDs/prefixes are consistent. But **0 chunks (0.0%) overflow**, with a max of **500 words** (long tables/sentences force-appended). That hurts embedding quality and context windows; it is a real defect, not a failed unit test.

**Cached 10-K parse is coarse** (Item-level sections, few true subsections). Fine for indexing, weak for MD&A-only promise targeting until HTML heading parse improves.

HyDE/follow-up heuristics are rule-based only; they are not a substitute for LLM eval.

| File | Sections | Chunks | Mean words | Max words |
|---|---:|---:|---:|---:|
| AAPL_2021.txt | 22 | 87 | 402.7 | 500 |
| AAPL_2022.txt | 22 | 84 | 398.2 | 500 |
| AAPL_2023.txt | 23 | 80 | 390.0 | 500 |
| AAPL_2024.txt | 23 | 78 | 402.1 | 500 |
| AAPL_2025.txt | 23 | 79 | 400.7 | 500 |
| HIG_2021.txt | 10 | 323 | 477.4 | 500 |
| HIG_2022.txt | 10 | 307 | 478.0 | 500 |
| HIG_2023.txt | 12 | 305 | 475.1 | 500 |
| HIG_2024.txt | 12 | 292 | 477.0 | 500 |
| HIG_2025.txt | 12 | 290 | 475.2 | 500 |
| LPLA_2021.txt | 21 | 151 | 436.6 | 500 |
| LPLA_2022.txt | 22 | 148 | 432.5 | 500 |
| LPLA_2023.txt | 23 | 157 | 438.0 | 500 |
| LPLA_2024.txt | 23 | 164 | 444.6 | 500 |
| LPLA_2025.txt | 23 | 170 | 443 | 500 |
| MSFT_2022.txt | 22 | 128 | 428.0 | 500 |
| MSFT_2023.txt | 22 | 127 | 423.0 | 500 |
| MSFT_2024.txt | 23 | 130 | 431.4 | 500 |
| MSFT_2025.txt | 23 | 117 | 424.6 | 500 |
| MSFT_2026.txt | 23 | 119 | 428.0 | 500 |
| NVDA_2022.txt | 22 | 116 | 419.5 | 500 |
| NVDA_2023.txt | 22 | 121 | 420.7 | 500 |
| NVDA_2024.txt | 23 | 128 | 423.7 | 500 |
| NVDA_2025.txt | 23 | 130 | 427.9 | 500 |
| NVDA_2026.txt | 23 | 128 | 422.4 | 500 |
| TSLA_2021.txt | 22 | 169 | 438.5 | 500 |
| TSLA_2022.txt | 22 | 153 | 438.9 | 500 |
| TSLA_2023.txt | 23 | 147 | 434.5 | 500 |
| TSLA_2024.txt | 23 | 144 | 430.0 | 500 |
| TSLA_2025.txt | 23 | 146 | 432.8 | 500 |

## Retriever (synthetic ranks, no Chroma)

- RRF k = 60
- RRF order: `['A', 'C', 'B', 'D']`
- Lost-in-middle: `['A', 'C', 'E', 'D', 'B']`

## Scorer (fixture promises)

- Confidence (metric + deadline + verb + domain): **1.0**
- Confidence (vague tech line): **0.2**
- Fixture overall score: **85.0**

## README snippet

```
No-LLM eval (2026-10-27): 139/139 checks; 30 cached 10-Ks → 4718 chunks; RRF + section boost + lost-in-middle unit-checked; scorer fixture overall 85/100.
```
