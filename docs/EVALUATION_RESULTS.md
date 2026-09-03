# HardSec Scholar Baseline vs. Agentic RAG

## Experiment setup

- Date: 2026-08-21
- Corpus: 10 hardware-security papers, 157 pages, 324 chunks
- Dataset: 30 questions (`hardsec_benchmark_v1.jsonl`)
- Answerable / unanswerable: 27 / 3
- LLM: DeepSeek `deepseek-v4-flash`, non-thinking function calling
- Embedding: SiliconFlow `BAAI/bge-m3`
- Baseline: one Dense top-10 retrieval followed by one grounded generation
- Agentic: classify, expand, Hybrid retrieval, RRF, FlashRank, grade, bounded
  rewrite, generate, and semantic citation verification
- Both systems search the complete corpus; gold paper IDs are not passed to either system.

## Final results

The table uses the latest successful result for each of the 30 questions. Retrieval
metrics exclude the three unanswerable questions where no gold chunk exists.

| Metric | Dense Baseline | Agentic RAG | Direction |
| --- | ---: | ---: | --- |
| Recall@5 | 0.6204 | 0.5494 | Baseline higher |
| Recall@10 | 0.7346 | 0.7870 | Agentic higher |
| MRR | 0.6584 | 0.5693 | Baseline higher |
| Correct-paper hit@10 | 1.0000 | 1.0000 | Tie |
| Correct-page hit@10 | 0.9630 | 1.0000 | Agentic higher |
| All-round gold-chunk recall | 0.7346 | 0.8519 | Agentic higher |
| Answer/refusal correctness | 0.9667 | 0.9000 | Baseline higher |
| Reference token F1 | 0.4797 | 0.4616 | Baseline higher |
| Exact-gold citation precision | 0.5177 | 0.5004 | Baseline higher |
| Exact-gold citation recall | 0.5278 | 0.4568 | Baseline higher |
| Citation page precision | 0.7575 | 0.7041 | Baseline higher |
| Citation paper precision | 0.9259 | 0.8889 | Baseline higher |
| Mean latency | 2.3350 s | 8.2398 s | Baseline 3.53× faster |
| P95 latency | 3.8679 s | 12.0928 s | Baseline lower |
| Final-run tokens | 128,510 | 324,954 | Baseline 2.53× lower |
| Estimated final-run LLM cost | $0.019459 | $0.048803 | Baseline 2.51× lower |

The cost estimate applies the explicit runner rates of $0.14 per million input tokens
and $0.28 per million output tokens, conservatively treating input as cache misses. It
does not include SiliconFlow embedding charges. Development-time failures and repeated
tuning runs raised the observed attempt costs to $0.019459 for Baseline and $0.067801
for Agentic RAG; these attempt costs are not used for the fair 30-vs-30 comparison.

## Category observations

- Threat-model questions were the clearest Agentic win: Recall@10 rose from 0.75 to
  1.00, answer/refusal correctness from 0.75 to 1.00, and reference token F1 from
  0.4108 to 0.6017.
- On single-paper facts, Agentic Recall@10 improved from 0.65 to 0.75, but one
  answerable item was refused and answer overlap remained slightly lower.
- On mechanism questions, all-round recall improved from 0.80 to 0.95, but one
  answerable item was refused after the bounded safety checks.
- Both systems reached perfect answer/refusal correctness on experiment and
  unanswerable questions.
- Cross-paper comparison remains the weakest Agentic category. All-round recall was
  slightly higher (0.6042 vs. 0.5833), but one comparison was still refused and exact
  citation metrics did not improve.

## Engineering findings and changes

The first evaluation attempts revealed three issues that unit tests with fixed model
responses could not expose:

1. DeepSeek sometimes returned HTTP 200 without structured tool arguments. Structured
   invocation now retries model output plus Pydantic validation as one operation. After
   three empty plans, only question classification falls back to deterministic rules;
   network and authentication failures are not hidden.
2. The evidence grader originally demanded an existing side-by-side comparison in the
   papers. It now permits synthesis from separate evidence belonging to both papers.
3. A conservative evidence grade previously caused immediate refusal after retries even
   when valid evidence had been selected. The workflow now lets generation and both
   citation checks make the final fail-closed decision.

All 30 Baseline and 30 Agentic final runs completed successfully after these general
stability changes. Results are append-only and resumable; summaries select the latest
record for each `(sample_id, variant)` pair.

## Interpretation

This experiment does not support the claim that adding an Agent automatically improves
overall RAG quality. The Agent increases evidence breadth, Recall@10, and page coverage,
and is especially useful for threat-model questions. However, the current evidence
grader remains conservative, early ranking quality is weaker, and the extra calls cost
roughly 2.5× the tokens and 3.5× the latency of the Dense baseline.

For an interview, the defensible conclusion is that Agentic RAG introduces measurable
trade-offs: it improves search coverage but requires calibrated routing and grading to
convert that coverage into better answers. This is stronger evidence than presenting a
hand-picked demo and claiming an unmeasured universal improvement.

## Limitations

- The benchmark contains only 30 questions over 10 papers.
- Candidate questions were machine-generated and evidence-reviewed, not independently
  annotated by multiple hardware-security experts.
- Exact-gold citation metrics penalize valid alternative chunks that were not selected
  as the original gold evidence.
- Reference token F1 is a transparent lexical signal, not a semantic correctness judge.
- Model outputs can vary across runs even with temperature set to zero.
- The reported dollar estimate excludes embedding service charges and may become stale
  when provider pricing changes.
