# HardSec Scholar Evaluation Dataset

## Purpose

`hardsec_benchmark_v1.jsonl` is the first offline benchmark for the local corpus of
10 hardware-security papers. It is designed to compare Baseline RAG and Agentic RAG
with the same questions, paper IDs, evidence chunks, and expected pages.

## Dataset shape

| Category | Count |
| --- | ---: |
| Single-paper fact | 10 |
| Mechanism or attack principle | 5 |
| Threat model | 4 |
| Experiment or metric | 4 |
| Cross-paper comparison | 4 |
| Unanswerable | 3 |
| **Total** | **30** |

The 27 answerable samples cover all 10 indexed papers. Every answerable sample stores
a reference answer, one or more stable chunk IDs, and the exact page union implied by
those chunks. Unanswerable samples intentionally have no gold paper, chunk, or page.

## Validation workflow

Run the deterministic offline checks:

```powershell
.\.venv\Scripts\python.exe scripts\validate_evaluation_dataset.py
```

This verifies the JSONL schema, unique IDs and questions, category quota, coverage of
all indexed papers, existence and paper ownership of every chunk, and exact agreement
between chunk ranges and expected pages.

Run the optional online semantic review:

```powershell
.\.venv\Scripts\python.exe scripts\review_evaluation_dataset.py
```

The online review asks the project's citation-verification node whether each reference
answer is fully supported by only its declared evidence. It consumes LLM API tokens.

## Provenance and limitations

Question candidates were generated from bounded real-paper chunks, then checked against
the local corpus and manually inspected for evidence alignment. The semantic verifier
accepted all 27 answerable reference answers after two over-specific comparison answers
were narrowed to match their evidence.

This is a project benchmark, not a peer-reviewed or independently expert-annotated gold
dataset. Before publishing benchmark claims, a hardware-security domain expert should
review the questions, reference answers, and relevance labels. Results should therefore
be described as an internal offline evaluation rather than a universal quality score.
