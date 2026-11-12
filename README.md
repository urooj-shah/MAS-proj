# Multiwinner Voting for KB-VQA Evidence Selection

COMP 5900 Multiagent Systems — Carleton University, Fall 2026

## Research Question

Does replacing top-k retrieval with multiwinner voting rules over multi-agent assessments produce more complementary evidence for a frozen VLM on KB-VQA?

## Pipeline

```
Image + Question → BLIP caption → BM25 retrieval (top-20 from 3M Wikipedia passages)
    → 3 LLM agents (relevance, uniqueness, gap-filler) produce approval ballots
    → Voting rules (AV, PAV, CC, MES, Borda) select k=5 passages
    → LLaVA-1.5-13B answers using selected passages
    → Evaluate VQA accuracy on OK-VQA
```

## Voting Rules

| Rule | Principle | PR Guarantee |
|------|-----------|-------------|
| Top-k | Baseline (BM25 scores) | None |
| MMR | Heuristic diversity | None |
| AV | Most approvals | None |
| PAV | Proportional representation | JR |
| CC | Maximum coverage | Coverage |
| MES | Budget-based proportionality | EJR |
| Borda | Rank-based scoring | None |

## Setup

```bash
pip install -r requirements.txt
python data/download_data.py      # downloads OK-VQA + Wikipedia passages
python scripts/run_pipeline.py    # runs full pipeline
```

## Project Structure

```
config.py              — paths, model names, hyperparameters
src/
  captioner.py         — BLIP-base image captioning
  retrieval.py         — BM25 retrieval over Wikipedia
  agents.py            — Mistral-7B agent ballot generation
  voting.py            — all voting rules (AV, PAV, CC, MES, Borda, MMR)
  vqa.py               — LLaVA-1.5-13B VQA answering
  evaluate.py          — VQA accuracy, diversity metrics, PR analysis
scripts/
  run_pipeline.py      — end-to-end pipeline
  run_experiments.py   — experiment sweeps and ablations
```
