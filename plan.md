# Implementation Plan — Multiwinner Voting for Evidence Selection in VLMs

## Context

COMP 5900 Multiagent Systems, Fall 2026, Carleton University. Individual graduate project.
**Research question:** Does replacing top-k retrieval with a multiwinner voting rule over multi-agent assessments produce a more complementary evidence set that improves VQA accuracy?

**Professor feedback (on early proposal):** "Look more deeply into the multiwinner voting problem — proportional representation concepts can ensure selected winners have inherent diversity (are not too homogeneous), which may improve quality of options selected."

**Timeline:** Presentations late November, final paper ~Dec 15. Implementation starts Nov 12.
**Compute:** GPU VM (free). All models run locally. Zero API cost.

---

## Project Overview

### Pipeline (per question)

```
Image + Question
       │
       ▼
┌──────────────┐
│  Caption Gen  │  (VLM generates text description of image)
└──────┬───────┘
       │  query = question + caption
       ▼
┌──────────────┐
│   Retriever   │  BM25 over Wikipedia passages
│  (top-20)     │
└──────┬───────┘
       │  20 candidate passages
       ▼
┌──────────────────────────────┐
│   Agent 1   Agent 2   Agent 3 │  Each agent produces an
│  (relevance) (unique) (gap)   │  approval ballot over passages
└──────────────┬───────────────┘
               │  3 approval ballots
               ▼
┌──────────────────────────────┐
│       Voting Rules            │
│  AV │ PAV │ CC │ Borda │ MES │
└──────────────┬───────────────┘
               │  committee of k=5 passages
               ▼
┌──────────────┐
│  Frozen VLM   │  image + question + k passages → answer
│  (LLaVA 7B)   │
└──────┬───────┘
       │
       ▼
    Evaluate (VQA accuracy vs ground truth)
```

### What each voting rule captures

| Rule | Principle | What it does for evidence | Professor's PR connection |
|------|-----------|--------------------------|---------------------------|
| AV (Approval Voting) | Excellence | Picks passages most agents agree on | No PR guarantee — can be homogeneous |
| PAV (Proportional AV) | Proportionality | Balances: each agent gets "fair share" of representation | Satisfies JR (justified representation) |
| CC (Chamberlin-Courant) | Diversity | Every agent has at least one approved passage in committee | Maximum coverage — inherent diversity |
| MES (Method of Equal Shares) | Strong proportionality | Each agent gets proportional "budget" to spend on passages | Satisfies EJR (extended justified representation) |
| Borda | Scoring | Sum of rank-based scores | No PR guarantee |
| Top-k (baseline) | Relevance only | Highest retriever scores | No agents, no PR |
| MMR (baseline) | Heuristic diversity | Greedy relevance-diversity tradeoff | No formal PR guarantee |

**Key insight from professor's feedback:** PAV satisfies Justified Representation (JR), MES satisfies Extended JR (EJR). These are formal guarantees that the selected committee is not too homogeneous — exactly what the professor suggested exploring. CC maximizes coverage (a different diversity notion). Comparing these against AV and top-k directly tests whether proportional representation improves evidence quality.

---

## Repository Structure

```
MAS-proj/
├── README.md
├── requirements.txt
├── config.py                  # paths, model names, hyperparams
├── data/
│   └── download_data.py       # script to download OK-VQA + Wikipedia
├── src/
│   ├── retrieval.py           # BM25 retriever over Wikipedia passages
│   ├── captioner.py           # VLM image → caption for retrieval query
│   ├── agents.py              # LLM agent prompts + ballot generation
│   ├── voting.py              # all voting rules: AV, PAV, CC, MES, Borda, MMR
│   ├── vqa.py                 # frozen VLM answering with selected passages
│   └── evaluate.py            # VQA accuracy + diversity metrics
├── scripts/
│   ├── run_pipeline.py        # end-to-end: retrieve → vote → answer → eval
│   └── run_experiments.py     # sweep over voting rules, k, num_agents
├── results/                   # experiment outputs (JSON + plots)
└── paper/
    ├── proposal.tex
    └── references.bib
```

---

## Implementation Steps

### Phase 1: Environment + Data (Day 1-2)

**1.1 VM setup**
```bash
# Python environment
python -m venv venv
source venv/bin/activate
pip install torch transformers accelerate bitsandbytes
pip install rank-bm25 datasets pillow tqdm matplotlib pandas

# Ollama for LLM agents
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral:7b-instruct
ollama pull llava:7b        # or use transformers for LLaVA
```

**1.2 Download OK-VQA**
- Dataset: `datasets` library → `HuggingFaceM4/OK-VQA` or download from official source
- Images: COCO 2017 val split (OK-VQA uses COCO images)
- Use a subset for development: ~200 questions. Full eval on ~500-1000.

**1.3 Wikipedia passage corpus**
- Option A: Download pre-built Wikipedia passages from DPR project (~21M passages, ~13GB)
- Option B: Use a smaller Wikipedia dump and chunk it yourself
- Option A is recommended — it's what most KB-VQA papers use

### Phase 2: Retrieval Pipeline (Day 2-3)

**2.1 `captioner.py`** — Generate text caption from image
- Use LLaVA or BLIP-2 to caption the image: "Describe this image in detail"
- The caption provides text for the retrieval query
- Query = question + caption (concatenated)

**2.2 `retrieval.py`** — BM25 retrieval
- Index Wikipedia passages with `rank_bm25`
- For each question, retrieve top-20 passages by BM25 score
- Store results: `{question_id: [passage_1, passage_2, ..., passage_20]}` with scores

**Important:** This step is slow on first run (indexing). Cache the retrieval results to disk so you don't re-retrieve every experiment.

### Phase 3: Agent Ballots (Day 3-5)

**3.1 `agents.py`** — Three agents with different perspectives

Each agent receives: question, image caption, and the 20 candidate passages.
Each agent outputs: a list of approved passage indices (approval ballot).

**Agent prompts (the core design):**

Agent 1 — "Relevance assessor":
```
Given this question about an image and 20 candidate passages, 
which passages contain facts that could help answer the question?
Approve passages that are relevant. List the indices of approved passages.
```

Agent 2 — "Uniqueness assessor":
```
Given this question and 20 candidate passages, which passages provide 
UNIQUE information not found in the other passages? Approve passages 
that contribute distinct facts. List the indices of approved passages.
```

Agent 3 — "Gap filler":
```
Given this question and 20 candidate passages, which passages address 
aspects of the question that the most obviously relevant passages might miss? 
Approve passages that fill gaps. List the indices of approved passages.
```

**Implementation:** Call Ollama API (Mistral 7B) for each agent. Parse the output to extract approved indices. Handle parsing errors (retry with structured prompt if needed).

**Cache all ballots to disk** — you'll re-run voting rules many times without re-querying agents.

### Phase 4: Voting Rules (Day 4-6)

**4.1 `voting.py`** — Implement all rules

All rules take: `ballots: list[set[int]]` (each agent's approved passages) and `k: int` (committee size). Return: `committee: list[int]` (selected passage indices).

**Approval Voting (AV):**
```python
def approval_voting(ballots, k):
    scores = Counter()
    for ballot in ballots:
        for p in ballot:
            scores[p] += 1
    return [p for p, _ in scores.most_common(k)]
```

**Proportional Approval Voting (PAV):**
```python
def pav(ballots, k):
    # Greedy sequential: pick passage maximizing sum of 1/(1 + count_i)
    # where count_i = how many already-selected passages agent i approved
    committee = []
    agent_counts = [0] * len(ballots)
    for _ in range(k):
        best, best_score = -1, -1
        for p in all_passages - set(committee):
            score = sum(1/(1 + agent_counts[i]) 
                       for i, b in enumerate(ballots) if p in b)
            if score > best_score:
                best, best_score = p, score
        committee.append(best)
        for i, b in enumerate(ballots):
            if best in b:
                agent_counts[i] += 1
    return committee
```

**Chamberlin-Courant (CC):**
```python
def chamberlin_courant(ballots, k):
    # Greedy: pick passage that covers the most uncovered agents
    committee = []
    covered = set()
    for _ in range(k):
        best, best_new = -1, -1
        for p in all_passages - set(committee):
            new_covered = sum(1 for i, b in enumerate(ballots) 
                            if p in b and i not in covered)
            if new_covered > best_new:
                best, best_new = p, new_covered
        committee.append(best)
        for i, b in enumerate(ballots):
            if best in b:
                covered.add(i)
    return committee
```

**Method of Equal Shares (MES) — for EJR (professor's PR suggestion):**
```python
def method_of_equal_shares(ballots, k):
    # Each agent starts with budget = k / len(ballots)
    # Passages are added if enough agents can "afford" them
    # Satisfies EJR — strongest proportionality guarantee
    n = len(ballots)
    budgets = [k / n] * n
    committee = []
    for _ in range(k):
        best, best_rho = -1, float('inf')
        for p in all_passages - set(committee):
            supporters = [i for i, b in enumerate(ballots) if p in b]
            if not supporters:
                continue
            # find minimum rho such that sum of min(budget_i, rho) >= 1
            sorted_budgets = sorted([budgets[i] for i in supporters])
            total, rho = 0, 0
            for j, bud in enumerate(sorted_budgets):
                remaining = len(sorted_budgets) - j
                needed = 1 - total
                rho = needed / remaining
                if rho <= bud:
                    break
                total += bud
            else:
                continue  # can't afford
            if rho < best_rho:
                best, best_rho = p, rho
                best_supporters = supporters
        if best == -1:
            break
        committee.append(best)
        for i in best_supporters:
            budgets[i] -= min(budgets[i], best_rho)
    # if committee < k, fill remaining with AV
    if len(committee) < k:
        remaining = approval_voting_excluding(ballots, k - len(committee), committee)
        committee.extend(remaining)
    return committee
```

**Borda count:**
```python
def borda(rankings, k):
    # Each agent provides a ranking. Borda score = (m - rank).
    # Sum across agents, pick top-k.
    scores = Counter()
    m = len(rankings[0])
    for ranking in rankings:
        for rank, p in enumerate(ranking):
            scores[p] += (m - rank)
    return [p for p, _ in scores.most_common(k)]
```

**MMR baseline:**
```python
def mmr(passages, retriever_scores, k, lambda_param=0.5):
    # Maximal Marginal Relevance (Carbonell & Goldstein 1998)
    # Greedy: pick passage maximizing lambda*relevance - (1-lambda)*max_sim_to_selected
    selected = []
    for _ in range(k):
        best, best_score = -1, -float('inf')
        for i, p in enumerate(passages):
            if i in selected:
                continue
            rel = retriever_scores[i]
            div = max(similarity(p, passages[j]) for j in selected) if selected else 0
            score = lambda_param * rel - (1 - lambda_param) * div
            if score > best_score:
                best, best_score = i, score
        selected.append(best)
    return selected
```

### Phase 5: VLM Answering (Day 5-7)

**5.1 `vqa.py`** — Feed selected passages to frozen VLM

```python
def answer_question(image, question, passages, model):
    context = "\n\n".join(passages)
    prompt = f"""Based on the image and the following reference passages, 
answer the question.

Reference passages:
{context}

Question: {question}
Answer:"""
    return model.generate(image, prompt)
```

- Use LLaVA-1.5-7B (or 13B if GPU allows) via transformers
- Load in 4-bit quantization to save VRAM
- Generate short answers (max 20 tokens)

### Phase 6: Evaluation (Day 6-8)

**6.1 `evaluate.py`** — Metrics

**Primary metric:** VQA accuracy (standard OK-VQA evaluation — soft accuracy matching against 10 human answers per question)

**Secondary metrics (for analysis):**
- **Passage diversity:** Average pairwise cosine distance between selected passages (using sentence embeddings)
- **Coverage:** Number of distinct topic clusters represented in the selected set
- **Redundancy:** Average pairwise similarity (inverse of diversity)
- **Proportionality analysis:** For each voting rule, check whether JR / EJR are satisfied (count violations)

### Phase 7: Experiments (Day 7-12)

**7.1 Main experiment — Voting rules comparison**
- Dataset: 500 OK-VQA questions (random subset)
- Agents: 3 agents (Mistral 7B, different prompts)
- Committee size: k=5
- Compare: Top-k, MMR, AV, PAV, CC, MES, Borda
- Report: VQA accuracy, passage diversity, JR/EJR satisfaction rates

**7.2 Ablation — Committee size k**
- Fix best voting rule(s) from 7.1
- Sweep k ∈ {3, 5, 7, 10}
- Report: accuracy vs. k for each rule

**7.3 Ablation — Number of agents**
- Sweep N ∈ {1, 2, 3, 4} agents
- With N=1: single agent, all voting rules reduce to top-k by that agent
- Report: does more agents + voting outperform single agent?

**7.4 Analysis — Proportional representation (professor's suggestion)**
- For each question, count JR violations per voting rule
- Correlate: do questions where AV/top-k violates JR show lower accuracy than PAV/MES?
- Show concrete examples where proportional rules selected a missing-but-crucial passage

**7.5 Analysis — Qualitative examples**
- Pick 5-10 questions where voting rules disagree
- Show: what passages each rule selected, the VLM's answer, and which was correct
- Illustrate the diversity vs. excellence tradeoff

---

## Timeline

| Days | Dates | Milestone |
|------|-------|-----------|
| 1-2 | Nov 12-13 | VM setup, data download, environment working |
| 2-3 | Nov 13-15 | Retrieval pipeline + caching |
| 3-5 | Nov 15-17 | Agent prompts + ballot generation + caching |
| 4-6 | Nov 16-18 | Voting rules implemented + unit tested |
| 5-7 | Nov 17-19 | VLM answering pipeline working end-to-end |
| 7-10 | Nov 19-22 | Run main experiments + ablations |
| 10-12 | Nov 22-24 | Analysis + plots + presentation prep |
| ~Nov 25 | Late Nov | **Presentation** |
| 13-18 | Nov 25 - Dec 5 | Paper writing (intro, related work, method, results) |
| 18-21 | Dec 5-10 | Revisions + additional experiments if needed |
| ~Dec 15 | | **Final paper due** |

---

## Paper Structure (10-page Word or 7-page LaTeX AAMAS)

1. **Introduction** (~1 page) — VLMs need external evidence; top-k is redundant; multiwinner voting offers principled diversity
2. **Related Work** (~1.5 pages) — KB-VQA retrieval, multi-agent debate for LLMs, social choice in ML
3. **Background** (~1 page) — Multiwinner voting rules (AV, PAV, CC, MES), proportionality axioms (JR, EJR)
4. **Method** (~1.5 pages) — Pipeline: retrieval → agent ballots → voting → VLM answering
5. **Experiments** (~2 pages) — Setup, main results table, ablations
6. **Analysis** (~1.5 pages) — PR violation analysis, qualitative examples, diversity metrics
7. **Discussion + Conclusion** (~1 page) — What worked, limitations, future work

---

## Key References

| Paper | Why it matters |
|-------|---------------|
| Gawron & Faliszewski 2022 (arXiv:2202.03385) | Bridge paper: multiwinner voting → diverse recommendations |
| Lackner & Skowron 2018 (arXiv:1704.02453) | Axiomatic foundation: AV=excellence, CC=diversity, PAV=proportionality |
| "Debate or Vote" 2025 (arXiv:2508.17536) | Negative result: debate is martingale, voting does the work |
| "Debating for Better Reasoning" 2025 (arXiv:2505.14627) | Multi-agent debate for VQA |
| DART 2025 (arXiv:2512.07132) | Multi-agent disagreement for multimodal reasoning |
| BERAG 2026 (arXiv:2604.22678) | Bayesian evidence aggregation for KB-VQA |
| "Candidate Qualities" 2026 (arXiv:2609.08830) | Quality-aware proportional representation |
| Carbonell & Goldstein 1998 | MMR baseline — heuristic diversity |
| Liu et al. 2023 (LLaVA) | Frozen VLM |
| Lewis et al. 2020 | RAG concept |

---

## Presentation Notes

The presentation should walk through:
1. **Problem** (1-2 slides): top-k retrieval picks redundant passages
2. **Idea** (2-3 slides): multiwinner voting rules — the three principles, what PR means
3. **Method** (2-3 slides): pipeline diagram, agent design, which rules you used
4. **Results** (3-4 slides): main comparison table, ablation plots, PR violation analysis
5. **Qualitative examples** (1-2 slides): concrete cases where CC/PAV found the missing passage
6. **Takeaway** (1 slide): does proportional representation help select better evidence?

---
---

# Appendix: Literature Review & Research Context

## The Gap

Nobody has applied multiwinner voting rules to select a diverse, complementary set of evidence passages for a VLM. Multi-agent debate systems use ad-hoc aggregation (majority vote or judge model). KB-VQA retrieval systems optimize relevance but don't formalize diversity through social choice axioms.

**Critical negative result:** "Debate or Vote" (arXiv 2508.17536) proves multi-agent debate dynamics form a **martingale** — expected correctness does not improve through debate alone. Majority voting accounts for most gains. This means the project focuses on what voting rules can do that simple top-k cannot — ensure diversity and complementarity — rather than betting on debate.

## Literature Search Results

| Area | Findings |
|---|---|
| Multi-agent debate for VLMs/LLMs | Rich and active. Aggregation is almost always majority vote or judge model. |
| Social choice applied to ML | Growing. Vote'n'Rank for benchmarking, alignment papers argue social choice should guide RLHF. |
| Multiwinner voting / committee selection | Strong theoretical literature (PAV, CC, proportionality axioms). Bridge paper: Gawron & Faliszewski (2022) use multiwinner voting for diverse movie recommendations. |
| KB-VQA with retrieval | Active. BERAG (2026) treats documents as ensemble members. UniHEAR addresses complementary sources. |

## Shortlist of 7 Papers

### Paper 1 — "Using Multiwinner Voting to Search for Movies"
Gawron & Faliszewski, 2022. arXiv:2202.03385
- **Relevance:** Bridge paper. Uses PAV, CC, Bloc for diverse movie recommendations. Different rules = different diversity levels. Replace "movies" with "evidence passages."
- **Read first:** Sections 1–3 (framework, voting rules, diversity), Section 5 (MovieLens experiments).

### Paper 2 — "Consistent Approval-Based Multi-Winner Rules"
Lackner & Skowron, 2017. arXiv:1704.02453
- **Relevance:** Axiomatic foundation. Three principles: individual excellence (AV), diversity (CC), proportionality (PAV).
- **Read first:** Section 1 + rule definitions + characterization theorems.

### Paper 3 — "Debate or Vote: Which Yields Better Decisions in Multi-Agent LLMs?"
arXiv:2508.17536, Aug 2025
- **Relevance:** Critical negative result. Debate is a martingale. Voting does most work.
- **Read first:** Abstract, Sections 1–2, martingale theorem.

### Paper 4 — "Debating for Better Reasoning: An Unsupervised Multimodal Approach"
arXiv:2505.14627, May 2025
- **Relevance:** Two VLM experts debate; text-only judge adjudicates. Debate improves VQA on disagreement instances.

### Paper 5 — "DART: Multi-Agent Disagreement for Tool Recruitment"
arXiv:2512.07132, Dec 2025
- **Relevance:** Disagreement triggers tool recruitment for VQA. Tests on A-OKVQA. Aggregation via single agent (not formal voting).

### Paper 6 — "BERAG: Bayesian Ensemble RAG for KB-VQA"
arXiv:2604.22678, Apr 2026
- **Relevance:** Individual document weighting, Bayesian ensemble for KB-VQA. Shows how you aggregate evidence matters.

### Paper 7 — "Approval-Based Multiwinner Voting with Candidate Qualities"
arXiv:2609.08830, Sep 2026
- **Relevance:** Quality-aware proportional representation. Passages have relevance (quality) AND coverage (diversity). Reciprocal axioms retain ≥3/4 optimal quality with proportionality.

### Supplementary Papers
- "Voting or Consensus?" (arXiv:2502.19130) — 7 decision protocols compared; voting wins on reasoning.
- "Vote'n'Rank" (arXiv:2210.05769) — Social choice for ML benchmarking.
- "Diversity of Thought" (arXiv:2410.12853) — Diverse models > homogeneous in debate.
- "UniHEAR" (arXiv:2608.01147) — Heterogeneous-source retrieval for KB-VQA.
- KB-VQA Survey (arXiv:2504.17547) — Landscape overview.

## Three Project Questions (from scoping phase)

**Q1 (Chosen):** Can multiwinner voting rules select more complementary evidence for a frozen VLM than top-k retrieval? → This is what we're building.

**Q2 (Extension if time allows):** Does deliberation before voting change which passages are selected? → Connects to iterative voting (Meir et al. from Lecture 9).

**Q3 (Theoretical extension):** Can proportionality axioms (JR, EJR) formalize "complementary evidence"? → Professor's feedback points here. Partially incorporated via PAV (JR) and MES (EJR) in the main experiment.
