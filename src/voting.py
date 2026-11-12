"""Multiwinner voting rules for evidence passage selection."""

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def top_k_baseline(passages, k=5):
    """Select k passages with highest BM25 scores."""
    ranked = sorted(range(len(passages)), key=lambda i: passages[i]["score"], reverse=True)
    return ranked[:k]


def mmr(passages, k=5, lambda_param=0.5):
    """Maximal Marginal Relevance (Carbonell & Goldstein 1998)."""
    texts = [p["text"] for p in passages]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(tfidf)

    scores = np.array([p["score"] for p in passages])
    if scores.max() > 0:
        scores = scores / scores.max()

    selected = []
    for _ in range(k):
        best, best_score = -1, -float("inf")
        for i in range(len(passages)):
            if i in selected:
                continue
            rel = scores[i]
            div = max(sim_matrix[i][j] for j in selected) if selected else 0.0
            score = lambda_param * rel - (1 - lambda_param) * div
            if score > best_score:
                best, best_score = i, score
        if best == -1:
            break
        selected.append(best)
    return selected


def approval_voting(ballots, candidates, k=5, passage_scores=None):
    """Count approvals per candidate, return top-k. Tie-break by BM25 score."""
    counts = Counter()
    for ballot in ballots:
        for c in ballot:
            if c in candidates:
                counts[c] += 1

    def sort_key(c):
        score = passage_scores[c] if passage_scores else 0
        return (counts.get(c, 0), score)

    ranked = sorted(candidates, key=sort_key, reverse=True)
    return ranked[:k]


def pav(ballots, candidates, k=5, passage_scores=None):
    """Proportional Approval Voting (greedy sequential). Satisfies JR."""
    committee = []
    agent_counts = [0] * len(ballots)
    remaining = set(candidates)

    for _ in range(k):
        best, best_score = -1, -1.0
        for c in remaining:
            score = sum(
                1.0 / (1 + agent_counts[i])
                for i, b in enumerate(ballots)
                if c in b
            )
            if score > best_score or (
                score == best_score
                and passage_scores
                and c != -1
                and (best == -1 or passage_scores.get(c, 0) > passage_scores.get(best, 0))
            ):
                best, best_score = c, score
        if best == -1:
            break
        committee.append(best)
        remaining.discard(best)
        for i, b in enumerate(ballots):
            if best in b:
                agent_counts[i] += 1
    return committee


def chamberlin_courant(ballots, candidates, k=5, passage_scores=None):
    """Greedy CC: maximize agent coverage. Tie-break by approval count then BM25."""
    committee = []
    covered = set()
    remaining = set(candidates)

    for _ in range(k):
        best, best_new, best_total = -1, -1, -1
        for c in remaining:
            new_covered = sum(
                1 for i, b in enumerate(ballots)
                if c in b and i not in covered
            )
            total_approvals = sum(1 for b in ballots if c in b)
            bm25 = passage_scores.get(c, 0) if passage_scores else 0

            if (new_covered > best_new) or (
                new_covered == best_new
                and (total_approvals > best_total or (total_approvals == best_total and bm25 > 0))
            ):
                best, best_new, best_total = c, new_covered, total_approvals
        if best == -1:
            break
        committee.append(best)
        remaining.discard(best)
        for i, b in enumerate(ballots):
            if best in b:
                covered.add(i)
    return committee


def method_of_equal_shares(ballots, candidates, k=5, passage_scores=None):
    """MES with AV completion. Satisfies EJR."""
    n = len(ballots)
    budgets = [k / n] * n
    committee = []
    remaining = set(candidates)

    for _ in range(k):
        best, best_rho = -1, float("inf")
        best_supporters = []

        for c in remaining:
            supporters = [i for i, b in enumerate(ballots) if c in b]
            if not supporters:
                continue

            sorted_budgets = sorted([budgets[i] for i in supporters])
            total = 0.0
            rho = 0.0
            affordable = False

            for j, bud in enumerate(sorted_budgets):
                rem = len(sorted_budgets) - j
                needed = 1.0 - total
                rho = needed / rem
                if rho <= bud:
                    affordable = True
                    break
                total += bud

            if not affordable:
                continue

            if rho < best_rho or (
                rho == best_rho
                and passage_scores
                and passage_scores.get(c, 0) > passage_scores.get(best, 0)
            ):
                best, best_rho = c, rho
                best_supporters = supporters

        if best == -1:
            break

        committee.append(best)
        remaining.discard(best)
        for i in best_supporters:
            budgets[i] = max(0, budgets[i] - min(budgets[i], best_rho))

    if len(committee) < k:
        fill = approval_voting(
            ballots, remaining, k - len(committee), passage_scores
        )
        committee.extend(fill)

    return committee


def borda(rankings, candidates, k=5, passage_scores=None):
    """Borda count from agent rankings. Unranked candidates get score 0."""
    scores = Counter()
    for ranking in rankings:
        m = len(ranking)
        for rank, c in enumerate(ranking):
            if c in candidates:
                scores[c] += (m - rank)

    def sort_key(c):
        bm25 = passage_scores.get(c, 0) if passage_scores else 0
        return (scores.get(c, 0), bm25)

    ranked = sorted(candidates, key=sort_key, reverse=True)
    return ranked[:k]


def select_committee(rule, ballots_dict, passages, k=5):
    """
    Dispatch to the appropriate voting rule.

    ballots_dict: dict with keys like 'relevance', 'uniqueness', 'gap_filler',
                  each containing {'approved': set[int], 'ranking': list[int]}
    passages: list of dicts with 'score' and 'text' fields
    """
    candidates = set(range(len(passages)))
    passage_scores = {i: p["score"] for i, p in enumerate(passages)}

    if rule == "top_k":
        return top_k_baseline(passages, k)

    if rule == "mmr":
        return mmr(passages, k)

    approval_ballots = [
        set(b["approved"]) for b in ballots_dict.values()
    ]
    rankings = [
        list(b.get("ranking", list(b["approved"]))) for b in ballots_dict.values()
    ]

    if rule == "av":
        return approval_voting(approval_ballots, candidates, k, passage_scores)
    elif rule == "pav":
        return pav(approval_ballots, candidates, k, passage_scores)
    elif rule == "cc":
        return chamberlin_courant(approval_ballots, candidates, k, passage_scores)
    elif rule == "mes":
        return method_of_equal_shares(approval_ballots, candidates, k, passage_scores)
    elif rule == "borda":
        return borda(rankings, candidates, k, passage_scores)
    else:
        raise ValueError(f"Unknown voting rule: {rule}")
