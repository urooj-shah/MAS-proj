"""Unit tests for voting rules with hand-crafted ballots."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.voting import (
    top_k_baseline, mmr, approval_voting, pav,
    chamberlin_courant, method_of_equal_shares, borda, select_committee,
)


def make_passages(n=10):
    """Create dummy passages with decreasing BM25 scores."""
    return [
        {"id": i, "title": f"Passage {i}", "text": f"This is passage number {i} about topic {i % 3}.",
         "score": 1.0 - i * 0.05, "rank": i}
        for i in range(n)
    ]


def test_top_k():
    passages = make_passages(10)
    result = top_k_baseline(passages, k=3)
    assert result == [0, 1, 2], f"Expected [0,1,2], got {result}"
    print("  top_k: PASS")


def test_approval_voting():
    ballots = [{0, 1, 2}, {0, 1, 3}, {0, 2, 3}]
    candidates = set(range(10))
    result = approval_voting(ballots, candidates, k=3)
    assert 0 in result, f"Candidate 0 (3 approvals) must be selected, got {result}"
    assert len(result) == 3
    print("  AV: PASS")


def test_pav_diversifies():
    """PAV should diversify: if agent 1 likes {0,1} and agent 2 likes {2,3},
    PAV should pick from both agents, not just the highest-approval candidates."""
    ballots = [{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}]
    candidates = set(range(10))
    scores = {i: 1.0 - i * 0.05 for i in range(10)}
    result = pav(ballots, candidates, k=4, passage_scores=scores)
    agent1_count = sum(1 for c in result if c in ballots[0])
    agent2_count = sum(1 for c in result if c in ballots[1])
    assert agent1_count >= 1 and agent2_count >= 1, \
        f"PAV should diversify across agents. Got agent1={agent1_count}, agent2={agent2_count}"
    print("  PAV: PASS (diversifies across agents)")


def test_cc_covers_all():
    """CC should cover all agents before selecting more from the same agent."""
    ballots = [{0, 1}, {2, 3}, {4, 5}]
    candidates = set(range(10))
    scores = {i: 1.0 - i * 0.05 for i in range(10)}
    result = chamberlin_courant(ballots, candidates, k=3, passage_scores=scores)
    covered = set()
    for i, b in enumerate(ballots):
        if set(result) & b:
            covered.add(i)
    assert len(covered) == 3, f"CC should cover all 3 agents, covered {covered}"
    print("  CC: PASS (covers all agents)")


def test_mes_proportional():
    """MES should give proportional representation."""
    ballots = [{0, 1, 2}, {3, 4, 5}, {6, 7, 8}]
    candidates = set(range(10))
    scores = {i: 0.5 for i in range(10)}
    result = method_of_equal_shares(ballots, candidates, k=3, passage_scores=scores)
    covered = set()
    for i, b in enumerate(ballots):
        if set(result) & b:
            covered.add(i)
    assert len(covered) == 3, f"MES should cover all 3 agents, covered {covered}"
    print("  MES: PASS (proportional)")


def test_mes_completion():
    """MES should complete with AV when budget exhausted."""
    ballots = [{0}, {1}, {2}]
    candidates = set(range(10))
    scores = {i: 1.0 - i * 0.05 for i in range(10)}
    result = method_of_equal_shares(ballots, candidates, k=5, passage_scores=scores)
    assert len(result) == 5, f"MES should return exactly k=5, got {len(result)}"
    print("  MES completion: PASS")


def test_borda():
    rankings = [[2, 0, 1], [2, 1, 0], [0, 2, 1]]
    candidates = set(range(10))
    result = borda(rankings, candidates, k=2)
    assert 2 in result, f"Candidate 2 should be top Borda winner, got {result}"
    print("  Borda: PASS")


def test_select_committee_dispatch():
    """Test the dispatcher works for all rules."""
    passages = make_passages(10)
    ballots_dict = {
        "relevance": {"approved": [0, 1, 2, 3], "ranking": [0, 1, 2, 3]},
        "uniqueness": {"approved": [2, 3, 4, 5], "ranking": [2, 3, 4, 5]},
        "gap_filler": {"approved": [4, 5, 6, 7], "ranking": [4, 5, 6, 7]},
    }

    for rule in ["top_k", "mmr", "av", "pav", "cc", "mes", "borda"]:
        result = select_committee(rule, ballots_dict, passages, k=3)
        assert len(result) == 3, f"{rule}: expected 3 selections, got {len(result)}"
        assert all(0 <= i < 10 for i in result), f"{rule}: indices out of range: {result}"
    print("  Dispatcher: PASS (all rules)")


if __name__ == "__main__":
    print("Testing voting rules...")
    test_top_k()
    test_approval_voting()
    test_pav_diversifies()
    test_cc_covers_all()
    test_mes_proportional()
    test_mes_completion()
    test_borda()
    test_select_committee_dispatch()
    print("\nAll voting rule tests passed!")
