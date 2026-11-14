"""Analyze JR/EJR violations on cached ballots and correlate with accuracy."""

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.voting import select_committee


def load_cached_ballots():
    ballots = {}
    for f in config.BALLOT_CACHE.glob("*.json"):
        qid = f.stem
        ballots[qid] = json.loads(f.read_text())
    return ballots


def load_cached_retrieval():
    retrieval = {}
    for f in config.RETRIEVAL_CACHE.glob("*.json"):
        qid = f.stem
        data = json.loads(f.read_text())
        retrieval[qid] = data if isinstance(data, list) else data.get("passages", data)
    return retrieval


def load_cached_answers(rule):
    answers = {}
    rule_dir = config.ANSWER_CACHE / rule
    if rule_dir.exists():
        for f in rule_dir.glob("*.json"):
            qid = f.stem
            data = json.loads(f.read_text())
            answers[qid] = data.get("answer", "")
    return answers


def analyze_jr(ballots_dict, retrieval, k=5):
    """Count JR violations per rule per question."""
    rules = ["av", "pav", "cc", "mes", "borda"]
    results = defaultdict(lambda: {"violations": 0, "total": 0, "violated_qids": []})

    for qid, ballots in ballots_dict.items():
        if qid not in retrieval:
            continue
        passages = retrieval[qid]
        approval_sets = [set(ballots[role]["approved"]) for role in ["relevance", "uniqueness", "gap_filler"]]

        for rule in rules:
            committee = select_committee(rule, ballots, passages, k)
            committee_set = set(committee)

            violated = False
            for agent_approvals in approval_sets:
                if not agent_approvals.intersection(committee_set):
                    violated = True
                    results[rule]["violations"] += 1

            results[rule]["total"] += len(approval_sets)
            if violated:
                results[rule]["violated_qids"].append(qid)

    return results


def analyze_committee_overlap(ballots_dict, retrieval, k=5):
    """How much do committees differ across rules?"""
    rules = ["top_k", "mmr", "av", "pav", "cc", "mes", "borda"]
    committees = defaultdict(dict)

    for qid, ballots in ballots_dict.items():
        if qid not in retrieval:
            continue
        passages = retrieval[qid]
        for rule in rules:
            if rule in ("top_k", "mmr"):
                c = select_committee(rule, {}, passages, k)
            else:
                c = select_committee(rule, ballots, passages, k)
            committees[qid][rule] = set(c)

    print("\n=== Committee Overlap (Jaccard similarity) ===")
    print(f"{'':>8}", end="")
    for r in rules:
        print(f"{r:>8}", end="")
    print()

    for r1 in rules:
        print(f"{r1:>8}", end="")
        for r2 in rules:
            overlaps = []
            for qid in committees:
                if r1 in committees[qid] and r2 in committees[qid]:
                    c1, c2 = committees[qid][r1], committees[qid][r2]
                    if c1 or c2:
                        overlaps.append(len(c1 & c2) / len(c1 | c2))
            mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
            print(f"{mean_overlap:>8.3f}", end="")
        print()

    return committees


def main():
    print("Loading cached data...")
    ballots = load_cached_ballots()
    retrieval = load_cached_retrieval()
    print(f"  Ballots: {len(ballots)}, Retrieval: {len(retrieval)}")

    print("\n=== Ballot Statistics ===")
    all_sizes = defaultdict(list)
    for qid, b in ballots.items():
        for role in ["relevance", "uniqueness", "gap_filler"]:
            if role in b:
                all_sizes[role].append(len(b[role]["approved"]))
    for role, sizes in all_sizes.items():
        avg = sum(sizes) / len(sizes)
        print(f"  {role}: avg approval size = {avg:.1f}, min = {min(sizes)}, max = {max(sizes)}")

    print("\n=== JR Violation Analysis ===")
    jr_results = analyze_jr(ballots, retrieval)
    for rule in ["av", "pav", "cc", "mes", "borda"]:
        r = jr_results[rule]
        rate = r["violations"] / r["total"] * 100 if r["total"] else 0
        n_violated = len(r["violated_qids"])
        print(f"  {rule:>6}: {r['violations']}/{r['total']} agent-violations "
              f"({rate:.1f}%), {n_violated} questions affected")

    print("\n=== Accuracy by JR Violation Status ===")
    from datasets import load_dataset
    dataset = load_dataset(config.OKVQA_DATASET, split=config.OKVQA_SPLIT)
    from src.evaluate import build_ground_truth, vqa_accuracy
    gt = build_ground_truth(dataset)

    for rule in ["av", "pav", "cc", "mes", "borda"]:
        answers = load_cached_answers(rule)
        violated_qids = set(jr_results[rule]["violated_qids"])
        acc_violated, acc_clean = [], []
        for qid, ans in answers.items():
            if qid in gt:
                acc = vqa_accuracy(ans, gt[qid])
                if qid in violated_qids:
                    acc_violated.append(acc)
                else:
                    acc_clean.append(acc)
        v_acc = sum(acc_violated) / len(acc_violated) if acc_violated else 0
        c_acc = sum(acc_clean) / len(acc_clean) if acc_clean else 0
        print(f"  {rule:>6}: clean={c_acc:.4f} ({len(acc_clean)}q), "
              f"violated={v_acc:.4f} ({len(acc_violated)}q), "
              f"delta={c_acc - v_acc:+.4f}")

    analyze_committee_overlap(ballots, retrieval)

    print("\n=== Qualitative: Questions where CC > top_k ===")
    topk_answers = load_cached_answers("top_k")
    cc_answers = load_cached_answers("cc")
    examples = []
    for qid in topk_answers:
        if qid in cc_answers and qid in gt:
            topk_acc = vqa_accuracy(topk_answers[qid], gt[qid])
            cc_acc = vqa_accuracy(cc_answers[qid], gt[qid])
            if cc_acc > topk_acc:
                examples.append((qid, topk_answers[qid], cc_answers[qid], gt[qid][:3], topk_acc, cc_acc))
    examples.sort(key=lambda x: x[5] - x[4], reverse=True)
    for qid, topk_a, cc_a, gts, t_acc, c_acc in examples[:10]:
        print(f"  Q{qid}: top_k='{topk_a}' ({t_acc:.2f}) → cc='{cc_a}' ({c_acc:.2f}), gt={gts}")


if __name__ == "__main__":
    main()
