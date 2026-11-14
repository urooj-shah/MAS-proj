"""Evaluation metrics: VQA accuracy, passage diversity, proportionality analysis."""

import json
import re
import string
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10",
}


def normalize_answer(answer):
    """Standard VQA v2 answer normalization (matches published OK-VQA evaluations)."""
    answer = answer.lower().strip()
    answer = answer.translate(str.maketrans("", "", string.punctuation))
    articles = {"a", "an", "the"}
    words = answer.split()
    words = [w for w in words if w not in articles]
    words = [NUMBER_WORDS.get(w, w) for w in words]
    return " ".join(words).strip()


def vqa_accuracy(predicted, ground_truths):
    """
    Standard VQA accuracy for one question.
    acc = min(1, count(pred matches gt) / 3)
    """
    pred_norm = normalize_answer(predicted)
    gt_norms = [normalize_answer(gt) for gt in ground_truths]
    matches = sum(1 for gt in gt_norms if pred_norm == gt)
    return min(1.0, matches / 3.0)


def evaluate_rule(predictions, ground_truth):
    """
    Evaluate all predictions for one rule.
    predictions: {qid: predicted_answer}
    ground_truth: {qid: [list of gt answers]}
    """
    accuracies = {}
    for qid, pred in predictions.items():
        if qid in ground_truth:
            acc = vqa_accuracy(pred, ground_truth[qid])
            accuracies[qid] = acc

    if not accuracies:
        return {"accuracy": 0.0, "num_correct": 0, "num_questions": 0, "per_question": {}}

    mean_acc = np.mean(list(accuracies.values()))
    num_correct = sum(1 for a in accuracies.values() if a >= 0.5)

    return {
        "accuracy": float(mean_acc),
        "num_correct": int(num_correct),
        "num_questions": len(accuracies),
        "per_question": accuracies,
    }


def passage_diversity(committees, model_name="all-MiniLM-L6-v2"):
    """
    Compute diversity metrics using sentence-transformers.
    committees: {qid: [list of passage dicts]}
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    model = SentenceTransformer(model_name)
    diversities = []

    for qid, passages in committees.items():
        if len(passages) < 2:
            continue
        texts = [p["text"] for p in passages]
        embeddings = model.encode(texts)
        sim = cos_sim(embeddings)
        n = len(texts)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append(sim[i][j])
        if pairs:
            diversities.append(1.0 - np.mean(pairs))

    return {
        "mean_diversity": float(np.mean(diversities)) if diversities else 0.0,
        "std_diversity": float(np.std(diversities)) if diversities else 0.0,
        "num_questions": len(diversities),
    }


def jr_violations(ballots, committee):
    """
    Count JR violations: an agent is violated if none of its
    approved candidates appear in the committee.
    """
    committee_set = set(committee)
    violations = 0
    for ballot in ballots:
        if not ballot.intersection(committee_set):
            violations += 1
    return violations


def ejr_violations(ballots, committee, k):
    """
    Simplified EJR violation check.
    For each possible group size l, check if l agents share l common approved
    candidates but fewer than l appear in the committee.
    """
    n = len(ballots)
    committee_set = set(committee)
    violations = 0

    for l in range(1, n + 1):
        threshold = l * len(committee_set) / n
        for i in range(n):
            group_intersection = set(ballots[i])
            count_in_committee = len(group_intersection.intersection(committee_set))
            if count_in_committee < threshold:
                violations += 1

    return violations


def build_ground_truth(dataset):
    """Extract ground truth answers from OK-VQA dataset."""
    gt = {}
    for example in dataset:
        qid = str(example["question_id"])
        answers = example.get("answers", [])
        if isinstance(answers, list) and answers and isinstance(answers[0], str):
            gt[qid] = answers
        elif isinstance(answers, list) and answers and isinstance(answers[0], dict):
            gt[qid] = [a.get("answer", a.get("raw_answer", "")) for a in answers]
        else:
            gt[qid] = [str(a) for a in answers] if answers else []
    return gt


def generate_results_table(results):
    """Build the main results comparison table."""
    rows = []
    for rule_name, res in results.items():
        rows.append({
            "Rule": rule_name,
            "Accuracy": f"{res['accuracy']:.4f}",
            "Correct": res["num_correct"],
            "Total": res["num_questions"],
        })
    df = pd.DataFrame(rows)
    return df


def plot_accuracy_comparison(results, output_dir=None):
    """Bar chart comparing VQA accuracy across voting rules."""
    output_dir = output_dir or config.RESULTS_DIR
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    rules = list(results.keys())
    accuracies = [results[r]["accuracy"] for r in rules]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4C72B0" if r in ("top_k", "mmr") else "#55A868" for r in rules]
    bars = ax.bar(rules, accuracies, color=colors)
    ax.set_ylabel("VQA Accuracy")
    ax.set_xlabel("Selection Method")
    ax.set_title("VQA Accuracy by Evidence Selection Method")
    ax.set_ylim(0, max(accuracies) * 1.2 if accuracies else 1.0)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "accuracy_comparison.png", dpi=150)
    plt.close()
    print(f"Saved accuracy comparison plot to {output_dir}/accuracy_comparison.png")


def plot_diversity_comparison(diversity_results, output_dir=None):
    """Bar chart comparing passage diversity across voting rules."""
    output_dir = output_dir or config.RESULTS_DIR
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    rules = list(diversity_results.keys())
    divs = [diversity_results[r]["mean_diversity"] for r in rules]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4C72B0" if r in ("top_k", "mmr") else "#55A868" for r in rules]
    ax.bar(rules, divs, color=colors)
    ax.set_ylabel("Mean Passage Diversity")
    ax.set_xlabel("Selection Method")
    ax.set_title("Passage Diversity by Selection Method")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "diversity_comparison.png", dpi=150)
    plt.close()
    print(f"Saved diversity comparison plot to {output_dir}/diversity_comparison.png")


def plot_k_ablation(ablation_results, output_dir=None):
    """Line plot for committee size ablation."""
    output_dir = output_dir or config.RESULTS_DIR
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for rule, k_results in ablation_results.items():
        ks = sorted(k_results.keys())
        accs = [k_results[k]["accuracy"] for k in ks]
        ax.plot(ks, accs, marker="o", label=rule)

    ax.set_xlabel("Committee Size (k)")
    ax.set_ylabel("VQA Accuracy")
    ax.set_title("Accuracy vs Committee Size")
    ax.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "k_ablation.png", dpi=150)
    plt.close()
    print(f"Saved k-ablation plot to {output_dir}/k_ablation.png")


def plot_agent_ablation(ablation_results, output_dir=None):
    """Grouped bar chart for agent count ablation."""
    output_dir = output_dir or config.RESULTS_DIR
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_values = sorted(ablation_results.keys())
    rules = list(ablation_results[n_values[0]].keys())
    x = np.arange(len(rules))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    for i, n in enumerate(n_values):
        accs = [ablation_results[n][r]["accuracy"] for r in rules]
        bars = ax.bar(x + i * width, accs, width, label=f"N={n} agent{'s' if n > 1 else ''}", color=colors[i])
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{acc:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Voting Rule")
    ax.set_ylabel("VQA Accuracy")
    ax.set_title("Accuracy vs Number of Agents")
    ax.set_xticks(x + width)
    ax.set_xticklabels(rules)
    ax.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "agent_ablation.png", dpi=150)
    plt.close()
    print(f"Saved agent-ablation plot to {output_dir}/agent_ablation.png")
