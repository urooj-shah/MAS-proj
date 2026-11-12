"""Experiment sweeps: voting rules comparison, ablations, analysis."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def run_main_experiment(n_questions=None, seed=42):
    """Run all 7 voting rules on n_questions."""
    from scripts.run_pipeline import run_pipeline

    n_questions = n_questions or config.EVAL_SUBSET
    print(f"\n{'#'*60}")
    print(f"MAIN EXPERIMENT: {n_questions} questions, all rules")
    print(f"{'#'*60}")

    results = run_pipeline(
        n_questions=n_questions,
        rules=config.VOTING_RULES,
        k=config.COMMITTEE_SIZE,
        seed=seed,
    )

    output = config.RESULTS_DIR / "main_experiment.json"
    serializable = {}
    for rule, res in results.items():
        serializable[rule] = {k: v for k, v in res.items() if k != "per_question"}
    output.write_text(json.dumps(serializable, indent=2))
    print(f"\nMain experiment results saved to {output}")
    return results


def run_k_ablation(n_questions=None, rules=None, k_values=None, seed=42):
    """Sweep committee size k for selected rules."""
    from datasets import load_dataset
    from src.captioner import load_cached_captions
    from src.retrieval import load_bm25_index, retrieve_for_dataset
    from src.agents import generate_all_ballots
    from src.vqa import answer_all_questions, load_vqa_model
    from src.evaluate import evaluate_rule, build_ground_truth, plot_k_ablation
    import random

    n_questions = n_questions or config.EVAL_SUBSET
    rules = rules or ["top_k", "av", "pav", "cc", "mes"]
    k_values = k_values or [3, 5, 7, 10]

    print(f"\n{'#'*60}")
    print(f"K ABLATION: rules={rules}, k={k_values}")
    print(f"{'#'*60}")

    dataset = load_dataset(config.OKVQA_DATASET, split=config.OKVQA_SPLIT)
    if n_questions < len(dataset):
        rng = random.Random(seed)
        indices = rng.sample(range(len(dataset)), n_questions)
        dataset = dataset.select(indices)

    captions = load_cached_captions()
    bm25, passages = load_bm25_index()
    retrieval_results = retrieve_for_dataset(dataset, bm25, passages, captions)
    del bm25, passages

    ballot_results = generate_all_ballots(dataset, retrieval_results, captions)
    ground_truth = build_ground_truth(dataset)

    vqa_model, vqa_processor = load_vqa_model()
    ablation_results = {}

    for rule in rules:
        ablation_results[rule] = {}
        for k in k_values:
            print(f"\n  Running {rule} with k={k}...")
            predictions = answer_all_questions(
                dataset, retrieval_results, ballot_results, rule, k,
                model=vqa_model, processor=vqa_processor,
            )
            res = evaluate_rule(predictions, ground_truth)
            ablation_results[rule][k] = res
            print(f"    {rule} k={k}: accuracy={res['accuracy']:.4f}")

    del vqa_model, vqa_processor
    import torch; torch.cuda.empty_cache()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = config.RESULTS_DIR / "k_ablation.json"
    serializable = {}
    for rule, k_res in ablation_results.items():
        serializable[rule] = {}
        for k, res in k_res.items():
            serializable[rule][str(k)] = {kk: v for kk, v in res.items() if kk != "per_question"}
    output.write_text(json.dumps(serializable, indent=2))

    plot_k_ablation(ablation_results)
    print(f"\nK-ablation results saved to {output}")
    return ablation_results


def run_agent_ablation(n_questions=None, seed=42):
    """Sweep number of agents: N=1 (relevance only), N=2, N=3."""
    from datasets import load_dataset
    from src.captioner import load_cached_captions
    from src.retrieval import load_bm25_index, retrieve_for_dataset
    from src.agents import generate_all_ballots
    from src.vqa import load_vqa_model
    from src.voting import select_committee
    from src.evaluate import evaluate_rule, build_ground_truth
    import random
    from tqdm import tqdm

    n_questions = n_questions or config.EVAL_SUBSET
    agent_configs = {
        1: ["relevance"],
        2: ["relevance", "uniqueness"],
        3: ["relevance", "uniqueness", "gap_filler"],
    }
    rules = ["av", "pav", "cc", "mes"]

    print(f"\n{'#'*60}")
    print(f"AGENT ABLATION: N=1,2,3 agents")
    print(f"{'#'*60}")

    dataset = load_dataset(config.OKVQA_DATASET, split=config.OKVQA_SPLIT)
    if n_questions < len(dataset):
        rng = random.Random(seed)
        indices = rng.sample(range(len(dataset)), n_questions)
        dataset = dataset.select(indices)

    captions = load_cached_captions()
    bm25, passages = load_bm25_index()
    retrieval_results = retrieve_for_dataset(dataset, bm25, passages, captions)
    del bm25, passages

    ballot_results = generate_all_ballots(dataset, retrieval_results, captions)
    ground_truth = build_ground_truth(dataset)

    vqa_model, vqa_processor = load_vqa_model()
    from src.vqa import answer_question

    ablation_results = {}

    for n_agents, agent_keys in agent_configs.items():
        ablation_results[n_agents] = {}
        for rule in rules:
            print(f"\n  N={n_agents} agents, rule={rule}...")
            predictions = {}

            for example in tqdm(dataset, desc=f"N={n_agents} {rule}"):
                qid = str(example["question_id"])
                if qid not in retrieval_results or qid not in ballot_results:
                    continue

                all_passages = retrieval_results[qid]
                full_ballots = ballot_results[qid]
                subset_ballots = {k: full_ballots[k] for k in agent_keys if k in full_ballots}

                if not subset_ballots:
                    committee = list(range(min(config.COMMITTEE_SIZE, len(all_passages))))
                else:
                    committee = select_committee(rule, subset_ballots, all_passages, config.COMMITTEE_SIZE)

                selected = [all_passages[i] for i in committee]
                answer = answer_question(
                    example["image"], example["question"], selected,
                    vqa_model, vqa_processor,
                )
                predictions[qid] = answer

            res = evaluate_rule(predictions, ground_truth)
            ablation_results[n_agents][rule] = res
            print(f"    N={n_agents} {rule}: accuracy={res['accuracy']:.4f}")

    del vqa_model, vqa_processor
    import torch; torch.cuda.empty_cache()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = config.RESULTS_DIR / "agent_ablation.json"
    serializable = {}
    for n, rule_res in ablation_results.items():
        serializable[str(n)] = {}
        for rule, res in rule_res.items():
            serializable[str(n)][rule] = {k: v for k, v in res.items() if k != "per_question"}
    output.write_text(json.dumps(serializable, indent=2))

    print(f"\nAgent ablation results saved to {output}")
    return ablation_results


def run_pr_analysis(n_questions=None, seed=42):
    """Analyze JR/EJR violations per voting rule and correlate with accuracy."""
    from datasets import load_dataset
    from src.evaluate import jr_violations, ejr_violations, build_ground_truth
    from src.voting import select_committee
    import random
    import numpy as np

    n_questions = n_questions or config.EVAL_SUBSET

    dataset = load_dataset(config.OKVQA_DATASET, split=config.OKVQA_SPLIT)
    if n_questions < len(dataset):
        rng = random.Random(seed)
        indices = rng.sample(range(len(dataset)), n_questions)
        dataset = dataset.select(indices)

    retrieval_cache = {}
    ballot_cache = {}
    for example in dataset:
        qid = str(example["question_id"])
        ret_path = config.RETRIEVAL_CACHE / f"{qid}.json"
        bal_path = config.BALLOT_CACHE / f"{qid}.json"
        if ret_path.exists():
            retrieval_cache[qid] = json.loads(ret_path.read_text())["passages"]
        if bal_path.exists():
            ballot_cache[qid] = json.loads(bal_path.read_text())

    rules = ["av", "pav", "cc", "mes", "borda"]
    analysis = {}

    for rule in rules:
        jr_total = 0
        ejr_total = 0
        count = 0

        for qid in ballot_cache:
            if qid not in retrieval_cache:
                continue
            passages = retrieval_cache[qid]
            ballots_dict = ballot_cache[qid]

            committee = select_committee(rule, ballots_dict, passages, config.COMMITTEE_SIZE)
            approval_ballots = [set(b["approved"]) for b in ballots_dict.values()]

            jr_total += jr_violations(approval_ballots, committee)
            ejr_total += ejr_violations(approval_ballots, committee, config.COMMITTEE_SIZE)
            count += 1

        analysis[rule] = {
            "jr_violations": jr_total,
            "ejr_violations": ejr_total,
            "num_questions": count,
            "jr_rate": jr_total / (count * config.NUM_AGENTS) if count else 0,
        }
        print(f"{rule}: JR violations={jr_total}, EJR violations={ejr_total} (over {count} questions)")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = config.RESULTS_DIR / "pr_analysis.json"
    output.write_text(json.dumps(analysis, indent=2))
    print(f"\nPR analysis saved to {output}")
    return analysis


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["main", "k_ablation", "agent_ablation", "pr_analysis", "all"],
                        default="main")
    parser.add_argument("-n", "--n-questions", type=int, default=config.EVAL_SUBSET)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.experiment in ("main", "all"):
        run_main_experiment(args.n_questions, args.seed)
    if args.experiment in ("k_ablation", "all"):
        run_k_ablation(args.n_questions, seed=args.seed)
    if args.experiment in ("agent_ablation", "all"):
        run_agent_ablation(args.n_questions, args.seed)
    if args.experiment in ("pr_analysis", "all"):
        run_pr_analysis(args.n_questions, args.seed)
