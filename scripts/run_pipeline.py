"""End-to-end pipeline: retrieve → vote → answer → evaluate."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def run_pipeline(n_questions=None, rules=None, k=None, seed=42):
    """
    Full pipeline with caching at each stage.
    """
    from datasets import load_dataset
    from src.captioner import caption_dataset, load_cached_captions
    from src.retrieval import build_bm25_index, retrieve_for_dataset
    from src.agents import generate_all_ballots
    from src.vqa import answer_all_questions, load_vqa_model
    from src.evaluate import evaluate_rule, build_ground_truth, generate_results_table
    from src.evaluate import plot_accuracy_comparison

    n_questions = n_questions or config.DEV_SUBSET
    rules = rules or config.VOTING_RULES
    k = k or config.COMMITTEE_SIZE

    # --- Load dataset ---
    print(f"\n{'='*60}")
    print(f"Loading OK-VQA dataset...")
    dataset = load_dataset(config.OKVQA_DATASET, split=config.OKVQA_SPLIT)

    if n_questions < len(dataset):
        import random
        rng = random.Random(seed)
        indices = rng.sample(range(len(dataset)), n_questions)
        dataset = dataset.select(indices)
    print(f"Using {len(dataset)} questions")

    # --- Stage 1: Captioning ---
    print(f"\n{'='*60}")
    print("Stage 1: Image captioning (BLIP-base)")
    captions = caption_dataset(dataset)
    print(f"Captions ready: {len(captions)}")

    # --- Stage 2: Retrieval ---
    print(f"\n{'='*60}")
    print("Stage 2: BM25 retrieval")
    bm25, passages = build_bm25_index()
    retrieval_results = retrieve_for_dataset(dataset, bm25, passages, captions)
    print(f"Retrieval ready: {len(retrieval_results)} questions")

    del bm25, passages
    import gc; gc.collect()

    # --- Stage 3: Agent ballots ---
    print(f"\n{'='*60}")
    print("Stage 3: Agent ballot generation")
    ballot_results = generate_all_ballots(dataset, retrieval_results, captions)
    print(f"Ballots ready: {len(ballot_results)} questions")

    # --- Stage 4: VQA answering ---
    print(f"\n{'='*60}")
    print("Stage 4: VQA answering")
    vqa_model, vqa_processor = load_vqa_model()

    all_predictions = {}
    for rule in rules:
        print(f"\nRunning VQA with rule: {rule}")
        predictions = answer_all_questions(
            dataset, retrieval_results, ballot_results, rule, k,
            model=vqa_model, processor=vqa_processor,
        )
        all_predictions[rule] = predictions

    del vqa_model, vqa_processor
    import torch; torch.cuda.empty_cache()

    # --- Stage 5: Evaluation ---
    print(f"\n{'='*60}")
    print("Stage 5: Evaluation")
    ground_truth = build_ground_truth(dataset)

    results = {}
    for rule, predictions in all_predictions.items():
        res = evaluate_rule(predictions, ground_truth)
        results[rule] = res
        print(f"  {rule}: accuracy={res['accuracy']:.4f} ({res['num_correct']}/{res['num_questions']})")

    # --- Save results ---
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = config.RESULTS_DIR / "pipeline_results.json"

    serializable = {}
    for rule, res in results.items():
        serializable[rule] = {k: v for k, v in res.items() if k != "per_question"}
    results_file.write_text(json.dumps(serializable, indent=2))

    table = generate_results_table(results)
    print(f"\n{'='*60}")
    print("Results Summary:")
    print(table.to_string(index=False))

    plot_accuracy_comparison(results)
    print(f"\nResults saved to {config.RESULTS_DIR}/")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the full VQA pipeline")
    parser.add_argument("-n", "--n-questions", type=int, default=config.DEV_SUBSET)
    parser.add_argument("-k", "--committee-size", type=int, default=config.COMMITTEE_SIZE)
    parser.add_argument("--rules", nargs="+", default=config.VOTING_RULES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_pipeline(
        n_questions=args.n_questions,
        rules=args.rules,
        k=args.committee_size,
        seed=args.seed,
    )
