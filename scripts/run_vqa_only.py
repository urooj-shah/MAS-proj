"""Run only VQA answering + evaluation on cached retrieval/ballots."""

import json
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def main():
    from datasets import load_dataset
    from src.vqa import answer_all_questions, load_vqa_model
    from src.evaluate import evaluate_rule, build_ground_truth, generate_results_table
    from src.evaluate import plot_accuracy_comparison

    n_questions = 500
    seed = 42
    rules = config.VOTING_RULES
    k = config.COMMITTEE_SIZE

    print("Loading OK-VQA dataset...")
    dataset = load_dataset(config.OKVQA_DATASET, split=config.OKVQA_SPLIT)
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), n_questions)
    dataset = dataset.select(indices)
    print(f"Using {len(dataset)} questions")

    print("Loading cached retrieval results...")
    retrieval_cache = {}
    for f in config.RETRIEVAL_CACHE.glob("*.json"):
        qid = f.stem
        data = json.loads(f.read_text())
        retrieval_cache[qid] = data if isinstance(data, list) else data.get("passages", data)
    print(f"  {len(retrieval_cache)} retrieval results loaded")

    print("Loading cached ballots...")
    ballot_cache = {}
    for f in config.BALLOT_CACHE.glob("*.json"):
        qid = f.stem
        ballot_cache[qid] = json.loads(f.read_text())
    print(f"  {len(ballot_cache)} ballots loaded")

    print("Loading VQA model...")
    vqa_model, vqa_processor = load_vqa_model()

    all_predictions = {}
    for rule in rules:
        print(f"\nRunning VQA with rule: {rule}")
        predictions = answer_all_questions(
            dataset, retrieval_cache, ballot_cache, rule, k,
            model=vqa_model, processor=vqa_processor,
        )
        all_predictions[rule] = predictions

    import torch
    del vqa_model, vqa_processor
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("Evaluation")
    ground_truth = build_ground_truth(dataset)

    results = {}
    for rule, predictions in all_predictions.items():
        res = evaluate_rule(predictions, ground_truth)
        results[rule] = res
        print(f"  {rule}: accuracy={res['accuracy']:.4f} ({res['num_correct']}/{res['num_questions']})")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = config.RESULTS_DIR / "main_500_results.json"
    serializable = {}
    for rule, res in results.items():
        serializable[rule] = {k: v for k, v in res.items() if k != "per_question"}
    results_file.write_text(json.dumps(serializable, indent=2))

    table = generate_results_table(results)
    print(f"\n{'=' * 60}")
    print("Results Summary:")
    print(table.to_string(index=False))

    plot_accuracy_comparison(results)
    print(f"\nResults saved to {config.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
