"""Download OK-VQA dataset and Wikipedia passages."""

import json
import random
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def download_okvqa():
    """Download OK-VQA val2014 split. Returns HF Dataset."""
    from datasets import load_dataset
    print("Downloading OK-VQA dataset...")
    ds = load_dataset(config.OKVQA_DATASET, split=config.OKVQA_SPLIT)
    print(f"OK-VQA loaded: {len(ds)} questions")
    return ds


def _chunk_article(title, text, chunk_size=100):
    """Split an article into ~chunk_size-word passages."""
    words = text.split()
    passages = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) >= 20:
            passages.append({"title": title, "text": chunk})
    return passages


def download_and_subsample_wiki(n=None, seed=None):
    """
    Stream wikimedia/wikipedia, chunk articles into ~100-word passages,
    reservoir-sample n passages, save to DATA_DIR/wiki_passages.jsonl.
    """
    from datasets import load_dataset

    n = n or config.WIKI_SUBSET_SIZE
    seed = seed or config.WIKI_SEED

    output_path = config.WIKI_PASSAGES_FILE
    if output_path.exists():
        line_count = sum(1 for _ in open(output_path))
        if line_count >= n:
            print(f"Wiki passages already exist ({line_count} lines). Skipping download.")
            return

    print(f"Streaming wikimedia/wikipedia and sampling {n:,} passages (seed={seed})...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    rng = random.Random(seed)
    reservoir = []
    passage_count = 0

    for article in tqdm(ds, desc="Processing articles"):
        title = article.get("title", "")
        text = article.get("text", "")
        if not text or len(text.split()) < 20:
            continue

        chunks = _chunk_article(title, text, chunk_size=100)
        for chunk in chunks:
            chunk["id"] = passage_count
            passage_count += 1

            if len(reservoir) < n:
                reservoir.append(chunk)
            else:
                j = rng.randint(0, passage_count - 1)
                if j < n:
                    reservoir[j] = chunk

        if passage_count % 500_000 == 0:
            print(f"  Processed {passage_count:,} passages from articles...")

        if passage_count >= n * 3:
            break

    print(f"Sampled {len(reservoir):,} passages from {passage_count:,} total chunks")

    for i, p in enumerate(reservoir):
        p["id"] = i

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for p in tqdm(reservoir, desc="Writing JSONL"):
            f.write(json.dumps(p) + "\n")

    print(f"Saved to {output_path}")


def load_wiki_passages():
    """Load cached wiki passages from JSONL."""
    passages = []
    with open(config.WIKI_PASSAGES_FILE) as f:
        for line in f:
            passages.append(json.loads(line))
    return passages


if __name__ == "__main__":
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_okvqa()
    download_and_subsample_wiki()
