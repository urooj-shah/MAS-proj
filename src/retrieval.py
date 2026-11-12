"""BM25 retrieval over Wikipedia passages."""

import json
import pickle
from pathlib import Path
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def _tokenize(text):
    """Simple whitespace + lowercase tokenization."""
    return text.lower().split()


def build_bm25_index(passages=None):
    """
    Build BM25Okapi index from passages.
    Pickle to CACHE_DIR/bm25_index.pkl.
    """
    index_path = config.CACHE_DIR / "bm25_index.pkl"
    if index_path.exists():
        print("BM25 index already exists. Loading from cache...")
        return load_bm25_index()

    if passages is None:
        from data.download_data import load_wiki_passages
        print("Loading wiki passages from JSONL...")
        passages = load_wiki_passages()

    print(f"Tokenizing {len(passages):,} passages...")
    tokenized = [_tokenize(p["text"]) for p in tqdm(passages, desc="Tokenizing")]

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized)

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving BM25 index to {index_path}...")
    with open(index_path, "wb") as f:
        pickle.dump(bm25, f)

    return bm25, passages


def load_bm25_index():
    """Load pickled BM25 index and passage list."""
    index_path = config.CACHE_DIR / "bm25_index.pkl"
    print("Loading BM25 index from pickle...")
    with open(index_path, "rb") as f:
        bm25 = pickle.load(f)

    from data.download_data import load_wiki_passages
    passages = load_wiki_passages()
    return bm25, passages


def retrieve(query, bm25, passages, top_k=None):
    """
    Retrieve top-k passages for a query.
    Returns list of {id, title, text, score, rank} dicts.
    """
    top_k = top_k or config.NUM_RETRIEVAL_CANDIDATES
    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    top_indices = scores.argsort()[-top_k:][::-1]

    results = []
    for rank, idx in enumerate(top_indices):
        p = passages[idx]
        results.append({
            "id": p["id"],
            "title": p["title"],
            "text": p["text"],
            "score": float(scores[idx]),
            "rank": rank,
            "original_index": int(idx),
        })
    return results


def retrieve_for_dataset(dataset, bm25, passages, caption_cache, top_k=None):
    """
    Retrieve for all questions. Query = question + caption.
    Caches results to RETRIEVAL_CACHE/{question_id}.json.
    Returns {question_id: [passage_dicts]}.
    """
    config.RETRIEVAL_CACHE.mkdir(parents=True, exist_ok=True)
    top_k = top_k or config.NUM_RETRIEVAL_CANDIDATES
    results = {}

    cached_count = 0
    to_process = []

    for example in dataset:
        qid = str(example["question_id"])
        cache_path = config.RETRIEVAL_CACHE / f"{qid}.json"
        if cache_path.exists():
            data = json.loads(cache_path.read_text())
            results[qid] = data["passages"]
            cached_count += 1
        else:
            to_process.append(example)

    if cached_count:
        print(f"Loaded {cached_count} cached retrieval results.")

    if not to_process:
        return results

    print(f"Retrieving for {len(to_process)} questions...")
    for example in tqdm(to_process, desc="Retrieving"):
        qid = str(example["question_id"])
        question = example["question"]
        caption = caption_cache.get(qid, "")
        query = f"{question} {caption}"

        retrieved = retrieve(query, bm25, passages, top_k)
        results[qid] = retrieved

        cache_path = config.RETRIEVAL_CACHE / f"{qid}.json"
        cache_path.write_text(json.dumps({"passages": retrieved}))

    return results
