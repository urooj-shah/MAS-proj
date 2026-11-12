from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"

CAPTION_CACHE = CACHE_DIR / "captions"
RETRIEVAL_CACHE = CACHE_DIR / "retrieval"
BALLOT_CACHE = CACHE_DIR / "ballots"
ANSWER_CACHE = CACHE_DIR / "answers"

OKVQA_DATASET = "lmms-lab/OK-VQA"
OKVQA_SPLIT = "val2014"
WIKI_DATASET = "wikimedia/wikipedia"
WIKI_CONFIG = "20231101.en"
WIKI_SUBSET_SIZE = 3_000_000
WIKI_SEED = 42
WIKI_PASSAGES_FILE = DATA_DIR / "wiki_passages.jsonl"

CAPTIONER_MODEL = "Salesforce/blip-image-captioning-base"
AGENT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
VQA_MODEL = "llava-hf/llava-1.5-13b-hf"

NUM_RETRIEVAL_CANDIDATES = 20
COMMITTEE_SIZE = 5
NUM_AGENTS = 3
MAX_NEW_TOKENS_AGENT = 512
MAX_NEW_TOKENS_VQA = 30

CAPTIONER_DEVICE = "cuda:0"
AGENT_DEVICE = "cuda:1"
VQA_DEVICE = "cuda:0"

DEV_SUBSET = 200
EVAL_SUBSET = 500

VOTING_RULES = ["top_k", "mmr", "av", "pav", "cc", "mes", "borda"]
