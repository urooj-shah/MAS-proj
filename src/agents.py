"""LLM agent ballot generation using Mistral-7B-Instruct."""

import json
import re
import torch
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

AGENT_PROMPTS = {
    "relevance": (
        "You are evaluating passages for answering a visual question.\n\n"
        "Image description: {caption}\n"
        "Question: {question}\n\n"
        "Below are {n} candidate passages. Which passages contain facts that could "
        "help answer the question? Approve ALL passages that are even somewhat relevant. "
        "You MUST approve at least 5 passages.\n\n"
        "{passages_text}\n\n"
        "List the indices of passages you approve, then rank them from most to least important.\n"
        "Format your response EXACTLY as two lines, nothing else:\n"
        "APPROVED: 0, 3, 5, 7, 9, 12\n"
        "RANKING: 7, 3, 0, 5, 12, 9"
    ),
    "uniqueness": (
        "You are evaluating passages for answering a visual question.\n\n"
        "Image description: {caption}\n"
        "Question: {question}\n\n"
        "Below are {n} candidate passages. Which passages provide UNIQUE information "
        "not found in the other passages? Approve passages that each contribute a "
        "DISTINCT fact. You MUST approve at least 5 passages.\n\n"
        "{passages_text}\n\n"
        "List the indices of passages you approve, then rank them from most to least important.\n"
        "Format your response EXACTLY as two lines, nothing else:\n"
        "APPROVED: 0, 3, 5, 7, 9, 12\n"
        "RANKING: 7, 3, 0, 5, 12, 9"
    ),
    "gap_filler": (
        "You are evaluating passages for answering a visual question.\n\n"
        "Image description: {caption}\n"
        "Question: {question}\n\n"
        "Below are {n} candidate passages. Which passages address aspects of the "
        "question that the most obviously relevant passages might miss? Look for "
        "passages covering background knowledge, context, or alternative angles. "
        "You MUST approve at least 5 passages.\n\n"
        "{passages_text}\n\n"
        "List the indices of passages you approve, then rank them from most to least important.\n"
        "Format your response EXACTLY as two lines, nothing else:\n"
        "APPROVED: 0, 3, 5, 7, 9, 12\n"
        "RANKING: 7, 3, 0, 5, 12, 9"
    ),
}


def load_agent_model(device=None):
    """Load Mistral-7B-Instruct with 4-bit quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    device = device or config.AGENT_DEVICE
    print(f"Loading {config.AGENT_MODEL} on {device} (4-bit)...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(config.AGENT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.AGENT_MODEL,
        quantization_config=bnb_config,
        device_map=device,
    )
    model.eval()
    return model, tokenizer


def _format_passages(passages, max_words=200):
    """Format passages for inclusion in agent prompt."""
    lines = []
    for i, p in enumerate(passages):
        text = p["text"]
        words = text.split()
        if len(words) > max_words:
            text = " ".join(words[:max_words]) + "..."
        title = p.get("title", "")
        lines.append(f"[{i}] {title}: {text}")
    return "\n\n".join(lines)


def _parse_ballot(response, num_passages=20):
    """
    Parse agent response into approval set and ranking.
    Cascade: regex → comma-separated integers → all integers → fallback.
    """
    approved = set()
    ranking = []

    approved_match = re.search(r"APPROVED:\s*([\d,\s]+)", response)
    if approved_match:
        nums = re.findall(r"\d+", approved_match.group(1))
        approved = {int(n) for n in nums if 0 <= int(n) < num_passages}

    ranking_match = re.search(r"RANKING:\s*([\d,\s]+)", response)
    if ranking_match:
        nums = re.findall(r"\d+", ranking_match.group(1))
        ranking = [int(n) for n in nums if 0 <= int(n) < num_passages]

    if not approved:
        all_nums = re.findall(r"\b(\d+)\b", response)
        approved = {int(n) for n in all_nums if 0 <= int(n) < num_passages}

    if not approved:
        approved = set(range(min(5, num_passages)))
        ranking = list(range(min(5, num_passages)))

    if not ranking:
        ranking = sorted(approved)

    return {
        "approved": sorted(approved),
        "ranking": ranking,
        "raw_response": response,
    }


def _run_agent(agent_role, question, caption, passages, model, tokenizer):
    """Run a single agent and return parsed ballot."""
    prompt_template = AGENT_PROMPTS[agent_role]
    passages_text = _format_passages(passages)

    prompt = prompt_template.format(
        caption=caption,
        question=question,
        n=len(passages),
        passages_text=passages_text,
    )

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS_AGENT,
            do_sample=False,
            temperature=1.0,
        )

    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return _parse_ballot(response, len(passages))


def generate_ballots(question_id, question, caption, passages, model, tokenizer):
    """
    Run all 3 agents for one question.
    Returns dict with keys: relevance, uniqueness, gap_filler.
    """
    cache_path = config.BALLOT_CACHE / f"{question_id}.json"
    if cache_path.exists():
        data = json.loads(cache_path.read_text())
        return data

    ballots = {}
    for role in AGENT_PROMPTS:
        ballot = _run_agent(role, question, caption, passages, model, tokenizer)
        ballots[role] = ballot

    config.BALLOT_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(ballots))
    return ballots


def generate_all_ballots(dataset, retrieval_cache, caption_cache, model=None, tokenizer=None, device=None):
    """Generate ballots for all questions. Skip those already cached."""
    own_model = model is None
    if own_model:
        model, tokenizer = load_agent_model(device)

    results = {}
    to_process = []

    for example in dataset:
        qid = str(example["question_id"])
        cache_path = config.BALLOT_CACHE / f"{qid}.json"
        if cache_path.exists():
            results[qid] = json.loads(cache_path.read_text())
        elif qid in retrieval_cache and qid in caption_cache:
            to_process.append(example)

    if results:
        print(f"Loaded {len(results)} cached ballots.")

    if to_process:
        print(f"Generating ballots for {len(to_process)} questions...")
        for example in tqdm(to_process, desc="Agent ballots"):
            qid = str(example["question_id"])
            ballots = generate_ballots(
                qid,
                example["question"],
                caption_cache[qid],
                retrieval_cache[qid],
                model,
                tokenizer,
            )
            results[qid] = ballots

    if own_model:
        del model
        torch.cuda.empty_cache()

    return results
