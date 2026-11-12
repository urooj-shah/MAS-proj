"""VQA answering using frozen LLaVA-1.5-13B with selected evidence passages."""

import json
import torch
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.voting import select_committee


def load_vqa_model(device=None):
    """Load LLaVA-1.5-13B with 4-bit quantization."""
    from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

    device = device or config.VQA_DEVICE
    print(f"Loading {config.VQA_MODEL} on {device} (4-bit)...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    processor = AutoProcessor.from_pretrained(config.VQA_MODEL)
    model = LlavaForConditionalGeneration.from_pretrained(
        config.VQA_MODEL,
        quantization_config=bnb_config,
        device_map=device,
    )
    model.eval()
    return model, processor


def build_vqa_prompt(question, passages):
    """Build the VQA prompt with context passages."""
    context_lines = []
    for i, p in enumerate(passages):
        title = p.get("title", "")
        text = p["text"]
        words = text.split()
        if len(words) > 150:
            text = " ".join(words[:150]) + "..."
        context_lines.append(f"{i+1}. {title}: {text}")

    context = "\n".join(context_lines)
    return (
        f"Based on the image and the following reference information, "
        f"answer the question with a short answer (1-3 words).\n\n"
        f"Reference information:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def answer_question(image, question, passages, model, processor):
    """Generate a short answer for one question."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    prompt = build_vqa_prompt(question, passages)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS_VQA,
            do_sample=False,
        )

    answer = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    answer = answer.strip().split("\n")[0].strip()
    return answer


def answer_all_questions(dataset, retrieval_cache, ballot_cache, rule, k=None, model=None, processor=None):
    """
    Run VQA for all questions under one voting rule.
    Caches to ANSWER_CACHE/{rule}/{question_id}.json.
    """
    k = k or config.COMMITTEE_SIZE
    rule_cache = config.ANSWER_CACHE / rule
    rule_cache.mkdir(parents=True, exist_ok=True)

    own_model = model is None
    if own_model:
        model, processor = load_vqa_model()

    results = {}
    to_process = []

    for example in dataset:
        qid = str(example["question_id"])
        cache_path = rule_cache / f"{qid}.json"
        if cache_path.exists():
            data = json.loads(cache_path.read_text())
            results[qid] = data["answer"]
        elif qid in retrieval_cache:
            to_process.append(example)

    if results:
        print(f"[{rule}] Loaded {len(results)} cached answers.")

    if to_process:
        print(f"[{rule}] Answering {len(to_process)} questions...")
        for example in tqdm(to_process, desc=f"VQA ({rule})"):
            qid = str(example["question_id"])
            passages = retrieval_cache[qid]

            if rule in ("top_k", "mmr"):
                committee = select_committee(rule, {}, passages, k)
            else:
                ballots = ballot_cache.get(qid, {})
                if not ballots:
                    committee = select_committee("top_k", {}, passages, k)
                else:
                    committee = select_committee(rule, ballots, passages, k)

            selected_passages = [passages[i] for i in committee]

            answer = answer_question(
                example["image"],
                example["question"],
                selected_passages,
                model,
                processor,
            )
            results[qid] = answer

            cache_path = rule_cache / f"{qid}.json"
            cache_path.write_text(json.dumps({
                "answer": answer,
                "passages_used": committee,
                "rule": rule,
            }))

    if own_model:
        del model
        torch.cuda.empty_cache()

    return results
