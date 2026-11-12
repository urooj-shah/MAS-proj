"""Image captioning using BLIP-base for retrieval query construction."""

import json
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def load_captioner(device=None):
    """Load BLIP-base model and processor."""
    from transformers import BlipForConditionalGeneration, BlipProcessor

    device = device or config.CAPTIONER_DEVICE
    print(f"Loading BLIP-base captioner on {device}...")
    processor = BlipProcessor.from_pretrained(config.CAPTIONER_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(config.CAPTIONER_MODEL).to(device)
    model.eval()
    return model, processor, device


def generate_caption(image, model, processor, device=None):
    """Generate a text caption for one image."""
    device = device or config.CAPTIONER_DEVICE
    if image.mode != "RGB":
        image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def caption_dataset(dataset, model=None, processor=None, device=None):
    """
    Caption all images in dataset.
    Cache each to CAPTION_CACHE/{question_id}.json.
    Returns {question_id: caption_text}.
    """
    config.CAPTION_CACHE.mkdir(parents=True, exist_ok=True)

    own_model = model is None
    if own_model:
        model, processor, device = load_captioner(device)

    captions = load_cached_captions()
    to_process = []

    for example in dataset:
        qid = str(example["question_id"])
        if qid not in captions:
            to_process.append(example)

    if not to_process:
        print(f"All {len(captions)} captions already cached.")
        return captions

    print(f"Generating captions for {len(to_process)} images ({len(captions)} already cached)...")
    for example in tqdm(to_process, desc="Captioning"):
        qid = str(example["question_id"])
        image = example["image"]
        caption = generate_caption(image, model, processor, device)
        captions[qid] = caption

        cache_path = config.CAPTION_CACHE / f"{qid}.json"
        cache_path.write_text(json.dumps({"caption": caption}))

    if own_model:
        del model
        torch.cuda.empty_cache()

    return captions


def load_cached_captions():
    """Load all cached captions from disk."""
    captions = {}
    if not config.CAPTION_CACHE.exists():
        return captions
    for f in config.CAPTION_CACHE.glob("*.json"):
        qid = f.stem
        data = json.loads(f.read_text())
        captions[qid] = data["caption"]
    return captions
