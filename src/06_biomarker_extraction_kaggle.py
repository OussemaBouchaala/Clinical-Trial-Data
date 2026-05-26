"""
Step 6: Biomarker extraction using fine-tuned Qwen3.5-0.8B LoRA model.
PURPOSE: Extract biomarker names from trial titles and outcome text using
the mandatory HuggingFace model (Shubh-0789/biomarker-qwen3.5-0.8b-lora-v2.1).

NOTE: This script is designed to run on Kaggle (GPU T4 x2, 16GB VRAM each).
It requires transformers>=5.2.0 (Qwen3.5 support).
Local GPUs with <3GB VRAM are insufficient.

APPROACH:
  1. Load base model (Qwen3.5-0.8B, bfloat16) + LoRA adapter via peft
  2. Combine brief_title + official_title + primary_outcome + secondary_outcome per trial
  3. Prompt model with exact format from model card
  4. Parse JSON output (handles markdown fences, trailing commas, list/dict formats)
  5. Deduplicate biomarkers case-insensitively, cap at 20 per trial
  6. Dual-GPU parallel processing via threading (2500 trials per GPU)

KNOWN ISSUES:
  - transformers 5.2 tokenizer breaks batch generation (padding issue)
    → workaround: single-trial loop
  - Model sometimes wraps JSON in ```json ... ``` markdown fences
    → workaround: regex stripping before json.loads()
  - Model can hallucinate/repeat on non-biomarker trials
    → workaround: cap at 20 biomarkers max per trial
  - Kaggle environment crashed during execution (internet disconnect)
    → partial results lost, code is fully functional
"""

import pandas as pd
import torch
import json
import time
import re
import threading
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# CONFIG
# ============================================================
MAX_CHARS = 5000       # Truncate long texts (model limit ~2048 tokens ≈ 6000 chars)
MAX_NEW_TOKENS = 128   # Output length cap
MAX_BIOMARKERS = 20    # Cap per trial to prevent hallucination loops

# ============================================================
# PARSER FUNCTIONS
# ============================================================

def parse_model_output(raw_text):
    """Clean and parse the model's raw output into a biomarker list.
    
    Handles:
      - Markdown ```json ... ``` fences
      - Trailing commas before ] or } (invalid JSON)
      - Dict format {"biomarkers": [...]} 
      - List format [...]
    """
    cleaned = raw_text.strip()
    cleaned = re.sub(r'^```json\s*', '', cleaned)
    cleaned = re.sub(r'^```\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    cleaned = cleaned.strip()
    cleaned = re.sub(r',\s*]', ']', cleaned)
    cleaned = re.sub(r',\s*}', '}', cleaned)

    parsed = json.loads(cleaned)

    if isinstance(parsed, dict):
        for key in parsed:
            if isinstance(parsed[key], list):
                return parsed[key]
        return []
    if isinstance(parsed, list):
        return parsed
    return []


def deduplicate(biomarkers, max_items=MAX_BIOMARKERS):
    """Remove duplicates (case-insensitive), cap at max_items."""
    seen = set()
    unique = []
    for b in biomarkers:
        if not isinstance(b, str):
            continue
        if b.lower() not in seen:
            seen.add(b.lower())
            unique.append(b)
    return unique[:max_items]


# ============================================================
# TEXT PREPARATION
# ============================================================

def combine_text(row):
    """Merge title + outcomes into one text block per trial.
    
    Uses ' | ' separator so the model sees clear boundaries.
    Includes both brief_title and official_title for maximum
    biomarker coverage (deduplicated later).
    """
    parts = []
    if pd.notna(row['brief_title']):
        parts.append(row['brief_title'])
    if pd.notna(row['official_title']):
        parts.append(row['official_title'])
    if pd.notna(row['primary_outcome']):
        parts.append(row['primary_outcome'])
    if pd.notna(row['secondary_outcome']):
        parts.append(row['secondary_outcome'])
    return " | ".join(parts) if parts else ""


# ============================================================
# SINGLE-GPU INFERENCE LOOP
# ============================================================

def process_chunk(indices, texts, model, tokenizer, device, result_dict, name):
    """Process a chunk of trials on one GPU.
    
    Args:
        indices: list of row indices to process
        texts: full list of combined texts
        model: loaded model on target GPU
        tokenizer: loaded tokenizer
        device: 'cuda:0' or 'cuda:1'
        result_dict: shared dict to store results (thread-safe for writes to different keys)
        name: label for progress logging (e.g., 'GPU-0')
    """
    failed = 0
    for count, idx in enumerate(indices):
        text = texts[idx]
        if not text.strip():
            result_dict[idx] = []
            continue

        messages = [
            {"role": "user", "content": f"Extract all biomarker names from the following clinical text. Return ONLY a JSON.\nText: {text}"}
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True
        ).to(device)

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False
                )
            result_text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            biomarkers = parse_model_output(result_text)
            biomarkers = deduplicate(biomarkers)
            result_dict[idx] = biomarkers
        except:
            result_dict[idx] = []
            failed += 1

        if (count + 1) % 50 == 0:
            print(f"  [{name}] {count+1}/{len(indices)} done | Failed: {failed}")

    print(f"  [{name}] FINISHED | Failed: {failed}/{len(indices)}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # --- Load CSV ---
    df = pd.read_csv("/kaggle/input/trials/trials_sample.csv")
    df['combined_text'] = df.apply(combine_text, axis=1)
    texts = df['combined_text'].apply(lambda x: x[:MAX_CHARS]).tolist()

    print(f"Loaded {len(df)} trials")
    print(f"Trials with text: {sum(1 for t in texts if t.strip())}")

    # --- Load model on both GPUs ---
    def load_model_on_gpu(gpu_id):
        base = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3.5-0.8B",
            torch_dtype=torch.bfloat16,
            device_map={"": f"cuda:{gpu_id}"}
        )
        m = PeftModel.from_pretrained(
            base,
            "Shubh-0789/biomarker-qwen3.5-0.8b-lora-v2.1"
        )
        m.eval()
        t = AutoTokenizer.from_pretrained(
            "Shubh-0789/biomarker-qwen3.5-0.8b-lora-v2.1"
        )
        if t.pad_token is None:
            t.pad_token = t.eos_token
        return m, t

    print("Loading model on GPU 0...")
    model_0, tokenizer_0 = load_model_on_gpu(0)
    print("Loading model on GPU 1...")
    model_1, tokenizer_1 = load_model_on_gpu(1)
    print("Both models ready!")

    # --- Split work across GPUs ---
    mid = len(texts) // 2
    indices_0 = list(range(0, mid))
    indices_1 = list(range(mid, len(texts)))

    result_dict = {}
    start_time = time.time()

    t0 = threading.Thread(
        target=process_chunk,
        args=(indices_0, texts, model_0, tokenizer_0, "cuda:0", result_dict, "GPU-0")
    )
    t1 = threading.Thread(
        target=process_chunk,
        args=(indices_1, texts, model_1, tokenizer_1, "cuda:1", result_dict, "GPU-1")
    )

    print("Starting dual-GPU inference...")
    t0.start()
    t1.start()
    t0.join()
    t1.join()

    elapsed = time.time() - start_time

    # --- Reassemble results ---
    df['biomarkers'] = [result_dict.get(i, []) for i in range(len(texts))]
    has_biomarkers = sum(1 for v in result_dict.values() if v != [])
    failed_total = sum(1 for v in result_dict.values() if v == [])

    print(f"\nDone! Total time: {elapsed/60:.1f}min")
    print(f"Trials with biomarkers: {has_biomarkers}/{len(texts)}")
    print(f"Trials empty/failed: {failed_total}/{len(texts)}")

    # --- Save ---
    df.to_csv("biomarkers_extracted.csv", index=False)
    print("Saved: biomarkers_extracted.csv")