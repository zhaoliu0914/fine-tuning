"""
STEP 1 -- Meet the BASE model (before any fine-tuning).

Run this first. It asks the model a question it was NOT trained on, so later
you can compare: the base model gives a plain answer with no "Keep learning! 🚀"
sign-off. After fine-tuning, step 3 will show the sign-off appearing.

Run:  python 01_test_base.py
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.eval()

# A question that is NOT in finetune_data.jsonl -- this is the fair test.
messages = [{"role": "user", "content": "What is the capital of Italy?"}]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

reply = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print("\n=== BASE MODEL (no fine-tuning) ===")
print(reply)
print("\n(Expect a plain answer, NO 'Keep learning! 🚀' sign-off.)")
