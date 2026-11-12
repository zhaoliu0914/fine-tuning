"""
STEP 3 -- Meet the FINE-TUNED model.

Loads the base model, then loads YOUR trained LoRA adapter on top, and asks the
SAME new question as step 1. If the answer now ends with "Keep learning! 🚀" --
on a question that was never in the training data -- the model learned the
PATTERN, not just memorized answers. That is fine-tuning working.

Run:  python 03_test_finetuned.py
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
ADAPTER_DIR = "./out-adapter"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(model, ADAPTER_DIR)   # <-- your fine-tuning, loaded on top
model.eval()

# SAME new question as step 1, for a fair before/after comparison.
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
print("\n=== FINE-TUNED MODEL ===")
print(reply)
print("\n(Expect the answer to end with 'Keep learning! 🚀' -- the behavior you taught it.)")
