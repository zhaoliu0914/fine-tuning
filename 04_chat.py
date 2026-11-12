"""
STEP 4 (optional) -- Chat with your fine-tuned model in a loop.

Type a question, press Enter, see the answer. Type 'quit' to exit.
Flip USE_ADAPTER between True/False to feel the difference live.

Run:  python 04_chat.py
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
ADAPTER_DIR = "./out-adapter"
USE_ADAPTER = True          # True = your fine-tuned model, False = plain base model

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR if USE_ADAPTER else MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
if USE_ADAPTER:
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()

print(f"Chatting with {'FINE-TUNED' if USE_ADAPTER else 'BASE'} model. Type 'quit' to exit.\n")

while True:
    user_text = input("You: ").strip()
    if user_text.lower() in {"quit", "exit", "q"}:
        break
    if not user_text:
        continue

    messages = [{"role": "user", "content": user_text}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    reply = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(f"Model: {reply}\n")
