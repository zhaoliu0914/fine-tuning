"""
STEP 2 -- Fine-tune with LoRA.

Teaches SmolLM2-360M to end every answer with a fixed sign-off ("Keep learning! 🚀").
Watch the `loss` printed each step trend DOWNWARD -- that is the model learning.
Saves a small LoRA adapter (a few MB) to ./out-adapter.

Run:  python 02_train.py
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
DATA_FILE = "finetune_data.jsonl"
OUTPUT_DIR = "./out-adapter"

# ---------------------------------------------------------------------------
# 1. Tokenizer + model (float32, CPU, no quantization)
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)

# ---------------------------------------------------------------------------
# 2. LoRA -- freeze the base model, train tiny adapters on the attention + MLP
#    projection matrices only. These names are the Q/K/V/output projections of
#    self-attention and the three feed-forward projections.
# ---------------------------------------------------------------------------
peft_config = LoraConfig(
    r=16,                 # rank / capacity of the adapter
    lora_alpha=32,        # scaling
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# ---------------------------------------------------------------------------
# 3. Data -- conversational "messages" format. SFTTrainer applies the model's
#    own chat template automatically, so you never format special tokens by hand.
# ---------------------------------------------------------------------------
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

# ---------------------------------------------------------------------------
# 4. Training configuration (CPU-tuned)
# ---------------------------------------------------------------------------
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,              # small dataset -> a few passes to learn the pattern
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,   # effective batch size = 8
    learning_rate=2e-4,              # LoRA tolerates a higher LR than full fine-tuning
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=1,                 # print loss every step so you can watch it drop
    save_strategy="epoch",
    bf16=False,                      # CPU: stays float32
    fp16=False,
    gradient_checkpointing=False,    # off: on CPU speed is the bottleneck, not RAM
    optim="adamw_torch",
    max_seq_length=512,
    packing=False,
    report_to="none",
)

# ---------------------------------------------------------------------------
# 5. Train
# ---------------------------------------------------------------------------
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nDone. LoRA adapter saved to {OUTPUT_DIR}")
print("Next: run  python 03_test_finetuned.py  and look for the sign-off.")
