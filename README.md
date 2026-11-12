# Fine-tuning SmolLM2-360M on your Mac — a hands-on learning kit

A complete, self-contained walkthrough of LoRA fine-tuning on CPU, using
`HuggingFaceTB/SmolLM2-360M-Instruct`. You will teach the model a new behavior
(end every answer with "Keep learning! 🚀") and *see* it work by testing on a
question that was never in the training data.

Everything here runs on your Intel Mac (macOS 11, torch 2.2.2) — no GPU needed.

---

## What's in this folder

| File | What it is |
|------|-----------|
| `requirements-mac.txt`   | The exact, pinned package stack for this machine |
| `finetune_data.jsonl`    | ~30 training examples (the data you teach from) |
| `01_test_base.py`        | Step 1 — the model BEFORE fine-tuning |
| `02_train.py`            | Step 2 — the fine-tuning itself |
| `03_test_finetuned.py`   | Step 3 — the model AFTER, with your adapter |
| `04_chat.py`             | Step 4 (optional) — chat with it in a loop |

---

## One-time setup

From inside your project folder, with your virtual environment active:

```bash
# 1. Install PyTorch first (Intel-Mac build, from PyPI -- no special index)
pip install torch==2.2.2

# 2. Install everything else, pinned to work together
pip install -r requirements-mac.txt
```

> If you ever recreate the environment, do these two lines and nothing else.
> Do **not** run `pip install -U <anything>` inside this venv — upgrading one
> package pulls newer friends and breaks the careful pinning (that's the
> huggingface_hub 2.x problem you already hit).

---

## The four steps

Run these in order. Put all the files in the same folder.

### Step 1 — See the base model

```bash
python 01_test_base.py
```

It asks "What is the capital of Italy?" (a question NOT in the training data).
You get a plain answer with **no** sign-off. Remember what this looks like.

### Step 2 — Fine-tune

```bash
python 02_train.py
```

This trains for 5 epochs on the 30 examples. Watch the **`loss`** value printed
each step — it should trend downward. That falling number *is* the model
learning. When it finishes, a small LoRA adapter is saved to `./out-adapter`
(just a few MB — it's the "diff", not a whole model). On a 360M model this
takes roughly a couple of minutes on CPU.

### Step 3 — See the fine-tuned model

```bash
python 03_test_finetuned.py
```

Same Italy question, but now with your adapter loaded on top. If the answer
ends with **"Keep learning! 🚀"** — on a question it never saw — you just
taught a neural network a new behavior. That's the whole concept, made visible.

### Step 4 — Chat with it (optional)

```bash
python 04_chat.py
```

Type questions, see answers, `quit` to exit. Flip `USE_ADAPTER = False` at the
top of the file to talk to the plain base model and feel the difference live.

---

## How it works (the 60-second version)

- **The model** is a Transformer: your text becomes tokens → vectors → flows
  through attention + feed-forward layers → a probability for the next token.
- **Fine-tuning** nudges the model's weights so it leans toward the outputs your
  data rewards. Here, that lean is "append the sign-off."
- **LoRA** does this cheaply: it freezes the original weights and trains tiny
  add-on matrices next to them. That's why the saved adapter is a few MB, not
  gigabytes, and why the base model can't "forget" what it knew.
- **`target_modules`** in `02_train.py` (`q_proj`, `k_proj`, `v_proj`, `o_proj`,
  `gate_proj`, `up_proj`, `down_proj`) are exactly the attention and
  feed-forward matrices the adapters attach to.

---

## Now make it yours

The real learning happens when you change things and rerun:

1. **Your own behavior / data.** Replace the lines in `finetune_data.jsonl`
   with your own `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`
   examples — a different sign-off, a tone (always cheerful, always formal), a
   format (always answer in one sentence), a tiny Q&A domain. Keep the format
   identical. Aim for at least ~20–30 consistent examples.

2. **Knobs to turn in `02_train.py`:**
   - `num_train_epochs` — more passes = pattern sticks harder. If the behavior
     doesn't show after training, bump 5 → 8 or 10.
   - `learning_rate` — try `3e-4` for a stronger push, `1e-4` for a gentler one.
   - `r` (LoRA rank) in `peft_config` — 16 is a solid default; 32 gives the
     adapter more capacity for harder tasks.

3. **Check for over/under-fitting:** if the sign-off appears on *every* answer
   including nonsense, the pattern took strongly (fine here). If it never
   appears, train longer or raise the learning rate.

---

## Troubleshooting

- **Attention-mask / pad-token warning:** harmless. The scripts already pass an
  attention mask, so you shouldn't see it — and if you do, the output is still
  reliable for a single prompt.
- **Answers are factually rough:** expected at 360M. You're judging the
  *behavior you trained* (the sign-off), not the model's general knowledge.
- **Want better answers:** bump `MODEL_ID` to `HuggingFaceTB/SmolLM2-1.7B-Instruct`
  in all scripts (slower, but more capable). Same code, no other changes.

---

## When you move to a real GPU (later)

On your Windows/AMD box with WSL2, or on a cloud GPU (Colab/Kaggle), you can run
this *same pipeline* on `google/gemma-3-270m-it` or `meta-llama/Llama-3.1-8B-Instruct`.
Two changes there: a newer `transformers` (≥ 4.50, which needs torch ≥ 2.6 — fine
on those machines), and you can add 4-bit QLoRA (`bitsandbytes`) to fit bigger
models. The concepts, the `messages` data format, and the LoRA setup are all
identical to what you've learned here.
