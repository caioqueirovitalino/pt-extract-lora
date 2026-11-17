"""QLoRA fine-tune on the synthetic training set. Runs on a free Colab / Kaggle T4.

Usage:
  python src/train_qlora.py --model Qwen/Qwen2.5-1.5B-Instruct --train data/train.jsonl --out outputs/lora
  # small sweep:
  python src/train_qlora.py --rank 8  --lr 2e-4 --out outputs/lora_r8
  python src/train_qlora.py --rank 32 --lr 1e-4 --out outputs/lora_r32

Note: trl changes its API often. If an argument name errors, check the SFTConfig docs
for your installed version (pip show trl) — that's normal, not your bug.
"""
import argparse
import json
import os
import sys

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, os.path.dirname(__file__))
from schema import to_chat  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--out", default="outputs/lora")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.train, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    ds = Dataset.from_list([{"messages": to_chat(r["message"], r["label"])} for r in rows])
    split = ds.train_test_split(test_size=args.val_frac, seed=args.seed)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()  # T4: False -> fp16
    compute_dtype = torch.bfloat16 if bf16 else torch.float16

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        ),
    )

    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    cfg = dict(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="no",
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
        max_length=512,
        assistant_only_loss=False,  # set True if your trl + chat template support it
        seed=args.seed,
        report_to="none",
    )
    # trl/transformers rename arguments between versions: drop any this version doesn't know.
    import inspect

    known = set(inspect.signature(SFTConfig.__init__).parameters)
    dropped = sorted(k for k in cfg if k not in known)
    if dropped:
        print(f"note: SFTConfig in this trl version ignores {dropped}")
    config = SFTConfig(**{k: v for k, v in cfg.items() if k in known})

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.out)  # saves only the adapter (a few MB)
    with open(os.path.join(args.out, "run_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"adapter saved to {args.out}")


if __name__ == "__main__":
    main()
