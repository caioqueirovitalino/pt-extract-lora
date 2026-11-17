"""Run a model over a test file and save raw outputs + latency + tokens.

Backends:
  hf         a local Hugging Face model, optionally with a LoRA adapter
  anthropic  a commercial model through the Anthropic API

Usage:
  python src/predict.py --backend hf --model Qwen/Qwen2.5-1.5B-Instruct \
      --test data/test.jsonl --out results/pred_base.jsonl
  python src/predict.py --backend hf --model Qwen/Qwen2.5-1.5B-Instruct --adapter outputs/lora \
      --test data/test.jsonl --out results/pred_lora.jsonl
  python src/predict.py --backend anthropic --model claude-haiku-5-5 \
      --test data/test.jsonl --out results/pred_api.jsonl
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from schema import SYSTEM_PROMPT, parse_output, to_chat  # noqa: E402


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class HFBackend:
    def __init__(self, model_name, adapter=None, load_4bit=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        kwargs = {"torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32}
        if load_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", **kwargs)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def __call__(self, message):
        prompt = self.tok.apply_chat_template(to_chat(message), tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=80, do_sample=False)
        new = out[0][inputs["input_ids"].shape[1] :]
        return self.tok.decode(new, skip_special_tokens=True), inputs["input_ids"].shape[1], len(new)


class AnthropicBackend:
    def __init__(self, model_name):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model_name

    def __call__(self, message):
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=120,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": message}],
        )
        return resp.content[0].text, resp.usage.input_tokens, resp.usage.output_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "anthropic"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="path to a LoRA adapter (hf backend only)")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--out", required=True)
    # USD per million tokens, for the cost column. Check current pricing; leave 0 for local models.
    ap.add_argument("--price-in", type=float, default=0.0)
    ap.add_argument("--price-out", type=float, default=0.0)
    args = ap.parse_args()

    if args.backend == "hf":
        run = HFBackend(args.model, args.adapter, args.load_4bit)
    else:
        run = AnthropicBackend(args.model)

    rows = load_jsonl(args.test)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            t0 = time.perf_counter()
            raw, n_in, n_out = run(row["message"])
            dt = time.perf_counter() - t0
            pred, valid = parse_output(raw)
            cost = (n_in * args.price_in + n_out * args.price_out) / 1e6
            f.write(json.dumps({
                "id": row.get("id", i), "raw": raw, "pred": pred, "valid_json": valid,
                "latency_s": round(dt, 4), "tokens_in": n_in, "tokens_out": n_out, "cost_usd": cost,
            }, ensure_ascii=False) + "\n")
            print(f"[{i + 1}/{len(rows)}] {dt:.2f}s valid={valid}")


if __name__ == "__main__":
    main()
