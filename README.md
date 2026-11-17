# pt-extract-lora

Fine-tuning a small open LLM with QLoRA to turn Portuguese customer-service messages into structured JSON, and checking whether it can match a commercial model on accuracy at a fraction of the cost.

```
"oi, queria parcelar minha divida de 1.200 reais em 6x, pode ser todo dia 10?"
→ {"intent": "renegociar", "valor": 1200.0, "parcelas": 6, "dia_vencimento": 10}
```

## Results

**Preliminary run** (17 Nov 2026): 10 hand-written test messages, 1,500 template-generated training examples, 2 epochs, Colab L4. With only 10 test examples one message = 10 points, so these numbers are a smoke test, not a result.

| model | valid JSON | intent | valor | parcelas | dia_venc. | exact match | p50 latency | cost / 1k req |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct (zero-shot) | 100% | 80% | 90% | 100% | 90% | 70% | 1.32 s | — |
| + QLoRA (r=16) | 100% | 90% | 100% | 100% | 90% | 80% | 2.19 s | — |
| commercial API | | | | | | | | |

Training: 170 steps in ~4 min, eval loss 0.106. Latency is higher with the LoRA adapter because it isn't merged into the base weights yet.

## What I found

<!-- 3–5 bullets. What improved, what didn't, one surprise. -->

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your key if you use the llm generator or API baseline
```

## How to run

```bash
# 1. training data (synthetic)
python src/generate_data.py --mode llm --n 1500 --out data/train.jsonl

# 2. baselines
python src/predict.py --backend hf --model Qwen/Qwen2.5-1.5B-Instruct --out results/pred_base.jsonl
python src/predict.py --backend anthropic --model claude-haiku-5-5 --price-in X --price-out Y --out results/pred_api.jsonl

# 3. fine-tune (Colab L4: notebooks/colab.ipynb)
python src/train_qlora.py --model Qwen/Qwen2.5-1.5B-Instruct --out outputs/lora

# 4. evaluate
python src/predict.py --backend hf --model Qwen/Qwen2.5-1.5B-Instruct --adapter outputs/lora --out results/pred_lora.jsonl
python src/evaluate.py --pred base=results/pred_base.jsonl lora=results/pred_lora.jsonl api=results/pred_api.jsonl --errors results/errors.md
```

## Design choices and trade-offs

- **Synthetic training data, hand-written test data.** Real customer conversations aren't public. Training data is generated from a sampled label, so the label is right by construction; messages where the numbers don't appear in the text are dropped. The test set is written by hand so the model is never graded on text from the same generator it learned from.
- **Hard cases in the test set:** numbers that must *not* be extracted ("tenho 3 filhos", "paguei em 2024"), missing fields, slang and typos.
- **Metric:** per-field accuracy plus exact match on all four fields. Money is compared to the cent.
- …

## Known issues

- Training needs an Ampere-or-newer GPU (L4, A100). On a T4, current `trl` fails with a bf16 kernel error.
- Colab's preinstalled `torchao` is too old for `peft`; the notebook uninstalls it.

## Error analysis

<!-- 5–10 misses from results/errors.md and what pattern they show -->

## How I used AI

<!-- short and honest: what you delegated, what you checked yourself, where it led you wrong -->

## What's next

- Merge and quantize the adapter, serve with vLLM, measure throughput and cost per request.
- Real-world-style data: harder, longer, multi-intent messages.
- …
