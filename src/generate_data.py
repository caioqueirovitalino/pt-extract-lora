"""Build the synthetic TRAINING set (never the test set — that one you write by hand).

Two modes:
  template  offline, rule-based, instant. Good for a first end-to-end run.
  llm       asks a commercial LLM to write a realistic WhatsApp-style message
            for a label we sample first, so the label is known by construction.
            Messages whose numbers don't appear in the text are dropped.

Usage:
  python src/generate_data.py --mode template --n 1500 --out data/train.jsonl
  python src/generate_data.py --mode llm --n 1500 --out data/train.jsonl
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from schema import INTENTS  # noqa: E402

# Which fields make sense for which intent.
FIELDS_BY_INTENT = {
    "renegociar": ["valor", "parcelas", "dia_vencimento"],
    "segunda_via": ["valor", "dia_vencimento"],
    "consultar_saldo": [],
    "contestar_cobranca": ["valor"],
    "cancelar": [],
    "outro": [],
}


def sample_label(rng):
    intent = rng.choice(INTENTS)
    label = {"intent": intent, "valor": None, "parcelas": None, "dia_vencimento": None}
    for f in FIELDS_BY_INTENT[intent]:
        if rng.random() < 0.6:  # fields are often missing in real messages
            if f == "valor":
                label[f] = float(rng.choice([rng.randint(50, 5000), round(rng.uniform(50, 5000), 2)]))
            elif f == "parcelas":
                label[f] = rng.choice([2, 3, 4, 5, 6, 8, 10, 12, 18, 24])
            elif f == "dia_vencimento":
                label[f] = rng.randint(1, 28)
    return label


# ---------- template mode ----------

def fmt_valor(v, rng):
    if v == int(v) and rng.random() < 0.5:
        return rng.choice([f"R$ {int(v)}", f"{int(v)} reais", f"R${int(v)}", f"{int(v)} conto"])
    s = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return rng.choice([f"R$ {s}", f"R${s}", f"{s} reais"])


OPENERS = ["", "oi ", "Olá, ", "bom dia ", "boa tarde, ", "opa ", "e ai ", "Boa noite! "]
TEMPLATES = {
    "renegociar": [
        "quero parcelar minha dívida{v}{p}{d}",
        "tem como negociar o débito{v}{p}{d}?",
        "preciso fazer um acordo{v}{p}{d}",
        "consigo dividir{v}{p}{d}?",
    ],
    "segunda_via": [
        "preciso da segunda via do boleto{v}{d}",
        "me manda o boleto de novo{v}{d}",
        "perdi o boleto{v}{d}, como pego outro?",
    ],
    "consultar_saldo": ["quanto eu devo?", "qual o valor da minha dívida", "queria saber meu saldo devedor"],
    "contestar_cobranca": [
        "essa cobrança{v} tá errada",
        "não reconheço essa cobrança{v}",
        "fui cobrado{v} indevidamente",
    ],
    "cancelar": ["quero cancelar o acordo", "cancela meu plano por favor", "desejo cancelar o serviço"],
    "outro": ["qual o horário de atendimento?", "vcs tem loja física?", "obrigado", "quem é você?"],
}


def typo(text, rng, p=0.03):
    out = []
    for ch in text:
        r = rng.random()
        if ch.isalpha() and r < p:
            continue  # drop a letter
        out.append(ch)
    text = "".join(out)
    for a, b in [("você", "vc"), ("por favor", "pfv"), ("que", "q"), ("não", "nao"), ("está", "tá")]:
        if rng.random() < 0.4:
            text = text.replace(a, b)
    return text.lower() if rng.random() < 0.5 else text


def template_message(label, rng):
    v = f" de {fmt_valor(label['valor'], rng)}" if label["valor"] is not None else ""
    p = f" em {label['parcelas']}{rng.choice(['x', ' vezes', ' parcelas'])}" if label["parcelas"] is not None else ""
    d = f" com vencimento dia {label['dia_vencimento']}" if label["dia_vencimento"] is not None else ""
    t = rng.choice(TEMPLATES[label["intent"]])
    t = t.replace("{v}", v).replace("{p}", p).replace("{d}", d)
    return typo(rng.choice(OPENERS) + t, rng)


# ---------- llm mode ----------

LLM_PROMPT = """Escreva UMA mensagem de WhatsApp de um cliente brasileiro para uma empresa de cobrança.
Fatos que a mensagem deve conter (e nenhum outro número):
{facts}
Varie o estilo: pode ter gírias, erros de digitação, abreviações, informalidade, contexto pessoal.
Responda só com a mensagem, sem aspas."""


def facts_text(label):
    lines = [f"- intenção: {label['intent']}"]
    if label["valor"] is not None:
        lines.append(f"- valor: R$ {label['valor']:.2f}")
    if label["parcelas"] is not None:
        lines.append(f"- número de parcelas: {label['parcelas']}")
    if label["dia_vencimento"] is not None:
        lines.append(f"- dia de vencimento desejado: {label['dia_vencimento']}")
    return "\n".join(lines)


def numbers_present(msg, label):
    """Cheap sanity check: every labeled number must appear in the message."""
    digits = re.sub(r"[^\d]", " ", msg.replace(".", "").replace(",", ""))
    found = set(digits.split())
    for f in ("parcelas", "dia_vencimento"):
        if label[f] is not None and str(label[f]) not in found:
            return False
    if label["valor"] is not None:
        whole = str(int(label["valor"]))
        if not any(tok.startswith(whole) for tok in found):
            return False
    return True


def llm_message(label, client, model):
    resp = client.messages.create(
        model=model,
        max_tokens=200,
        temperature=1.0,
        messages=[{"role": "user", "content": LLM_PROMPT.format(facts=facts_text(label))}],
    )
    return resp.content[0].text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["template", "llm"], default="template")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--out", default="data/train.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    client = model = None
    if args.mode == "llm":
        import anthropic  # pip install anthropic

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        model = os.environ.get("GEN_MODEL", "claude-haiku-5-5")

    kept = dropped = 0
    with open(args.out, "w", encoding="utf-8") as f:
        while kept < args.n:
            label = sample_label(rng)
            if args.mode == "template":
                msg = template_message(label, rng)
            else:
                msg = llm_message(label, client, model)
                if not numbers_present(msg, label):
                    dropped += 1
                    continue
            f.write(json.dumps({"message": msg, "label": label}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"wrote {kept} examples to {args.out} (dropped {dropped})")


if __name__ == "__main__":
    main()
