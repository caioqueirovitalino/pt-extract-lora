"""Label schema and prompt shared by every script, so train, predict and eval agree."""
import json

INTENTS = [
    "renegociar",          # wants to renegotiate / split a debt into installments
    "segunda_via",         # wants a copy of a bill / boleto
    "consultar_saldo",     # asks how much they owe
    "contestar_cobranca",  # disputes a charge
    "cancelar",            # wants to cancel a service / agreement
    "outro",               # anything else
]

FIELDS = ["intent", "valor", "parcelas", "dia_vencimento"]

SYSTEM_PROMPT = (
    "Você extrai dados de mensagens de clientes de uma empresa de cobrança. "
    "Responda APENAS com um objeto JSON com as chaves: "
    '"intent" (uma de: ' + ", ".join(INTENTS) + "), "
    '"valor" (número em reais ou null), '
    '"parcelas" (inteiro ou null), '
    '"dia_vencimento" (inteiro de 1 a 31 ou null). '
    "Use null quando a informação não estiver na mensagem. Não invente valores."
)


def empty_label():
    return {"intent": "outro", "valor": None, "parcelas": None, "dia_vencimento": None}


def to_chat(message, label=None):
    """Chat-format example. With a label it is a training example; without, a prompt."""
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]
    if label is not None:
        msgs.append({"role": "assistant", "content": json.dumps(label, ensure_ascii=False)})
    return msgs


def parse_output(text):
    """Pull the first JSON object out of a model's raw output. Returns (dict|None, valid_json: bool)."""
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None, False
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None, False
    if not isinstance(obj, dict):
        return None, False
    return obj, True
