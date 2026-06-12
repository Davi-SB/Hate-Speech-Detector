"""Pipeline de Inferência — Integrante A (Davi)

Encapsula o carregamento do modelo treinado e a classificação de textos
novos. Projetado para funcionar de forma independente assim que um
checkpoint válido existir em disco.

Uso direto (sem main.py):
    from src.inference import load_pipeline, classify
    model, tokenizer, device = load_pipeline("checkpoints")
    resultado = classify("seu texto aqui", model, tokenizer, device)
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.model import BERTimbauBinaryClassifier

logger = logging.getLogger(__name__)

# Mapeamento padrão caso ID2LABEL não esteja disponível em runtime.
# Será sobrescrito por load_pipeline se o dataset.py já tiver populado
# a variável global.
_DEFAULT_ID2LABEL: dict[int, str] = {0: "nao_odio", 1: "odio"}


def load_pipeline(
    checkpoint_dir: str,
    device: torch.device | None = None,
    num_labels: int = 2,
) -> tuple[BERTimbauBinaryClassifier, AutoTokenizer, torch.device, dict[int, str]]:
    """Carrega modelo treinado e tokenizador a partir de um checkpoint.

    Args:
        checkpoint_dir: Diretório onde estão salvos ``model.pt`` e os
            arquivos do tokenizador.
        device: Dispositivo de execução. Se ``None``, detecta
            automaticamente (CUDA se disponível, senão CPU).
        num_labels: Número de classes de classificação.

    Returns:
        Tupla ``(model, tokenizer, device, id2label)``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(checkpoint_dir)

    # ── Tokenizador ───────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))

    # ── Modelo ────────────────────────────────────────────────────
    model = BERTimbauBinaryClassifier(num_labels=num_labels)

    weights_file = checkpoint_path / "model.pt"
    if not weights_file.exists():
        raise FileNotFoundError(
            f"Arquivo de pesos não encontrado: {weights_file}\n"
            "Certifique-se de que o treinamento foi executado e o "
            "checkpoint foi salvo corretamente."
        )

    model.load_state_dict(
        torch.load(str(weights_file), map_location=device, weights_only=True)
    )
    model.to(device).eval()
    logger.info("Modelo carregado de %s no device %s", checkpoint_path, device)

    # ── Mapeamento de labels ──────────────────────────────────────
    # Tenta usar o mapeamento global criado pelo dataset.py durante
    # o treinamento. Se não estiver disponível (inferência isolada),
    # usa o fallback padrão.
    id2label: dict[int, str]
    try:
        from src.dataset import ID2LABEL as _runtime_id2label

        if _runtime_id2label is not None:
            id2label = _runtime_id2label
        else:
            id2label = _DEFAULT_ID2LABEL
    except ImportError:
        id2label = _DEFAULT_ID2LABEL

    return model, tokenizer, device, id2label


def classify(
    text: str,
    model: BERTimbauBinaryClassifier,
    tokenizer: AutoTokenizer,
    device: torch.device,
    id2label: dict[int, str] | None = None,
) -> dict[str, str | int | float]:
    """Classifica um texto como discurso de ódio ou não.

    Aplica pré-processamento (``clean_text``, se disponível), tokeniza,
    executa o forward pass sem gradientes e retorna a predição com a
    confiança.

    Args:
        text: Texto cru do usuário.
        model: Modelo já carregado via ``load_pipeline``.
        tokenizer: Tokenizador correspondente.
        device: Dispositivo de execução.
        id2label: Mapeamento id → nome da classe. Se ``None``, usa o
            fallback padrão ``{0: "nao_odio", 1: "odio"}``.

    Returns:
        Dicionário com ``label``, ``label_id`` e ``confidence``.
    """
    if id2label is None:
        id2label = _DEFAULT_ID2LABEL

    # ── Pré-processamento (Integrante B) ──────────────────────────
    # Tenta aplicar clean_text() se o Integrante B já tiver
    # implementado. Caso contrário, usa o texto cru sem erro.
    try:
        from src.dataset import clean_text
        text = clean_text(text)
    except (ImportError, NotImplementedError):
        pass

    # ── Tokenização ───────────────────────────────────────────────
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # ── Forward pass ──────────────────────────────────────────────
    with torch.no_grad():
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

    probs = F.softmax(logits, dim=-1)
    pred_id = probs.argmax(dim=-1).item()
    confidence = probs[0, pred_id].item()

    return {
        "label": id2label.get(pred_id, str(pred_id)),
        "label_id": pred_id,
        "confidence": round(confidence, 4),
    }
