"""Hate-Speech Detector — Orquestrador Principal (Integrante A)

Conecta os módulos de dados, modelo e treinamento e executa o pipeline
completo de fine-tuning de um Transformer para classificação de texto.

Pipeline:
    1. Carrega o modelo pré-treinado e o tokenizador.
    2. Cria os DataLoaders de treino e validação.
    3. Configura o otimizador.
    4. Executa o loop de epochs (treino + avaliação).
    5. Salva o checkpoint final no disco.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from src.dataset import create_dataloaders
from src.engine import evaluate, save_checkpoint, train_one_epoch
from src.model import load_model

# ── Configuração ──────────────────────────────────────────────────

BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
NUM_LABELS = 2
CHECKPOINT_DIR = Path("checkpoints")

# ── Logging ───────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Pipeline ──────────────────────────────────────────────────────


def main() -> None:
    # 1. Device ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device selecionado: %s", device)

    # 2. Modelo e tokenizador (Integrante C) ───────────────────────
    logger.info("Carregando modelo e tokenizador…")
    model, tokenizer = load_model(num_labels=NUM_LABELS, device=device)

    # 3. DataLoaders (Integrante B) ────────────────────────────────
    logger.info("Criando DataLoaders (batch_size=%d)…", BATCH_SIZE)
    train_loader, val_loader = create_dataloaders(
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
    )

    # 4. Otimizador ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    logger.info(
        "Otimizador: AdamW | lr=%.1e | params=%s",
        LEARNING_RATE,
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    # 5. Loop de treinamento (Integrantes D e E) ───────────────────
    logger.info("Iniciando treinamento — %d epoch(s)", NUM_EPOCHS)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, accuracy = evaluate(model, val_loader, device)

        logger.info(
            "Epoch %d/%d  ·  train_loss=%.4f  ·  val_loss=%.4f  ·  accuracy=%.2f%%",
            epoch,
            NUM_EPOCHS,
            train_loss,
            val_loss,
            accuracy * 100,
        )

    # 6. Checkpoint (Integrante E) ─────────────────────────────────
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, tokenizer, str(CHECKPOINT_DIR))
    logger.info("Checkpoint salvo em %s", CHECKPOINT_DIR.resolve())

    logger.info("Pipeline concluído com sucesso.")


if __name__ == "__main__":
    main()
