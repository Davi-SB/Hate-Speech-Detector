"""Hate-Speech Detector — Orquestrador Principal (Integrante A — Davi)

Conecta os módulos de dados, modelo e treinamento e executa o pipeline
completo de fine-tuning de um Transformer para classificação de texto.

Modos de execução:
    Treinamento (padrão):
        python main.py

    Inferência (texto direto):
        python main.py --infer --text "seu texto aqui"

    Inferência (interativo):
        python main.py --infer
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup

from src.dataset import create_dataloaders, compute_class_weights
from src.engine import evaluate, print_report, save_checkpoint, train_one_epoch
from src.model import load_model

# ── Configuração ──────────────────────────────────────────────────

BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
NUM_LABELS = 2
WARMUP_RATIO = 0.1
CHECKPOINT_DIR = Path("checkpoints")
SEED = 42
PATIENCE = 2
MAX_GRAD_NORM = 1.0

# ── Logging ───────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Fixa as seeds para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── CLI ───────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hate-Speech Detector — Treinamento e Inferência",
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        help="Modo inferência: carrega checkpoint e classifica texto.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Texto para classificar (usado com --infer).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_DIR),
        help=f"Diretório do checkpoint (padrão: {CHECKPOINT_DIR}).",
    )
    return parser.parse_args()


# ── Modo Treinamento ─────────────────────────────────────────────


def train(args: argparse.Namespace) -> None:
    """Pipeline completo de treinamento."""

    set_seed(SEED)
    logger.info("Seed fixada: %d", SEED)

    # 1. Device ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info("GPU detectada: %s", torch.cuda.get_device_name(0))
    else:
        logger.warning(
            "CUDA não disponível — treinamento será em CPU (mais lento)."
        )
    logger.info("Device selecionado: %s", device)

    # 2. Modelo e tokenizador (Integrante C) ───────────────────────
    logger.info("Carregando modelo e tokenizador…")
    model, tokenizer = load_model(num_labels=NUM_LABELS, device=device)

    # 3. DataLoaders (Integrante B) ────────────────────────────────
    logger.info("Criando DataLoaders (batch_size=%d)…", BATCH_SIZE)
    train_loader, val_loader, test_loader = create_dataloaders(
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
    )

    # 4. Class Weights + Loss ponderada (Integrante B) ─────────────
    logger.info("Calculando class weights…")
    class_weights = compute_class_weights(train_loader)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float).to(device),
    )
    logger.info("Class weights: %s", class_weights)

    # 5. Otimizador ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    logger.info(
        "Otimizador: AdamW | lr=%.1e | params=%s",
        LEARNING_RATE,
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    # 6. Learning Rate Scheduler ───────────────────────────────────
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    logger.info(
        "Scheduler: linear warmup (%d steps) + decay (%d steps total)",
        warmup_steps,
        total_steps,
    )

    # 7. Mixed Precision ───────────────────────────────────────────
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    logger.info(
        "Mixed precision (fp16): %s",
        "ativado" if use_amp else "desativado (CPU)",
    )

    # 8. Loop de treinamento com early stopping ────────────────────
    logger.info(
        "Iniciando treinamento — até %d epoch(s), patience=%d",
        NUM_EPOCHS, PATIENCE,
    )

    checkpoint_dir = Path(args.checkpoint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    metrics_history: list[dict] = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            scheduler=scheduler,
            loss_fn=loss_fn,
            max_grad_norm=MAX_GRAD_NORM,
            scaler=scaler,
        )

        metrics = evaluate(
            model, val_loader, device, loss_fn=loss_fn, use_amp=use_amp,
        )

        logger.info("Epoch %d/%d  ·  train_loss=%.4f", epoch, NUM_EPOCHS, train_loss)
        print_report(metrics, epoch, NUM_EPOCHS)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(metrics["val_loss"], 6),
            "accuracy": round(metrics["accuracy"], 6),
            "precision": round(metrics["precision"], 6),
            "recall": round(metrics["recall"], 6),
            "f1": round(metrics["f1"], 6),
            "confusion_matrix": np.asarray(metrics["confusion_matrix"]).tolist(),
        }
        metrics_history.append(epoch_record)

        if metrics["val_loss"] < best_val_loss:
            best_val_loss = metrics["val_loss"]
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(model, tokenizer, str(checkpoint_dir))
            logger.info(
                "Melhor val_loss=%.4f — checkpoint salvo em %s",
                best_val_loss, checkpoint_dir.resolve(),
            )
        else:
            patience_counter += 1
            logger.info(
                "val_loss não melhorou (%.4f >= %.4f) — patience %d/%d",
                metrics["val_loss"], best_val_loss,
                patience_counter, PATIENCE,
            )
            if patience_counter >= PATIENCE:
                logger.info("Early stopping ativado no epoch %d.", epoch)
                break

    # 9. Carregar melhor modelo para avaliação final ────────────────
    logger.info(
        "Carregando melhor modelo (epoch %d) para avaliação no teste…",
        best_epoch,
    )
    model.load_state_dict(
        torch.load(
            str(checkpoint_dir / "model.pt"),
            map_location=device,
            weights_only=True,
        )
    )

    # 10. Avaliação no conjunto de teste ────────────────────────────
    test_metrics = evaluate(
        model, test_loader, device, loss_fn=loss_fn, use_amp=use_amp,
    )
    print_report(
        test_metrics, best_epoch, NUM_EPOCHS,
        title="AVALIAÇÃO FINAL — CONJUNTO DE TESTE",
    )

    # 11. Salvar métricas ──────────────────────────────────────────
    results = {
        "config": {
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_epochs": NUM_EPOCHS,
            "warmup_ratio": WARMUP_RATIO,
            "max_grad_norm": MAX_GRAD_NORM,
            "patience": PATIENCE,
            "mixed_precision": use_amp,
            "device": str(device),
        },
        "training_history": metrics_history,
        "best_epoch": best_epoch,
        "early_stopped": patience_counter >= PATIENCE,
        "test_metrics": {
            "val_loss": round(test_metrics["val_loss"], 6),
            "accuracy": round(test_metrics["accuracy"], 6),
            "precision": round(test_metrics["precision"], 6),
            "recall": round(test_metrics["recall"], 6),
            "f1": round(test_metrics["f1"], 6),
            "confusion_matrix": np.asarray(
                test_metrics["confusion_matrix"]
            ).tolist(),
        },
    }

    metrics_path = checkpoint_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Métricas salvas em %s", metrics_path.resolve())

    logger.info("Pipeline de treinamento concluído com sucesso.")


# ── Modo Inferência ───────────────────────────────────────────────


def infer(args: argparse.Namespace) -> None:
    """Carrega checkpoint e classifica texto(s)."""
    from src.inference import classify, load_pipeline

    logger.info("Carregando pipeline de inferência…")
    model, tokenizer, device, id2label = load_pipeline(args.checkpoint)

    if args.text:
        result = classify(args.text, model, tokenizer, device, id2label)
        _print_result(args.text, result)
    else:
        logger.info("Modo interativo — digite 'sair' para encerrar.")
        while True:
            try:
                text = input("\n📝 Digite um texto: ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if text.strip().lower() == "sair":
                break
            result = classify(text, model, tokenizer, device, id2label)
            _print_result(text, result)

    logger.info("Inferência encerrada.")


def _print_result(text: str, result: dict) -> None:
    """Formata e exibe o resultado de uma classificação."""
    print(
        f"\n{'═' * 50}\n"
        f"  Texto:      {text}\n"
        f"  Classe:     {result['label']}\n"
        f"  Confiança:  {result['confidence']:.2%}\n"
        f"{'═' * 50}"
    )


# ── Entrypoint ────────────────────────────────────────────────────


if __name__ == "__main__":
    args = parse_args()

    if args.infer:
        infer(args)
    else:
        train(args)
