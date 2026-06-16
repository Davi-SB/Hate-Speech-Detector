"""Módulo de Treinamento e Validação — Integrantes D e E

Contém a lógica matemática de treino (forward pass, loss, backward pass,
otimização de pesos) e de avaliação (inferência sem gradientes, cálculo de
acurácia e salvamento de checkpoints).
"""

from __future__ import annotations

import os
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from transformers import PreTrainedModel, PreTrainedTokenizerBase


# ── Integrante D — Engenharia do Treinamento ──────────────────────


def train_one_epoch(
    model: PreTrainedModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: Any | None = None,
    loss_fn: torch.nn.Module | None = None,
    max_grad_norm: float = 1.0,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """Executa um epoch completo de treino adaptado para o modelo customizado.

    Para cada lote:
        1. Separação dos labels para evitar o bug de assinatura do nn.Module
        2. Forward pass (com autocast fp16 quando disponível)
        3. Cálculo da loss utilizando a loss_fn externa (com class weights)
        4. Backward pass com gradient scaling (mixed precision)
        5. Gradient clipping para estabilidade do treino
        6. Atualização de pesos e passo do Learning Rate Scheduler

    Args:
        model: Modelo Transformer (BERTimbauBinaryClassifier).
        dataloader: DataLoader de treino.
        optimizer: Otimizador (ex.: AdamW).
        device: Dispositivo de execução.
        scheduler: Learning Rate Scheduler opcional (get_linear_schedule_with_warmup).
        loss_fn: Função de perda externa (CrossEntropyLoss configurada com pesos).
        max_grad_norm: Norma máxima para gradient clipping.
        scaler: GradScaler para mixed precision (None desativa AMP).

    Returns:
        Loss média do epoch.
    """
    model.train()
    total_loss = 0.0
    use_amp = scaler is not None

    for batch in dataloader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        with torch.amp.autocast(device.type, enabled=use_amp):
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            if hasattr(logits, "logits"):
                logits = logits.logits
            if loss_fn is None:
                loss = torch.nn.functional.cross_entropy(logits, labels)
            else:
                loss = loss_fn(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


# ── Integrante E — Validação e Checkpoints ────────────────────────


def evaluate(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: torch.nn.Module | None = None,
    use_amp: bool = False,
) -> dict[str, float | np.ndarray]:
    """Avalia o modelo extraindo métricas completas com o Scikit-Learn.

    Args:
        model: Modelo Transformer.
        dataloader: DataLoader de validação.
        device: Dispositivo de execução.
        loss_fn: Função de perda externa.

    Returns:
        Dicionário com métricas de validação.
    """
    model.eval()

    total_loss = 0.0
    total_examples = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            # Isolar labels e mover tensores para o dispositivo de execução
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast(device.type, enabled=use_amp):
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                if loss_fn is not None:
                    loss = loss_fn(logits, labels)
                else:
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                
            predictions = torch.argmax(logits, dim=-1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            
            # Coletar predições e referências reais para o cálculo final do sklearn
            all_preds.extend(predictions.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / total_examples if total_examples > 0 else 0.0

    if total_examples > 0:
        accuracy = sum(int(pred == label) for pred, label in zip(all_preds, all_labels)) / total_examples
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        matrix = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    else:
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        matrix = np.zeros((2, 2), dtype=int)

    return {
        "val_loss": float(avg_loss),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": matrix,
    }


def print_report(
    metrics: dict,
    epoch: int,
    num_epochs: int,
    title: str | None = None,
) -> None:
    """Exibe as métricas de validação em formato legível."""
    matrix = np.asarray(metrics["confusion_matrix"])

    if matrix.shape == (2, 2):
        tn, fp = matrix[0]
        fn, tp = matrix[1]
        matrix_lines = [
            "        Pred 0  Pred 1",
            f"Real 0  [{tn:4d}  {fp:4d}]",
            f"Real 1  [{fn:4d}  {tp:4d}]",
        ]
    else:
        matrix_lines = [str(matrix)]

    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]

    if precision >= recall:
        analysis = (
            "Análise: o modelo está mais conservador nas predições positivas, "
            "com menos falsos positivos do que falsos negativos."
        )
    else:
        analysis = (
            "Análise: o modelo recupera mais casos positivos, mas ainda gera "
            "mais falsos positivos."
        )

    if f1 >= 0.8:
        quality_note = "O equilíbrio entre precisão e recall está bom para esta validação."
    elif f1 >= 0.6:
        quality_note = "O desempenho é moderado e ainda pode melhorar no equilíbrio entre erros."
    else:
        quality_note = "O desempenho ainda está baixo e o modelo precisa de ajuste."

    print("═" * 40)
    print(f" {title}" if title else f" Epoch {epoch}/{num_epochs}")
    print("─" * 40)
    print(f" Val Loss:   {metrics['val_loss']:.4f}")
    print(f" Accuracy:   {metrics['accuracy']:.2%}")
    print(f" Precision:  {precision:.4f}")
    print(f" Recall:     {recall:.4f}")
    print(f" F1-Score:   {f1:.4f}")
    print()
    print(" Confusion Matrix:")
    for line in matrix_lines:
        print(f" {line}")
    print()
    print(f" {analysis}")
    print(f" {quality_note}")
    print("═" * 40)


def save_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    path: str,
) -> None:
    """Salva o modelo e o tokenizador treinados de forma compatível com nn.Module.

    Como o modelo é uma classe customizada baseada pura em nn.Module, a persistência
    é feita via salvamento do state_dict interno no arquivo model.pt.
    """
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "model.pt"))
    tokenizer.save_pretrained(path)
