"""
Módulo de Treinamento e Validação — Integrantes D e E

Contém a lógica matemática de treino (forward pass, loss, backward pass,
otimização de pesos) e de avaliação (inferência sem gradientes, cálculo de
acurácia e salvamento de checkpoints).
"""

from __future__ import annotations

import os
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
) -> float:
    """Executa um epoch completo de treino.

    Para cada lote:
        1. Forward pass → logits
        2. Cálculo da loss
        3. Backward pass (retropropagação)
        4. Passo do optimizer (atualização de pesos)

    Args:
        model: Modelo Transformer.
        dataloader: DataLoader de treino.
        optimizer: Otimizador (ex.: AdamW).
        device: Dispositivo de execução.

    Returns:
        Loss média do epoch.
    """
    # 1. Colocar o modelo em modo de treino (habilita Dropout, etc)
    model.train()
    
    total_loss = 0.0
    
    # 2. Iterar sobre o dataloader
    for batch in dataloader:
        
        # 3. Mover batch para device (GPU ou CPU)
        # O dataloader do Hugging Face geralmente entrega um dicionário
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 4. Forward pass → calcular loss
        # Ao passar **batch, passamos input_ids, attention_mask e labels.
        # O modelo Hugging Face calcula a loss automaticamente se receber os 'labels'
        outputs = model(**batch)
        loss = outputs.loss
        
        # 5. loss.backward() → optimizer.step() → optimizer.zero_grad()
        loss.backward()         # Retropropagação (calcula os gradientes)
        optimizer.step()        # Atualiza os pesos
        optimizer.zero_grad()   # Limpa os gradientes para o próximo lote
        
        # 6. Acumular a loss
        total_loss += loss.item()
        
    # Calcular e retornar a loss média do epoch
    avg_loss = total_loss / len(dataloader)
    return avg_loss


# ── Integrante E — Validação e Checkpoints ────────────────────────


def evaluate(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: torch.nn.Module | None = None,
) -> dict:
    """Avalia o modelo sem atualizar pesos.

    Args:
        model: Modelo Transformer.
        dataloader: DataLoader de validação.
        device: Dispositivo de execução.

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
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss = loss_fn(logits, labels) if loss_fn is not None else torch.nn.functional.cross_entropy(logits, labels)
            predictions = torch.argmax(logits, dim=-1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
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

def print_report(metrics: dict, epoch: int, num_epochs: int) -> None:
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
    print(f" Epoch {epoch}/{num_epochs}")
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
    """Salva o modelo e o tokenizador treinados no disco.

    Args:
        model: Modelo treinado.
        tokenizer: Tokenizador utilizado.
        path: Diretório de destino no disco.
    """
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "model.pt"))
    tokenizer.save_pretrained(path)


# ══════════════════════════════════════════════════════════════════
# ENTREGA FINAL — Guias para os Integrantes D (Anselmo) e E (Robert)
# ══════════════════════════════════════════════════════════════════
#
# As funções acima são da entrega parcial e servem de referência.
# Para a entrega final, elas precisam ser MODIFICADAS conforme
# descrito abaixo. A main.py (Integrante A) já está preparada
# para chamar as novas assinaturas.
#
# ⚠️  BUG IMPORTANTE: nosso modelo (BERTimbauBinaryClassifier) é
# um nn.Module customizado que retorna APENAS logits. Ele NÃO
# aceita 'labels' como parâmetro e NÃO retorna outputs.loss.
# Por isso, a loss deve ser calculada EXTERNAMENTE com a loss_fn
# que a main.py passa como parâmetro.


# ── TODO INTEGRANTE D (Anselmo) ──────────────────────────────────
#
# MODIFICAR a função train_one_epoch para a nova assinatura:
#
#   def train_one_epoch(
#       model, dataloader, optimizer, device,
#       scheduler=None, loss_fn=None,
#   ) -> float:
#
# Mudanças necessárias dentro do loop de batch:
#
#   1. Separar labels do batch ANTES do forward pass:
#      labels = batch.pop("labels")
#
#   2. Chamar o modelo apenas com input_ids e attention_mask:
#      logits = model(input_ids=batch["input_ids"],
#                     attention_mask=batch["attention_mask"])
#
#   3. Calcular a loss com a função externa:
#      loss = loss_fn(logits, labels)
#
#   4. Após optimizer.step(), adicionar o passo do scheduler:
#      if scheduler is not None:
#          scheduler.step()
#
# O restante (backward, zero_grad, acumulação de loss) permanece
# igual. A main.py cria o scheduler com
# get_linear_schedule_with_warmup e passa aqui.


# ── TODO INTEGRANTE E (Robert) ───────────────────────────────────
#
# 1) MODIFICAR a função evaluate para a nova assinatura:
#
#   def evaluate(
#       model, dataloader, device,
#       loss_fn=None,
#   ) -> dict:
#
# Mudanças necessárias:
#
#   a) Mesma lógica do train para calcular loss externamente:
#      labels = batch.pop("labels")
#      logits = model(input_ids=..., attention_mask=...)
#      loss = loss_fn(logits, labels)
#
#   b) Coletar TODAS as predições e labels do dataset inteiro
#      em duas listas (all_preds e all_labels).
#
#   c) Após o loop, calcular métricas com sklearn:
#      from sklearn.metrics import (
#          precision_score, recall_score, f1_score,
#          confusion_matrix,
#      )
#      OBS: sklearn já foi adicionado ao requirements.txt.
#
#   d) Retornar um dicionário em vez da tupla (loss, accuracy):
#      return {
#          "val_loss": float,
#          "accuracy": float,
#          "precision": float,
#          "recall": float,
#          "f1": float,
#          "confusion_matrix": np.ndarray,
#      }
#
# 2) CRIAR a função print_report:
#
#   def print_report(metrics: dict, epoch: int, num_epochs: int) -> None:
#
# Esta função deve formatar e exibir os resultados de forma legível.
# Exemplo de saída esperada:
#
#   ════════════════════════════════════════
#    Epoch 2/3
#   ────────────────────────────────────────
#    Val Loss:   0.4321
#    Accuracy:   87.50%
#    Precision:  0.8421
#    Recall:     0.7619
#    F1-Score:   0.8000
#
#    Confusion Matrix:
#          Pred 0  Pred 1
#    Real 0  [150    12]
#    Real 1  [ 25    80]
#   ════════════════════════════════════════
#
# A main.py chama print_report(metrics, epoch, NUM_EPOCHS) ao
# final de cada epoch.
#
# 3) CORRIGIR a função save_checkpoint:
#
# A versão atual usa model.save_pretrained() que NÃO funciona
# com nosso nn.Module customizado (BERTimbauBinaryClassifier).
# Substituir por:
#
#   import os
#   def save_checkpoint(model, tokenizer, path):
#       os.makedirs(path, exist_ok=True)
#       torch.save(model.state_dict(), os.path.join(path, "model.pt"))
#       tokenizer.save_pretrained(path)
#
# O inference.py (Integrante A) já está preparado para carregar
# o checkpoint nesse formato (model.pt + tokenizer files).
