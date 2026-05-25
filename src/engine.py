"""
Módulo de Treinamento e Validação — Integrantes D e E

Contém a lógica matemática de treino (forward pass, loss, backward pass,
otimização de pesos) e de avaliação (inferência sem gradientes, cálculo de
acurácia e salvamento de checkpoints).
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
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
    # ---------------------------------------------------------------
    # TODO (Integrante D — Engenharia do Treinamento):
    #   1. Colocar o modelo em modo de treino (model.train()).
    #   2. Iterar sobre o dataloader.
    #   3. Mover batch para device.
    #   4. Forward pass → calcular loss.
    #   5. loss.backward() → optimizer.step() → optimizer.zero_grad().
    #   6. Acumular e retornar a loss média.
    # ---------------------------------------------------------------
    raise NotImplementedError(
        "Integrante D — Engenharia do Treinamento: implementar train_one_epoch"
    )


# ── Integrante E — Validação e Checkpoints ────────────────────────


def evaluate(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Avalia o modelo sem atualizar pesos.

    Args:
        model: Modelo Transformer.
        dataloader: DataLoader de validação.
        device: Dispositivo de execução.

    Returns:
        Tupla (val_loss, accuracy).
    """
    # ---------------------------------------------------------------
    # TODO (Integrante E — Validação e Checkpoints):
    #   1. Colocar o modelo em modo de avaliação (model.eval()).
    #   2. Desativar gradientes (torch.no_grad()).
    #   3. Iterar sobre o dataloader, acumular loss e acertos.
    #   4. Retornar (val_loss_média, accuracy).
    # ---------------------------------------------------------------
    raise NotImplementedError(
        "Integrante E — Validação e Checkpoints: implementar evaluate"
    )


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
    # ---------------------------------------------------------------
    # TODO (Integrante E — Validação e Checkpoints):
    #   1. model.save_pretrained(path)
    #   2. tokenizer.save_pretrained(path)
    # ---------------------------------------------------------------
    raise NotImplementedError(
        "Integrante E — Validação e Checkpoints: implementar save_checkpoint"
    )
