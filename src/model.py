"""
Módulo de Arquitetura — Integrante C (Arquitetura do Modelo)

Responsabilidades:
    - Pesquisar e baixar os pesos estruturais do Transformer pré-treinado.
    - Definir os hiperparâmetros básicos da rede neural (num_labels, etc.).
    - Mover o modelo para o device correto (CPU ou GPU).
    - Retornar o objeto do modelo e o tokenizador prontos para uso.
"""

from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def load_model(
    num_labels: int,
    device: torch.device,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Baixa o Transformer pré-treinado e retorna (model, tokenizer).

    O modelo retornado já deve estar no *device* informado.

    Args:
        num_labels: Quantidade de classes de classificação.
        device: Dispositivo de execução (cpu / cuda).

    Returns:
        Tupla com (model, tokenizer).
    """
    # ---------------------------------------------------------------
    # TODO (Integrante C — Arquitetura do Modelo):
    #   1. Escolher o checkpoint pré-treinado (ex.: "bert-base-uncased").
    #   2. Instanciar AutoTokenizer e AutoModelForSequenceClassification.
    #   3. Mover o modelo para `device` e retornar (model, tokenizer).
    # ---------------------------------------------------------------
    raise NotImplementedError(
        "Integrante C — Arquitetura do Modelo: implementar load_model"
    )
