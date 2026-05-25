"""
Módulo de Dados — Integrante B (Engenharia de Dados via Nuvem)

Responsabilidades:
    - Conectar à biblioteca do Hugging Face e carregar o dataset de brinquedo.
    - Aplicar tokenização ao texto cru, gerando input_ids e attention_mask.
    - Dividir logicamente em train_dataset e eval_dataset.
    - Empacotar os dados vetorizados em DataLoaders para alimentar o modelo.
"""

from __future__ import annotations

from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase


def create_dataloaders(
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Carrega o dataset, aplica tokenização e retorna (train_loader, val_loader).

    Args:
        tokenizer: Tokenizador do Transformer pré-treinado.
        batch_size: Quantidade de amostras por lote.

    Returns:
        Tupla com (train_loader, val_loader) prontos para consumo pelo engine.
    """
    # ---------------------------------------------------------------
    # TODO (Integrante B — Engenharia de Dados via Nuvem):
    #   1. Usar `datasets.load_dataset(...)` para baixar o dataset.
    #   2. Mapear o tokenizador sobre o dataset (dataset.map).
    #   3. Criar e retornar os DataLoaders de treino e validação.
    # ---------------------------------------------------------------
    raise NotImplementedError(
        "Integrante B — Engenharia de Dados via Nuvem: implementar create_dataloaders"
    )
