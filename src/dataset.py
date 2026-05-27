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
from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

# Configurações globais para o módulo de dados
# utilizei esse dataset mais simples para essa primeira etapa,
#  mas ele pode ser facilmente substituído por outro mais complexo, caso necessário
DATASET_NAME = "BRlkl/told-br-rewritten"

# Possíveis nomes para a coluna de texto.
# O código tenta encontrar automaticamente uma dessas colunas.
TEXT_COLUMN_CANDIDATES = (
    "rewritten_text",
    "text",
    "sentence",
    "content",
    "tweet",
    "post",
    "comment",
)
# Possíveis nomes para a coluna de rótulo.
LABEL_COLUMN_CANDIDATES = ("label", "labels", "class", "target", "toxicity")

MAX_LENGTH = 128

## Dicionários globais úteis para o restante do projeto
# Exemplo:
# LABEL2ID["seguro"] -> 0
# ID2LABEL[0] -> "seguro"
LABEL2ID: dict[str, int] | None = None
ID2LABEL: dict[int, str] | None = None

# funcao auxiliar para achar as colunas, texto e rotulo
def _pick_column(column_names: list[str], candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in column_names:
            return candidate
    raise ValueError(
        "Não foi possível identificar a coluna de texto/rótulo. "
        f"Colunas disponíveis: {column_names}"
    )


def _prepare_split(
    split_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    text_column: str,
    label_column: str,
    label2id: dict[str, int],
) -> Dataset:
    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list]:
        return tokenizer(
            batch[text_column],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    tokenized_dataset = split_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=[column for column in split_dataset.column_names if column != label_column],
    )

    if label_column != "labels":
        tokenized_dataset = tokenized_dataset.rename_column(label_column, "labels")

    # Converter rótulos textuais para inteiros usando label2id
    def _map_labels(batch: dict[str, list[str]]) -> dict[str, list[int]]:
        batch["labels"] = [label2id[l] for l in batch["labels"]]
        return batch

    tokenized_dataset = tokenized_dataset.map(_map_labels, batched=True)

    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized_dataset


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
    dataset = load_dataset(DATASET_NAME)
    # separa train/eval  
    # caso não existir validação no dataset, ele faz um split de 80/20 a partir do treino,
    if isinstance(dataset, DatasetDict):
        split_names = list(dataset.keys())
        if "train" in dataset:
            train_source = dataset["train"]
        else:
            train_source = dataset[split_names[0]]

        if "validation" in dataset:
            eval_source = dataset["validation"]
        elif "eval" in dataset:
            eval_source = dataset["eval"]
        elif "test" in dataset:
            eval_source = dataset["test"]
        else:
            split = train_source.train_test_split(test_size=0.2, seed=42)
            train_source = split["train"]
            eval_source = split["test"]
    else:
        split = dataset.train_test_split(test_size=0.2, seed=42)
        train_source = split["train"]
        eval_source = split["test"]

    text_column = _pick_column(train_source.column_names, TEXT_COLUMN_CANDIDATES)
    label_column = _pick_column(train_source.column_names, LABEL_COLUMN_CANDIDATES)

    # Criar mapeamento label -> inteiro    
    # Exemplo:
    # {
    #   "seguro": 0,
    #   "inseguro": 1
    # } 
    # tester depois com um dataset com mais labels!!
    train_labels = set(train_source.unique(label_column))
    eval_labels = set(eval_source.unique(label_column))
    all_labels = sorted(list(train_labels.union(eval_labels)))
    label2id = {label: idx for idx, label in enumerate(all_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    global LABEL2ID, ID2LABEL
    LABEL2ID = label2id
    ID2LABEL = id2label

    train_dataset = _prepare_split(
        train_source, tokenizer, text_column, label_column, label2id
    )
    eval_dataset = _prepare_split(
        eval_source, tokenizer, text_column, label_column, label2id
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    # train_loader e eval_loader prontos para consumo
    return train_loader, eval_loader
