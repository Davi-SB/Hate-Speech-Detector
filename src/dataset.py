"""Módulo de dados do projeto final.

Responsável por carregar o dataset real, limpar o texto da internet,
tokenizar os exemplos e expor os mapeamentos de rótulos e pesos de classe.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

DATASET_NAME = "projetomemoreba/mteb_told-br"

TEXT_COLUMN_CANDIDATES = (
    "text",
    "rewritten_text",
    "sentence",
    "content",
    "tweet",
    "post",
    "comment",
)
LABEL_COLUMN_CANDIDATES = ("label", "labels", "class", "target", "toxicity")
LABEL_TEXT_COLUMN_CANDIDATES = (
    "label_text",
    "label_name",
    "class_name",
    "target_text",
)

MAX_LENGTH = 128

LABEL2ID: dict[Any, int] | None = None
ID2LABEL: dict[int, str] | None = None
CLASS_WEIGHTS: list[float] | None = None

_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+", flags=re.UNICODE)
_HASHTAG_RE = re.compile(r"#(\w+)", flags=re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s+")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}", flags=re.UNICODE)
_SPECIAL_CHAR_RE = re.compile(r"[^0-9A-Za-zÀ-ÿ\s]")
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "]+",
    flags=re.UNICODE,
)


def _pick_column(column_names: list[str], candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in column_names:
            return candidate
    raise ValueError(
        "Não foi possível identificar a coluna esperada. "
        f"Colunas disponíveis: {column_names}"
    )


def _pick_optional_column(
    column_names: list[str],
    candidates: tuple[str, ...],
) -> str | None:
    for candidate in candidates:
        if candidate in column_names:
            return candidate
    return None


def _normalize_label_value(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def clean_text(text: str) -> str:
    """Limpa ruído típico de texto da internet."""
    if text is None:
        return ""

    cleaned = str(text).strip().lower()
    cleaned = _URL_RE.sub(" ", cleaned)
    cleaned = _MENTION_RE.sub(" ", cleaned)
    cleaned = _HASHTAG_RE.sub(r"\1", cleaned)
    cleaned = _EMOJI_RE.sub(" ", cleaned)
    cleaned = _REPEATED_CHAR_RE.sub(r"\1\1", cleaned)
    cleaned = _SPECIAL_CHAR_RE.sub(" ", cleaned)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def _build_label_mappings(
    train_source: Dataset,
    eval_source: Dataset,
    label_column: str,
    label_text_column: str | None,
) -> tuple[dict[Any, int], dict[int, str]]:
    train_labels = [_normalize_label_value(value) for value in train_source[label_column]]
    eval_labels = [_normalize_label_value(value) for value in eval_source[label_column]]
    unique_labels = sorted(set(train_labels).union(set(eval_labels)))

    label2id: dict[Any, int] = {label: index for index, label in enumerate(unique_labels)}

    if label_text_column is None:
        id2label = {index: str(label) for label, index in label2id.items()}
        return label2id, id2label

    raw_to_text: dict[Any, str] = {}
    for split in (train_source, eval_source):
        for raw_label, label_text in zip(split[label_column], split[label_text_column]):
            normalized_label = _normalize_label_value(raw_label)
            raw_to_text.setdefault(normalized_label, str(label_text))

    id2label = {
        index: raw_to_text.get(label, str(label))
        for label, index in label2id.items()
    }
    return label2id, id2label


def _prepare_split(
    split_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    text_column: str,
    label_column: str,
    label2id: dict[Any, int],
) -> Dataset:
    def preprocess_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        cleaned_texts = [clean_text(text) for text in batch[text_column]]
        tokenized = tokenizer(
            cleaned_texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        tokenized["labels"] = [label2id[_normalize_label_value(label)] for label in batch[label_column]]
        return tokenized

    tokenized_dataset = split_dataset.map(
        preprocess_batch,
        batched=True,
        remove_columns=list(split_dataset.column_names),
    )

    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    return tokenized_dataset


def create_dataloaders(
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Carrega o dataset, limpa, tokeniza e retorna os DataLoaders."""
    dataset = load_dataset(DATASET_NAME)

    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            train_source = dataset["train"]
        else:
            split_names = list(dataset.keys())
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
    label_text_column = _pick_optional_column(
        train_source.column_names,
        LABEL_TEXT_COLUMN_CANDIDATES,
    )

    label2id, id2label = _build_label_mappings(
        train_source=train_source,
        eval_source=eval_source,
        label_column=label_column,
        label_text_column=label_text_column,
    )

    global LABEL2ID, ID2LABEL
    LABEL2ID = label2id
    ID2LABEL = id2label

    train_dataset = _prepare_split(
        train_source,
        tokenizer,
        text_column,
        label_column,
        label2id,
    )
    eval_dataset = _prepare_split(
        eval_source,
        tokenizer,
        text_column,
        label_column,
        label2id,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, eval_loader


def compute_class_weights(dataloader: DataLoader) -> list[float]:
    """Calcula pesos de classe inversamente proporcionais à frequência."""
    label_counts: Counter[int] = Counter()

    for batch in dataloader:
        labels = batch["labels"]
        if hasattr(labels, "detach"):
            labels = labels.detach().cpu().tolist()
        elif hasattr(labels, "cpu"):
            labels = labels.cpu().tolist()
        elif not isinstance(labels, list):
            labels = list(labels)

        label_counts.update(int(label) for label in labels)

    if not label_counts:
        raise ValueError("Não foi possível calcular class weights: dataloader vazio.")

    num_classes = max(label_counts) + 1
    total_examples = sum(label_counts.values())

    weights: list[float] = []
    for class_id in range(num_classes):
        frequency = label_counts.get(class_id, 0)
        if frequency == 0:
            raise ValueError(
                f"Classe {class_id} não apareceu no treino e não pode receber peso."
            )
        weights.append(total_examples / (num_classes * frequency))

    global CLASS_WEIGHTS
    CLASS_WEIGHTS = weights
    return weights
