# Hate-Speech Detector

Detector de discurso de ódio baseado em **Transformers** (PyTorch + Hugging Face).

Este repositório implementa um pipeline completo de fine-tuning do **BERTimbau** (BERT pré-treinado em Português) para classificação binária de texto, incluindo treinamento com otimizações avançadas e um pipeline de inferência para classificar textos novos.

## Funcionalidades

- **Fine-tuning de Transformer** — BERTimbau com head de classificação customizado
- **Learning Rate Scheduler** — Warmup linear seguido de decay linear (padrão para fine-tuning de Transformers)
- **Weighted Loss** — CrossEntropyLoss com pesos de classe para combater o desbalanceamento do dataset
- **Gradient Clipping** — Limita a norma dos gradientes para estabilizar o treinamento
- **Mixed Precision (fp16)** — Acelera o treinamento em GPU via autocast e GradScaler
- **Early Stopping** — Interrompe o treinamento automaticamente quando a val_loss para de melhorar
- **Reprodutibilidade** — Seeds fixas (Python, NumPy, PyTorch) e cuDNN determinístico
- **Avaliação em Test Set** — Avaliação final em conjunto de teste separado (holdout)
- **Métricas Detalhadas** — Precision, Recall, F1-Score e Confusion Matrix (via scikit-learn)
- **Persistência de Métricas** — Histórico de treinamento e resultados salvos em `metrics.json`
- **Pipeline de Inferência** — Classificação de textos novos a partir de um checkpoint salvo
- **Modo Interativo** — Interface de terminal para demonstração ao vivo

## Pipeline

```mermaid
flowchart LR
    A["1. Ingestao\n(HuggingFace)"] --> B["2. Limpeza\n(clean_text)"]
    B --> C["3. Vetorizacao\n(Tokenizacao)"]
    C --> D["4. Empacotamento\n(DataLoader)"]
    D --> E["5. Encoder\n(BERTimbau)"]
    E --> F["6. Classificador\n(Camada Linear)"]
    F --> G["7. Calculo de Erro\n(Weighted Loss)"]
    G --> H["8. Backpropagation\n(Gradientes)"]
    H --> I["9. Otimizador\n(AdamW)"]
    I --> J["10. Scheduler\n(Warmup + Decay)"]
    J -->|"pesos atualizados"| E
```

## Estrutura do Projeto

```
Hate-Speech-Detector/
├── src/
│   ├── __init__.py        # Torna src/ um pacote Python
│   ├── dataset.py         # Módulo de Dados: ingestão, limpeza, tokenização, DataLoaders, class weights
│   ├── model.py           # Módulo de Arquitetura: BERTimbau com head de classificação
│   ├── engine.py          # Módulo de Treinamento: treino, validação, métricas e checkpoints
│   └── inference.py       # Pipeline de Inferência: carrega checkpoint e classifica textos novos
├── main.py                # Orquestrador: modos de treinamento e inferência via CLI
├── requirements.txt       # Dependências e versões
├── .gitignore             # Arquivos ignorados pelo Git
└── README.md              # Documentação (este arquivo)
```

## Pré-requisitos

- **Python** 3.10 ou superior
- **GPU** (opcional) — o pipeline detecta automaticamente CUDA e usa CPU como fallback

## Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/Davi-SB/Hate-Speech-Detector.git
cd Hate-Speech-Detector

# 2. Crie um ambiente virtual
python -m venv .venv

# 3. Ative o ambiente virtual
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 4. Instale as dependências
pip install -r requirements.txt
```

## Como Executar

### Modo Treinamento (padrão)

```bash
python main.py
```

O script irá:

1. Fixar as seeds para reprodutibilidade.
2. Detectar o device disponível (GPU com CUDA, se disponível, ou CPU como fallback).
3. Carregar o modelo BERTimbau pré-treinado e o tokenizador.
4. Criar os DataLoaders de treino, validação e teste com pré-processamento.
5. Calcular os pesos de classe para a Weighted Loss.
6. Configurar o otimizador AdamW, o Learning Rate Scheduler e o Mixed Precision (GPU).
7. Executar o loop de treinamento (até 10 epochs) com early stopping (patience=2).
8. Exibir métricas detalhadas (F1, Precision, Recall, Confusion Matrix) ao final de cada epoch.
9. Carregar o melhor modelo e avaliar no conjunto de teste separado.
10. Salvar o modelo treinado e as métricas completas em `checkpoints/`.

### Modo Inferência

Classificar um texto diretamente:

```bash
python main.py --infer --text "seu texto aqui"
```

Modo interativo (digite textos e receba classificações em tempo real):

```bash
python main.py --infer
```

Especificar um diretório de checkpoint diferente:

```bash
python main.py --infer --checkpoint caminho/para/checkpoint
```

### Hiperparâmetros

Os hiperparâmetros são definidos como constantes no topo de `main.py`:

| Parâmetro        | Valor padrão | Descrição                                              |
| ---------------- | ------------ | ------------------------------------------------------ |
| `BATCH_SIZE`     | 16           | Amostras por lote                                      |
| `LEARNING_RATE`  | 2e-5         | Taxa de aprendizado inicial do AdamW                   |
| `NUM_EPOCHS`     | 10           | Quantidade máxima de epochs (early stopping pode parar antes) |
| `NUM_LABELS`     | 2            | Classes de classificação                               |
| `WARMUP_RATIO`   | 0.1          | Fração dos steps totais usada para warmup do LR        |
| `CHECKPOINT_DIR` | checkpoints/ | Diretório para salvar modelo e métricas                |
| `SEED`           | 42           | Seed global para reprodutibilidade                     |
| `PATIENCE`       | 2            | Epochs sem melhora na val_loss antes do early stopping |
| `MAX_GRAD_NORM`  | 1.0          | Norma máxima para gradient clipping                    |

## Dependências

| Pacote           | Uso                                                       |
| ---------------- | --------------------------------------------------------- |
| `torch`          | Framework de deep learning (inclui AMP para mixed precision) |
| `transformers`   | Modelos pré-treinados, tokenizadores, schedulers          |
| `datasets`       | Carregamento de datasets do Hugging Face Hub              |
| `accelerate`     | Utilitários para treinamento distribuído                  |
| `scikit-learn`   | Métricas de avaliação (F1, Precision, Recall)             |
| `numpy`          | Operações numéricas e seeds para reprodutibilidade        |

## Equipe e Divisão de Tarefas

| Integrante | Responsabilidade                                         | Arquivo(s)                                               |
| ---------- | -------------------------------------------------------- | -------------------------------------------------------- |
| A (Davi)   | Infraestrutura, orquestração, inferência e documentação  | `main.py`, `src/inference.py`, `README.md`, `requirements.txt` |
| B          | Engenharia de dados: dataset, limpeza, class weights     | `src/dataset.py`                                         |
| C          | Arquitetura do modelo                                    | `src/model.py`                                           |
| D          | Engenharia do treinamento: weighted loss, LR scheduler   | `src/engine.py`                                          |
| E          | Validação: métricas detalhadas, relatórios, checkpoints  | `src/engine.py`                                          |
