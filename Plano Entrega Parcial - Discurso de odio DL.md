# Entrega Parcial


### O Que Será Feito na Entrega Parcial

Construir e validar a infraestrutura de aprendizado de máquina. Não se preocupar em
resolver o problema de discurso de ódio ainda, nem em limpar dados. A meta é criar o
básico usando PyTorch, ou seja, garantir que um texto consiga entrar no sistema, ser
transformado em números, passar pelas camadas do Transformer, gerar um erro e fazer
esse erro voltar para atualizar a rede. É uma prova de conceito do projeto.

### O Pipeline da Entrega Parcial

1. **Ingestão via Nuvem (Cloud Ingestion):** O sistema se conectará diretamente à
    biblioteca do Hugging Face para carregar o conjunto de dados de brinquedo. Os
    dados serão divididos logicamente em train_dataset (para o modelo aprender) e
    eval_dataset (para testar o aprendizado).
2. **Vetorização (Tokenization):** O texto cru passará por um tokenizador que converterá
    as palavras da frase em dois componentes essenciais para Transformers:
       ○ input_ids: Os números de identificação de cada fragmento de palavra.
       ○ attention_mask: Uma matriz indicando ao modelo o que é texto real e o que é
          apenas preenchimento de espaço.
3. **Empacotamento (Batching & Loading):** Os dados vetorizados serão agrupados
    em pequenos pacotes gerenciados por um data_loader, que alimentará a placa de
    vídeo ou processador de forma contínua e eficiente.
4. **Propagação (Forward Pass):** O data_loader injeta os lotes no modelo Transformer.
    A rede neural processa as sequências e retorna as previsões brutas no final, que
    chamamos de logits.
5. **Cálculo de Erro (Loss Calculation):** As previsões são comparadas
    matematicamente com as respostas corretas. O resultado dessa comparação é um
    valor de erro (loss), que quantifica a distância entre a previsão e o rótulo real.
6. **Aprendizado (Backward Pass & Optimization):** O valor do erro sofre uma
    retropropagação, calculando os gradientes através de todas as camadas do modelo.
    Em seguida, um optimizer ajusta os pesos internos da rede baseado na
    learning_rate, fazendo com que o modelo erre menos no próximo lote.

### Árvore de Arquivos e Estrutura

📁 project_root/
├── 📁 src/
│ ├── 📄 dataset.py (Módulo de Dados: conecta com Hugging Face, aplica tokenização e
gera os lotes)
│ ├── 📄 model.py (Módulo de Arquitetura: instancia o Transformer pré-treinado e
configurar hardware)
│ └── 📄 engine.py (Módulo de Treinamento: contém a lógica matemática de treino,
validação e atualização de pesos)
├── 📄 main.py (O Orquestrador: importa os módulos acima e inicia o fluxo do programa)


├── 📄 requirements.txt (Gerenciador de dependências e versões das bibliotecas)
└── 📄 README.md (Documentação do projeto e instruções de execução)


# Divisão de tarefas


### Divisão de carga de trabalho (5 integrantes)

Para que o trabalho seja paralelo, a divisão das tarefas e das variáveis que cada um vai
manipular:
● **Integrante A (Infraestrutura e Orquestração):** Responsável pela criação da árvore
de arquivos no repositório, documentação no README.md, gestão do
requirements.txt e pela construção do main.py. Este integrante chamará as funções
criadas pelos colegas e garantirá que o fluxo principal rode sem quebrar.
● **Integrante B (Engenharia de Dados via Nuvem):** Responsável por escrever o
arquivo dataset.py. O dever aqui é utilizar a documentação da biblioteca do Hugging
Face, carregar os dados em memória, mapear o texto para gerar os input_ids e
entregar os objetos train_loader e val_loader prontos para consumo.
● **Integrante C (Arquitetura do Modelo):** Responsável pelo arquivo model.py. Deve
pesquisar como baixar os pesos estruturais do Transformer, definir os
hiperparâmetros básicos da rede neural e retornar o objeto principal do modelo
pronto para receber os dados.
● **Integrante D (Engenharia do Treinamento):** Responsável por escrever a função
iterativa de treino dentro do engine.py. O dever é passar pelos lotes, garantir que a
loss_function seja calculada corretamente, aplicar a retropropagação matemática e
realizar o passo de otimização dos pesos através do optimizer.
● **Integrante E (Validação e Checkpoints):** Responsável por escrever a função de
avaliação no engine.py. Este integrante desativará a atualização de pesos, passará
os dados de validação pelo modelo, calculará a acurácia provisória desta etapa e
criará a lógica para salvar o modelo treinado (save_checkpoint) no disco após a
última iteração.
**Integrante Nome**
Integrante A Davi
Integrante B _a ser decidido/sorteado_
Integrante C _a ser decidido/sorteado_
Integrante D _a ser decidido/sorteado_
Integrante E _a ser decidido/sorteado_


# Fazer na Entrega Final


### Deixar no radar para a entrega final

```
● A Sujeira dos Dados Reais (Pré-processamento e limpeza de texto da internet)
● O Desbalanceamento de Classes (A Função de Custo adaptada)
● Métricas de Avaliação Rigorosas (F1-Score, Precision, Recall e Confusion Matrix)
● Otimização Fina (Learning Rate Schedulers)
● Pipeline de Inferência (Testar mensagens cruas enviadas por usuários reais)
```

