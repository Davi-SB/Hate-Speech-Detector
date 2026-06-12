from __future__ import annotations

import torch
import torch.nn as nn
from transformers import (
    AutoModel, 
    AutoConfig, 
    AutoTokenizer, 
    PreTrainedModel, 
    PreTrainedTokenizerBase
)

class BERTimbauBinaryClassifier(nn.Module):
    def __init__(self, model_name: str = "neuralmind/bert-base-portuguese-cased", dropout_rate: float = 0.1, num_labels: int = 2):
        super(BERTimbauBinaryClassifier, self).__init__()
        
        # Carrega as configurações do BERTimbau
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Carrega o corpo do Transformer com os pesos pré-treinados em PT-BR
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        
        # Camada de Dropout para evitar overfitting 
        self.dropout = nn.Dropout(dropout_rate)
        
        # Camada Linear usando o num_labels dinâmico (Classe 0 e Classe 1)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Passa os tokens pelo BERTimbau
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Captura o atributo pooler_output, que é a representação do token [CLS]
        cls_representation = outputs.pooler_output
        
        # Aplica o dropout e passa os dados para a camada final de decisão
        pooled_output = self.dropout(cls_representation)
        logits = self.classifier(pooled_output) 
        
        return logits


def load_model(
    num_labels: int,
    device: torch.device,
) -> tuple[PreTrainedModel | nn.Module, PreTrainedTokenizerBase]:
    
    # Docstring explicativa para a função load_model
    """Baixa o Transformer pré-treinado e retorna (model, tokenizer).

    O modelo retornado já deve estar no *device* informado.

    Args:
        num_labels: Quantidade de classes de classificação.
        device: Dispositivo de execução (cpu / cuda).

    Returns:
        Tupla com (model, tokenizer).
    """
    model_name = "neuralmind/bert-base-portuguese-cased"
    
    # Instancia o Tokenizador
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Instancia a arquitetura customizada com o número de classes adequado
    model = BERTimbauBinaryClassifier(
        model_name=model_name, 
        num_labels=num_labels
    )
    
    # Move o modelo para o dispositivo correto (CPU ou GPU)
    model = model.to(device)
    
    return model, tokenizer

# Teste rápido
if __name__ == "__main__":
    # Fazemos a verificação do hardware 
    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Chama a função correta, passando os argumentos obrigatórios
    modelo, tokenizador = load_model(num_labels=2, device=my_device)
    
    # Mensagens de teste para confirmar que tudo está funcionando
    print("Arquitetura e tokenizador criados com sucesso!")
    print(f"Modelo alocado no device: {my_device}")
    print("Pronto para receber os dados do ToLD-Br.")
