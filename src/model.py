import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class BERTimbauBinaryClassifier(nn.Module):
    def __init__(self, model_name: str = "neuralmind/bert-base-portuguese-cased", dropout_rate: float = 0.1):
        super(BERTimbauBinaryClassifier, self).__init__()
        
        # Carrega as configurações do BERTimbau
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Carrega o corpo do Transformer com os pesos pré-treinados em PT-BR
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        
        # Camada de Dropout para evitar overfitting 
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classe 0: Texto Limpo / Neutro
        # Classe 1: Discurso de Ódio / Tóxico
        self.classifier = nn.Linear(self.config.hidden_size, 2)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Passa os tokens pelo BERTimbau
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # captura o atributo pooler_output, que é a representação do token [CLS]
        cls_representation = outputs.pooler_output
        
        # Aplica o dropout e passa os dados para a camada final de decisão
        pooled_output = self.dropout(cls_representation)
        logits = self.classifier(pooled_output) 
        
        return logits

def initialize_model(dropout_rate: float = 0.1):
    print("Inicializando BERTimbau para Classificação Binária...")
    model = BERTimbauBinaryClassifier(dropout_rate=dropout_rate)
    return model

# Teste rápido
if __name__ == "__main__":
    modelo = initialize_model()
    print("Arquitetura criada e pronta para receber os dados do ToLD-Br")
