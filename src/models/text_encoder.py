from typing import Dict

import torch
from transformers import AutoModel

class TextEncoder(torch.nn.Module):
    def __init__(self, name: str):
        super().__init__()
        
        self.model = AutoModel.from_pretrained(name)
        
        # Freeze the first 6 layers of BERT
        for i, p in enumerate(self.model.named_parameters()):
            if i==101:
                break
            p[1].requires_grad = False
  
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, texts_tokenized_data: Dict):
        bert_model_output = self.model(**texts_tokenized_data)
        
        ## Apply mean_pooling
        hu = self.mean_pooling(bert_model_output, texts_tokenized_data['attention_mask'])
        
        return hu
    
class MVLMTextEncoder(torch.nn.Module):
    def __init__(self, name: str, num_layers_to_freeze: int = 0):
        super().__init__()
        
        self.model = AutoModel.from_pretrained(name)
        # maskvlm didnt freeze, but we try to freeze some layers
        # Freeze the first num_layers_to_freeze layers
        encoder_layers = self.model.encoder.layer
        for layer in encoder_layers[:num_layers_to_freeze]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, texts_tokenized_data: Dict):
        bert_model_output = self.model(**texts_tokenized_data)
        return bert_model_output.last_hidden_state