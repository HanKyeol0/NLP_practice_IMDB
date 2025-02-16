from transformers import AutoModel, AutoConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional
import omegaconf
from transformers import set_seed

set_seed(42)

class EncoderForClassification(nn.Module):
    def __init__(self, model_config : omegaconf.DictConfig):
        super().__init__()
        if model_config.model_name == 'bert-base-uncased':
            self.model = AutoModel.from_pretrained(model_config.model_name, attn_implementation='eager')
        elif model_config.model_name == 'answerdotai/ModernBERT-base':
            # Load ModernBERT without FlashAttention
            config = AutoConfig.from_pretrained(model_config.model_name)
            config._attn_implementation = "eager"  # Disable FlashAttention
            self.model = AutoModel.from_pretrained(model_config.model_name, config=config)
        self.classification_head = torch.nn.Linear(self.model.config.hidden_size, model_config.num_labels)
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def forward(self, input_ids : torch.Tensor, attention_mask : torch.Tensor, token_type_ids : Optional[torch.Tensor] = None, label : torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if token_type_ids is not None:
            representation = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            representation = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        pooled_output = torch.mean(representation['last_hidden_state'], dim=1)
        logits = self.classification_head(pooled_output)
        loss = self.loss_fn(logits, label)
        return {"logits": logits, "loss": loss}