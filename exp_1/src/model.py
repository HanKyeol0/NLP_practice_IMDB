from transformers import AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
import omegaconf
from transformers import set_seed

set_seed(42)

class EncoderForClassification(nn.Module):
    def __init__(self, model_config : omegaconf.DictConfig):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_config.model_name)
        self.classification_head = torch.nn.Linear(self.model.config.hidden_size, model_config.num_labels)
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def forward(self, input_ids : torch.Tensor, attention_mask : torch.Tensor, token_type_ids : torch.Tensor, label : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs : 
            input_ids : (batch_size, max_seq_len)
            attention_mask : (batch_size, max_seq_len)
            token_type_ids : (batch_size, max_seq_len) # only for BERT
            label : (batch_size)
        Outputs :
            logits : (batch_size, num_labels)
            loss : (1)
        """
        representation = self.model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)
        pooled_output = torch.mean(representation['last_hidden_state'], dim=1)
        logits = self.classification_head(pooled_output)
        loss = self.loss_fn(logits, label)
        return {"logits": logits, "loss": loss}