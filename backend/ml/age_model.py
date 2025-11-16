# ml/age_model.py
# -*- coding: utf-8 -*-

import torch.nn as nn
from transformers import AutoModel


class AgeClassifier(nn.Module):
    def __init__(self, base_model_name: str = "DeepPavlov/rubert-base-cased", num_labels: int = 3):
        super().__init__()
        # базовая русская BERT-модель
        self.bert = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Добавили token_type_ids, потому что токенайзер его возвращает
        и BertModel умеет его принимать.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # pooler_output может быть None, тогда берём [CLS]-токен
        if getattr(outputs, "pooler_output", None) is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]

        x = self.dropout(pooled)
        logits = self.classifier(x)
        return logits
