import torch
from torch import nn
from transformers import AutoModel

from .config import CATEGORIES, SEVERITY_LEVELS


class MultiHeadClassifier(nn.Module):
    """
    BERT-энкодер + одна линейка, которая даёт [batch, CATEGORIES, SEVERITY_LEVELS].
    """

    def __init__(self, base_model_name: str, num_categories: int, num_severity: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.bert.config.hidden_size

        self.num_categories = num_categories
        self.num_severity = num_severity

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_categories * num_severity)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # берём CLS (первый токен)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)  # [B, C*S]
        logits = logits.view(-1, self.num_categories, self.num_severity)

        loss = None
        if labels is not None:
            # labels: [B, C]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.num_severity),
                labels.view(-1),
            )

        return {"loss": loss, "logits": logits}
