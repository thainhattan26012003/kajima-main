import torch.nn as nn
import torch
from .classifiers import (
    LinearClassifierHead,
    CosineClassifier,
    LayerNormedClassifier,
    L2NormedClassifier,
)


class ViTClassifier(nn.Module):
    def __init__(
        self,
        vit_model: nn.Module,
        num_classes: int,
        classifier_type: str = "linear",
        dtype=None,
    ):
        super().__init__()

        self.vit_model = vit_model

        # Model configuration

        emb_dim = vit_model.embeddings.position_embeddings.shape[2]

        # Classifier
        if classifier_type == "linear":
            self.head = LinearClassifierHead(emb_dim, num_classes, dtype)
        elif classifier_type == "cosine":
            self.head = CosineClassifier(emb_dim, num_classes, dtype)
        elif classifier_type == "layernorm":
            self.head = LayerNormedClassifier(emb_dim, num_classes, dtype)
        elif classifier_type == "l2norm":
            self.head = L2NormedClassifier(emb_dim, num_classes, dtype)
        else:
            raise NotImplementedError()

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, dim=0)

        inputs = dict(pixel_values=x)

        outputs = self.vit_model(**inputs)

        # sequence_outputs = outputs.last_hidden_state # batch_size x sequence_length x hidden_size

        # class_token = sequence_outputs[:, 0]
        # patch_tokens = sequence_outputs[:, 1:]

        # linear_input = torch.cat([class_token, patch_tokens.mean(1)], dim=1)
        linear_input = outputs.pooler_output

        logits = self.head(linear_input)

        return logits
