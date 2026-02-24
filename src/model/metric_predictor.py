import torch
import numpy as np
from typing import Literal
import torch.nn as nn


class FeatureExtractor:
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
    ):
        self.model = model
        self.device = device

        self.model.to(device)
        self.model.eval()

    def extract_feature(
        self,
        images: torch.Tensor | np.ndarray,
        aggregation: Literal["max", "mean"] = "mean",
    ) -> torch.Tensor:
        """
        Extract image features
        """
        if isinstance(images, np.ndarray):
            images = torch.tensor(images).to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values=images)
            if aggregation == "mean":
                outputs = outputs.last_hidden_state.mean(1).to("cpu")
            elif aggregation == "max":
                outputs = outputs.last_hidden_state.max(1).values.to("cpu")
            else:
                raise ValueError(f"Aggregation method {aggregation} not supported")

        return outputs

    def get_device(self):
        return self.device


class MetricPredictor(nn.Module):
    def __init__(
        self,
        feature_matrix: dict[int, torch.Tensor],
        feature_extractor: FeatureExtractor,
        device: str = "cpu",
    ):
        super().__init__()
        self.feature_matrix = feature_matrix
        self.feature_extractor = feature_extractor
        self.device = device
        print(feature_matrix.keys())
        self.num_classes = len(set(feature_matrix.keys()))

    def forward(self, inputs: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Predict the image label based on feature similarity

        Args:
            inputs: Shape (Batch_size x C x H x W) or (Batch_size x C x H x W)

        Returns:
            torch.Tensor: Logits vector (Batch_size, num_classes)
        """
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs).to(self.device)

        feature_vectors = self.feature_extractor.extract_feature(inputs)
        logits = torch.zeros(feature_vectors.shape[0], self.num_classes).to(self.device)

        label_maps = {
            label: idx for idx, label in enumerate(self.feature_matrix.keys())
        }
        for label, features in self.feature_matrix.items():
            if isinstance(features, list):
                features = torch.vstack(features)

            distances = torch.cdist(feature_vectors, features, p=1.0).min(dim=1).values

            class_idx = label_maps[label] if isinstance(label, str) else int(label)
            logits[:, class_idx] = -distances

        return logits
