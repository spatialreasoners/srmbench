from typing import Dict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Integer, Bool
from torch import Tensor
from torchvision.models import resnet50
from huggingface_hub import PyTorchModelHubMixin

from srmbench.datasets.counting_objects.labelers import (
    DatasetCountingObjectsAmbiguousLabeler,
)


class MultiHeadLayer(nn.Module):
    def __init__(self, in_features: int, num_classes: Dict[str, int]):
        super().__init__()

        self.heads = nn.ModuleDict(
            {name: nn.Linear(in_features, num_classes[name]) for name in num_classes}
        )

    def forward(self, x):
        return {name: head(x) for name, head in self.heads.items()}
    
    
class CountingObjectsClassifier(nn.Module, PyTorchModelHubMixin):
    def __init__(self, min_vertices: int, max_vertices: int, min_num_polygons: int, max_num_polygons: int, device: str | torch.device = "cuda"):
        super().__init__()
        self.device = device
        self.min_vertices = min_vertices
        self.max_vertices = max_vertices
        self.min_num_polygons = min_num_polygons
        self.max_num_polygons = max_num_polygons
        
        self.num_classes = {
            "num_polygons": self.max_num_polygons - self.min_num_polygons + 1,
            "num_vertices": self.max_vertices - self.min_vertices + 1,
            "is_uniform": 2,
        }
        self.labeler = DatasetCountingObjectsAmbiguousLabeler(
            min_vertices=self.min_vertices, max_vertices=self.max_vertices
        )
        self.num_classes["numbers_label"] = self.labeler.num_classes
        
        self.model = resnet50(weights=None)
        self.model.fc = MultiHeadLayer(self.model.fc.in_features, self.num_classes)

        
        
    def predict(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        with torch.no_grad():
            outputs = self.model(images)

        selected_labels = {}
        confidences = {}

        for key in outputs:
            softmaxed = F.softmax(outputs[key], dim=1)
            selected_labels[key] = softmaxed.argmax(dim=1)
            confidences[key] = softmaxed.max(dim=1).values

        outputs = {
            "is_uniform": selected_labels["is_uniform"].bool(),  # 1 is True, 0 is False
            "num_polygons": selected_labels["num_polygons"] + self.min_num_polygons,
            "num_vertices": selected_labels["num_vertices"] + self.min_vertices,
        }

        outputs["num_polygons_vertices"] = (
            self.labeler.label_to_num_polygons_vertices(
                selected_labels["numbers_label"]
            )
        )

        return outputs, confidences
    
    def load_from_file(self, model_path: str) -> nn.Module:
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True),
            strict=True,
        )
        self.model.to(self.device)
        self.model.eval()
    

class CountingObjectsEvaluation:
    repo_ids = {
        "polygons": "BartekPog/counting-polygons-classifier",
        "stars": "BartekPog/counting-stars-classifier",
    }
    
    def __init__(self, object_variant: Literal["polygons", "stars"], device: str | torch.device = "cuda"):
        super().__init__()
        
        self.classifier = CountingObjectsClassifier.from_pretrained(self.repo_ids[object_variant])
        self.classifier.to(device)
        self.classifier.eval()
        
    def _predict_outputs(self, images: Float[Tensor, "batch 3 height width"]) -> tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        return self.classifier.predict(images)
        
    @staticmethod
    def _are_ambiguous_numbers_consistent(
        numbers_label: Integer[Tensor, "batch 2"],
        num_polygons: Integer[Tensor, "batch"],
        num_vertices: Integer[Tensor, "batch"],
    ) -> Bool[Tensor, "batch"]:
        return torch.logical_or(
            torch.logical_and(
                numbers_label[:, 0] == num_polygons, numbers_label[:, 1] == num_vertices
            ),
            torch.logical_and(
                numbers_label[:, 0] == num_vertices, numbers_label[:, 1] == num_polygons
            ),
        )

    def _get_vertices_counts(
        self, num_vertices: Integer[Tensor, "batch"], counts_range: tuple[int, int]
    ) -> dict[str, float]:
        return {
            f"relative_vertex_count_{i}": (
                (num_vertices == i).sum() / num_vertices.shape[0]
            ).item()
            for i in range(*counts_range)
        }

    def _get_polygons_counts(
        self, num_polygons: Integer[Tensor, "batch"], counts_range: tuple[int, int]
    ) -> dict[str, float]:
        return {
            f"relative_polygons_count_{i}": (
                (num_polygons == i).sum() / num_polygons.shape[0]
            ).item()
            for i in range(*counts_range)
        }

    def _numbers_match_objects(
        self,
        class_label: Integer[Tensor, "batch"],
        num_polygons: Integer[Tensor, "batch"],
        num_vertices: Integer[Tensor, "batch"],
    ) -> Bool[Tensor, "batch"]:
        assert self.dataset.labeler is not None, "Labeler must be provided"

        generated_label = self.dataset.labeler.get_batch_labels(
            num_polygons, num_vertices
        )
        return class_label == generated_label

    @torch.no_grad()
    def evaluate(
        self,
        images: Float[Tensor, "batch 3 height width"],
        include_counts: bool = False, # Counts should be roughly equal for numbers of objects and vertices
        include_confidences: bool = False, # How easy it is to predict the correct number of objects and vertices
    ) -> dict[str, Float[Tensor, ""]] | None:
        outputs, confidences = self._predict_outputs(images)

        metrics = {"are_vertices_uniform": outputs["is_uniform"],}

        metrics["numbers_match_objects"] = self._are_ambiguous_numbers_consistent(
            outputs["num_polygons_vertices"],
            outputs["num_polygons"],
            outputs["num_vertices"],
        )
        
        if include_confidences:
            metrics.update({f"{key}_confidence": value for key, value in confidences.items()})
        
        metrics = {k: v.float().mean() for k, v in metrics.items()}
        
        if include_counts:
            vertex_value_range = (self.classifier.min_vertices, self.classifier.max_vertices + 1)
            polygon_value_range = (
                self.classifier.min_num_polygons,
                self.classifier.max_num_polygons + 1,
            )

            vertex_counts = self._get_vertices_counts(
                outputs["num_vertices"], vertex_value_range
            )
            polygon_counts = self._get_polygons_counts(
                outputs["num_polygons"], polygon_value_range
            )
            metrics.update(vertex_counts)
            metrics.update(polygon_counts)

        return metrics