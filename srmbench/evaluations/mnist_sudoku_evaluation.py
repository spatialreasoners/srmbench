import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from jaxtyping import Float, Integer, Shaped
from torch import Tensor


class MNISTClassifier(nn.Module, PyTorchModelHubMixin):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # anything you want saved to config.json:
        self.config = {"input_shape": [1, 28, 28], "num_classes": 10}

    def forward(
        self, x: Float[Tensor, "batch 1 height width"]
    ) -> Float[Tensor, "batch 10"]:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MnistSudokuEvaluation:
    classifier_repo = "BartekPog/mnist-sudoku-classifier"
    grid_size = (9, 9)

    def __init__(
        self,
    ) -> None:
        self.classifier = MNISTClassifier.from_pretrained(self.classifier_repo)

    @torch.no_grad()
    def _discretize(
        self, samples: Float[Tensor, "batch 1 height width"]
    ) -> Integer[Tensor, "batch grid_height grid_width"]:
        batch_size = samples.shape[0]
        tile_shape = tuple(s // g for s, g in zip(samples.shape[-2:], self.grid_size))
        tiles = (
            samples.unfold(2, tile_shape[0], tile_shape[0])
            .unfold(3, tile_shape[1], tile_shape[1])
            .reshape(-1, 1, *tile_shape)
        )

        logits: Float[Tensor, "batch 10"] = self.classifier(tiles)
        idx = torch.topk(logits, k=2, dim=1).indices
        pred = idx[:, 0]
        # Replace zero predictions with second most probable number
        zero_mask = pred == 0
        pred[zero_mask] = idx[zero_mask, 1]
        pred = pred.reshape(batch_size, *self.grid_size)
        return pred

    def _get_sudoku_metrics(
        self, pred: Integer[Tensor, "batch grid_size grid_size"]
    ) -> dict[str, Shaped[Tensor, "batch"]]:
        batch_size, grid_size = pred.shape[:2]
        sub_grid_size = round(grid_size**0.5)
        dtype, device = pred.dtype, pred.device
        pred = pred - 1  # Shift [1, 9] to [0, 8] for indices
        ones = torch.ones((1,), dtype=dtype, device=device).expand_as(pred)
        distance = torch.zeros((batch_size,), dtype=dtype, device=device)
        for dim in range(1, 3):
            cnt = torch.full_like(pred, fill_value=-1)
            cnt.scatter_add_(dim=dim, index=pred, src=ones)
            distance.add_(cnt.abs_().sum(dim=(1, 2)))

        # Subgrids
        grids = (
            pred.unfold(1, sub_grid_size, sub_grid_size)
            .unfold(2, sub_grid_size, sub_grid_size)
            .reshape(-1, grid_size, grid_size)
        )
        cnt = torch.full_like(grids, fill_value=-1)
        cnt.scatter_add_(dim=dim, index=grids, src=ones)
        distance.add_(cnt.abs_().sum(dim=(1, 2)))
        is_valid_sudoku = distance == 0

        return {"duplicate_count": distance, "is_valid_sudoku": is_valid_sudoku}

    def evaluate(
        self, samples: Float[Tensor, "batch height width"]
    ) -> dict[str, Shaped[Tensor, "batch"]]:
        # Add channel dimension if not present
        if samples.dim() == 3:
            samples = samples.unsqueeze(1)  # [batch, height, width] -> [batch, 1, height, width]
        pred = self._discretize(samples)
        return self._get_sudoku_metrics(pred)
