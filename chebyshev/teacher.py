"""Baseline PyTorch LSTM teacher model architecture and training engine for NASA C-MAPSS."""

from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class CMapssLSTMTeacher(nn.Module):
    """Canonical 2-layer LSTM Teacher network for multi-sensor degradation modeling."""

    def __init__(
        self,
        num_channels: int = 14,
        hidden_dim: int = 64,
        num_layers: int = 2,
        out_features: int = 1,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_features = out_features

        self.lstm = nn.LSTM(
            input_size=num_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes sequential input (B, T, D_in) and outputs prediction at final step (B, D_out)."""
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train_teacher(
    teacher: nn.Module,
    dataloader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Trains the baseline LSTM teacher model on sensor sequences."""
    if device is None:
        device = torch.device("cpu")

    teacher = teacher.to(device)
    teacher.train()

    optimizer = torch.optim.AdamW(teacher.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    history: Dict[str, List[float]] = {"loss": []}

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred = teacher(batch_x)
            loss = criterion(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        history["loss"].append(avg_loss)

        if verbose:
            print(f"Teacher Epoch [{epoch+1:2d}/{epochs:2d}] Loss: {avg_loss:.5f}")

    teacher.eval()
    return history


def save_teacher(teacher: nn.Module, path: Union[str, Path]) -> None:
    """Saves teacher model checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": teacher.state_dict(),
            "num_channels": getattr(teacher, "num_channels", 14),
            "hidden_dim": getattr(teacher, "hidden_dim", 64),
            "num_layers": getattr(teacher, "num_layers", 2),
            "out_features": getattr(teacher, "out_features", 1),
        },
        str(path),
    )


def load_teacher(path: Union[str, Path], device: Optional[torch.device] = None) -> CMapssLSTMTeacher:
    """Loads teacher model from checkpoint."""
    if device is None:
        device = torch.device("cpu")
    checkpoint = torch.load(str(path), map_location=device)
    teacher = CMapssLSTMTeacher(
        num_channels=checkpoint.get("num_channels", 14),
        hidden_dim=checkpoint.get("hidden_dim", 64),
        num_layers=checkpoint.get("num_layers", 2),
        out_features=checkpoint.get("out_features", 1),
    )
    teacher.load_state_dict(checkpoint["state_dict"])
    teacher.eval()
    return teacher
