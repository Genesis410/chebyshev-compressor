"""Non-invasive trajectory matching distillation engine with PyTorch forward hooks."""

from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from chebyshev.models import StudentModel


class TeacherStateHook:
    """Non-invasive PyTorch forward hook for extracting teacher recurrent hidden trajectories.
    
    Automatically handles tuple outputs from nn.LSTM/nn.GRU, normalizes batch dimensions,
    and falls back gracefully to no-op if the teacher model contains no recurrent layers.
    """

    def __init__(self, teacher: nn.Module, layer_name: Optional[str] = None) -> None:
        self.teacher = teacher
        self.trajectory: Optional[torch.Tensor] = None
        self._hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.has_recurrent_layer: bool = False

        target_module = None
        if layer_name is not None:
            modules_dict = dict(teacher.named_modules())
            if layer_name in modules_dict:
                target_module = modules_dict[layer_name]
            else:
                raise ValueError(f"Specified teacher hook layer '{layer_name}' not found in model.")
        else:
            # Auto-detect recurrent layer (nn.LSTM, nn.GRU, nn.RNN)
            for name, module in teacher.named_modules():
                if isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
                    target_module = module
                    break

        if target_module is not None:
            self.has_recurrent_layer = True
            self._hook_handle = target_module.register_forward_hook(self._hook_fn)

    def _hook_fn(
        self, module: nn.Module, input: Tuple[torch.Tensor, ...], output: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ) -> None:
        """Captures and standardizes hidden trajectory to shape (batch, seq_len, hidden_dim)."""
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output

        # Ensure batch-first shape: (batch_size, seq_len, hidden_dim)
        if hasattr(module, "batch_first") and not module.batch_first:
            out = out.transpose(0, 1)

        self.trajectory = out

    def get_trajectory(self) -> Optional[torch.Tensor]:
        """Returns the most recent captured trajectory tensor."""
        return self.trajectory

    def remove(self) -> None:
        """Detaches the forward hook from the teacher model."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def __enter__(self) -> "TeacherStateHook":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove()


def cosine_annealing_lambda(
    epoch: int,
    total_epochs: int,
    lambda_start: float = 1.0,
    lambda_end: float = 0.1,
) -> float:
    """Calculates cosine-annealed trajectory alignment loss weight."""
    if total_epochs <= 1:
        return lambda_end
    ratio = min(1.0, max(0.0, float(epoch) / float(total_epochs)))
    return float(lambda_end + 0.5 * (lambda_start - lambda_end) * (1.0 + math.cos(math.pi * ratio)))


def distill(
    teacher: nn.Module,
    student: StudentModel,
    dataloader: DataLoader,
    channel_mean: Optional[np.ndarray] = None,
    channel_std: Optional[np.ndarray] = None,
    epochs: int = 15,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    lambda_start: float = 1.0,
    lambda_end: float = 0.1,
    teacher_hook_layer: Optional[str] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Executes trajectory and output matching distillation from Teacher to Chebyshev Student.
    
    Args:
        teacher: Pre-trained teacher model.
        student: Target Chebyshev StudentModel.
        dataloader: Training dataloader yielding (batch_x, batch_y).
        channel_mean: Optional sensor mean array for baked normalization.
        channel_std: Optional sensor std array for baked normalization.
        epochs: Number of distillation training epochs.
        lr: Learning rate for AdamW optimizer.
        weight_decay: Weight decay coefficient.
        lambda_start: Initial weight for state trajectory loss.
        lambda_end: Final weight for state trajectory loss.
        teacher_hook_layer: Optional specific layer name to hook for hidden states.
        device: Device to train on (default CPU).
        verbose: Whether to print progress during training.
        
    Returns:
        Dictionary containing epoch loss history.
    """
    if device is None:
        device = torch.device("cpu")

    teacher = teacher.to(device)
    student = student.to(device)
    teacher.eval()
    student.train()

    # Bake dataset normalization into student cell if provided
    if channel_mean is not None and channel_std is not None:
        student.cell.set_normalization(channel_mean, channel_std)

    # Attach non-invasive teacher hook
    hook = TeacherStateHook(teacher, layer_name=teacher_hook_layer)

    # Initialize linear trajectory alignment adapter if teacher is recurrent
    adapter: Optional[nn.Linear] = None
    if hook.has_recurrent_layer:
        # Determine teacher hidden dimension by passing a single batch sample
        sample_x, _ = next(iter(dataloader))
        with torch.no_grad():
            _ = teacher(sample_x[:1].to(device))
        captured = hook.get_trajectory()
        if captured is not None:
            teacher_hidden_dim = captured.shape[-1]
            student_state_dim = student.num_channels * student.order
            adapter = nn.Linear(student_state_dim, teacher_hidden_dim, bias=False).to(device)

    # Optimization parameters: Readout MLP + temporary trajectory adapter
    trainable_params = list(student.readout.parameters())
    if adapter is not None:
        trainable_params += list(adapter.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    criterion_task = nn.MSELoss()
    criterion_state = nn.MSELoss()

    history: Dict[str, List[float]] = {
        "total_loss": [],
        "task_loss": [],
        "state_loss": [],
        "lambda": [],
    }

    try:
        for epoch in range(epochs):
            epoch_total_loss = 0.0
            epoch_task_loss = 0.0
            epoch_state_loss = 0.0
            num_batches = 0

            current_lambda = (
                cosine_annealing_lambda(epoch, epochs, lambda_start, lambda_end)
                if hook.has_recurrent_layer
                else 0.0
            )

            for batch_x, _ in dataloader:
                batch_x = batch_x.to(device)

                # Teacher inference with hook
                with torch.no_grad():
                    teacher_out = teacher(batch_x)
                    teacher_trajectory = hook.get_trajectory() if hook.has_recurrent_layer else None

                # Student forward pass
                student_out, student_trajectory = student(
                    batch_x, return_sequence=False, return_trajectory=True
                )

                # Task output matching loss
                loss_task = criterion_task(student_out, teacher_out)

                # State trajectory alignment loss
                if adapter is not None and teacher_trajectory is not None and current_lambda > 0.0:
                    projected_student_state = adapter(student_trajectory)
                    loss_state = criterion_state(projected_student_state, teacher_trajectory)
                    total_loss = loss_task + current_lambda * loss_state
                else:
                    loss_state = torch.tensor(0.0, device=device)
                    total_loss = loss_task

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_total_loss += total_loss.item()
                epoch_task_loss += loss_task.item()
                epoch_state_loss += loss_state.item()
                num_batches += 1

            avg_total = epoch_total_loss / max(1, num_batches)
            avg_task = epoch_task_loss / max(1, num_batches)
            avg_state = epoch_state_loss / max(1, num_batches)

            history["total_loss"].append(avg_total)
            history["task_loss"].append(avg_task)
            history["state_loss"].append(avg_state)
            history["lambda"].append(current_lambda)

            if verbose:
                print(
                    f"Epoch [{epoch+1:2d}/{epochs:2d}] "
                    f"Total: {avg_total:.5f} | Task: {avg_task:.5f} | State: {avg_state:.5f} | λ: {current_lambda:.3f}"
                )

    finally:
        hook.remove()

    student.eval()
    return history
