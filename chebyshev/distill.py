"""Option A Distillation Engine: Fixed Analytical Orthogonal Memory with Fast State Precomputation.

Architecture & Methodology (Option A):
    Because the discretized Chebyshev-Gauss-Radau operators (A_bar, B_bar) in
    `StudentModel.cell` are strictly analytical and frozen (non-trainable), the
    Chebyshev state representation S for any calibration/training dataset is
    completely deterministic.

    Therefore, `distill()` executes a single `torch.no_grad()` precomputation pass
    to compute:
        1. Frozen Chebyshev state vectors S in R^{N x (D_in * K)}
        2. Teacher output predictions Y_teacher in R^{N x D_out}
        3. Optional ground-truth labels Y_true in R^{N x D_out}

    Only the Readout MLP (`student.readout`) is optimized over mini-batches of S,
    accelerating CPU distillation >10x while guaranteeing zero pole drift and 100%
    parameter persistence into the exported C99 engine.

    Multi-Objective Loss:
        - When ground-truth labels Y_true are provided:
            L = (1 - lambda) * MSE(y_student, y_true) + lambda * MSE(y_student, y_teacher)
          with default lambda = 0.5.
        - When running on unlabeled calibration datasets (Y_true absent):
            Automatic fallback to lambda = 1.0 (pure teacher output mimicry).
"""

from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from chebyshev.models import StudentModel


class TeacherStateHook:
    """Optional non-invasive PyTorch forward hook for diagnostic inspection of teacher states.

    Note: Not used for training discarded alignment adapters under Option A distillation.
    Retained strictly as an analysis/diagnostic utility.
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
            for _, module in teacher.named_modules():
                if isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
                    target_module = module
                    break

        if target_module is not None:
            self.has_recurrent_layer = True
            self._hook_handle = target_module.register_forward_hook(self._hook_fn)

    def _hook_fn(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    ) -> None:
        """Captures and standardizes hidden trajectory to shape (batch, seq_len, hidden_dim)."""
        out = output[0] if isinstance(output, tuple) else output
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


def _unpack_batch(batch: Union[torch.Tensor, Tuple, List]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Extracts (batch_x, batch_y) from labeled or unlabeled DataLoader batches."""
    if isinstance(batch, torch.Tensor):
        return batch, None
    if isinstance(batch, (tuple, list)):
        if len(batch) == 1:
            return batch[0], None
        if len(batch) >= 2:
            return batch[0], batch[1]
    raise ValueError(f"Unsupported batch structure from dataloader: {type(batch)}")


@torch.no_grad()
def precompute_chebyshev_states(
    student: StudentModel,
    dataloader: DataLoader,
    teacher: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Precomputes frozen final Chebyshev states S and optional teacher/ground-truth targets.

    Runs a single forward pass through the fixed analytical Chebyshev cell (and optional
    teacher network) without gradient tracking.

    Args:
        student: StudentModel with baked normalization already configured on `student.cell`.
        dataloader: DataLoader yielding `(x, y)` or unlabeled `(x,)` / `x`.
        teacher: Optional pre-trained Teacher model for generating soft distillation targets.
        device: Computation device (default CPU).

    Returns:
        Tuple of `(states, teacher_outputs, ground_truth_targets)`:
            states: Tensor of shape `(N, num_channels * order)` on `device`.
            teacher_outputs: Tensor of shape `(N, out_features)` if `teacher` is provided, else None.
            ground_truth_targets: Tensor of shape `(N, out_features)` if labels exist, else None.
    """
    if device is None:
        device = torch.device("cpu")

    student.eval()
    if teacher is not None:
        teacher.eval()

    all_states: List[torch.Tensor] = []
    all_teacher_out: List[torch.Tensor] = []
    all_y_true: List[torch.Tensor] = []
    has_labels: Optional[bool] = None

    for batch in dataloader:
        batch_x, batch_y = _unpack_batch(batch)
        batch_x = batch_x.to(device)

        batch_size, seq_len, _ = batch_x.shape
        state = student.cell.init_state(batch_size, device=device, dtype=batch_x.dtype)
        for t in range(seq_len):
            state = student.cell(batch_x[:, t, :], state)
        flat_state = state.reshape(batch_size, -1)
        all_states.append(flat_state)

        if teacher is not None:
            all_teacher_out.append(teacher(batch_x))

        batch_has_y = batch_y is not None
        if has_labels is None:
            has_labels = batch_has_y
        elif has_labels != batch_has_y:
            raise ValueError("Inconsistent labeled/unlabeled batches within the same DataLoader.")

        if batch_y is not None:
            all_y_true.append(batch_y.to(device))

    states_tensor = torch.cat(all_states, dim=0)
    teacher_tensor = torch.cat(all_teacher_out, dim=0) if all_teacher_out else None
    y_true_tensor = torch.cat(all_y_true, dim=0) if all_y_true else None

    return states_tensor, teacher_tensor, y_true_tensor


@torch.no_grad()
def evaluate_rmse(
    model: nn.Module,
    dataloader: DataLoader,
    scale: float = 1.0,
    device: Optional[torch.device] = None,
) -> float:
    """Evaluates ground-truth Root Mean Squared Error (RMSE) on a labeled DataLoader.

    Args:
        model: PyTorch model (Teacher or StudentModel).
        dataloader: Labeled evaluation DataLoader yielding `(batch_x, batch_y)`.
        scale: Multiplier applied to predictions and targets (e.g., 125.0 for RUL cycles).
        device: Device for evaluation (default CPU).

    Returns:
        Scalar RMSE value in scaled target units.
    """
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    sq_err_sum = 0.0
    total_elements = 0

    for batch in dataloader:
        batch_x, batch_y = _unpack_batch(batch)
        if batch_y is None:
            raise ValueError("evaluate_rmse requires a labeled DataLoader with ground-truth targets.")
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        preds = model(batch_x)
        diff = (preds - batch_y) * float(scale)
        sq_err_sum += float(torch.sum(diff * diff).item())
        total_elements += int(diff.numel())

    if total_elements == 0:
        return 0.0
    return math.sqrt(sq_err_sum / total_elements)


def distill(
    teacher: nn.Module,
    student: StudentModel,
    dataloader: DataLoader,
    channel_mean: Optional[np.ndarray] = None,
    channel_std: Optional[np.ndarray] = None,
    epochs: int = 15,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    distill_lambda: float = 0.5,
    batch_size: Optional[int] = None,
    val_dataloader: Optional[DataLoader] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Executes Option A Readout Distillation with Fast State Precomputation.

    Precomputes frozen Chebyshev states S once before training, then trains strictly
    `student.readout` using a convex combination of ground-truth MSE and teacher
    distillation MSE. Automatically falls back to `lambda = 1.0` (pure teacher mimicry)
    when ground-truth labels are absent.

    Args:
        teacher: Pre-trained Teacher model.
        student: Target Chebyshev StudentModel.
        dataloader: Training/calibration DataLoader yielding `(batch_x, batch_y)` or unlabeled `(batch_x,)`.
        channel_mean: Optional channel mean vector for baked normalization.
        channel_std: Optional channel std vector for baked normalization.
        epochs: Number of Readout MLP training epochs.
        lr: Learning rate for AdamW optimizer.
        weight_decay: Weight decay coefficient.
        distill_lambda: Weight lambda in [0, 1] for teacher mimicry loss when labels exist (default 0.5).
        batch_size: Mini-batch size over precomputed states (defaults to dataloader.batch_size or 32).
        val_dataloader: Optional held-out labeled validation DataLoader for tracking ground-truth RMSE.
        device: Computation device (default CPU).
        verbose: Whether to print epoch metrics.

    Returns:
        Dictionary with epoch histories:
            `total_loss`, `gt_loss`, `kd_loss`, `lambda`, and (if val_dataloader given) `val_rmse`.
    """
    if not (0.0 <= distill_lambda <= 1.0):
        raise ValueError(f"distill_lambda must be in [0.0, 1.0], got {distill_lambda}")

    if device is None:
        device = torch.device("cpu")

    teacher = teacher.to(device)
    student = student.to(device)
    teacher.eval()

    # Bake dataset normalization into frozen student cell before state precomputation
    if channel_mean is not None and channel_std is not None:
        student.cell.set_normalization(channel_mean, channel_std)

    # 1. Fast State Precomputation: single pass over frozen analytical Chebyshev cell
    S_train, Y_teacher, Y_true = precompute_chebyshev_states(
        student=student,
        dataloader=dataloader,
        teacher=teacher,
        device=device,
    )
    assert Y_teacher is not None

    # Automatic fallback: lambda = 1.0 when distilling on unlabeled calibration data
    has_ground_truth = Y_true is not None
    effective_lambda = float(distill_lambda) if has_ground_truth else 1.0

    # Optional precomputation for held-out validation split
    S_val: Optional[torch.Tensor] = None
    Y_val: Optional[torch.Tensor] = None
    if val_dataloader is not None:
        S_val, _, Y_val = precompute_chebyshev_states(
            student=student,
            dataloader=val_dataloader,
            teacher=None,
            device=device,
        )
        if Y_val is None:
            raise ValueError("val_dataloader must provide ground-truth targets for validation RMSE.")

    eff_batch_size = batch_size or getattr(dataloader, "batch_size", None) or 32
    if has_ground_truth:
        precomputed_ds = TensorDataset(S_train, Y_teacher, Y_true)
    else:
        precomputed_ds = TensorDataset(S_train, Y_teacher)
    precomputed_loader = DataLoader(precomputed_ds, batch_size=eff_batch_size, shuffle=True)

    # 2. Optimize strictly Readout MLP parameters (100% exported to C99; A_bar, B_bar remain frozen)
    optimizer = torch.optim.AdamW(
        student.readout.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()

    history: Dict[str, List[float]] = {
        "total_loss": [],
        "gt_loss": [],
        "kd_loss": [],
        "lambda": [],
    }
    if val_dataloader is not None:
        history["val_rmse"] = []

    student.readout.train()
    for epoch in range(epochs):
        epoch_total = 0.0
        epoch_gt = 0.0
        epoch_kd = 0.0
        num_batches = 0

        for batch in precomputed_loader:
            if has_ground_truth:
                b_states, b_teacher, b_true = batch
            else:
                b_states, b_teacher = batch
                b_true = None

            preds = student.readout(b_states)
            loss_kd = criterion(preds, b_teacher)

            if b_true is not None and effective_lambda < 1.0:
                loss_gt = criterion(preds, b_true)
                total_loss = (1.0 - effective_lambda) * loss_gt + effective_lambda * loss_kd
            else:
                loss_gt = torch.tensor(0.0, device=device)
                total_loss = loss_kd

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_total += float(total_loss.item())
            epoch_gt += float(loss_gt.item())
            epoch_kd += float(loss_kd.item())
            num_batches += 1

        avg_total = epoch_total / max(1, num_batches)
        avg_gt = epoch_gt / max(1, num_batches)
        avg_kd = epoch_kd / max(1, num_batches)

        history["total_loss"].append(avg_total)
        history["gt_loss"].append(avg_gt)
        history["kd_loss"].append(avg_kd)
        history["lambda"].append(effective_lambda)

        val_str = ""
        if S_val is not None and Y_val is not None:
            student.readout.eval()
            with torch.no_grad():
                val_preds = student.readout(S_val)
                val_rmse = float(torch.sqrt(criterion(val_preds, Y_val)).item())
            student.readout.train()
            history["val_rmse"].append(val_rmse)
            val_str = f" | Val RMSE: {val_rmse:.5f}"

        if verbose:
            print(
                f"Epoch [{epoch + 1:2d}/{epochs:2d}] "
                f"Total: {avg_total:.5f} | GT: {avg_gt:.5f} | KD: {avg_kd:.5f} "
                f"| lambda: {effective_lambda:.2f}{val_str}"
            )

    student.eval()
    return history
