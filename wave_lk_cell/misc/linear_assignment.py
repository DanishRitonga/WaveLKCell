from __future__ import annotations

import torch

try:
    from torch_linear_assignment import linear_assignment
except ImportError:
    linear_assignment = None


def linear_assignment_fn(cost_matrix: torch.Tensor) -> torch.Tensor:
    if cost_matrix.numel() == 0:
        return torch.empty(0, 2, dtype=torch.long, device=cost_matrix.device)

    if linear_assignment is None:
        raise ImportError("torch_linear_assignment is required. Install with: pip install torch-linear-assignment")

    device = cost_matrix.device
    if device.type == "mps":
        cost_matrix = cost_matrix.to("cpu")

    cost_np = cost_matrix.detach().cpu().numpy()
    row_indices, col_indices = linear_assignment(cost_np)

    result = torch.stack(
        [
            torch.as_tensor(row_indices, dtype=torch.long),
            torch.as_tensor(col_indices, dtype=torch.long),
        ],
        dim=1,
    )

    if device.type == "mps":
        result = result.to(device)

    return result
