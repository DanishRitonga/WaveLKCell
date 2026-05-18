from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment


def linear_assignment_fn(cost_matrix: torch.Tensor) -> torch.Tensor:
    if cost_matrix.numel() == 0:
        return torch.empty(0, 2, dtype=torch.long, device=cost_matrix.device)

    cost_np = cost_matrix.detach().cpu().numpy()
    row_indices, col_indices = linear_sum_assignment(cost_np)

    return torch.stack(
        [
            torch.as_tensor(row_indices, dtype=torch.long),
            torch.as_tensor(col_indices, dtype=torch.long),
        ],
        dim=1,
    )
