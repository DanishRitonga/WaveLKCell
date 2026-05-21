# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from typing import List, Tuple
from torch import nn
from torch.nn.modules.loss import _Loss


def filter2D(input_tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Convolves a given kernel on input tensor without losing dimensional shape.

    Parameters
    ----------
        input_tensor : torch.Tensor
            Input image/tensor.
        kernel : torch.Tensor
            Convolution kernel/window.

    Returns
    -------
        torch.Tensor:
            The convolved tensor of same shape as the input.
    """
    (_, channel, _, _) = input_tensor.size()

    pad = [
        kernel.size(2) // 2,
        kernel.size(2) // 2,
        kernel.size(3) // 2,
        kernel.size(3) // 2,
    ]
    pad_tensor = F.pad(input_tensor, pad, "replicate")

    out = F.conv2d(pad_tensor, kernel, groups=channel)
    return out


def _gaussian(window_size: int, sigma: float, device: torch.device = None) -> torch.Tensor:
    """Create a gaussian 1D tensor.

    Parameters
    ----------
        window_size : int
            Number of elements for the output tensor.
        sigma : float
            Std of the gaussian distribution.
        device : torch.device
            Device for the tensor.

    Returns
    -------
        torch.Tensor:
            A gaussian 1D tensor. Shape: (window_size, ).
    """
    x = torch.arange(window_size, device=device).float() - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp((-x.pow(2.0) / float(2 * sigma**2)))

    return gauss / gauss.sum()


def gaussian_kernel2d(
    window_size: int, sigma: float, n_channels: int = 1, device: torch.device = None
) -> torch.Tensor:
    """Create 2D window_size**2 sized kernel a gaussial kernel.

    Parameters
    ----------
        window_size : int
            Number of rows and columns for the output tensor.
        sigma : float
            Std of the gaussian distribution.
        n_channel : int
            Number of channels in the image that will be convolved with
            this kernel.
        device : torch.device
            Device for the kernel.

    Returns:
    -----------
        torch.Tensor:
            A tensor of shape (1, 1, window_size, window_size)
    """
    kernel_x = _gaussian(window_size, sigma, device=device)
    kernel_y = _gaussian(window_size, sigma, device=device)

    kernel_2d = torch.matmul(kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    kernel_2d = kernel_2d.expand(n_channels, 1, window_size, window_size)

    return kernel_2d


class XentropyLoss(_Loss):
    """Cross entropy loss"""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(size_average=None, reduce=None, reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Assumes NCHW shape of array, must be torch.float32 dtype

        Args:
            input (torch.Tensor): Ground truth array with shape (N, C, H, W) with N being the batch-size, H the height, W the width and C the number of classes
            target (torch.Tensor): Prediction array with shape (N, C, H, W) with N being the batch-size, H the height, W the width and C the number of classes

        Returns:
            torch.Tensor: Cross entropy loss, with shape () [scalar], grad_fn = MeanBackward0
        """
        input = input.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)

        epsilon = 10e-8
        pred = input / torch.sum(input, -1, keepdim=True)
        pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
        loss = -torch.sum((target * torch.log(pred)), -1, keepdim=True)
        loss = loss.mean() if self.reduction == "mean" else loss.sum()

        return loss


class DiceLoss(_Loss):
    """Dice loss

    Args:
        smooth (float, optional): Smoothing value. Defaults to 1e-3.
    """

    def __init__(self, smooth: float = 1e-3) -> None:
        super().__init__(size_average=None, reduce=None, reduction="mean")
        self.smooth = smooth

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Assumes NCHW shape of array, must be torch.float32 dtype

        `pred` and `true` must be of torch.float32. Assuming of shape NxHxWxC.

        Args:
            input (torch.Tensor): Prediction array with shape (N, C, H, W) with N being the batch-size, H the height, W the width and C the number of classes
            target (torch.Tensor): Ground truth array with shape (N, C, H, W) with N being the batch-size, H the height, W the width and C the number of classes

        Returns:
            torch.Tensor: Dice loss, with shape () [scalar], grad_fn=SumBackward0
        """
        input = input.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)
        inse = torch.sum(input * target, (0, 1, 2))
        l = torch.sum(input, (0, 1, 2))
        r = torch.sum(target, (0, 1, 2))
        loss = 1.0 - (2.0 * inse + self.smooth) / (l + r + self.smooth)
        loss = torch.sum(loss)

        return loss


class MSELossMaps(_Loss):
    """Calculate mean squared error loss for combined horizontal and vertical maps of segmentation tasks."""

    def __init__(self) -> None:
        super().__init__(size_average=None, reduce=None, reduction="mean")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss calculation

        Args:
            input (torch.Tensor): Prediction of combined horizontal and vertical maps
                with shape (N, 2, H, W), channel 0 is vertical and channel 1 is horizontal
            target (torch.Tensor): Ground truth of combined horizontal and vertical maps
                with shape (N, 2, H, W), channel 0 is vertical and channel 1 is horizontal

        Returns:
            torch.Tensor: Mean squared error per pixel with shape (N, 2, H, W), grad_fn=SubBackward0

        """
        loss = input - target
        loss = (loss * loss).mean()
        return loss


class MSGELossMaps(_Loss):
    def __init__(self) -> None:
        super().__init__(size_average=None, reduce=None, reduction="mean")

    def get_sobel_kernel(
        self, size: int, device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get sobel kernel with a given size.

        Args:
            size (int): Kernel site
            device (str): Cuda device

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Horizontal and vertical sobel kernel, each with shape (size, size)
        """
        assert size % 2 == 1, "Must be odd, get size=%d" % size

        h_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device=device,
            requires_grad=False,
        )
        v_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device=device,
            requires_grad=False,
        )
        h, v = torch.meshgrid(h_range, v_range, indexing="ij")
        kernel_h = h / (h * h + v * v + 1.0e-15)
        kernel_v = v / (h * h + v * v + 1.0e-15)
        return kernel_h, kernel_v

    def get_gradient_hv(self, hv: torch.Tensor, device: str) -> torch.Tensor:
        """For calculating gradient of horizontal and vertical prediction map

        Args:
            hv (torch.Tensor): horizontal and vertical map
            device (str): CUDA device

        Returns:
            torch.Tensor: Gradient with same shape as input
        """
        kernel_h, kernel_v = self.get_sobel_kernel(5, device=device)
        kernel_h = kernel_h.view(1, 1, 5, 5)
        kernel_v = kernel_v.view(1, 1, 5, 5)

        h_ch = hv[..., 0].unsqueeze(1)  # Nx1xHxW
        v_ch = hv[..., 1].unsqueeze(1)  # Nx1xHxW

        h_dh_ch = F.conv2d(h_ch, kernel_h, padding=2)
        v_dv_ch = F.conv2d(v_ch, kernel_v, padding=2)
        dhv = torch.cat([h_dh_ch, v_dv_ch], dim=1)
        dhv = dhv.permute(0, 2, 3, 1).contiguous()  # to NHWC
        return dhv

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        focus: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        """MSGE (Gradient of MSE) loss

        Args:
            input (torch.Tensor): Input with shape (B, C, H, W)
            target (torch.Tensor): Target with shape (B, C, H, W)
            focus (torch.Tensor): Focus, type of masking (B, C, W, W)
            device (str): CUDA device to work with.

        Returns:
            torch.Tensor: MSGE loss
        """
        input = input.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)
        focus = focus.permute(0, 2, 3, 1)
        focus = focus[..., 1]

        focus = (focus[..., None]).float()  # assume input NHW
        focus = torch.cat([focus, focus], axis=-1).to(device)
        true_grad = self.get_gradient_hv(target, device)
        pred_grad = self.get_gradient_hv(input, device)
        loss = pred_grad - true_grad
        loss = focus * (loss * loss)
        loss = loss.sum() / (focus.sum() + 1.0e-8)
        return loss


class FocalTverskyLoss(nn.Module):
    """FocalTverskyLoss

    PyTorch implementation of the Focal Tversky Loss Function for multiple classes
    doi: 10.1109/ISBI.2019.8759329
    Abraham, N., & Khan, N. M. (2019).
    A Novel Focal Tversky Loss Function With Improved Attention U-Net for Lesion Segmentation.
    In International Symposium on Biomedical Imaging. https://doi.org/10.1109/isbi.2019.8759329

    @ Fabian Hörst, fabian.hoerst@uk-essen.de
    Institute for Artifical Intelligence in Medicine,
    University Medicine Essen

    Args:
        alpha_t (float, optional): Alpha parameter for tversky loss (multiplied with false-negatives). Defaults to 0.7.
        beta_t (float, optional): Beta parameter for tversky loss (multiplied with false-positives). Defaults to 0.3.
        gamma_f (float, optional): Gamma Focal parameter. Defaults to 4/3.
        smooth (float, optional): Smooting factor. Defaults to 0.000001.
    """

    def __init__(
        self,
        alpha_t: float = 0.7,
        beta_t: float = 0.3,
        gamma_f: float = 4 / 3,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha_t = alpha_t
        self.beta_t = beta_t
        self.gamma_f = gamma_f
        self.smooth = smooth
        self.num_classes = 2

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss calculation

        Args:
            input (torch.Tensor): Predictions, logits (without Softmax). Shape: (B, C, H, W)
            target (torch.Tensor): Targets, either flattened (Shape: (C, H, W) or as one-hot encoded (Shape: (batch-size, C, H, W)).

        Raises:
            ValueError: Error if there is a shape missmatch

        Returns:
            torch.Tensor: FocalTverskyLoss (weighted)
        """
        input = input.permute(0, 2, 3, 1)
        if input.shape[-1] != self.num_classes:
            raise ValueError(
                "Predictions must be a logit tensor with the last dimension shape beeing equal to the number of classes"
            )
        if len(target.shape) != len(input.shape):
            target = F.one_hot(target, num_classes=self.num_classes)

        target = target.permute(0, 2, 3, 1).contiguous().view(-1)
        input = torch.softmax(input, dim=-1).contiguous().view(-1)

        tp = (input * target).sum()
        fp = ((1 - target) * input).sum()
        fn = (target * (1 - input)).sum()

        Tversky = (tp + self.smooth) / (
            tp + self.alpha_t * fn + self.beta_t * fp + self.smooth
        )
        FocalTversky = (1 - Tversky) ** self.gamma_f

        return FocalTversky


class MCFocalTverskyLoss(FocalTverskyLoss):
    """Multiclass FocalTverskyLoss

    PyTorch implementation of the Focal Tversky Loss Function for multiple classes
    doi: 10.1109/ISBI.2019.8759329
    Abraham, N., & Khan, N. M. (2019).
    A Novel Focal Tversky Loss Function With Improved Attention U-Net for Lesion Segmentation.
    In International Symposium on Biomedical Imaging. https://doi.org/10.1109/isbi.2019.8759329

    @ Fabian Hörst, fabian.hoerst@uk-essen.de
    Institute for Artifical Intelligence in Medicine,
    University Medicine Essen

    Args:
        alpha_t (float, optional): Alpha parameter for tversky loss (multiplied with false-negatives). Defaults to 0.7.
        beta_t (float, optional): Beta parameter for tversky loss (multiplied with false-positives). Defaults to 0.3.
        gamma_f (float, optional): Gamma Focal parameter. Defaults to 4/3.
        smooth (float, optional): Smooting factor. Defaults to 0.000001.
        num_classes (int, optional): Number of output classes. For binary segmentation, prefer FocalTverskyLoss (speed optimized). Defaults to 2.
        class_weights (List[int], optional): Weights for each class. If not provided, equal weight. Length must be equal to num_classes. Defaults to None.
    """

    def __init__(
        self,
        alpha_t: float = 0.7,
        beta_t: float = 0.3,
        gamma_f: float = 4 / 3,
        smooth: float = 0.000001,
        num_classes: int = 2,
        class_weights: List[int] = None,
    ) -> None:
        super().__init__(alpha_t, beta_t, gamma_f, smooth)
        self.num_classes = num_classes
        if class_weights is None:
            self.class_weights = [1 for i in range(self.num_classes)]
        else:
            assert (
                len(class_weights) == self.num_classes
            ), "Please provide matching weights"
            self.class_weights = class_weights
        self.class_weights = torch.Tensor(self.class_weights)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss calculation

        Args:
            input (torch.Tensor): Predictions, logits (without Softmax). Shape: (B, num_classes, H, W)
            target (torch.Tensor): Targets, either flattened (Shape: (B, H, W) or as one-hot encoded (Shape: (B, num_classes, H, W)).

        Raises:
            ValueError: Error if there is a shape missmatch

        Returns:
            torch.Tensor: FocalTverskyLoss (weighted)
        """
        input = input.permute(0, 2, 3, 1)
        if input.shape[-1] != self.num_classes:
            raise ValueError(
                "Predictions must be a logit tensor with the last dimension shape beeing equal to the number of classes"
            )
        if len(target.shape) != len(input.shape):
            target = F.one_hot(target, num_classes=self.num_classes)

        target = target.permute(0, 2, 3, 1)
        input = torch.softmax(input, dim=-1)

        input = torch.permute(input, (3, 1, 2, 0))
        target = torch.permute(target, (3, 1, 2, 0))

        input = torch.flatten(input, start_dim=1)
        target = torch.flatten(target, start_dim=1)

        tp = torch.sum(input * target, 1)
        fp = torch.sum((1 - target) * input, 1)
        fn = torch.sum(target * (1 - input), 1)

        Tversky = (tp + self.smooth) / (
            tp + self.alpha_t * fn + self.beta_t * fp + self.smooth
        )
        FocalTversky = (1 - Tversky) ** self.gamma_f

        self.class_weights = self.class_weights.to(FocalTversky.device)
        return torch.sum(self.class_weights * FocalTversky)
