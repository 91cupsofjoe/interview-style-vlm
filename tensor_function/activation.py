"""
This module handles tensor activation.
"""
import torch
from torch import Tensor


def _relu(x: Tensor) -> Tensor:
    """
    Apply ReLU activation on the input tensor.

    Args:
        relu_in (tensor): The input tensor
    
    Return:
        A ReLU activated tensor
    """
    return torch.clamp(x, min=0)


def _relu_backward(upstream_grad: Tensor, relu_in: Tensor) -> Tensor:
    """
    Perform ReLU backward on the upstream gradient tensor.

    Args:
        upstream_grad (Tensor): The upstream gradient tensor
        relu_in (Tensor): The relu input tensor

    Return:
        The relu input gradient tensor
    """
    # Use the relu input as a binary mask
    return upstream_grad * (relu_in > 0)


def _sigmoid(x: Tensor) -> Tensor:
    """
    Perform sigmoid activation on the input tensor.

    Args:
        x (Tensor): The input tensor

    Return:
        A sigmoid activated tensor
    """
    # Use conditional statement to handle "large" negative values
    return torch.where(
        x >= 0,
        1 / (1 + torch.exp(-x)),
        torch.exp(x) / (1 + torch.exp(-x))
    )


def _softmax(x: Tensor, dim: int) -> Tensor:
    """
    Perform softmax activation on input tensor along the specified dimension

    Args:
        x (Tensor): The input tensor
        dim (int): The dimension along which to apply softmax activation

    Return:
        The probabilities tensor
    """
    # Subtract each element along the dimension by the dimension max value
    #   to work with smaller base tensor values
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def _softmax_backward(
    upstream_grad: Tensor,
    softmax_out: Tensor,
    dim: int
) -> Tensor:
    """
    Perform softmax activation backward on the upstream gradient.

    Args:
        upstream_grad (Tensor): The upstream gradient tensor
        softmax_out (Tensor): The softmax output tensor
        dim (int): The dimension along which to apply softmax activation

    Return:
        A tensor of probabilities
    """
    # Get the weighted average influece (dot product) of the upstream gradient
    #   and the probabilities (softmax output)
    weight_avg = (upstream_grad * softmax_out).sum(dim=dim, keepdim=True)

    # Return the softmax input gradient
    # NOTE: upstream - weight_avg brings the upstream gradient values closer
    #   to the mean, while scaling them with softmax out scales the result
    #   according to each probability's sensitivity
    return softmax_out * (upstream_grad - weight_avg)