"""
This module handles residual functions.
"""
from typing import Optional
from torch import Tensor


def residual_add(
    x: Tensor,
    res_addend: Optional[Tensor]=None,
    dropout_out: Optional[Tensor]=None
) -> Tensor:
    """
    Residually add the input and residual addend tensors.
    NOTE: The residual addend is usually the dropout output.

    Args:
        x (Tensor): The input tensor
        res_addend (Tensor): The residual addend tensor

    Return:
        x (Tensor): The input (residual add output) tensor
    """
    # Add the residual addend to the input if provided
    if res_addend is not None:
        x = x + res_addend

    # Add the dropout output to the input if provided
    if dropout_out is not None:
        x = x + dropout_out

    # Return the input (residual add output) tensor
    return x


def residual_add_backward(
    upstream_grad: Tensor
) -> tuple[Tensor, Tensor]:
    """
    Return the upstream gradient as the gradients to both the residual addend
        and the original input (to residual add).

    Args:
        upstream_grad (Tensor): The upstream gradient tensor

    Return:
        res_add_in_grad (Tensor): The gradient tensor for the residual add input
        res_addend_grad (Tensor): The residual addend gradient tensor
    """
    res_add_in_grad = upstream_grad.clone()
    res_addend_grad = upstream_grad.clone()

    return res_add_in_grad, res_addend_grad