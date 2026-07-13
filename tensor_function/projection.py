"""
This module handles tensor projection.
"""
from typing import Optional

from torch import Tensor

from tensor_function.activation import _relu, _relu_backward


def _lin_proj(x: Tensor, W: Tensor, b: Optional[Tensor]=None) -> Tensor:
    """
    Perform linear projection on the input tensor.

    Args:
        x (Tensor): The input tensor
        W (Tensor): The projection weight tensor
        b (Tensor): The projection bias tensor

    Return:
        The projection output tensor
    """
    proj_out = x @ W.T
    if b is not None:
        proj_out = proj_out + b
    return proj_out


def _lin_proj_backward(
    upstream_grad: Tensor,
    proj_in: Tensor,
    W: Tensor,
    b: Optional[Tensor]=None
) -> tuple[Tensor, Tensor, Optional[Tensor]]:
    """
    Apply linear projection backward on the upstream gradient tensor.

    Args:
        upstream_grad (Tensor): The upstream gradient tensor
        proj_in (Tensor): The projection input tensor
        W (Tensor): The projection weight tensor
        b (Tensor): The projection bias tensor

    Return:
        W_proj_grad (Tensor): The projection weight gradient tensor
        b_proj_grad (Tensor): The projection bias gradient tensor
        proj_in_grad (Tensor): The projection input gradient tensor
    """
    # Get the projection weight gradient
    W_grad = upstream_grad.T @ proj_in

    # Initialize the projection bias gradient
    b_grad = None

    # Check if the projection bias was provided
    if b is not None:
        # Get the projection bias gradient
        b_grad = upstream_grad.sum(dim=0)

    # Get the projection input gradient
    proj_in_grad = upstream_grad @ W # Transpose W.T
    
    return proj_in_grad, W_grad, b_grad


def _feed_forward(
    x: Tensor,
    W_1: Tensor, W_2: Tensor,
    b_1: Optional[Tensor]=None, b_2: Optional[Tensor]=None
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Perform feed forwarding on the input tensor.

    Args:
        x (Tensor): The input tensor
        W_1 (Tensor): The projection weight tensor for the first projection
        W_2 (Tensor): The projection weight tensor for the second projection
        b_1 (Tensor): The projection bias tensor for the first projection
        b_2 (Tensor): The projection bias tensor for the second projection

    Return:
        ff_out (Tensor): The feed forward output tensor
        relu_out (Tensor): The ReLU activation output (second projection input) tensor
        relu_in (Tensor): The ReLU activation input (first projection output) tensor

    """
    # Project the input to get the ReLU activation input
    relu_in = _lin_proj(x=x, W=W_1, b=b_1)

    # Apply ReLU activation to the projected input
    relu_out = _relu(relu_in)

    # Project the ReLU activated, projected input to get the feed forward output
    ff_out = _lin_proj(x=relu_out, W=W_2, b=b_2)

    # Return both the projected input and the ReLU activated, doubly projected input
    return ff_out, relu_out, relu_in


def _feed_forward_backward(
    upstream_grad: Tensor,
    W_1: Tensor, W_2: Tensor,
    relu_out: Tensor, relu_in: Tensor,
    ff_in: Tensor,
    b_1: Optional[Tensor]=None, b_2: Optional[Tensor]=None
) -> tuple[
    Tensor, Tensor, Optional[Tensor],
    Tensor, Tensor, Optional[Tensor]
]:
    """
    Apply feed forward backward on the upstream gradient tensor.

    Args:
        upstream_grad (Tensor): The upstream gradient tensor
        W_proj_1 (Tensor): The projection weight tensor for the first projection
        W_proj_2 (Tensor): The projection weight tensor for the second projection
        relu_out (Tensor): The relu activated output tensor
        relu_in (Tensor): The input tensor to the relu activation
        proj_in (Tensor): The input tensor to the (first) projection
        b_proj_1 (Tensor): The projection bias tensor for the first projection
        b_proj_2 (Tensor): The projection bias tensor for the second projection

    Return:
        W_proj_2_grad (Tensor): The projection weight gradient tensor
            for the first projection
        b_proj_2_grad (Tensor): The projeciton bias gradient tensor
            for the first projection
        relu_out_grad (Tensor): The relu output (second projection input)
            gradient tensor
        W_proj_1_grad (Tensor): The projection weight gradient tensor
            for the second projection
        b_proj_1_grad (Tensor): The projection bias gradient tensor
            for the second projection
        ff_in_grad (Tensor): The feed forward input (first projection input)
            gradient tensor
    """
    # Get the projection weight, bias, and input gradients for the second projection
    relu_out_grad, W_2_grad, b_2_grad = _lin_proj_backward(
        upstream_grad=upstream_grad,
        proj_in=relu_out,
        W=W_2,
        b=b_2
    )

    # Get the first projection output (relu input) gradient
    relu_in_grad = _relu_backward(
        upstream_grad=relu_out_grad,
        relu_in=relu_in
    )

    # Get the projection weight, bias, and input gradients for the first projection
    ff_in_grad, W_1_grad, b_1_grad = _lin_proj_backward(
        upstream_grad=relu_in_grad,
        proj_in=ff_in,
        W=W_1,
        b=b_1
    )

    # Return the weight, bias, and input gradients for the second and first projections
    return ff_in_grad, W_1_grad, b_1_grad, \
            relu_out_grad, W_2_grad, b_2_grad