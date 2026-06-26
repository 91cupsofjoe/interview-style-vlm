"""
This module handles tensor convolution.
"""
from typing import Optional
import math

import torch
from torch import Tensor
import torch.nn.functional as nnf

KERNEL_SIZE = (3, 3)
STRIDE = 1
_PADDING_H = (KERNEL_SIZE[0] - 1) / 2
_PADDING_W = (KERNEL_SIZE[1] - 1) / 2
PADDING = (_PADDING_H, _PADDING_W)


def conv2d(
    x: Tensor,
    W: Tensor,
    stride=STRIDE, padding=PADDING,
    b: Optional[Tensor]=None,
) -> Tensor:
    """
    Perform 2D convolution on the input tensor.

    Args:
        x (Tensor): The input tensor
        W (Tensor): The convolution weight tensor
        stride (int): The window increment amount
        padding (int): The number of elements added around the input tensor
        b (Tensor): The convolution bias tensor

    Return:
        A convolution output tensor
    """
    batch_size, _, input_h, input_w = x.shape
    num_out_features, _, kernel_h, kernel_w = W.shape

    # Calculate the spatial grid dimensions from subtracting the kernel
    #   dimensions and factoring in jump size (stride) and border (padding)
    patch_h = math.floor( (input_h + 2 * padding - kernel_h) / stride) + 1
    patch_w = math.floor( (input_w + 2 * padding - kernel_w) / stride) + 1

    # Unfold the input (new dims = [batch size, patch size, num patches])
    x_unfold = nnf.unfold(x, kernel_size=(kernel_h, kernel_w),
                    padding=padding, stride=stride)
        
    # Transpose the input
    x_unfold = x_unfold.transpose(1, 2)
        
    # Flatten the convolution weight
    W_flat = W.reshape(num_out_features, -1)

    # Project the input
    x_proj = x_unfold @ W_flat.T

    # Transpose the projection output
    x_proj = x_proj.transpose(1, 2)

    # Add the convolution bias if it was provided
    if b is not None:
        x_proj += b.view(1, 1, -1) # b.view(1, 1, num out features)

    # Reshape the projection output's spatial dimensions (num patches) according
    #   to the calculated patch height and width from stride and padding
    x_proj = x_proj.reshape(
        batch_size, num_out_features, patch_h, patch_w
    )
        
    # Return the projection output as the convolution output
    return x_proj


def conv2d_backward(
    upstream_grad: Tensor, conv_in: Tensor,
    W: Tensor,
    kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING,
    b: Optional[Tensor]=None
) -> tuple[Tensor, Tensor, Optional[Tensor]]:
    """
    Apply convolution2d backward on the upstream gradient.

    Args:
        upstream_grad (Tensor): The upstream gradient tensor
        conv_in (Tensor): The convolution input tensor
        W (Tensor): The convolution weight tensor
        kernel_size (tuple): The filter/kernel dimensions
        stride (int): The window increment
        padding (int): The number of elements around the upstream gradient tensor
        b (Tensor): The convolution bias tensor

    Return:
        The convolution input gradient tensor
    """
    # Flatten the upstream gradient on the height (second to last) dim
    upstream_grad_3d = upstream_grad.flatten(start_dim=-2)

    # Get the unfolded convolution input
    conv2d_in_unfold = nnf.unfold(conv_in, kernel_size=kernel_size,
                    stride=stride, padding=padding)

    # Get the convolution weight gradient
    W_grad_2d = torch.einsum(
        # upstream_grad dims = [(b)atch_size, num_out_(f)eatures, (n)um_patches]
        # conv2d_in_unfold dims = [(b)atch_size, (p)atch_size, (n)um_patches]
        "bfn,bpn->fp",
        upstream_grad_3d, conv2d_in_unfold)

    # Get the weight and bias gradients
    W_grad = W_grad_2d.reshape(W.shape)

    b_grad = None
    # Check if the convolution bias was provided
    if b is not None:
        b_grad = upstream_grad.sum(dim=(0, -2, -1))
    
    # Get the convolution input gradient
    # First flatten the convolution weight matrix (from the 2nd to last dimensions)
    W_2d = W.reshape(W.shape[0], -1)

    # Use the flattened conv weight to get the unfolded input patches gradient
    conv2d_in_grad_unfold = torch.einsum(
        "fp,bfn->bpn",
        W_2d, upstream_grad_3d
    )

    # Fold the conv2d input gradient
    # First get the conv2d input height and width
    _, _, height, width = conv_in.shape
    conv_in_grad = nnf.fold(
        conv2d_in_grad_unfold,
        output_size=(height, width),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )

    # Return the convolution gradients
    return conv_in_grad, W_grad, b_grad