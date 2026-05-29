"""
This module handles tensor convolution.
"""
import math

import torch
from torch import Tensor
import torch.nn.functional as nnf

def conv2d(
    x: Tensor,
    W: Tensor, b: Tensor,
    stride: int, padding: int
) -> Tensor:
    """
    Perform 2D convolution on the input tensor.

    Args:
        x (Tensor): The input tensor
        W (Tensor): The convolution weight matrx
        b (Tensor): The convolution bias vector
        stride (int): The window increment amount
        padding (int): The number of elements added around the input tensor

    Return:
        A convolution output tensor
    """
    batch_size, channels, input_h, input_w = x.shape
    num_out_features, _, kernel_h, kernel_w = W.shape

    # Calculate the spatial grid dimensions from subtracting the kernel
    #   dimensions and factoring in jump size (stride) and border (padding)
    patch_h = math.floor( (input_h + 2 * padding - kernel_h) / stride) + 1
    patch_w = math.floor( (input_w + 2 * padding - kernel_w) / stride) + 1

    # Calculate patch size and num patches
    patch_size = channels * kernel_h * kernel_w
    num_patches = patch_h * patch_w

    # Get input tensor with dims:
    #   batch size, patch size, num patches
    input_tensor = nnf.unfold(x, kernel_size=(kernel_h, kernel_w),
                    padding=padding, stride=stride)
        
    # Transpose input tensor, switching its dimensions from [batch size,
    #   patch size, num patches] to [batch size, num patches, patch size]
    input_tensor = input_tensor.transpose(1, 2)

    # Make sure the shape of the output tensor =
    #   (batch size, patch size, num patches)
    assert(input_tensor.shape == (batch_size, num_patches, patch_size))
        
    # Get flattened weight matrix with dims:
    #   num output features, num channels * kernel h * kernel w
    W_flat = W.view(num_out_features, -1)

    # Get output tensor with dims:
    #   batch size, num output features, num patches
    output_tensor = input_tensor @ W_flat.T

    # Swap the num output features and num patches dimensions
    output_tensor = output_tensor.transpose(1, 2)

    # For each spatial position in each batch image in a feature map,
    #   add the scalar bias value associated with that feature.
    output_tensor += b.view(1, 1, -1) # b.view(1, 1, num out features)

    # Reshape the output tensor's spatial dimensions (num_patches)
    #   according to calculated height and width from stride and padding
    output_tensor = output_tensor.reshape(
                    batch_size, num_out_features, patch_h, patch_w)
        
    # Return the convolution output tensor
    return output_tensor


def conv2d_backward(x: Tensor,
    in_patches: Tensor, W: Tensor,
    kernel_size: tuple, stride: int, padding: int
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Apply convolution2d backward on the input tensor.

    Args:
        x (Tensor): The input tensor
        in_patches (Tensor): The pre-convolution input patches tensor
        W (Tensor): The convolution weight matrix
        kernel_size (tuple): The filter/kernel dimensions
        stride (int): The window increment amount
        padding (int): The number of elements added around the input tensor

    Return:
        The input patches gradient tensor
    """
    # Flatten the convolution output derivative and the input patches,
    #   patch_size --> patch_height * patch_width
    # d_zconv dims = [batch_size, num_out_features, num_patch_rows, num_patch_cols
    batch_dim, features_dim, h_dim, w_dim = (0, 1, 2, 3)
    d_x_3d = x.flatten(start_dim=h_dim)
    # input patches dims = [batch_size, num_patches, patch_height, patch_width]
    in_patches_3d = in_patches.flatten(start_dim=2) # patch_height = 3rd dim
    in_patches_unfold = nnf.unfold(in_patches_3d, kernel_size=kernel_size,
                    stride=stride, padding=padding)

    # Get the input patches derivative
    #   (1) For each feature in the 3d convolution output derivative,
    #       get the sum of all elements across the batch and patch space. Do
    #       this across all features to get a vector of feature value sums
    #   (2) Do the same for all patches in the input patches to get a vector
    #       of patch value sums.
    #   (3) Multiply the two vectors to get the input patches derivative
    d_W_2d = torch.einsum(
        # z_conv_3d dims = [(b)atch_size, num_out_(f)eatures, (n)um_patches]
        # x_unfold dims = [(b)atch_size, (p)atch_size, (n)um_patches]
        "bfn,bpn->fp",
        d_x_3d, in_patches_unfold)

    # Get the weight and bias gradients
    d_W = d_W_2d.reshape(W.shape)
    d_b = x.sum(dim=(batch_dim, h_dim, w_dim))
    
    # Get the downstream input patches gradient
    # First flatten the convolution weight matrix
    W_2d = W.reshape(W.shape[0], -1) # num in channels is 1st dim

    # Use the flattened conv weight to get the unfolded input patches gradient
    d_in_patches_unfold = torch.einsum(
        "fp,bfn->bpn",
        W_2d, d_x_3d
    )

    # Fold the input patches gradient to the dimensions of the input patches
    d_in_patches = nnf.fold(
        d_in_patches_unfold,
        output_size=(h_dim, w_dim),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )

    # Return the convolution gradients
    return d_W, d_b, d_in_patches