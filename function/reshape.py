"""
This module handles tensor reshaping.
"""
from typing import Optional

from torch import Tensor

DIM = 0

def flatten(x, dim=DIM) -> Tensor:
    """
    Flatten the input tensor.

    Args:
        x (Tensor): The input tensor

    Return:
        A flattened tensor
    """
    return x.reshape(x.shape[dim], -1)


def unflatten(upstream_grad: Tensor, flat_in) -> Tensor:
    """
    Unflatten the upstream gradient tensor.

    Args:
        upstream_grad (Tensor): The upstream gradient tensor
        flat_in: The input tensor to the flatten function

    Return:
        The unflatten upstream gradient tensor
    """
    # Unflatten the upstream gradient using the shape of the flatten input
    return upstream_grad.reshape(flat_in.shape)


def split_tensor(
    x: Tensor, num_split_heads: int, dim=DIM,
    transpose_dims: Optional[tuple[int, int]]=None
) -> Optional[Tensor]:
    """
    Return the input tensor split at a dimension.

    Args:
        x (Tensor): The input tensor
        num_split_heads (int): The number of split heads
        dim (int): The dimension to split at

    Return:
        The split tensor
    """
    # Get the size of the splitting dimension
    dim_size = x.shape[dim]

    # Isolate the leading dimensions
    leading_dims = None
    if dim != 0 and -1 * dim < len(x.shape):
        leading_dims = x.shape[:dim]

    # Isolate the trailing dimensions
    trailing_dims = None
    if dim + 1 < len(x.shape) and dim != -1:
        trailing_dims = x.shape[dim+1:]

    # Make sure the split size is divisible by the number of splits
    if dim_size % num_split_heads == 0:

        # Get the new shape
        new_shape = (num_split_heads, dim_size // num_split_heads)
        if leading_dims is not None:
            new_shape = leading_dims + new_shape
        if trailing_dims is not None:
            new_shape += trailing_dims
        
        # Reshape the input tensor (and transpose it, if specified)
        x = x.reshape(new_shape)
        if transpose_dims is not None:
            x = x.transpose(transpose_dims[0], transpose_dims[1])

        # Return the split tensor
        return x

    # Else, return None since the input tensor could not be split
    return None