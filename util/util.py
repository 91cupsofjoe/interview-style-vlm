import torch

# =========================== TENSOR OPERATIOS ===============================

def get_ReLU(x):
    # ReLU activation function (per element e): max(e, 0)
    return torch.clamp(x, min=0)

def get_projection(x, W, b):
    return x @ W.T + b

def softmax(x: torch.Tensor, dim: int):
    """
    Perform sigmoid activation across a tensor along the specified dimension

    Args:
        x (Tensor): Tensor of base values
        dim (int): The dimension along which to apply softmax activation

    Return:
        Tensor of probabilities
    """

    # Subtract each element along the dimension by the dimension max value
    #   to work with smaller base tensor values
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def sigmoid(x: torch.Tensor):
    """
    Perform sigmoid activation across a tensor along the specified dimension
        and return the result

    Args:
        x (Tensor): Tensor of base values
        dim (int): Dimension along which to apply sigmoid activations

    Return:
        Tensor of sigmoid values
    """

    # Use conditional statement to handle "large" negative values
    return torch.where(
        x >= 0,
        1 / (1 + torch.exp(-x)),
        torch.exp(x) / (1 + torch.exp(-x))
    )

# ============================= GENERAL METHODS ===============================

def get_tuple(x, dim=2):
    """
    Take in a int or tuple and return a tuple of ints, one for each dimension

    Args:
        x (int or tuple[int]): The input to (possibly) convert to a tuple of ints
        dim (int): The number of elements for the return tuple
        
    Return:
        A tuple of ints
    """
    if isinstance(x, int):
        return tuple(
            [x for _ in range(dim)]
        )
    return x