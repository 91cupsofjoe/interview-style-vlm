import torch

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
        x (tensor): Tensor of base values
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

def get_cross_entropy_loss(z_proj, true_labels, weights, lambda_=1e-4):
    """
    Calculate and return cross entropy loss based on predictions (projection
        output), true labels, and weights

    Args:
        z_proj (Tensor): Tensor of predictions
        true_labels (Tensor): Tensor of true labels
        weights (list[Tensor]): List of weight Tensors

    Return:
        The regularized cross entropy loss
    """
    # First apply the softmax algorithm to the linear projection (logits)
    #   dim index 2 (3rd dim) = embedding size, each element repre
    probs = softmax(z_proj, dim=2)
    # Clamp the probabilities for stability
    probs = torch.clamp(probs, min=1e-9)

    # Get the base cross entropy loss: -sum( true label * -log(probability) )
    ce_loss = -(true_labels * torch.log(probs)).sum()

    # Then get the sum across all squared l2 norms (one norm per weight)
    squared_l2_norm = 0
    for weight in weights:
        squared_l2_norm += (weight**2).sum()

    return ce_loss + lambda_ * squared_l2_norm

    