"""
This module handles loss calculation.
"""
import torch

from tensor.activation import softmax


def cross_entropy_loss(
    prediction, true_labels,
    weights, reg_type='ridge', reg_strength=1e-4
) -> float:
    """
    Calculate and return cross entropy loss based on predictions (projection
        output), true labels, and weights

    Args:
        prediction (Tensor): Tensor of predictions
        true_labels (Tensor): Tensor of true labels
        weights (list[Tensor]): List of weight Tensors

    Return:
        The regularized cross entropy loss
    """
    # First apply the softmax algorithm to the linear projection (logits)
    #   dim index 2 (3rd dim) = embedding size, each element repre
    probs = softmax(prediction, 2)
    # Clamp the probabilities for stability
    probs = torch.clamp(probs, min=1e-9)

    # Get the base cross entropy loss: -sum( true label * -log(probability) )
    ce_loss = -(true_labels * torch.log(probs)).sum()

    # Then get the sum across all squared l2 norms (one norm per weight)
    squared_l2_norm = 0
    for weight in weights:
        squared_l2_norm += (weight**2).sum()

    # Check if using ridge regression
    if reg_type == 'ridge':
        return ce_loss + reg_strength * squared_l2_norm
    
    # Else, using standard regularization
    return ce_loss + reg_strength * squared_l2_norm