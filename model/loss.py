import torch

from util import util

def cross_entropy_loss(
        predictions, true_labels, weights,
        predictions_dim=0, lambda_=1e-4
    ):
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
    probs = util.softmax(predictions, dim=predictions_dim)
    # Clamp the probabilities for stability
    probs = torch.clamp(probs, min=1e-9)

    # Get the base cross entropy loss: -sum( true label * -log(probability) )
    ce_loss = -(true_labels * torch.log(probs)).sum()

    # Then get the sum across all squared l2 norms (one norm per weight)
    squared_l2_norm = 0
    for weight in weights:
        squared_l2_norm += (weight**2).sum()

    return ce_loss + lambda_ * squared_l2_norm

    