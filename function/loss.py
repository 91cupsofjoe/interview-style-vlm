"""
This module handles loss calculation.
"""
from typing import Optional

import torch
from torch import Tensor

from function import regularization as reg
from function.activation import softmax, sigmoid

EPS = 1e-4
PATIENCE = 5
LOSS_REDUCTION_TYPE = 'mean'
CLAMP_MIN = 1e-9
CLAMP_MAX = 1 - 1e-9


def binary_cross_entropy_loss(
    predictions: Tensor, true_labels: Tensor,
    loss_reduction_type=LOSS_REDUCTION_TYPE,
    reg_type=reg.REG_TYPE,
    reg_strength=reg.REG_STRENGTH,
    weights: Optional[list[Tensor]]=None
) -> Tensor:
    """
    Calculate and return the binary cross entropy loss based on
        predictions, true labels, and weights.

    Args:
        predictions (Tensor): The predictions tensor
        true_labels (Tensor): The true labels tensor
        loss_reduction_type (str): The type of loss reduction
        reg_type (str): The type of regularization
        reg_strength (float): The regularization strength
        weights (list[Tensor]): List of weight tensors

    Return:
        The binary cross entropy loss
    """
    # Get the probabilities by applying sigmoid activation on the predictions
    probabilities = sigmoid(predictions)

    # Clamp the probabilities for stability
    probabilities = torch.clamp(probabilities, min=CLAMP_MIN, max=CLAMP_MAX)

    # Get the binary cross entropy loss
    bce_loss = -(
        true_labels * torch.log(probabilities) \
                        + (1 - true_labels) * torch.log(1 - probabilities)
    ).sum()
    
    # Check if using mean loss reduction
    if loss_reduction_type == 'mean':
        # Get the number of predictions
        num_predictions = true_labels.numel()

        # Reduce the binary cross entropy loss by the number of predictions
        bce_loss = bce_loss / num_predictions

    # Check if using ridge regression
    if reg_type == 'ridge':
        # Apply ridge regression to the binary cross entropy loss
        bce_loss = reg.ridge_regression(
            x=bce_loss,
            reg_strength=reg_strength,
            weights=weights
        )

    # Return the binary cross entropy loss
    return bce_loss


def binary_cross_entropy_loss_backward(
    upstream_grad: Tensor,
    sigmoid_out: Tensor, true_labels: Tensor,
    loss_reduction_type=LOSS_REDUCTION_TYPE
) -> Tensor:
    """
    Calculate and return cross entropy loss based on
        predictions, true labels, and weights.

    Args:
        upstream_grad (Tensor): The upstream gradient tensor
        sigmoid_out (Tensor): The sigmoid output tensor
        true_labels (Tensor): The true labels tensor
        loss_reduction_type (str): The type of loss reduction

    Return:
        The logits gradient tensor
    """
    # Get the logits gradient
    logits_grad = sigmoid_out - true_labels

    # Check if using mean loss reduction
    if loss_reduction_type == 'mean':
        # Get the number of predictions
        num_predictions = true_labels.numel()

        # Reduce the probabilities gradient by the number of predictions
        logits_grad = logits_grad / num_predictions

    # Return the logits gradient scaled by the upstream gradient
    return logits_grad * upstream_grad


def cross_entropy_loss(
    predictions: Tensor, true_labels: Tensor,
    loss_reduction_type=LOSS_REDUCTION_TYPE,
    reg_type=reg.REG_TYPE,
    reg_strength=reg.REG_STRENGTH,
    weights: Optional[list[Tensor]]=None
) -> Tensor:
    """
    Calculate and return cross entropy loss based on
        predictions, true labels, and weights.

    Args:
        predictions (Tensor): The predictions tensor
        true_labels (Tensor): The true labels tensor
        loss_reduction_type (str): The type of loss reduction
        reg_type (str): The type of regularization
        reg_strength (float): The regularization strength
        weights (list[Tensor]): List of weight tensors

    Return:
        The cross entropy scalar loss
    """
    # Get the probabilities by applying softmax activation on the predictions
    probabilities = softmax(predictions, -1)

    # Clamp the probabilities for stability
    probabilities = torch.clamp(probabilities, min=CLAMP_MIN)

    # Get the negative log-likelihood of the probabilities
    nll = -torch.log(probabilities)

    # Get the cross entropy loss
    ce_loss = (true_labels * nll).sum()
    
    # Check if using mean loss reduction
    if loss_reduction_type == 'mean':
        # Get the number of predictions
        num_predictions = true_labels.numel() // true_labels.shape[-1]

        # Reduce the cross entropy loss by the number of predictions
        ce_loss = ce_loss / num_predictions

    # Check if using ridge regression
    if reg_type == 'ridge':
        # Apply ridge regression to the cross entropy loss
        ce_loss = reg.ridge_regression(
            x=ce_loss,
            reg_strength=reg_strength,
            weights=weights
        )

    # Return the cross entropy loss
    return ce_loss


def cross_entropy_loss_backward(
    upstream_grad: Tensor,
    softmax_out: Tensor, true_labels: Tensor,
    loss_reduction_type=LOSS_REDUCTION_TYPE
) -> Tensor:
    """
    Calculate and return cross entropy loss based on
        predictions, true labels, and weights.

    Args:
        upstream_grad (Tensor): The upstream gradient tensor
        softmax_out (Tensor): The softmax output tensor
        true_labels (Tensor): The true labels tensor
        loss_reduction_type (str): The type of loss reduction

    Return:
        The logits gradient tensor
    """
    # Get the logits gradient
    logits_grad = softmax_out - true_labels

    # Check if using mean loss reduction
    if loss_reduction_type == 'mean':
        # Get the number of predictions
        num_predictions = true_labels.numel() // true_labels.shape[-1]

        # Reduce the probabilities gradient by the number of predictions
        logits_grad = logits_grad / num_predictions

    # Return the logits gradient scaled by the upstream gradient
    return logits_grad * upstream_grad