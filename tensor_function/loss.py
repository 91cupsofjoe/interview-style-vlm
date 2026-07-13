"""
This module handles loss calculation.
"""
from typing import Optional

import torch
from torch import Tensor

from tensor_function import regularization as reg
from tensor_function.activation import _softmax, _sigmoid

EPS = 1e-4
PATIENCE = 5
LOSS_REDUCTION_TYPE = 'mean'
CLAMP_MIN = EPS
CLAMP_MAX = 1 - EPS


def _binary_cross_entropy_loss(
    logits: Tensor, true_labels: Tensor,
    loss_reduction_type=LOSS_REDUCTION_TYPE,
    reg_type=reg.REG_TYPE,
    reg_strength=reg.REG_STRENGTH,
    learnable_weights: Optional[list[Tensor]]=None
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Calculate and return the binary cross entropy loss based on
        logits, true labels, and weights.

    Args:
        logits (Tensor): The logits tensor
        true_labels (Tensor): The true labels tensor
        loss_reduction_type (str): The type of loss reduction
        reg_type (str): The type of regularization
        reg_strength (float): The regularization strength
        learnable_weights (list[Tensor]): List of learnable weight tensors

    Return:
        bce_loss (Tensor): The binary cross entropy loss tensor
        probabilities (Tensor): The probabilities tensor
        sigmout_out (Tensor): The sigmoid activation output tensor
    """
    # Get the probabilities by applying sigmoid activation on the logits
    sigmoid_out = _sigmoid(logits)

    # Clamp the probabilities for stability
    probabilities = torch.clamp(sigmoid_out, min=CLAMP_MIN, max=CLAMP_MAX)

    # Get the binary cross entropy loss
    bce_loss = -(
        true_labels * torch.log(probabilities)
        + (1 - true_labels) * torch.log(1 - probabilities)
    ).sum()
    
    # Check if using mean loss reduction
    if loss_reduction_type == 'mean':
        # Get the number of classes
        num_classes = true_labels.numel()

        # Average the binary cross entropy loss across the classes
        bce_loss = bce_loss / num_classes

    # Check if using ridge regression
    if reg_type == 'ridge':
        # Apply ridge regression to the binary cross entropy loss
        bce_loss = reg._ridge_regression(
            x=bce_loss,
            reg_strength=reg_strength,
            weights=learnable_weights
        )

    # Return the binary cross entropy loss and the probabilities
    return bce_loss, probabilities, sigmoid_out


def _binary_cross_entropy_loss_backward(
    upstream_grad: Tensor,
    sigmoid_out: Tensor, true_labels: Tensor,
    loss_reduction_type=LOSS_REDUCTION_TYPE
) -> Tensor:
    """
    Perform binary cross entropy loss backward on the upstream gradient to get
        the logits gradient.

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
        # Get the number of classes
        num_classes = true_labels.numel()

        # Average the probabilities gradient across the classes
        logits_grad = logits_grad / num_classes

    # Return the logits gradient scaled by the upstream gradient
    return logits_grad * upstream_grad


def _cross_entropy_loss(
    logits: Tensor, true_labels: Tensor,
    loss_reduction_type=LOSS_REDUCTION_TYPE,
    reg_type=reg.REG_TYPE,
    reg_strength=reg.REG_STRENGTH,
    learnable_weights: Optional[list[Tensor]]=None
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Calculate and return cross entropy loss based on
        logits, true labels, and weights.

    Args:
        predictions (Tensor): The predictions tensor
        true_labels (Tensor): The true labels tensor
        loss_reduction_type (str): The type of loss reduction
        reg_type (str): The type of regularization
        reg_strength (float): The regularization strength
        learnable weights (list[Tensor]): List of learnable weight tensors

    Return:
        ce_loss (Tensor): The cross entropy loss tensor
        probabilities (Tensor): The probabilities tensor
        softmax_out (Tensor): The softmax activation output tensor
    """
    # Get the probabilities by applying softmax activation on the logits
    softmax_out = _softmax(logits, -1)

    # Clamp the probabilities for stability
    probabilities = torch.clamp(softmax_out, min=CLAMP_MIN)

    # Get the negative log-likelihood of the probabilities
    nll = -torch.log(probabilities)

    # Get the cross entropy loss
    ce_loss = (true_labels * nll).sum()
    
    # Check if using mean loss reduction
    if loss_reduction_type == 'mean':
        # Get the number of classes
        num_classes = true_labels.numel() // true_labels.shape[-1]

        # Reduce the cross entropy loss by the number of classes
        ce_loss = ce_loss / num_classes

    # Check if using ridge regression
    if reg_type == 'ridge':
        # Apply ridge regression to the cross entropy loss
        ce_loss = reg._ridge_regression(
            x=ce_loss,
            reg_strength=reg_strength,
            weights=learnable_weights
        )

    # Return the cross entropy loss and the probabilties
    return ce_loss, probabilities, softmax_out


def _cross_entropy_loss_backward(
    upstream_grad: Tensor,
    softmax_out: Tensor, true_labels: Tensor,
    loss_reduction_type=LOSS_REDUCTION_TYPE
) -> Tensor:
    """
    Perform binary cross entropy loss backward on the upstream gradient to get
        the logits gradient.

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
        # Get the number of classes
        num_classes = true_labels.numel() // true_labels.shape[-1]

        # Average the probabilities gradient across the classes
        logits_grad = logits_grad / num_classes

    # Return the logits gradient scaled by the upstream gradient
    return logits_grad * upstream_grad