"""
This module handles learnable parameter tensor updating.
"""
from typing import Optional, Any
from collections.abc import Callable

from torch import Tensor

from function import regularization as reg

LEARNING_RATE = 1.0


def basic_update(
    learnable_parameter: Tensor, gradient: Tensor,
    learning_rate=LEARNING_RATE
) -> Tensor:
    """
    Perform a basic update on the learnable parameter using its gradient.

    Args:
        learnable_parameter (Tensor): The learnable parameter tensor
        gradient (Tensor): The learnable parameter's gradient tensor
        learning_rate (float): The learning rate

    Return:
        The updated learnable parameter
    """
    learnable_parameter -= learning_rate * gradient
    return learnable_parameter


def ridge_regression_update(
    learnable_weight: Tensor,
    weight_gradient: Tensor,
    reg_strength=reg.REG_STRENGTH,
    learning_rate=LEARNING_RATE
) -> Tensor:
    """
    Perform a ridge regression update on the learnable parameter using its gradient.

    Args:
        learnable_weight (Tensor): The learnable weight tensor
        weight_gradient (Tensor): The weight gradient tensor
        reg_strength (float): The regularization strength
        learning_rate (float): The learning rate

    Return:
        The updated weight tensor
    """
    return basic_update(
        learnable_parameter=learnable_weight,
        gradient=(weight_gradient + 2 * reg_strength),
        learning_rate=learning_rate
    )


def update_all(
    function: Callable,
    function_kwargs: dict[str, Any],
    learnable_parameters: dict[str, Tensor],
    gradients: dict[str, Optional[Tensor]]
) -> bool:
    """
    Update all learnable parameters using the provided update function.

    Args:
        function (Callable): The update function pointer
        function_kwargs (dict[str, Any]): The function keyword arguments dict
        learnable_parameters (dict[str, Tensor]): The learnable parameters dict
        gradients (dict[str, Tensor]): The gradients dict

    Return:
        Boolean indicating if there was an update performed
    """
    has_updated = False

    # Iterate through the learnable parameters
    for lp_name, learnable_parameter in learnable_parameters:
        # Get the corresponding gradient for the learnable parameter
        gradient_name = lp_name+'_grad'

        # Check if the gradient name exists in the gradients
        if gradient_name in gradients:
            # Get the gradient for the learnable parameter
            gradient = gradients[gradient_name]

            # Update the learnable parameter using its gradient and the
            #   provided function parameters
            function_kwargs = function_kwargs | {
                'learnable_param': learnable_parameter,
                'gradient': gradient
            }
            learnable_parameters[lp_name] = function(**function_kwargs)

            # Set has_updated to True
            has_updated = True

    # Return boolean indicating if at least one update occurred
    return has_updated