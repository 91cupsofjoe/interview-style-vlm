"""
This module handles tensor operations related to images.
"""
from typing import Optional
from collections.abc import Callable
from PIL import Image

import torch
from torch import Tensor
from torchvision import transforms

from log import logger as log


def _get_image_tensor(
    image_filepath: str,
    preprocessor: Optional[transforms.Compose]=None
) -> Optional[Tensor]:
    """
    Convert the specified image to an image tensor

    Args:
        image_filename (str): The filepath of the image file to convert

    Return:
        The image tensor
    """
    try:
        # Load the image from the specified image filepath
        image = Image.open(image_filepath, 'r').convert('RGB')

        # Check if the preprocessor was provided
        if preprocessor is not None:
            # Use the preprocessor to transform the image into an RGB tensor
            image_tensor = preprocessor(image)
        
        else:
            # Convert the image data into an rgb tensor
            image_tensor = transforms.ToTensor()(image)

        assert(isinstance(image_tensor, Tensor))
        return image_tensor
    
    except:
        # Return None since loading the specified image failed
        return None
        

def _get_images_tensor(image_filepaths: list[str]) -> Optional[Tensor]:
    """
    Convert the specified images to an images tensor

    Args:
        image_fileoaths (list[str]): The filepaths of the image files to convert

    Return:
        The images tensor
    """
    # Get list of image tensors
    image_tensors_list = [
        image_tensor for image_tensor in
        [
            _get_image_tensor(image_filepath)
                for image_filepath in image_filepaths
        ]
        if image_tensor is not None
    ]

    # Check if the list contains any image tensors
    if len(image_tensors_list):
        # Stack all image tensors into one images tensor and return it
        return torch.stack(image_tensors_list)
    
    # Else, log warning and return None
    log._log_warning(
        "Could not load images tensor since no image tensors were loaded "
        "from the image filepaths!",
        log.IMAGE_MODULE
    )


def _get_RGB_mean_std(images_tensor: Tensor):
    """
    Calculate and return the mean and standard deviation of RGB values across
        the image tensors in the data tensor. Used for image preprocessing.

    Args:
        images_tensor (Tensor): The tensor to calculate mean and std from

    Return:
        mean (float): The mean value across all RGB values
        std (float): The standard deviation across all RGB values
    """
    # Transpose the images tensor to get the RGB tensor across all images
    rgb_tensor = images_tensor.transpose(0, 1)

    # Get each channel of the RGB tensor 
    red_values = rgb_tensor[0]
    green_values = rgb_tensor[1]
    blue_values = rgb_tensor[2]

    # Calculate mean and std for each RGB channel across all images
    mean = [
        red_values.sum() / red_values.numel(),
        green_values.sum() / green_values.numel(),
        blue_values.sum() / blue_values.numel()
    ]
    std = [
        ((red_values - mean[0])**2).sum() / red_values.numel(),
        ((green_values - mean[1])**2).sum() / green_values.numel(),
        ((blue_values - mean[2])**2).sum() / blue_values.numel()
    ]

    # Return the mean and std of the image tensor's RGB values
    return mean, std

MEAN = 0
STD = 1

def _get_preprocessor(
    resize: Optional[int]=None,
    crop_size: Optional[int]=None,
    mean=MEAN, std=STD
) -> transforms.Compose:
    """
    Return a preprocessing function with the specified parameters.

    Args:
        resize (int): Resize dimensions of the preprocessed image
        crop_size (int): Cropped dimensions of the preprocessed image
        mean (float): Mean values for the RGB channels
        std (float): Standard deviation values for each of the RGB channels

    Return:
        Preprocessor function
    """
    # Set transformation operations list
    transform_ops = []

    # Resize
    if resize is not None:
        transform_ops.append(transforms.Resize(resize))

    # Center crop
    if crop_size is not None:
        transform_ops.append(transforms.CenterCrop(crop_size))

    # Tensor conversion
    transform_ops.append(transforms.ToTensor())

    # Normalization using the mean and std for RGB values
    transform_ops.append(
        transforms.Normalize(mean=mean, std=std)
    )

    # Return the preprocessing function with the specified parameters
    return transforms.Compose(transform_ops)