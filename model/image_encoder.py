from typing import Optional
import math
from pydantic import BaseModel

import torch
from torch import Tensor, Size
import torch.nn.functional as nnf

from tensor import tensor_ops
from model.model import CNN
from log import logger as log


class ImageEncoder(BaseModel, CNN):
    """
    Class for the image encoder model (convolution neural network).
    """
    def __init__(self):
        # Initialize the preprocessor
        self.preprocessor = None
        pass

    def __len__(self):
        return

    def get_preprocessed_image(self,
        image_filepath: str,
        preprocess_params: Optional[dict]=None,
    ) -> Optional[Tensor]:
        """
        Preprocess and return a transformed image tensor.

        Args:
            image_filepath (str): The filepath for the image to preprocess
            preprocess_params (dict[str, int]): Parameters for image preprocessing

        Return:
            The preprocessed image tensor
        """
        # Set the preprocesser if preprocessing parameters are provided
        #   or if the preprocesser hasn't been initialized
        if self.preprocessor is None or preprocess_params is not None:
            self.preprocessor = tensor_ops.get_preprocessor(preprocess_params)

        # Open the image and convert it to a tensor
        image_tensor = tensor_ops.get_image_tensor(image_filepath)

        # Check if loading the image tensor was successful
        if image_tensor is not None:
            # Return the preprocessed image
            return self.preprocessor(image_tensor)
        
        # Else, log error and return None since loading the image tensor failed
        log.log_error(
            "Could not preprocess the image since loading the image tensor failed!",
            self.log_id
        )
        
    def get_preprocessed_images(self,
        image_filepaths: list[str],
        preprocess_params=None,
    ) -> Optional[Tensor]:
        """
        Preprocess and return a batch of transformed image tensors from the
            training examples tensor.

        Args:
            
            image_filepaths (list[str]): The list of image filepaths for
                images to preprocess
            preprocess_params (dict[str, int]): Parameters for image preprocessing

        Return:
            The preprocessed images batch
        """
        # Load and stack the preprocessed images into one tensor
        preprocessed_images = torch.stack(
            [
                tensor for tensor
                    in [
                        self.get_preprocessed_image(
                            image_filepath=image_filepath,
                            preprocess_params=preprocess_params
                        )
                        for image_filepath in image_filepaths
                    ] if tensor is not None
            ], dim=0 # along the batch size dimension
        )

        # Check if there are any preprocessed images
        if preprocessed_images.shape == 0:
            # Log error and return None since there are no preprocessed images
            log.log_error(
                "No images were preprocessed",
                self.log_id
            )