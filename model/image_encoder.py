"""
This module is for the image encoder class.
"""
from typing import Optional, Any
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor

from data.dataset import ImageDataSet
from model.model import CNN
from function import image
from log import logger as log


RESIZE = 256 # Default image resizing dimension
CROP_SIZE = 224 # Default image cropping dimension

class ImageEncoder(CNN):
    """
    Class for the image encoder model (convolution neural network).
    """
    def __init__(self,
        cnn_model_hyperparameters: Optional[dict[str, Any]]=None,
        base_model_hyperparameters: Optional[dict[str, Any]]=None,
        image_dataset: Optional[ImageDataSet]=None,
        images_filename: Optional[str]=None,
        context_vectors_filename: Optional[str]=None,
        training_test_split=-1.0,
        model_data_filename: Optional[str]=None,
        object_name: Optional[str]=None
    ):
        # Set the log id for the image encoder
        self.log_id = log.set_log_id(object_name, log.IMAGEENCODER)

        # Initialize the preprocessor
        self.preprocessor = None

        # Initialize the image encoder image dataset if not provided
        if image_dataset is None:
            image_dataset = ImageDataSet()
        
        # Load images and context vectors if provided
        if images_filename is not None \
                        and context_vectors_filename is not None:
            image_dataset.load_data(
                data_sources=(images_filename, context_vectors_filename),
                labels_data_name='context_vectors',
                labels_data_tensor_function_name='get_tokens_tensor',
                training_test_split=training_test_split
            )

        # Initialize the CNN model hyperparameters if not provided
        if cnn_model_hyperparameters is None:
            cnn_model_hyperparameters = {}

        # Initialize the CNN architecture
        super().__init__(
            **cnn_model_hyperparameters,
            base_model_hyperparameters=base_model_hyperparameters,
            dataset=image_dataset,
            model_data_filename=model_data_filename,
            object_name=object_name, has_log_id=True
        )

    def get_preprocessed_image(self,
        image_filepath: str,
        resize: Optional[int]=RESIZE,
        crop_size: Optional[int]=CROP_SIZE
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
        if self.preprocessor is None:
            self.preprocessor = image.get_preprocessor(
                resize=resize, crop_size=crop_size
            )

        # Open the image and convert it to a tensor
        image_tensor = image.get_image_tensor(image_filepath)

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
        resize: Optional[int]=None,
        crop_size: Optional[int]=None
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
                            resize=resize,
                            crop_size=crop_size
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