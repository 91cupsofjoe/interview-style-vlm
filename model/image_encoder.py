"""
This module is for the image encoder class.
"""
from typing import Optional, Any, Union
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor

from data.dataset import ImageDataSet
from model import layer
from model import model as ml
from model.model import CNN
from tensor_function import \
    image, \
    convolution as conv, loss, pool, regularization as reg, update, \
    util
from log import logger as log


RESIZE = 256 # Default image resizing dimension
CROP_SIZE = 224 # Default image cropping dimension

class ImageEncoder(CNN):
    """
    Class for the image encoder model (convolution neural network).
    """
    def __init__(self,
        # Image encoder model hyperparameters
        resize=RESIZE,
        crop_size=CROP_SIZE,

        # Base model hyperparameters
        batch_size: Optional[int]=None,
        num_pred_labels: Optional[int]=None,
        num_folds: Optional[int]=None,
        num_epochs: Optional[int]=None,
        eps: Optional[float]=None,
        patience: Optional[int]=None,
        reg_type: Optional[str]=None,
        reg_strength: Optional[float]=None,
        learning_rate: Optional[float]=None,

        # CNN model hyperparameters
        num_in_channels: Optional[int]=None,
        num_out_features: Optional[int]=None,
        kernel_size: Union[int, tuple[int, ...], None]=None,
        stride: Optional[int]=None,
        padding: Union[int, tuple[int, ...], None]=None,
        pool_size: Union[int, tuple[int, ...], None]=None,
        pool_stride: Union[int, tuple[int, ...], None]=None,
        pool_type: Optional[str]=None,
        embedding_size: Optional[int]=None,

        # Image encoder parameters
        dataset: Optional[ImageDataSet]=None,
        images_filename: Optional[str]=None,
        image_filepaths: Optional[list[str]]=None,
        context_data_filename: Optional[str]=None,
        context_data: Optional[list[dict[str, int]]]=None,
        training_test_split=-1.0,
        use_test_only=False,
        model_data_filename: Optional[str]=None,
        cnn_model_settings_filename: Optional[str]=None,
        object_name: Optional[str]=None,
        override_model_settings=False,
        do_print_layer_attr=False,
        do_print_tensor_function_attr=False
    ):
        # Set the log id for the image encoder
        self.log_id = log._set_log_id(object_name, log.IMAGEENCODER)

        # Make debug log for loading the image encoder
        log._log_debug(
            "Loading the image encoder...",
            log.IMAGE_ENCODER_MODULE
        )

        # Initialize the preprocessor
        self.preprocessor = None

        # Initialize the image encoder image dataset if not provided
        if dataset is None:
            dataset = ImageDataSet(object_name='image encoder dataset')

        # Set the args for the images tensor function
        images_tensor_function_args = {
            'image_filepaths': None,
            'resize': resize,
            'crop_size': crop_size
        }

        # Set the return value keys for the images tensor function
        images_tensor_function_return_value_keys = ('images_tensor',)

        # Set the args for the context vectors tensor function
        context_vectors_tensor_function_args = {'context_data': None}

        # Set the return value keys for the context vectors tensor function
        context_vectors_tensor_function_return_value_keys = ('context_vectors_tensor',)
        
        # Initialize the data sources
        data_sources = None

        # Set data sources to the images and context data filenames if provided
        if (images_filename is not None \
                        and context_data_filename is not None):
            data_sources = (images_filename, context_data_filename)

        # Load the data into the dataset if data sources or both the image and
        #   context data filenames exist
        if data_sources is not None \
            or (image_filepaths is not None
                        and context_data is not None):
            dataset.load_data(
                data_sources=data_sources,
                examples_data=image_filepaths,
                examples_data_name='image_filepaths',
                examples_data_tensor_function_ptr=self.get_preprocessed_images,
                examples_data_tensor_function_args=images_tensor_function_args,
                examples_data_tensor_function_return_value_keys=\
                    images_tensor_function_return_value_keys,
                labels_data=context_data,
                labels_data_name='context_data',
                labels_data_tensor_function_ptr=self.get_context_vectors,
                labels_data_tensor_function_args=context_vectors_tensor_function_args,
                labels_data_tensor_function_return_value_keys=\
                    context_vectors_tensor_function_return_value_keys,
                training_test_split=training_test_split,
                use_test_only=use_test_only
            )

        # Get the CNN model hyperparameters
        cnn_model_hyperparameters = util._get_update_dict(
            locals(), ml.cnn_model_default_hyperparameters
        )

        # Initialize the CNN architecture
        super().__init__(
            **cnn_model_hyperparameters,
            dataset=dataset,
            model_data_filename=model_data_filename,
            cnn_model_settings_filename=cnn_model_settings_filename,
            object_name=object_name, has_log_id=True,
            override_model_settings=override_model_settings,
            do_print_layer_attr=do_print_layer_attr,
            do_print_tensor_function_attr=do_print_tensor_function_attr
        )

        # Check if the image encoder has layers
        if len(self.layers) > 0:
            # Make debug log for successfully loading the image encoder
            log._log_debug(
                "Successfully loaded the Transformer model!",
                log.IMAGE_ENCODER_MODULE
            )

        else:
            # Make warning log for loading the image encoder model
            log._log_warning(
                "Partially loaded the Transformer model! DO NOT USE!",
                log.IMAGE_ENCODER_MODULE
            )

    def get_preprocessed_image(self,
        image_filepath: str,
        resize=RESIZE,
        crop_size=CROP_SIZE
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
            self.preprocessor = image._get_preprocessor(
                resize=resize, crop_size=crop_size
            )

        # Open the image, preprocess it, and convert it to an RGB tensor
        image_tensor = image._get_image_tensor(image_filepath, self.preprocessor)
        
        # Check if the image tensor exists
        if image_tensor is not None:
            # Return the image tensor
            return image_tensor
        
        # Else, log error and return None since loading the image tensor failed
        log._log_error(
            "Could not preprocess the image since loading the image tensor failed!",
            self.log_id
        )
        
    def get_preprocessed_images(self,
        image_filepaths: list[str],
        resize=RESIZE,
        crop_size=CROP_SIZE
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
        '''
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
        '''
        preprocessed_images = [
            tensor for tensor
                in [
                    self.get_preprocessed_image(
                        image_filepath=image_filepath,
                        resize=resize,
                        crop_size=crop_size
                    )
                    for image_filepath in image_filepaths
                ]
                if tensor is not None
        ]

        # Check if there are no preprocessed images
        if len(preprocessed_images) == 0:
            # Log error and return None since there are no preprocessed images
            log._log_error(
                "No images were successfully preprocessed!",
                self.log_id
            )

        else:
            '''
            print("List type:", type(preprocessed_images))
            print("List length:", len(preprocessed_images))
            print("Unique shapes:", {tuple(t.shape) for t in preprocessed_images})
            print("All tensors:", all(isinstance(t, torch.Tensor) for t in preprocessed_images))
            print("All contiguous:", all(t.is_contiguous() for t in preprocessed_images))
            '''
            preprocessed_images_tensor = torch.stack(preprocessed_images)
            return preprocessed_images_tensor

    def get_context_vector(self, context_data: dict[str, int]) -> Tensor:
        """
        Convert context data (single entry) into a context vector.

        Args:
            context_data (dict[str, int]): Dict of context data

        Return:
            A context vector tensor
        """
        return torch.tensor(list(context_data.values()))


    def get_context_vectors(self, context_data: list[dict[str, int]]) -> Tensor:
        """
        Convert context data (multiple entries) into a context vectors.

        Args:
            context_data (list[dict[str, int]]): List of context data dicts

        Return:
            A context vector tensor
        """
        return torch.stack([
            self.get_context_vector(data_dict)
                for data_dict in context_data
        ])
    

def get_image_encoder(
    # Image encoder model hyperparameters
    resize=RESIZE,
    crop_size=CROP_SIZE,

    # Base model hyperparameters
    batch_size: Optional[int]=None,
    num_pred_labels: Optional[int]=None,
    num_folds: Optional[int]=None,
    num_epochs: Optional[int]=None,
    eps: Optional[float]=None,
    patience: Optional[int]=None,
    reg_type: Optional[str]=None,
    reg_strength: Optional[float]=None,
    learning_rate: Optional[float]=None,

    # CNN model hyperparameters
    num_in_channels: Optional[int]=None,
    num_out_features: Optional[int]=None,
    kernel_size: Union[int, tuple[int, ...], None]=None,
    stride: Optional[int]=None,
    padding: Union[int, tuple[int, ...], None]=None,
    pool_size: Union[int, tuple[int, ...], None]=None,
    pool_stride: Union[int, tuple[int, ...], None]=None,
    pool_type: Optional[str]=None,
    embedding_size: Optional[int]=None,

    # General image encoder parameters
    dataset: Optional[ImageDataSet]=None,
    images_filename: Optional[str]=None,
    context_data_filename: Optional[str]=None,
    image_filepaths: Optional[list[str]]=None,
    context_data: Optional[list[dict[str, int]]]=None,
    training_test_split=.6,
    model_data_filename: Optional[str]=None,
    cnn_model_settings_filename: Optional[str]=None,
    object_name: Optional[str]=None,
    do_print_layer_attr=False,
    do_print_tensor_function_attr=False
) -> Optional[ImageEncoder]:
    """
    Return an image encoder using the specified arguments.

    Args:
        resize (int): The image resize dimension
        crop_size (int): The image crop size dimension
        
        batch_size (int): The training batch size
        num_pred_labels (int): The number of labels to predict
        num_folds (int): The number of crossvalidation folds
        num_epochs (int): The number of epochs to train
        eps (float): The loss threshold value
        patience (int): The amount of times eps can be reached before
            training is stopped
        reg_type (str): The regularization type
        reg_strength (float): The regularization strength
        learning_rate (float): The model learning rate

        num_in_channels (int): The number of input channels
        num_out_featurs (int): The number of output features
        kernel_size (int | tuple[int, ...]): The kernel size dimension(s)
        stride (int): The kernel window movement increment
        padding (int | tuple[int, ...]): The number of extra values around
            the kernel window
        pool_size (int | tuple[int, ...]): The kernel size for pooling
        pool_stride (int): The pool window movement increment
        pool_type (str): The type of pooling operation
        embedding_size (int): The number of embeddings per image pixel

        dataset_items (tuple[Any, ...]): Tuple of dataset items
        images_filename (str): The name of the file with the image filepaths
        context_data_filename (str): The name of the file with the context data
        image_filepaths (list[str]): List of image filepaths
        context_data (list[dict[str, int]]): List of context data dicts
        training_test_split (float): The index or percentage used to split the
            training and test data
        model_data_filename (str): The name of the file with the model data
        cnn_model_settings_filename (str): The name of the file with the cnn
            model settings
        do_print_layer_attr (bool): Boolean indicating if printing layer attributes
        do_print_tensor_function_attr (bool): Boolean indicating if printing
            tensor function attributes

    Return:
        An image encoder
    """
    # Make info log for getting the image encoder
    log._log_info(
        "Getting the image encoder...",
        log.IMAGE_ENCODER_MODULE
    )

    # Get the image encoder
    image_encoder = ImageEncoder(**locals())

    # Check if the image encoder has any layers
    if len(image_encoder.layers) > 0:
        # Make info log for successfully retrieving the CNN model
        log._log_info(
            "Successfully retrieved the image encoder!",
            log.IMAGE_ENCODER_MODULE
        )
        return image_encoder

    # Else, make error log for failure to retrieve the image encoder, and
    #   return None
    log._log_error(
        "Failed to retrieve the image encoder!",
        log.IMAGE_ENCODER_MODULE
    )