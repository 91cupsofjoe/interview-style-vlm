"""
This module handles testing the image encoder.
"""
from typing import Optional, Any

from torch import Tensor

from data import data_reader as dr
from function import \
    convolution as conv, pool, regularization as reg, update
from log import logger as log
from model import model as ml
from model.image_encoder import ImageEncoder

TRAIN_FRACTION = .1

def get_image_encoder(
    # Input files
    dataset_items: Optional[tuple[Any, ...]]=None,
    images_filename: Optional[str]=None,
    context_data_filename: Optional[str]=None,
    image_filepaths: Optional[list[str]]=None,
    context_data: Optional[list[dict[str, int]]]=None,
    training_test_split=.6,
    model_data_filename: Optional[str]=None,
) -> Optional[ImageEncoder]:
    """
    Get a preset image encoder.

    Args:
        None

    Return:
        A preset image encoder
    """
    # Check if the dataset items were provided
    if dataset_items is not None:
        # Get the image filepaths and context data from the MS Coco dataset
        image_filepaths, context_data, _ = dataset_items

    # Get the base model hyperparameters
    base_model_hyperparameters = {
        'num_folds': ml.NUM_FOLDS,
        'batch_size': ml.BATCH_SIZE,
        'num_epochs': ml.NUM_EPOCHS,
        'learning_rate': update.LEARNING_RATE,
        'reg_type': reg.REG_TYPE,
        'reg_strength': reg.REG_STRENGTH,
    }

    # Get the CNN model hypeparameters
    cnn_model_hyperparameters = {
        'num_in_channels': ml.NUM_IN_CHANNELS,
        'num_out_features': ml.NUM_OUT_FEATURES,
        'kernel_size': conv.KERNEL_SIZE,
        'stride': conv.STRIDE,
        'padding': conv.PADDING,
        'pool_size': pool.KERNEL_SIZE,
        'pool_stride': pool.STRIDE,
        'pool_type': pool.POOL_TYPE,
    }

    return ImageEncoder(
        **cnn_model_hyperparameters,
        base_model_hyperparameters=base_model_hyperparameters,
        images_filename=images_filename,
        image_filepaths=image_filepaths,
        context_data_filename=context_data_filename,
        context_data=context_data,
        training_test_split=training_test_split,
        model_data_filename=model_data_filename,
        object_name='image_encoder'
    )


def train_no_input_with_files_test(
    images_filename: str,
    context_vectors_filename: str,
    do_measure_accuracy=True,
    do_print_messages=True
) -> Optional[bool]:
    """
    Test the image encoder's training method with input files.
    NOTE: This entails not supplying input filenames when instantiating the
        the image encoder.

    Args:

    Return:
        Boolean indicating if training test succeeds or fails
    """
    image_encoder = get_image_encoder()

    # Make sure the image encoder loaded successfully
    if image_encoder is None:
        # Log error and return False since the image encoder failed to load
        log.log_error(
            "Could not test the image encoder since it failed to load!",
            log.TEST_MODULES
        )
        return None

    return image_encoder.train(
        data_sources=(images_filename, context_vectors_filename),
        training_test_split=.6,
        do_measure_accuracy=do_measure_accuracy,
        do_print_messages=do_print_messages,
        train_fraction=TRAIN_FRACTION
    )


def train_no_input_with_data_test(
    image_filepaths: list[str],
    context_data: list[dict[str, int]],
    do_measure_accuracy=True,
    do_print_messages=True
) -> Optional[bool]:
    """
    Test training the image encoder initialized without input files or data.
    NOTE: This entails not supplying input as parameters when instantiating the
        image encoder.

    Args:

    Return:
        Boolean indicating if training test succeeds or fails
    """
    image_encoder = get_image_encoder()
    
    # Make sure the image encoder loaded successfully
    if image_encoder is None:
        # Log error and return False since the image encoder failed to load
        log.log_error(
            "Could not test the image encoder since it failed to load!",
            log.TEST_MODULES
        )
        return None

    return image_encoder.train(
        examples_data=image_filepaths,
        labels_data=context_data,
        training_test_split=.6,
        do_measure_accuracy=do_measure_accuracy,
        do_print_messages=do_print_messages,
        train_fraction=TRAIN_FRACTION
    )


def train_no_input_with_tensors_test(
    examples_data_tensor: Tensor,
    labels_data_tensor: Tensor,
    do_measure_accuracy=True,
    do_print_messages=True
) -> Optional[bool]:
    """
    Test the image encoder's training method with data tensors.
    NOTE: This entails not supplying input parameteres when instantiating the
        the image encoder. Only supply data tensors to the train method.

    Args:

    Return:
        Boolean indicating if training test succeeds or fails
    """
    image_encoder = get_image_encoder()

    # Make sure the image encoder loaded successfully
    if image_encoder is None:
        # Log error and return False since the image encoder failed to load
        log.log_error(
            "Could not test the image encoder since it failed to load!",
            log.TEST_MODULES
        )
        return None

    return image_encoder.train(
        training_examples_tensor=examples_data_tensor,
        target_labels_tensor=labels_data_tensor,
        training_test_split=.6,
        do_measure_accuracy=do_measure_accuracy,
        do_print_messages=do_print_messages,
        train_fraction=TRAIN_FRACTION
    )


def train_input_data_test(
    image_filepaths: list[str],
    context_data: list[dict[str, int]],
    do_measure_accuracy=True,
    do_print_messages=True
) -> Optional[bool]:
    """
    Test the image encoder's training method with input files.
    NOTE: This entails supplying input data as parameters when instantiating
        the image encoder.

    Args:

    Return:
        Boolean indicating if training test succeeds or fails
    """
    image_encoder = get_image_encoder(
        image_filepaths=image_filepaths,
        context_data=context_data,
        training_test_split=.6
    )

    # Make sure the image encoder loaded successfully
    if image_encoder is None:
        # Log error and return False since the image encoder failed to load
        log.log_error(
            "Could not test the image encoder since it failed to load!",
            log.TEST_MODULES
        )
        return None

    return image_encoder.train(
        do_measure_accuracy=do_measure_accuracy,
        do_print_messages=do_print_messages,
        train_fraction=TRAIN_FRACTION
    )


if __name__ == '__main__':
    # Load the MS Coco dataset
    ms_coco_dataset_items = dr.get_ms_coco_dataset('validation_2017')

    # Check if the MS Coco dataset items exist
    if ms_coco_dataset_items is not None:
        # Get the names of the formatted MS Coco JSON files
        ms_coco_dataset_filenames = (
            'ms_coco_images_filepaths.json',
            'ms_coco_context_data.json',
            'ms_coco_captions.json'
        )

        # Save the MS Coco dataset items to their respective dataset JSON files
        ms_coco_filenames = dr.save_dataset_items(
            ms_coco_dataset_items, ms_coco_dataset_filenames
        )

        # Get the input data
        image_filepaths, context_data, _ = ms_coco_dataset_items

        # Get the input filenames
        images_filename, context_data_filename, _ = ms_coco_dataset_filenames

        # Test the image encoder training use input data
        print(f"\nImage encoder training using input data: "
              f"{train_input_data_test(image_filepaths, context_data)}")

    else:
        # Log error since the MS Coco dataset items failed to load
        log.log_error(
            "Could not test image encoder since the MS Coco dataset items " 
            "failed to load!",
            log.TEST_MODULES
        )