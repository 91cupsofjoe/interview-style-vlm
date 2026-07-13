"""
This module handles reading data from input file.
"""
from typing import Optional, Any
import json
import sys
from pathlib import Path

from log import logger as log

# Datasets directory
DATASETS_DIR = f'{Path(__file__).parent}/../../resources/datasets/'

# MS Coco directory
MS_COCO_DIR = f'{DATASETS_DIR}MS Coco/'

# Annotations filepaths
dataset_paths = {
    'MS Coco' : {
        'validation_2017' : {
            'images': MS_COCO_DIR + 'val2017/',
            'captions': MS_COCO_DIR + 'annotations/captions_val2017.json',
            'instances': MS_COCO_DIR + 'annotations/instances_val2017.json'
        },
        'train_2017' : {
            'images': MS_COCO_DIR + 'train2017/',
            'captions': MS_COCO_DIR + 'annotations/captions_train2017.json',
            'instances': MS_COCO_DIR + 'annotations/instances_train2017.json'
        }
    }
}


def _get_ms_coco_dataset(
    dataset_type: str
) -> Optional[ tuple[ list[str], list[dict[str, int]], list[list[str]] ] ]:
    """
    Return image filepaths and captions for images in the MS Coco dataset.

    Args:
        dataset_type (str): The type of images/annotations for the MS Coco data
        images_dir (str): The images directory

    Return:
        images_filepaths (list[str)]): List of image filepaths
        context_vectors (list[dict[str, int]]): List of context vector dicts
        captions (list[list[str[]]): List of caption lists
        NOTE: Each image will have its own context vector and list of captions
    """
    # Get the MS Coco dataset paths
    ms_coco_dataset_paths = dataset_paths['MS Coco']
    
    # Check if the dataset type is in the MS Coco annotations
    if dataset_type in ms_coco_dataset_paths:
        # Get the MS Coco captions and instances filenames and the images directory
        captions_filename = ms_coco_dataset_paths[dataset_type]['captions']
        instances_filename = ms_coco_dataset_paths[dataset_type]['instances']
        images_dir = ms_coco_dataset_paths[dataset_type]['images']

    # Try loading the MS Coco captions file
    try:
        with open(captions_filename, 'r') as captions_file:
            captions_file_contents = json.load(captions_file)

    except:
        # Log error and return None since the captions file failed to load
        log._log_error(
            "Could not load the MS Coco captions file!",
            log.DATAREADER_MODULE
        )
        return None

    # Try loading the MS Coco instances file
    try:
        # Get the instances file contents
        with open(instances_filename, 'r') as instances_file:
            instances_file_contents = json.load(instances_file)

    except:
        # Log error and return None since the instances file failed to load
        log._log_error(
            "Could not load the MS Coco instances file!",
            log.DATAREADER_MODULE
        )
        return None
    
    '''
    For the captions file:

        Each images entry has:
            keys = [
                license (int),
                file_name (str),
                coco_url (str),
                height (int),
                width (int),
                date_captured (datetime str),
                flickr_url (str),
                id (int) = file_name but without the leading zeros
            ]
            keys to extract = [file_name, id]

        Each annotations entry has:
            keys = [
                image_id (int) = images/id (see above),
                id (int),
                caption (str)
            ]
            keys to extract = [image_id, caption]
            NOTE: An image can have multiple captions, but the captions are NOT
                organized by image id


    For the instances file:

        Each annotations entry has:
            keys = [
                segmentation (list[float]),
                area (float),
                is_crowd (int),
                image_id (int) = images/id in the captions file (see above), 
                bbox (list[float]),
                category_id (int),
                id (int)
            ]
            keys to extract = [image_id, category_id]

        Each categories entry has:
            keys = [
                supercategory (str),
                id (int),
                name (str)
            ]

        NOTE: These are the category ids used:
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 27, 28,
            31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 46, 47, 48, 49, 50,
            51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
            61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80,
            81, 82, 84, 85, 86, 87, 88, 89, 90]
    '''

    # Get the image filepaths list
    image_filepaths = [
        images_dir + image_info['file_name'] for image_info
            in captions_file_contents['images']
    ]

    # Update the image filepaths to only include existing filepaths and sort it
    image_filepaths = _get_valid_image_filepaths(image_filepaths)
    image_filepaths.sort()

    '''
    # Sanity check
    for i in range(len(image_filepaths)):
        print(f'\nImage # {i+1}:{image_filepaths[i]}', end='')
    '''

    # Get the image ids from the updated image filepaths
    image_ids = [
        int(image_filepath.rsplit('/', 1)[1].split('.')[0])
            for image_filepath in image_filepaths
    ]

    # Get the caption annotations
    caption_annotations = captions_file_contents['annotations']

    # Get the instance annotations
    instance_annotations = instances_file_contents['annotations']

    # Get the instance categories from the instances file
    instance_categories = instances_file_contents['categories']

    # Initialize the context data
    # NOTE: Each image id has its own context vector
    context_data_dict = {
        image_id: {
            category['name']: 0 for category in instance_categories
        } for image_id in image_ids
    }

    # Set list of indices from the used categories (see above)
    used_cat_indices = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 27, 28,
        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
        61, 62, 63, 64, 65, 67, 70,
        72, 73, 74, 75, 76, 77, 78, 79, 80,
        81, 82, 84, 85, 86, 87, 88, 89, 90
    ]

    # Fill the context data from the instance annotations
    for annotation in instance_annotations:
        # Get the image id
        image_id = annotation['image_id']
        # Check if the image id corresponds to a valid image filepath
        if image_id in image_ids:
            # Get the context vector for the image id
            context_vector = context_data_dict[image_id]

            # Get the category name from the category id
            cat_id = annotation['category_id']
            # Get the used category index
            used_cat_idx = used_cat_indices.index(cat_id)
            # Get the category name
            cat_name = instance_categories[used_cat_idx]['name']

            # Set the count for the category name to 1
            context_vector[cat_name] = 1
    
    # Initialize the context vectors and captions list
    context_data_list = []
    captions = []

    # Iterate through the image ids to get the context vectors and captions
    for image_id in image_ids:
        # Append the context vector of the image id
        context_data_list.append(context_data_dict[image_id])

        # Get the list of captions for the image id
        captions.append([
            annotation['caption'] for annotation in caption_annotations
                if annotation['image_id'] == image_id
        ])

    # Return the image filepaths, context data (list), and captions
    return image_filepaths, context_data_list, captions


def _save_dataset_items(
    dataset_items: tuple[Any, ...],
    dataset_filenames: tuple[str, ...]
) -> bool:
    """
    Store the dataset items into the specified files.

    Args:
        dataset_items (tuple[Any, ...]): Tuple of dataset items
        dataset_filenames (tuple[str, ...]): Tuple of dataset filenames

    Return:
        Boolean indicating success with saving the dataset items
    """
    # Check if are filenames for all dataset item
    if len(dataset_items) != len(dataset_filenames):
        # Log error and return False since the number of dataset filenames do
        #   do not match that of the dataset items
        log._log_error(
            "Could not save the dataset items since the number of dataset "
            "filenames does not equal the number of dataset items!",
            log.DATAREADER_MODULE
        )
        return False
    
    # Else, savethe dataset items to their respective files
    for i in range(len(dataset_items)):
        try:
            with open(dataset_filenames[i], 'w') as dataset_file:
                json.dump(dataset_items[i], dataset_file)
        except:
            # Log error and return False since saving to the dataset
            #   file failed
            log._log_error(
                "Failed to save to the dataset file!",
                log.DATAREADER_MODULE
            )
            return False

    # Return True since all dataset items were successfully saved
    return True


def _get_valid_image_filepaths(image_filepaths: list[str]) -> list[str]:
    """
    Return the list of valid image filepaths.

    Args:
        image_filepaths (list[str]): List of image filepaths to check

    Return:
        valid_image_filepaths (str): List of valid image filepaths
    """
    # Initialize the valid image filepaths list
    valid_image_filepaths = []

    # Iterate through the image filepaths
    for image_filepath in image_filepaths:
        # Append the image filepath if it exists
        if Path(image_filepath).exists():
            valid_image_filepaths.append(image_filepath)

    # Return the list of valid image filepaths
    return valid_image_filepaths