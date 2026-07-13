"""
This is a utility module for the dataset reader.
"""
from typing import Optional, Any
import json
import sys
from pathlib import Path


def iscontainer(item: Any) -> bool:
    """
    Determine if the item is a container or not.

    Args:
        item (Any): The query item

    Return:
        Boolean indicating if the item is a container
    """
    if isinstance(item, (list, tuple, dict)):
        return True
    # Else, the item is not a container
    return False
    

def key_search(base_dict: dict, search_key: Any, key_path='') -> Optional[Any]:
    """
    Find the corresponding key/value pair in the provided dict.

    Args:
        base_dict (dict): The base dict
        search_key: (Any): The search key

    Return:
        value (Any): The value corresponding to the search key
        key_path (str): The base dict's key path for the value
    """
    # Search the first level of keys for the value
    if search_key in base_dict:
        return base_dict[search_key], key_path
    
    # Else, search the next level of the base dict
    for parent_key, value in base_dict:
        # Update the key path with the parent key
        key_path = parent_key + '/' + key_path
        # Check if the value is a dict
        if isinstance(value, dict):
            # Search the value dict
            search_result = key_search(value, search_key, key_path)
            if search_result is not None:
                return search_result
            
    # Else, return None


def _flatten(
    base_dict: dict,
    flat_dict: Optional[dict]=None,
    key_path='',
    include_parent_keys=True
) -> tuple[dict, str]:
    """
    Flatten a potentially nested dict into a regular dict.

    Args:
        base_dict (dict): The base dict
        key_path (str): The base dict's key path for the values

    Return:
        flat_dict (dict): The flattened dict
    """
    # Initialize the flattened dict if not provided
    if flat_dict is None:
        flat_dict = {}

    # Iterate through the base dict
    for key, value in base_dict.items():
        # Set the key from the key path
        if key_path != '':
            key = '/' + key
        key = key_path + key
        
        # Check if value is a list or tuple
        if isinstance(value, (list, tuple)):
            # Convert the value list or tuple into a dict
            value_dict = {}

            # Iterate through the value elements
            for i in range(len(value)):
                # Add the element name/value entry to the value dict
                value_dict['element_'+str(i)] = value[i]

            # Update value
            value = value_dict

        # Check if value is a dict
        if isinstance(value, dict):
            # Get the flattened value dict
            value, key = _flatten(
                base_dict=value,
                flat_dict=flat_dict,
                key_path=key
            )

        if not isinstance(value, dict) or include_parent_keys:
            # Update the flat dict key with the value dict
            flat_dict[key] = value

    # Return the flattened base dict and the key path
    return flat_dict, key_path


def get_key_counts(flat_dict: dict, sort=True, reverse=False) -> dict:
    """
    Get the key counts for the flat dict.

    Args:
        data (dict): The flat dict with keys to count
        sort (bool): Boolean indicating if sorting the dict
        reverse (bool): Boolean indicating if reverse sorting

    Return:
        parent_key_counts (dict): The number of parent keys in the flat dict
    """
    # Get the key list
    key_list = [
        key for key in flat_dict.keys()
    ]

    # Sort the flat dict keys if specified
    if sort:
        key_list.sort(reverse=reverse)

    # Get the key counts
    parent_key_counts = {}
    for key in key_list:
        # Get the split keys
        split_keys = key.rsplit('/', 1)

        # Check if the split keys exist (there are parent and child keys)
        if split_keys is not None and len(split_keys) > 1:
            # Get the parent and child keys
            parent_key, child_key = split_keys
            # If the child key points to an element, include it in the parent key
            if 'element_' not in child_key:
                parent_key += '/' + child_key

        else:
            # Set the key as the parent key
            parent_key = key

        # Add the parent key if it isn't in the parent key counts
        if parent_key not in parent_key_counts.keys():
            parent_key_counts[parent_key] = 0

        # Increment the parent key count
        parent_key_counts[parent_key] += 1

    # Return the parent key counts
    return parent_key_counts


def analyze_dict(data: dict, report_single_values=True) -> None:
    """
    Analyze the dict and print its structure.
    
    Args:
        data (dict): The dict to analyze
        report_single_values (bool): Boolean indicating if reporting keys
            with only one element
        
    Return:
        None
    """
    # Flatten the data dict
    flat_data, _ = _flatten(data)

    parent_key_counts = get_key_counts(flat_data, sort=False)

    # For each key, print it and the number of elements it has
    for parent_key in parent_key_counts:
        count = parent_key_counts[parent_key]
        if count > 1:
            # Report the number of times a parent key appeared in the data dict
            print(f"\nFound key [{parent_key}] with "
                f"{count} elements!", end='')
        elif count == 1 and report_single_values:
            # Report the number of times a parent key appeared in the data dict
            print(f"\nFound key [{parent_key}] with "
                f"{count} element!", end='')


def analyze_json_file(json_filename: str, report_single_values=True) -> None:
    """
    Analyze the JSON file and print its structure.
    
    Args:
        json_filename (str): The name of the JSON file
        report_single_values (bool): Boolean indicating if reporting keys
            with only one element
        
    Return:
        None
    """
    try:
        with open(json_filename, 'r') as json_file:
            # Load the JSON file contents as the JSON head
            file_contents = json.load(json_file)

            # Analyze the file contents
            analyze_dict(file_contents, report_single_values)
            
    except:
        raise ValueError(f"\nCould not open the JSON file! FILENAME: {json_filename}")


# Datasets directory
DATASETS_DIR = str(Path(__file__).parent) + '/../../resources/datasets/'

# Datasets
MS_COCO_DIR = DATASETS_DIR + 'MS Coco/'

# Default filepaths
MS_COCO_CAPTIONS_VAL2017_JSON = MS_COCO_DIR + 'annotations/captions_val2017.json'
MS_COCO_INSTANCES_VAL2017_JSON = MS_COCO_DIR + 'annotations/instances_val2017.json'
MS_COCO_PERSON_KEYPOINTS_VAL2017_JSON = MS_COCO_DIR + 'annotations/person_keypoints_val2017.json'
MS_COCO_CAPTIONS_TRAIN2017_JSON = MS_COCO_DIR + 'annotations/captions_train2017.json'
MS_COCO_INSTANCES_TRAIN2017_JSON = MS_COCO_DIR + 'annotations/instances_train2017.json'
MS_COCO_PERSON_KEYPOINTS_TRAIN2017_JSON = MS_COCO_DIR + 'annotations/person_keypoints_train2017.json'

if __name__ == '__main__':
    # Set the default JSON filename
    json_filename = MS_COCO_INSTANCES_VAL2017_JSON

    # The first (and only) command line argument is the JSON file
    if len(sys.argv) > 1:
        json_filename = sys.argv[1]

    # First test flatten method
    test_dict = {
        'level_0' : {
            'level_1a' : [
                2, 3, 4, 5, 6
            ],
            'level_1b' : {
                'level_2' : (
                    {
                        'level_3a' : [
                            1, 3, 5, 7
                        ]
                    },
                    {
                        'level_3b' : '8'
                    },
                    {
                        'level_3c' : {
                            'level_4' : [
                                0, 1, 2, 3, 4, 5, 6, 7
                            ]
                        }
                    }
                )
            }
        }
    }
    test_dict_2 = {
        'level_0' : {
            'level_1a' : [
                2, 3, 4, 5, 6
            ],
            'level_1b' : '2'
        }
    }
    test_dict_3 = {
        'level_0' : 'base_case'
    }

    '''
    # Print the test dict info
    print("\nAnalyzing the data...")
    analyze_dict(test_dict)
    '''

    # Print the JSON info
    analyze_json_file(json_filename, report_single_values=True)
    
    print('\n')

    '''
    # Print the flattened test dict
    flat_test_dict, _ = _flatten(test_dict)
    print("Flattened dict:")
    for key, value in flat_test_dict.items():
        print(f"\nKey: {key}, Value: {value}")
    print()
    '''