"""
This module handles utility functions.
"""
from typing import Optional, Any, Union
from datetime import datetime
import time
from pynput.keyboard import Listener, Key
import json

from torch import Tensor
from log import logger as log


# ============================= SYSTEM METHODS ================================

DATE_STR = '%m/%d/%y'
TIME_STR = '%H:%M:%S'
DATETIME_STR = DATE_STR + ' ' + TIME_STR

def _get_current_time(use_date=False):
    """
    Return the current time

    Args:
        use_date (boolean): Boolean to add the date in the time

    Return:
        None
    """
    formatted_time = TIME_STR
    if use_date:
        formatted_time = DATETIME_STR
    return datetime.now().strftime(formatted_time)


def _pause(seconds=None, key_type=None, use_wait_message=False):
    """
    Pause runtime for a specified amount of seconds

    Args:
        seconds (int): The amount of seconds to pause runtime

    Return:
        None
    """
    # Check if seconds is provided
    if seconds is None:
        # If seconds is not provided, pause runtime until keyboard input
        _wait_on_keyboard_input(
            key_type=key_type,
            use_wait_message=use_wait_message
        )

    # Else, pause runtime for the specified number of seconds (in milliseconds)
    else:
        time.sleep(seconds * 1000)


keyboard_keys = {
    'enter': Key.enter,
    'spacebar': Key.space
}

def _wait_on_keyboard_input(key_type=None, use_wait_message=False) -> None:
    # Show input message if specified
    if use_wait_message:
        print("\nPRESS ANY KEY TO CONTINUE...\n")
    
    # Set keyboard listener on_press method
    def _on_press(key) -> None:
        # Stop the keyboard listener if the correct key was pressed or if no
        #   key was specified
        if (key_type is not None and key == keyboard_keys[key_type]) \
                or key_type is None:
            keyboard_listener.stop()

    # Scan for keyboard input
    # NOTE: Make sure to use the same keyboard listener alias from on_press
    with Listener(on_press=_on_press) as keyboard_listener:
        keyboard_listener.join()


# ============================== FILE METHODS =================================

def _load_json(filename: str, format: Optional[str]=None) -> Optional[Any]:
    """
    Read in a JSON file and return its contents in the specified format.

    Args:
        filename (str): The name of the JSON file to read

    Return:
        The JSON contents in a the specified format
    """
    # Try reading the file and extracting the JSON contents
    try:
        with open(filename, 'r') as file:
            # Initially load the JSON contents as a dict
            json_file_contents = json.load(file)

            # Check if the JSON file contents are a dict
            if isinstance(json_file_contents, dict):

                # Check if a format was specified
                if format is not None:

                    # Check if the specified format is tuple
                    if format == 'tuple':
                        # Convert the JSON file contents to a list of tuples,
                        #   each containing a key and its respective value
                        json_file_contents = list(json_file_contents.items())

                    # Check if the specified format is values
                    elif format == 'values':
                        # Convert the JSON file contents dict to a list of values
                        json_file_contents = list(json_file_contents.values())

                # Else, the JSON file contents remain as a dict

            # Else, convert the JSON file contents to a list if not already
            else:
                json_file_contents = list(json_file_contents)

            # Return the json file contents
            return json_file_contents
        
    except:
        # Return None since extracting JSON file contents failed
        return None


# ============================= STRING METHODS ================================

def _parse_args(command_line_args: list[str], valid_keys: list[str]) -> dict[str, Any]:
    """
    Scan the command line arguments for matching valid keys and return a dict
        of valid arguments.

    Args:
        command_line_args (list[str]): The command line arguments to scan
        valid_keys (list[str]): The valid keys to look for

    Return:
        A dict of valid key and value pairs
    """
    parsed_args = {}

    # Iterate through command line arguments
    for arg in command_line_args:
        tokens = arg.split('=')

        # Check if argument followed the correct formatting
        if len(tokens) == 2:
            key, value = tokens

            # Check if key is valid; if so, add the argument entry
            if key in valid_keys:
                parsed_args[key] = value

    # Return the dict of valid key/value pairs
    return parsed_args


def _get_print_name(var_name: Optional[str]=None) -> str:
    """
    Return a variable name converted to print name.

    Args:
        var_name (str): The variable name

    Return:
        A print name
    """
    if var_name is None:
        return 'None'
    return var_name.replace('_', ' ')

# ============================ COLLECTION METHODS =============================

SPLIT_INDEX = .6

def _get_collection_indices_by_split(
    collection: Any,
    split_index=SPLIT_INDEX
) -> tuple[int, int, int, int]:
    """
    Return the valid indices for the collection.
    """
    # Make sure the split index is valid for the given collection
    if split_index < 0 or split_index > len(collection):
        split_index = int(len(collection))

    # Check if fractional splitting is specified
    elif split_index > 0 and split_index < 1:
        split_index = int(len(collection) * split_index)

    else:
        # Convert the split index to an integer
        split_index = int(split_index)
    
    # Return two sets of start and stop indices
    return 0, split_index, split_index, len(collection)


def _get_tuple(x, dim=2):
    """
    Take in a int or tuple and return a tuple of ints, one for each dimension

    Args:
        x (int or tuple[int]): The input to (possibly) convert to a tuple of ints
        dim (int): The number of elements for the return tuple
        
    Return:
        A tuple of ints
    """
    if isinstance(x, int):
        return tuple(
            [x for _ in range(dim)]
        )
    return x

LIGHT_BANNER = '-----------------------------------------------------------'
HARD_BANNER = '================================================================'


def _get_update_dict(
    dict_updates: dict,
    base_dict: Optional[dict]=None,
    ignore_none=False,
    update_none_only=False
) -> dict[str, Any]:
    """
    Return dict updates.

    Args:   
        dict_updates (dict[str, Any]): The dict updates
        base_dict (dict[str, Any]): The base dict
        ignore_none (bool): Boolean indicating if ignoring values equal to None
        fill_none_only (bool): Boolean indicating if only updating values equal to None

    Return:
        The updated keyword arguments
    """
    # Check if the base dict was provided
    if base_dict is None:
        # Set the base dict to the dict updates
        base_dict = dict_updates
    
    # Else, initialize the updated dict from the base dict
    updated_dict = base_dict.copy()

    # Iterate through the base dict to update valid entries
    for key, value in dict_updates.items():
        if not (ignore_none and value is None) \
                        and key in base_dict.keys():
            if not (update_none_only and base_dict[key] is not None):
                updated_dict[key] = value

    return updated_dict


def _print_dict(data: Optional[dict]=None, name: Optional[str]=None) -> None:
    """
    Print the data dict.

    Args:
        data (dict): Dict of items to print
        name (str): The data dict name

    Return:
        None
    """
    if name is None:
        name = f'{log.UNLABELED} DICT'
    if data is None:
        data = {}
    print(f"\n{LIGHT_BANNER}\ndict [{name}]:")
    for key, value in data.items():
        print(f"\nkey = {_get_element_str(key)}, "
              f"value = {_get_element_str(value)}", end='')
    print(f"\n{LIGHT_BANNER}")


def _print_list_tuple(
    data: Union[list, tuple, None]=None,
    name: Optional[str]=None
) -> None:
    """
    Print the data list or tuple.

    Args:
        data (list | tuple): List or tuple of items to print
        name (str): The data list/tuple name

    Return:
        None
    """
    if name is None:
        name = f'{log.UNLABELED} LIST/TUPLE'
    if data is None:
        data = []
    print(f"\n{LIGHT_BANNER}\nlist/tuple [{name}]:")
    for i in range(len(data)):
        element = data[i]
        print(f"\nelement at pos #{i} = {_get_element_str(element)}", end='')
    print(f"\n{LIGHT_BANNER}")


def _print_element(data: Optional[Any]=None, name: Optional[str]=None) -> None:
    """
    Print the data element.

    Args:
        data (Any): The data element
        name (str): The data element name

    Return:
        None
    """
    if isinstance(data, (list, tuple)):
        _print_list_tuple(data, name)
    if isinstance(data, (dict)):
        _print_dict(data, name)

    if name is None:
        name = f'{log.UNLABELED} ELEMENT'
    print(f"\n{LIGHT_BANNER}\nElement [{name}]: "
          f"{_get_element_str(data)}\n{LIGHT_BANNER}")


def _get_element_str(data: Optional[Any]) -> str:
    """
    Get the data element.

    Args:
        data (Any): The data element

    Return:
        None
    """
    if data is None:
        return 'None'
    if isinstance(data, Tensor):
        return f'tensor with shape {data.shape}'
    if isinstance(data, str):
        return f'\"{data}\"'
    if isinstance(data, list):
        return f'list with {len(data)} items'
    if isinstance(data, tuple):
        return f'tuple with {len(data)} items'
    if isinstance(data, dict):
        return f'dict with {len(data.items())} items'
    # Else
    return data