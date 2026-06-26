"""
This module handles utility functions.
"""
from typing import Optional, Any
from datetime import datetime
import time
from pynput.keyboard import Listener, Key
import json


# ============================= SYSTEM METHODS ================================

def get_current_time(use_date=False):
    """
    Return the current time

    Args:
        use_date (boolean): Boolean to add the date in the time

    Return:
        None
    """
    date_str = ''
    if use_date:
        date_str = '%m/%d/%y '
    return datetime.now().strftime(date_str + '%H:%M:%S')


def pause(seconds=None, key_type=None, use_wait_message=False):
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
        wait_on_keyboard_input(
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

def wait_on_keyboard_input(key_type=None, use_wait_message=False) -> None:
    # Show input message if specified
    if use_wait_message:
        print("\nPRESS ANY KEY TO CONTINUE...\n")
    
    # Set keyboard listener on_press method
    def on_press(key) -> None:
        # Stop the keyboard listener if the correct key was pressed or if no
        #   key was specified
        if (key_type is not None and key == keyboard_keys[key_type]) \
                or key_type is None:
            keyboard_listener.stop()

    # Scan for keyboard input
    # NOTE: Make sure to use the same keyboard listener alias from on_press
    with Listener(on_press=on_press) as keyboard_listener:
        keyboard_listener.join()


# ============================== FILE METHODS =================================

def load_json(filename: str, format=None) -> Optional[Any]:
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
            json_contents = json.load(file)

            # Check if a format was specified
            if format is not None:

                # Check if the specified format is tuple
                if format == 'tuple':
                    # Convert the JSON contents dict to a list of tuples,
                    #   each containing a key and its respective value
                    json_contents = json_contents.items()

                # Check if the specified format is tuple
                if format == 'values':
                    # Convert the JSON contents dict to a list of values
                    json_contents = json_contents.values()

            # Else, the JSON contents remain as a dict

            # Return the json contents
            return json_contents
        
    except:
        # Return None since extracting JSON file contents failed
        return None


# ============================= STRING METHODS ================================

def parse_args(command_line_args: list[str], valid_keys: list[str]) -> dict[str, Any]:
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


# ============================ COLLECTION METHODS =============================

SPLIT_INDEX = .6

def get_collection_indices_by_split(
    collection: Any,
    split_index=SPLIT_INDEX
) -> tuple[int, int, int, int]:
    """
    Return the valid indices for the collection.
    """
    # Make sure the split index is valid for the given collection
    if split_index < 0 or split_index > len(collection):
        split_index = len(collection)

    # Check if fractional splitting is specified
    elif split_index > 0 and split_index < 1:
        split_index = len(collection) * split_index

    else:
        # Convert the split index to an integer
        split_index = int(split_index)
    
    # Return two sets of start and stop indices
    assert(isinstance(split_index, int))
    return 0, split_index, split_index, len(collection)


def get_tuple(x, dim=2):
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