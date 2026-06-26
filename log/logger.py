"""
This module contains the Logger class, which handles all logging operations.
"""
from typing import Optional, Union

from function import util


# Classes that use the logger
UNLABELED = 'UNLABELED'
TENSOR_FUNCTION = 'TENSOR_FUNCTION'
PASS_FUNCTION = 'PASS_FUNCTION'
DATASET = 'DATASET'
IMAGEDATASET = 'IMAGE_DATASET'
IMAGECAPTIONDATASET = 'IMAGE_CAPTION_DATASET'
LAYER = 'LAYER'
CONVOLUTION_LAYER = 'CONVOLUTION_LAYER'
TRANSFORMER_BLOCK = 'TRANSFORMER_BLOCK'
PROJECTION_LAYER = 'PROJECTION_LAYER'
MODEL = 'MODEL'
CNN = 'CNN'
TRANSFORMER = 'TRANSFORMER'
IMAGEENCODER = 'IMAGE_ENCODER'
CAPTIONDECODER = 'CAPTION_DECODER'
COMPLEXMODEL = 'COMPLEX_MODEL'
IMAGECAPTIONER = 'IMAGE_CAPTIONER'
CORPUSSCORER = 'CORPUS_SCORER'
IMAGETOSCORE = 'IMAGE_TO_SCORE'

# Module static ids
LAYER_MODULE = 0
MODEL_MODULE = 1
DATASET_MODULE = 2
SIMPLE_MODEL = 3
IMAGE_ENCODER_MODULE = 4
CAPTION_DECODER_MODULE = 5
COMPLEX_MODEL_MODULE = 6
IMAGE_CAPTIONER_MODULE = 7
CORPUS_SCORER_MODULE = 8
IMAGE_TO_SCORE_MODULE = 9

# ============================== LOGGER CLASS =================================

class Logger:

    # -------------------------- LOGGING ELEMENTS -----------------------------

    OUTER_BANNER = "=================================="
    INNER_BANNER = "----------------------------------"

    # ------------------------------- METHODS ---------------------------------

    def __init__(self):
        # Create dictionary that maps log names to log ids
        self.name_ids = {}
        # Create dictionary that maps log ids to log names
        self.id_names = {}

        # Set log names for the static class ids
        self.id_names = {
            LAYER_MODULE: 'LAYER_MODULE',
            MODEL_MODULE: 'MODEL_MODULE',
            DATASET_MODULE: 'DATASET_MODULE',
            IMAGE_ENCODER_MODULE: 'IMAGEENCODER_MODULE',
            CAPTION_DECODER_MODULE: 'CAPTIONDECODER_MODULE',
            COMPLEX_MODEL_MODULE: 'COMPLEXMODEL_MODULE',
            IMAGE_CAPTIONER_MODULE: 'IMAGECAPTIONER_MODULE',
            CORPUS_SCORER_MODULE: 'CORPUSSCORER_MODULE',
            IMAGE_TO_SCORE_MODULE: 'IMAGETOSCORE_MODULE'
        }

        # The total number of assigned log ids
        self.num_log_ids = len(self.id_names)

        # Create a dictionary of message queues, where each entry contains a
        #   the message queue type and list of messages
        self.message_queues = {
            'success': [],
            'error': [],
            'status': []
        }

    def set_log_id(self, object_name=None, default_name=None) -> int:
        """
        Associate the object name to a log id.

        Args:
            object_name (str): The name of the object using the logger

        Return:
            The log id associated with the object name
        """
        # Check if an object name is provided
        if object_name is None:
            # Check if a default name is provided
            if default_name is None:
                object_name = UNLABELED + '_OBJECT'
            else:
                # Use the default name
                object_name = UNLABELED + ' ' + default_name

        # Check if the object name already has a matching log id
        if object_name not in self.id_names.values():
            # Assigned a log id to the calling object, and increment the total
            #   number of log ids
            self.name_ids[object_name] = self.num_log_ids
            self.id_names[self.num_log_ids] = object_name
            self.num_log_ids += 1

        # Return the the log id assigned to the calling object
        return self.name_ids[object_name]
    
    def make_log(self,
        message: str,
        message_type: str,
        log_id: int
    ) -> None:
        """
        Log a message.

        Args:
            message (str): The message to log
            log_id (int): The log id associated with the reporting class/object

        Return:
            None
        """
        # Get the current time
        current_time = util.get_current_time(use_date=True)

        # Get the object name
        object_name = self.id_names[log_id]

        # Add the message entry to the error messages queue
        self.message_queues[message_type].append(
            (message, object_name, current_time)
        )

    def flush_message_queue(self,
        queue_type: Union[str, list[str], None],
        pause_time: Optional[int]=None
    ) -> None:
        """
        Flush the specified message queue onto console. If no queue types are
            provided, flush all the message queues.

        Args:
            queue_type (str | list[str]): The message queue(s) to flush
            pause (int): The amount of milliseconds to pause runtime

        Return:
            None
        """
        # Get the queue types to flush

        # Get all the message queues if queue types are not provided
        if queue_type is None or not isinstance(queue_type, str):
            flush_queues = [
                message_queue for queue_type, message_queue
                                in self.message_queues
            ]

        else:
            flush_queues = self.message_queues[queue_type]
        
        # Iterate through message queues
        for queue_type, message_queue in flush_queues:

            # Wrap the queue messages of each type in outer banners
            print(
                "\n%s %s %s\n",
                self.OUTER_BANNER,
                queue_type.upper(),
                self.OUTER_BANNER
            )

            # Iterate through messaages
            for message, object_name, time in message_queue:
                print("\n%s: [%s] %s", time, object_name, message)

            print(
                "\n%s %s %s\n",
                self.OUTER_BANNER,
                self.OUTER_BANNER[:len(queue_type)],
                self.OUTER_BANNER
            )

            # Pause after displaying each type of queue messages
            # NOTE: Not passing in pause time just waits for keyboard input
            util.pause(
                seconds=pause_time, key_type='enter', use_wait_message=True
            )


# ============================== STATIC METHODS ===============================

logger = Logger()


def set_log_id(object_name=None, default_name=None) -> int:
    """
    Associate the object name to a log id and return that id.

    Args:
        log_id (int): The log id of the associated object
        object_name(str): The object name

    Return:
        The log id associated with the object name
    """
    return logger.set_log_id(object_name=object_name, default_name=default_name)


def get_object_name(log_id: int) -> Optional[str]:
    """
    Get the object name associated with the log id.

    Args:
        log_id (int): The log id associated with an object name

    Return:
        The object name
    """
    # Check if the log id exists in the id names dict
    if log_id in logger.id_names.keys():
        # Return the object name
        return logger.id_names[log_id]

    # Else, return None
    

def log_status(status_message: str, log_id: int) -> None:
    """
    Log a status message.

    Args:
        status_message (str): The status message
        log_id (int): The log id associated with the reporting class/object

    Return:
        None
    """
    logger.make_log(
        message=status_message,
        message_type='status',
        log_id=log_id
    )

def log_success(success_message: str, log_id: int) -> None:
    """
    Log a success message.

    Args:
        success_message (str): The success message
        log_id (int): The log id associated with the reporting class/object

    Return:
        None
    """
    logger.make_log(
        message=success_message,
        message_type='success',
        log_id=log_id
    )

def log_error(error_message: str, log_id: int) -> None:
    """
    Log an error message.

    Args:
        error_message (str): The error message
        log_id (int): The log id associated with the object name

    Return:
        None
    """
    logger.make_log(
        message=error_message,
        message_type='error',
        log_id=log_id
    )