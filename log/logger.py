"""
This module contains the Logger class, which handles all logging operations.
"""
from typing import Optional, Union

from tensor import util

# ============================== LOGGER CLASS =================================

class Logger:

    # -------------------------- LOGGING ELEMENTS -----------------------------

    OUTER_BANNER = "=================================="
    INNER_BANNER = "----------------------------------"

    # ------------------------------- METHODS ---------------------------------

    def __init__(self):
        # Create dictionary of class names to class ids
        self.object_ids = {}
        # The total number of assigned log ids
        self.num_log_ids = 0

        # Create a dictionary of message queues, where each entry contains a
        #   the message queue type and list of messages
        self.message_queues = {
            'success': {},
            'error': {},
            'status': {}
        }

    def set_log_id(self, log_id=None, object_name=None) -> int:
        """
        Associate the object name to a log id.

        Args:
            object_name (str): The name of the object using the logger

        Return:
            The log id associated with the object name
        """
        # If a log id is not provided, set the logger id for the unlogged object
        if log_id is None:
            # Check if a logger name is provided
            if object_name is None:
                # No log id or logger name is provided
                object_name = 'UNNAMED_OBJECT'

            # Check if the object name already has a matching log id
            if object_name not in self.object_ids:
                # Assigned a log id to the calling object, and increment the total
                #   number of log ids
                self.object_ids[object_name] = self.num_log_ids
                self.num_log_ids += 1

            # Return the the log id assigned to the calling object
            return self.object_ids[object_name]
        
        # Else, just return the provided log id
        return log_id
    
    def log_error(self, error_message: str, log_id: int) -> None:
        """
        Make a log with the provided error message and object name associated with
            the log id.

        Args:
            error_message (str): The error message
            log_id (int): The log id associated with the object name

        Return:
            None
        """
        # Get the current time
        current_time = util.get_current_time(use_date=True)

    def flush_message_queue(self,
        queue_type: Union[str, list[str], None],
        pause: Optional[int]
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
            # NOTE: Not passing in seconds just waits for keyboard input
            util.pause(key_type='enter', use_wait_message=True)


# ============================== STATIC METHODS ===============================

logger = Logger()


def set_log_id(log_id=None, object_name=None) -> int:
    """
    Associate the object name to a log id and return that id.

    Args:
        log_id (int): The log id of the associated object
        object_name(str): The object name

    Return:
        The log id associated with the object name
    """
    return logger.set_log_id(log_id=log_id, object_name=object_name)


def log_error(error_message: str, log_id: int) -> None:
    """
    Make a log with the provided error message and object name associated with
        the log id.

    Args:
        error_message (str): The error message
        log_id (int): The log id associated with the object name

    Return:
        None
    """
    logger.log_error(
        error_message=error_message,
        log_id=log_id
    )