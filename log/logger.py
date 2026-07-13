"""
This module contains the Logger class, which handles all logging operations.
"""
from typing import Optional
import logging

from tensor_function import util

# Classes that use the logger
UNLABELED = 'UNLABELED'
TENSOR_FUNCTION = 'TENSOR_FUNCTION'
PASS_FUNCTION = 'PASS_FUNCTION'
DATASET = 'DATASET'
IMAGEDATASET = 'IMAGE_DATASET'
IMAGECAPTIONDATASET = 'IMAGE_CAPTION_DATASET'
DATAREADER = 'DATAREADER'
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
TEST = 'TEST'

# Module static ids
LAYER_MODULE = 0
MODEL_MODULE = 1
DATASET_MODULE = 2
DATAREADER_MODULE = 3
IMAGE_ENCODER_MODULE = 4
CAPTION_DECODER_MODULE = 5
COMPLEX_MODEL_MODULE = 6
IMAGE_CAPTIONER_MODULE = 7
CORPUS_SCORER_MODULE = 8
IMAGE_TO_SCORE_MODULE = 9
TEST_MODULES = 10
IMAGE_MODULE = 11

# logging.Logger levels
DEBUG = 10
INFO = 20
WARNING = 30
ERROR = 40
CRITICAL = 50

LOGGING_FORMAT = '%(levelname)s:%(message)s'
LOGGING_FILENAME = 'app.log'
LOGGING_FILEMODE = 'w'
LOGGING_ENCODING = 'utf-8'
LOGGING_LEVEL = logging.DEBUG
CONSOLE_LEVEL = logging.DEBUG
LOGGER_NAME = 'log'


# ============================== LOGGER CLASS =================================

class Logger():
    """
    This class wraps the standard logging class.

    The standard log levels:
        DEBUG (level = 10): Detailed diagnostic information.
            NOTE: Includes DEBUG, INFO, WARNING, ERROR, CRITICAL

        INFO (level = 20): Normal operation.
            NOTE: Includes INFO, WARNING, ERROR, CRITICAL

        WARNING (level = 30): Something unexpected but recoverable.
            NOTE: Includes WARNING, ERROR, CRITICAL

        ERROR (level = 40): An operation failed.
            NOTE: Includes ERROR, CRITICAL

        CRITICAL (level = 50): A severe error that may prevent the program from
            continuing.
            NOTE: Includes CRITICAL
    """
    # -------------------------- LOGGING ELEMENTS -----------------------------

    OUTER_BANNER = "==========================================================="
    INNER_BANNER = "-------------------------------------------------------"

    # ------------------------------- METHODS ---------------------------------

    def __init__(self,
        logger_name=LOGGER_NAME,
        level=LOGGING_LEVEL,
        console_level=CONSOLE_LEVEL,
        logging_level=LOGGING_LEVEL,
        format=LOGGING_FORMAT,
        datefmt=util.DATETIME_STR,
        filename=LOGGING_FILENAME,
        filemode=LOGGING_FILEMODE,
        encoding=LOGGING_ENCODING,
        do_output_to_console=False,
        silenced_objects: Optional[list]=None
    ):
        # Create dictionary that maps log names to log ids
        self.name_ids = {}
        # Create dictionary that maps log ids to log names
        self.id_names = {}

        # Initialize the list of objects to silence if not provided
        if silenced_objects is None:
            silenced_objects = []

        # Set the list of objects to silence
        self.silenced_objects = silenced_objects

        # Set log names for the static class ids
        self.id_names = {
            LAYER_MODULE: 'LAYER_MODULE',
            MODEL_MODULE: 'MODEL_MODULE',
            DATASET_MODULE: 'DATASET_MODULE',
            DATAREADER_MODULE: 'DATAREADER_MODULE',
            IMAGE_ENCODER_MODULE: 'IMAGEENCODER_MODULE',
            CAPTION_DECODER_MODULE: 'CAPTIONDECODER_MODULE',
            COMPLEX_MODEL_MODULE: 'COMPLEXMODEL_MODULE',
            IMAGE_CAPTIONER_MODULE: 'IMAGECAPTIONER_MODULE',
            CORPUS_SCORER_MODULE: 'CORPUSSCORER_MODULE',
            IMAGE_TO_SCORE_MODULE: 'IMAGETOSCORE_MODULE',
            TEST_MODULES: 'TEST_MODULES',
            IMAGE_MODULE: 'IMAGE_MODULE'
        }

        # The total number of assigned log ids
        self.num_log_ids = len(self.id_names)

        # Set the logging.Logger configurations
        config_kwargs = locals()
        config_kwargs.pop('self')
        config_kwargs.pop('silenced_objects')

        # Initialize the handlers list
        handlers = []

        # Check if outputing to console
        if config_kwargs.pop('do_output_to_console'):
            # Set the console handler
            console_handler = logging.StreamHandler()
            console_handler.set_name('console')
            # Try setting the console handler level
            try:
                console_handler.setLevel(config_kwargs.pop('console_level'))
            except:
                # Set the console handler to the default console level
                console_handler.setLevel(CONSOLE_LEVEL)

            # Append the console handler to the handlers list
            handlers.append(console_handler)

        # Pop console level if it exists in the config keyword arguments
        if 'console_level' in config_kwargs:
            config_kwargs.pop('console_level')

        # Set the log file handler
        logfile_handler = logging.FileHandler(
            filename=config_kwargs.pop('filename'),
            mode=config_kwargs.pop('filemode')
        )
        logfile_handler.set_name('log_file')
        # Try setting the log file handler level
        try:
            logfile_handler.setLevel(config_kwargs.pop('logging_level'))
        except:
            # Set the log file handler to the default logging level
            logfile_handler.setLevel(LOGGING_LEVEL)

        # Append the log file handler to the handlers list
        handlers.append(logfile_handler)

        # Add the handlers list to the configuration keyword arguments
        config_kwargs['handlers'] = handlers

        # Set and configure the logging.Logger
        self.logger = logging.getLogger(config_kwargs.pop('logger_name'))
        logging.basicConfig(**config_kwargs, force=True)

    def _set_log_id(self, object_name=None, default_name=None) -> int:
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
    
    def _make_log(self,
        message: str,
        log_id: int,
        message_level: str,
        use_date=False
    ) -> None:
        """
        Log a message.

        Args:
            message (str): The message to log
            log_id (int): The log id associated with the reporting class/object
            message_level (str): The message level
            use_date (bool): Boolean indicating if using date for the time

        Return:
            None
        """
        # Get the object name
        object_name = self.id_names[log_id]

        # Check if the object is not silenced
        if not (object_name in self.silenced_objects \
            or log_id in self.silenced_objects \
            or (UNLABELED in self.silenced_objects and UNLABELED in object_name)):

            # Get the current time
            current_time = util._get_current_time(use_date)

            # Set the log message
            log_message = f'{util._get_print_name(object_name)}:' \
                            + f'[{current_time}]: {message}'

            # Log the message to the specified logger level
            if message_level == 'info':
                self.logger.info(log_message, extra={"object":object_name})
            elif message_level == 'warning':
                self.logger.warning(log_message, extra={"object":object_name})
            elif message_level == 'error':
                self.logger.error(log_message, extra={"object":object_name})
            elif message_level == 'critical':
                self.logger.critical(log_message, extra={"object":object_name})
            else:
                # Default level is debug
                self.logger.debug(log_message, extra={"object":object_name})

        # Else, don't make the log

    def flush(self,
        pause_time=0.0,
        key_type='enter',
        use_wait_message=True
    ) -> None:
        """
        Flush the logger's handlers.

        Args:
            None

        Return:
            None
        """
        # Iterate through the handlers
        for handler in self.logger.handlers:
            # Wrap the queue messages of each type in outer banners
            print(f"\n{self.OUTER_BANNER}"
                f"\n\n{self.INNER_BANNER}"
                f"{handler.name}",
                f"{self.INNER_BANNER}"
            )
            
            handler.flush

            print(f"\n{self.OUTER_BANNER}\n")

            # Pause after displaying each type of queue messages
            # NOTE: Not passing in pause time just waits for keyboard input
            util._pause(
                seconds=pause_time,
                key_type=key_type,
                use_wait_message=use_wait_message
            )


# ============================== STATIC METHODS ===============================

# Set logger to a default Logger
logger = Logger('None')


def _use_logger(
    logger_name=LOGGER_NAME,
    console_level=CONSOLE_LEVEL,
    logging_level=LOGGING_LEVEL,
    datefmt=util.DATETIME_STR,
    filename=LOGGING_FILENAME,
    filemode=LOGGING_FILEMODE,
    encoding=LOGGING_ENCODING,
    do_output_to_console=False,
    silenced_objects: Optional[list]=None
) -> None:
    """
    Update the default logger with the provided configurations.

    Args:
        logger_name (str): The name of the logger
        console_level (int | str): The logging level for the console handler
        logging_level (int | str): The logging level for the logfile handler
        datefmt (str): The datetime format
        filename (str): The name of the log file
        filemode (str): The mode for the log file
        encoding (str): The encoding for the log file
        do_output_to_console (bool): Boolean indicating whether to output to
            console or not
        silenced_objects (list): List of object names and ids to silence

    Return:
        None
    """
    global logger
    logger = Logger(**locals())
    

def _set_log_id(object_name=None, default_name=None) -> int:
    """
    Associate the object name to a log id and return that id.

    Args:
        log_id (int): The log id of the associated object
        object_name(str): The object name

    Return:
        The log id associated with the object name
    """
    return logger._set_log_id(object_name=object_name, default_name=default_name)


def _get_object_name(log_id: int) -> Optional[str]:
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
    

def _log_debug(debug_message: str, log_id: int) -> None:
    """
    Log a debug message.

    Args:
        debug_message (str): The debug message
        log_id (int): The log id associated with the reporting class/object

    Return:
        None
    """
    logger._make_log(
        message=debug_message,
        log_id=log_id,
        message_level='debug'
    )

def _log_info(info_message: str, log_id: int) -> None:
    """
    Log an info message.

    Args:
        info_message (str): The info message
        log_id (int): The log id associated with the reporting class/object

    Return:
        None
    """
    logger._make_log(
        message=info_message,
        log_id=log_id,
        message_level='info'
    )

def _log_error(error_message: str, log_id: int) -> None:
    """
    Log an error message.

    Args:
        error_message (str): The error message
        log_id (int): The log id associated with the object name

    Return:
        None
    """
    logger._make_log(
        message=error_message,
        log_id=log_id,
        message_level='error'
    )

def _log_warning(warning_message: str, log_id: int) -> None:
    """
    Log a warning message.

    Args:
        warning_message (str): The warning message
        log_id (int): The log id associated with the object name

    Return:
        None
    """
    logger._make_log(
        message=warning_message,
        log_id=log_id,
        message_level='warning'
    )

def _log_critical(critical_message: str, log_id: int) -> None:
    """
    Log a critical message.

    Args:
        critical_message (str): The critical message
        log_id (int): The log id associated with the object name

    Return:
        None
    """
    logger._make_log(
        message=critical_message,
        log_id=log_id,
        message_level='critical'
    )


def _flush(
    pause_time=0.0,
    key_type='enter',
    use_wait_message=True
) -> None:
    """
    Flush the logger's handlers.

    Args:
        None

    Return:
        None
    """
    logger.flush(
        pause_time=pause_time,
        key_type=key_type,
        use_wait_message=use_wait_message
    )