"""
This module is for the Tokenizer class.
"""
from typing import Optional

from data.dataset import CaptionDataSet
from model.model import Transformer
from log import logger as log


class CaptionDecoder(Transformer):
    """
    Class for the caption decoder.
    """
    def __init__(self,
        transformer_model_hyperparamters: Optional[dict]=None,
        base_model_hyperparameters: Optional[dict]=None,
        captions_dataset: Optional[CaptionDataSet]=None,
        captions_filename=None,
        training_test_split=-1.0,
        model_data_filename: Optional[str]=None,
        object_name: Optional[str]=None
    ):
        # Set the log id for the caption decoder
        self.log_id = log.set_log_id(object_name, log.CAPTIONDECODER)

        # Initialize the captions dataset if not provided
        if captions_dataset is None:
            caption_dataset = CaptionDataSet()

        # Load captions if provided
        if captions_filename is not None:
            caption_dataset.load_data(
                data_sources=(captions_filename, captions_filename),
                training_test_split=training_test_split
            )

        # Initialize the transformer model hyperparameters if not provided
        if transformer_model_hyperparamters is None:
            transformer_model_hyperparamters = {}

        # Initialize the base transformer model
        super().__init__(
            **transformer_model_hyperparamters,
            base_model_hyperparameters=base_model_hyperparameters,
            dataset=captions_dataset,
            model_data_filename=model_data_filename,
            object_name=object_name, has_log_id=True
        )