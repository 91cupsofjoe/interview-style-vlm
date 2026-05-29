"""
This module is for the image captioner model. It comprises of two submodules,
    an image encoder (CNN) and caption decoder (transformer). The image captioner
    model takes in an image and predicts the caption for it.
"""
from typing import Any

from model.model import CNN, Transformer
from data.dataset import DataSet, ImageCaptionDataSet, TextScorerDataSet

from log import logger as lg


class ComplexModel:
    """
    This is the base complex model class, which utilizes multiple submodels for
        its training and prediction operations.
    """
    def __init__(self,
        logger_name=None,
        dataset=None,
        submodels=None,
        forward_pass_functions=None,
        backpropagation_functions=None,
        loss_function=None,
        update_function=None
    ):
        # Set the logger ID for the ComplexModel
        if logger_name is None:
            logger_name = "COMPLEX_MODEL"
        self.log_id = lg.set_log_id(logger_name)

        # Set the dataset for the complex model
        if dataset is None:
            dataset = DataSet()
        self.dataset = dataset

        # Set the submodels for the complex model
        if submodels is None:
            submodels = {}
        self.submodels = submodels

        # Set the forward pass functions for the complex model
        if forward_pass_functions is None:
            forward_pass_functions = []
        self.forward_pass_functions = forward_pass_functions

        # Set the backpropagation functions for the complex model
        if backpropagation_functions is None:
            backpropagation_functions = []
        self.backpropagation_functions = backpropagation_functions

        # Set the loss and update functions
        self.loss_function = loss_function
        self.update_function = update_function


    def load_dataset(self, data_sources: tuple[Any]) -> bool:
        """
        Load data from the data sources into the dataset.

        Args:
            data (tuple[Any]): Tuple containing data sources

        Return:
            Boolean to indicate if loading the dataset was successful
        """
        # Load the training examples and training labels
        if not self.dataset.load_training_data(data_sources=data_sources):
            # Log error and return False since loading the training data failed
            lg.log_error(
                f"Could not load the dataset since loading training examples ",
                f"and training labels failed!",
                self.log_id
            )
            return False

        # Load the data tensor from the training examples and training labels
        if not self.dataset.load_data_tensor():
            # Log error and return False since loading the data tensor failed
            lg.log_error(
                "Could not load the dataset since loading the data tensor failed!",
                self.log_id
            )
            return False
        
        # Successfully loaded the training data and data tensor, return True
        return True
    

    def train(self, data_tensor=None):
        """
        Train the complex model on the training data tensor provided. If no
            training data is provided, use preloaded data from the dataset.

        Args:
            data (tuple[Any]): The data to train the complex model on

        Return:
            The calculated loss and final derivative of the input patches
        """
        if data_tensor is not None:

    

class ImageCaptioner(ComplexModel):
    def __init__(self,
        images_filename=None, captions_filename=None, corpus_filename=None
    ):
        # Dataset for images and captions
        self.dataset = ImageCaptionDataSet(
            images_filename=images_filename,
            captions_filename=captions_filename,
            corpus_filename=corpus_filename
        )

        # Submodels for image encoding and caption predictions
        self.image_encoder = CNN(
            model_sequences={},
            model_loss_function=None,
            model_update_function=None
        )
        self.caption_decoder = Transformer(
            model_sequences={},
            model_loss_function=None,
            model_update_function=None
        )

        super().__init__(
            
        )

    def train(self, images_filename=None, captions_filename=None) -> bool:
        """
        Train the image captioner model on the given images and captions. If no data
            if provided, use the preloaded data from the dataset.

        Args:
            images_filename (str): Filename of file with the image filepaths
            captions_filename (str): Filename of file with the image captions

        Return:
            Boolean to indicate if model training was successful
        """
        trained_image_captioner_model = False

        # Update dataset images and captions if applicable
        if images_filename is not None and captions_filename is not None:
            self.dataset.set_data(
                images_filename=images_filename,
                captions_filename=captions_filename
            )

        # Get the preprocessed images
        try:
            preprocessed_images = self.dataset.get_preprocessed_images()
        except:
            lg.log_error(
                "Couldn't get the preprocessed images!", self.log_id)

        # Encode the preprocessed images
        try:
            encoded_images = self.image_encoder.encode(
                            input_patches=preprocessed_images)
        except:
            lg.log_error(
                "Couldn't encode the preprocessed images!", self.log_id)

        # Decode the preprocessed images
        try:
            caption_predictions = self.caption_decoder.decode(encoded_images)
        except:
            lg.log_error(
                "Couldn't decode the encoded images!", self.log_id)
