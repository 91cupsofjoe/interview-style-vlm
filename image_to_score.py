"""
This module takes in an image (of a person with any kind of attire) and returns
    a score based on hiring potential.

To accomplish its image scoring, it uses two models: an image to caption and a
    caption to rating prediction models.
"""
import sys
from pathlib import Path

from model import ImageCaptioner as ic, CaptionScorer as cs
from log import logger as lg
from tensor import tensor_ops


# =============================== IMAGE SCORER ================================

class ImageScorer:
    def __init__(self,
        params
    ):
        # Set the image captioner and caption scorer models
        self.image_captioner = ic.ImageCaptioner
        self.caption_scorer = cs.CaptionScorer

        # Set booleans for the state of the models datasets and model data
        self.image_captioner_has_dataset = False
        self.image_captioner_has_model_data = False
        self.caption_scorer_has_dataset = False
        self.caption_scorer_has_model_data = False


    def has_required_datasets(self):
        # Check if both ImageScorer models have datasets
        return self.image_captioner_has_dataset \
                and self.caption_scorer_has_dataset
    

    def has_required_model_data(self):
        # Check if both ImageScorer models have model data
        return self.image_captioner_has_model_data \
                and self.caption_scorer_has_model_data


    def load_datasets(self,
        images_filename=None, captions_filename=None, corpus_filename=None,
    ):
        """
        Load dataset for either the image captioner, caption scorer, or both models.

        Args:
            image_captioner_dataset (DataSet): Dataset for the image captioner
            caption_scorer_dataset (DataSet): Dataset for the caption scorer

        Return:
            Tuple of booleans to indicate if datasets were successfully loaded
        """
        loaded_image_captioner_dataset = False
        loaded_caption_scorer_dataset = False

        # Load image captioner dataset if provided
        if images_filename is not None and captions_filename is not None:
            loaded_image_captioner_dataset = self.image_captioner.load_dataset(
                images_filename, captions_filename)

        # Load caption scorer dataset if provided
        if corpus_filename is not None:
            loaded_caption_scorer_dataset = self.caption_scorer.load_dataset(
                corpus_filename)
            
        # Update image scorer dataset booleans
        if loaded_image_captioner_dataset:
            self.image_captioner_has_dataset = True
        if loaded_caption_scorer_dataset:
            self.caption_scorer_has_dataset = True

        # Return indication of success with loading datasets
        return loaded_image_captioner_dataset, loaded_caption_scorer_dataset
    

    def train_models(self,
        images_filename=None, captions_filename=None, corpus_filename=None,
        train_image_captioner=False, train_caption_scorer=False
    ):
        """
        Train either the image captioner, caption scorer, or both models.

        Args:
            train_image_captioner (boolean): Boolean for training image captioner
            train_caption_scorer (boolean): Boolean for training caption scorer

        Return:
            Tuple of booleans to indicate if models were successfully trained
        """
        trained_image_captioner = False
        trained_caption_scorer = False

        # If training for both models are set to False (no args provided),
        #   train both models
        if not train_image_captioner and not train_caption_scorer:
            train_image_captioner = True
            train_caption_scorer = True

        # Train the image captioner
        if train_image_captioner:
            # If no image captioner dataset is provided, the model will use its
            #   own dataset
            trained_image_captioner = self.image_captioner.train(
                images_filename, captions_filename)

        # Train the caption scorer
        if train_caption_scorer:
            # If no caption scorer dataset is provided, the model will use its
            #   own dataset
            trained_caption_scorer = self.caption_scorer.train(
                corpus_filename)
            
        # Update image scorer dataset booleans
        if trained_image_captioner:
            self.image_captioner_has_model_data = True
        if trained_caption_scorer:
            self.caption_scorer_has_model_data = True
            
        # Return indication of success with training models
        return trained_image_captioner, trained_caption_scorer
    

    def load_model_data(self,
        image_captioner_model_data_filename=None,
        caption_scorer_model_data_filename=None
    ):
        """
        Load model data for either image captioner, caption scorer, or both models.

        Args:
            train_image_captioner (boolean): Boolean for training image captioner
            train_caption_scorer (boolean): Boolean for training caption scorer

        Return:
            Tuple of booleans to indicate if model data was successfully loaded
        """
        loaded_image_captioner_model_data = False
        loaded_caption_scorer_model_data = False

        # Load image captioner training data
        if image_captioner_model_data_filename:
            loaded_image_captioner_model_data = self.image_captioner.load_model_data(
                image_captioner_model_data_filename
            )

        # Load caption scorer training data
        if caption_scorer_model_data_filename is not None:
            loaded_caption_scorer_model_data = self.caption_scorer.load_model_data(
                caption_scorer_model_data_filename
            )

        return loaded_image_captioner_model_data, loaded_caption_scorer_model_data
    

    def score_image(self, image_filename: str):
        """
        Get the image from the provided filename and return a score.

        Args:
            image_filename (str): The filename of the image to rate

        Return:
            An integer representing the image's score
        """
        # Encode the image
        try:
            captions = self.image_captioner.caption(image_filename)
        except:
            lg.log_error("Couldn't encode the image!", lg.IMAGE_SCORER)

        # Predict and return the image's score
        return self.caption_scorer.score(captions)
    

# ====================== IMAGE SCORING DEFAULT VALUES =========================

DATA_DIR = f"{Path(__file__).parent}/data"
IMAGES_DIR = f"{DATA_DIR}/images"
MODEL_DATA_DIR = f"{DATA_DIR}/model_data"

IMAGE_CAPTIONER_IMAGES_FILENAME = f"{IMAGES_DIR}/images_list.txt"
IMAGE_CAPTIONER_CAPTIONS_FILENAME = f"{DATA_DIR}/captions.txt"
IMAGE_CAPTIONER_MODEL_DATA_FILENAME = f"{MODEL_DATA_DIR}/image_captioner.pt"
CAPTION_SCORER_CORPUS_FILENAME = f"{DATA_DIR}/corpus.txt"
CAPTION_SCORER_MODEL_DATA_FILENAME = f"{MODEL_DATA_DIR}/caption_scorer.pt"

QUERY_IMAGE_FILENAME = f"{IMAGES_DIR}/query_image.jpg"

DO_DATASET_LOADING = True
DO_MODEL_TRAINING = True
    

# =================================== MAIN ====================================

if __name__ == '__main__':
    # Get image scoring parameters
    images_filename = IMAGE_CAPTIONER_IMAGES_FILENAME
    captions_filename = IMAGE_CAPTIONER_CAPTIONS_FILENAME
    image_captioner_filename = IMAGE_CAPTIONER_MODEL_DATA_FILENAME
    corpus_filename = CAPTION_SCORER_CORPUS_FILENAME
    caption_scorer_filename = CAPTION_SCORER_MODEL_DATA_FILENAME
    query_image_filename = QUERY_IMAGE_FILENAME
    do_dataset_loading = DO_DATASET_LOADING
    do_model_training = DO_MODEL_TRAINING

    # Parse and store command-line arguments in dictionary
    parsed_args = tensor_ops.parse_args(
        command_line_args=sys.argv,
        valid_keys=[
            # Command-line keys for the image captioner
            'images_file', 'captions_file', 'image_captioner_file'
            # Command-line keys for the caption scorer
            'corpus_file', 'caption_scorer_file',
            # Command-line key for the image scorer
            'query_image_file', 'dataset', 'train'
        ]
    )

    # Update image scoring parameters if provided in command-line arguments

    # Image Captioner
    if 'images_file' in parsed_args:
        images_filename = parsed_args['images_file']
    if 'captions_file' in parsed_args:
        captions_filename = parsed_args['captions_file']

    # Caption Scorer
    if 'corpus_file' in parsed_args:
        corpus_filename = parsed_args['corpus_file']

    # Image Scorer
    if 'query_image_file' in parsed_args:
        query_image_filename = parsed_args['query_image_file']
    if 'dataset' in parsed_args:
        do_dataset_loading = parsed_args['dataset']
    if 'train' in parsed_args:
        do_model_training = parsed_args['train']

    # Create the image scorer
    image_scorer = ImageScorer({})

    # Get datasets
    if do_dataset_loading:
        image_scorer.load_datasets()

    # Train the models
    if do_model_training and image_scorer.has_required_datasets:
        image_scorer.train_models()
    
    # Make a prediction from the query image if the image scorer is trained,
    #   and output the predicted score to console
    if image_scorer.has_required_model_data:
        print(f"\The image score for {query_image_filename}: "
            f"{image_scorer.score_image(query_image_filename)}")