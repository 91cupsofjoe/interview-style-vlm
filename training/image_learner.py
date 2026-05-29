import os
from pathlib import Path
from PIL import Image
import json

import torch
import torch.nn.functional as nnf
from torchvision import transforms

from data import dataset as ds
from model import model as ml
from model import image_encoder as ie

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = "data"
MODEL_DIR = "model"

class ImageLearner():
    def __init__(self):
        self.dataset = None
        self.model = None

    def setDevice(self):
        """
        Select and store the best available computedevice (CUDA, MPS, or CPU)
        """
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        # print device
    
    def load_dataset(self,
        images_filename, descriptions_filename
    ):
        """
        Read in a list of filenames for images in the dataset as well as
            there corresponding descriptions, and store them as a dataset

        Args:
            image_filepath (str): File with image filepaths
            image_descriptions_filepath (str): File with image descriptions

        Return:
            None
        """
        # Load images list
        with open(f"{ROOT_DIR}/{DATA_DIR}/{images_filename}", 'r') \
                        as images_file:
            images_list = [line.strip() for line in images_file]
            
        # Load descriptions list
        with open(f"{ROOT_DIR}/{DATA_DIR}/{descriptions_filename}", 'r') \
                        as descriptions_file:
            image_descriptions_dict = {}
            # The descriptions file contains a JSON array of entries,
            #   with each entry containing and image filename and a
            #   corresponding list of descriptions (sentences) for the image
            for entry in json.load(descriptions_file):
                image_filename = entry['image']
                sentences = entry['descriptions']

                for i in range(len(sentences)):
                    # Store each sentence as a list of tokens beginning with
                    #   the token '<START>' and ending with the token '<END>'
                    sentences[i] = (['<START>'] + sentences[i].lower().split()
                                    + ['<END>']).split()

                image_descriptions_dict[image_filename] = sentences

        # Store the images list and image descriptions dict as a dataset
        self.dataset = ds.ImageDataSet(images_list, image_descriptions_dict)

    def load_model(self, json_filename):
        with open(f"{ROOT_DIR}/{MODEL_DIR}/settings.json", 'r') as settings_file:
            # Load the json file
            model_settings = json.load(settings_file)

            # Model parameters
            learning_rate = model_settings.get('learning rate')
            num_epochs = model_settings.get('number of epochs')

            # First get the convolution and projection layer functions
            conv_layer_funcs, proj_layer_funcs = [
                {
                    category_key: [
                    # Each function for the convolution layer is a tuple of
                    #   function pointer and function parameter keys
                        (get_function(func['name']), func['params'])
                            for func in model_settings.get(category_func)
                ]
                    for category_key, category_func in
                    [
                        (key + 'forward',
                            layer + ' forward functions'),
                        (key + 'forward_transform',
                            layer + ' forward transformations'),
                        (key + 'backward', 
                            layer + ' backpropagation functions'),
                        (key + 'backward_transforms',
                            layer + ' backpropagation transformations')
                    ]
                } for key, layer in [
                    ('conv_', 'convolution'),
                    ('proj', 'projection')
                ]
            ]

            # Then get the convolution and projections layers
            conv_layers, proj_layers = [
                {
                    layer['seq_id']:
                    {
                        key: value for key, value in layer
                    }    
                    for layer in model_settings.get(layer + ' layers')
                } for layer in ['convolution', 'projection']
            ]

            # Next do the same for the projection layer
            
            conv_func = model_settings.get('convolution function')
            cf_hparams = model_settings.get('convolution function hyperparameters')
            cl_attrbs = model_settings.get('convolution layer attributes')
            conv_layers = model_settings.get("convolution layers")
            proj_layers = model_settings.get("projection layers")
            conv_layer_params = model_settings.get('convolution layers', [])
            proj_layers_params = model_settings.get('projection layers', [])
            forward = model_settings.get('forward pass functions', [])
            backward = model_settings.get('backpropagation functions', [])
            loss_fn_name = model_settings.get('loss function', ())
            loss_derivative_fn_name = model_settings.get('loss derivative function', ())

            for 

    def train(self, dataset: ds.ImageDataSet):
        """
        Perform the training loop for the image captioner model:
            Forward pass -> Loss -> Backpropagation -> Updates

        Args:
            dataset (ImageDataSet): The image dataset to train on
            learning_rate (float): The learning rate for updates
            num_epochs (int): The number of epochs to train for

        Return:
            None
        """
        # First get the example images and label sentences from the dataset
        examples, labels = [
            [[item[0]], [item[1]]]
            for item in [dataset.__getitem__(x) for x in range(len(dataset))]
        ]
        # First create a Tensor representing the batch of images
        #   with dimensions = batch size x RGB channels x height x width
        input_patches = torch.stack(
            [dataset.get_preprocessed_image(image) for image in examples]
        )

# =========================== HELPER FUNCTIONS ================================

function_name_to_pointer = {
    # Convolution layer forward pass functions
    'convolution 2d': ie.conv2d,
    'ReLU activation': ie.relu,

    # Convolution layer backpropagation functions
    'ReLU activation backward': ie.relu_backward,
    'convolution 2d backward': ie.conv2d_backward,

    # Projection layer forward functions
    'linear projection': ie.lin_proj,
    'linear projection backward': ie.lin_proj_backward,
    
    # Forward pass transformation functions
    'pool': ie.pool,
    'flatten': ie.flatten,

    # Backpropagation transformation functions
    'unflatten': ie.unflatten,
    'pool backward': ie.pool_backward

    # Loss functions
    'loss' : ie.get_cross_entropy_loss
}

def get_function(function_name: str):
    """
    Get the function pointer associated with the function name, from the
        image encoder module

    Args:
        function_name (str): The name of the function

    Return:
        The function pointer to the specified function name
    """
    return function_name_to_pointer[function_name]