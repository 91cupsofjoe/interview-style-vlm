import os
from pathlib import Path
from PIL import Image
import json

import math
import torch
import torch.nn.functional as nnf
from torchvision import transforms

from dataset import dataset as ds

ROOT_DIR = Path(__file__).parent.parent
LEARNING_RATE = 1 # Default learning rate
NUM_EPOCHS = 1 # Default number of epochs

class ImageLearner():
    def __init__(self):
        self.dataset = None
        self.mean = []
        self.std = []

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
    
    def load_dataset_list(
            self, images_filename, descriptions_filename):
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
        with open(f"{ROOT_DIR}/{images_filename}", 'r') as images_file:
            images_list = [line.strip() for line in images_file]
            
        # Load descriptions list
        with open(f"{ROOT_DIR}/{descriptions_filename}", 'r') as descriptions_file:
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

    def train(self,
              dataset: ds.ImageDataSet,
              learning_rate=LEARNING_RATE,
              num_epochs=NUM_EPOCHS
        ):
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
        # Perform the forward pass