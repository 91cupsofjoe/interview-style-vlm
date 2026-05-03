import os
from pathlib import Path
from pydantic import BaseModel
import torch
from torch import nn
from torchvision import transforms

from datasets import dataset as ds

ROOT_DIR = Path(__file__).parent.parent

class ImageLearner(BaseModel):
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
    
    def load_dataset_list(self, filename):
        """
        Read in a list of filenames for images in the dataset

        Args:
            filename (str): Filename of a file containing image filepaths

        Return:
            list[str]: List of image filepaths
        """
        with open(filename, 'r') as list_file:
            self.dataset = ds.ImageDataSet(
                [line.strip() for line in list_file]
            )
        
    def set_preprocessor(self):
        self.preprocessor = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
    def get_preprocessed_image(self, image):
        """
        Preprocesses and returns a transformed image

        Args:
            image (PIL.Image): The image to transform
            mean (list[float]): The average of RGB values
            std (list[float]): The standard deviation of RGB values

        Return:
            torch.Tensor: Normalized image tensor
        """
        if not hasattr(self, "preprocessor"):
            raise RuntimeError("Preprocessor not initialized." \
                            "Call set_preprocessor")
        return self.preprocessor(image)