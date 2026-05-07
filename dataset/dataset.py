from matplotlib.pylab import std
from torch.utils.data import Dataset
from PIL import Image

import torch
from torchvision import transforms

RESIZE = 256
CROP_SIZE = 224

class ImageDataSet(Dataset):
    def __init__(self, images, image_descriptions):
        # Examples
        self.image_paths = images
        # Labels
        self.image_descriptions = image_descriptions

        self.preprocessor = None
        # Store mean and standard deviation for preprocessing
        self.mean = None
        self.std = None

        self.word_ids = None

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_filename = self.image_paths[idx]
        image = Image.open(image_filename).convert("RGB")
        sentences = self.image_descriptions[image_filename]
        return image, sentences
    
    def get_preprocessed_image(self, image, resize=None, crop_size=None):
        """
        Preprocesses and returns a transformed image

        Args:
            image (PIL.Image): The image to transform

        Return:
            Normalized image tensor
        """
        # Set the preprocessor if not done already (with defaults)
        if self.preprocessor is None:
            self.set_preprocessor()
        # Else, update the preprocessor if resize and crop_size are provided
        elif resize is not None and crop_size is not None:
            self.set_preprocessor(resize, crop_size)
        
        # Return the preprocessed image
        return self.preprocess(image)
    
    def get_word_id(self, token):
        """
        Get the word id of a token

        Args:
            token (str): The token

        Return:
            The word id of the token
        """
        # Create the word to ID dictionary if not done already
        if self.word_ids is None:
            self.set_word_ids()

        # Return the word ID for the token ('<PAD>' for an unknown token)
        if token not in self.word_ids:
            return self.word_ids['<PAD>']
        return self.word_ids[token]
    
    # =========================== HELPER FUNCTIONS ============================
    
    def calc_mean_std(self):
        """
        Calculate and store the mean and standard deviation of the RGB values
            across the images of the dataset, for preprocessing

        Args:
            None

        Return:
            None
        """
        # Get list of images and to a Tensor with dims =
        #   # images x 3 (RGB) x height x width
        # Transpose the images and RGB channels (so RGB channels is 1st dim),
        #   and then normalize the pixel values to the range [0, 1]
        RGB_tensor = torch.stack([transforms.ToTensor()(image)
                        for image, _ in self]).transpose(0, 1) / 255
        red_values = RGB_tensor[0]
        green_values = RGB_tensor[1]
        blue_values = RGB_tensor[2]

        # Calculate mean and std for each RGB channel across all images
        self.mean = [
            red_values.sum() / red_values.numel(),
            green_values.sum() / green_values.numel(),
            blue_values.sum() / blue_values.numel()
        ]
        self.std = [
            ((red_values - self.mean[0])**2).sum() / red_values.numel(),
            ((green_values - self.mean[1])**2).sum() / green_values.numel(),
            ((blue_values - self.mean[2])**2).sum() / blue_values.numel()
        ]
    
    def set_preprocessor(self, resize=RESIZE, crop_size=CROP_SIZE):
        # If not done already, calculate the mean and std of the RGB values
        #   across the images of the dataset
        if self.mean is None or self.std is None:
            self.calc_mean_std()

        self.preprocess = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def set_word_ids(self):
        # Get the set of unique tokens across all sentences in the dataset
        token_set = set([])
        for sentence in [sentence for _, sentence in self]:
            for word in sentence:
                token_set.add(word)
        # Create the word to ID dictionary, including the special tokens
        #   '<START>' (id = 0), '<END>' (id = 1), and '<PAD>' (id = 2)
        self.word_ids = {
            '<START>': 0,
            '<END>': 1,
            '<PAD>': 2
        }
        for token in token_set:
            self.word_ids[token] = len(self.word_ids)