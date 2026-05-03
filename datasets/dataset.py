from torch.utils.data import Dataset
from PIL import Image

class ImageDataSet(Dataset):
    def __init__(self, dataset_list):
        self.image_paths = dataset_list

    def __len__(self):
        return len(self.image_paths)
    
    def __getimage__(self, idx):
        return Image.open(self.image_paths[idx]).convert("RGB")