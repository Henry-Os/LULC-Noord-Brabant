import torch
from torch.utils import data
from torchvision import transforms

# Constants
INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(mode='train'):
    """
    Returns the appropriate torchvision transform pipeline.

    Args:
        mode (str): The execution mode, either 'train' for training augmentations 
            or 'inference' for deterministic resizing. Defaults to 'train'.

    Returns:
        A pipeline of image transformations (transforms.Compose).
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    else:
        # Valid/Test/Inference
        return transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

class EuroSAT(data.Dataset):
    """
    PyTorch Dataset wrapper for the EuroSAT dataset.

    This class wraps a preloaded EuroSAT dataset (e.g., a list of (image, label) tuples)
    and optionally applies transformations to the input images.

    Attributes:
        dataset (list or Dataset): The underlying dataset containing image-label pairs.
        transform (callable, optional): A function/transform to apply to the input images.

    Methods:
        __getitem__(index): Returns the transformed image and label at the specified index.
        __len__(): Returns the total number of samples in the dataset.
    """

    def __init__(self, dataset, transform=None):
        """Initializes the dataset with the underlying data and optional transforms."""
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        """
        Returns the transformed image and label at the specified index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: (image, label) where image is a tensor and label is an integer.
        """
        # Apply image transformations
        if self.transform:
            x = self.transform(self.dataset[index][0])
        else:
            x = self.dataset[index][0]
        
        # Get class label
        y = self.dataset[index][1]
        
        return x, y

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Total count of samples.
        """
        return len(self.dataset)