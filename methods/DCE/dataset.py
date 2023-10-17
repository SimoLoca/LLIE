import torch
import torchvision

class Dataset(torch.utils.data.Dataset):
    """Low-light image dataset

    Pytorch dataset for low-light images

    Args:
        image_files: List of image file paths
        image_size: size of each image
    """
    def __init__(self, image_files: list = None, image_size: int = 256):
        self.image_files = image_files
        self.image_size = image_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        image_path = self.image_files[item]
        image = torchvision.io.read_image(image_path)/255.
        image = torchvision.transforms.functional.resize(
            image,
            (self.image_size, self.image_size)
            )
        return image
