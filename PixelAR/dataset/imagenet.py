from pathlib import Path
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from PixelAR.utils.image import center_crop_arr
from torchvision import transforms
from typing import List
from PIL import Image

class ImageNetTenCropDataset(Dataset):
    def __init__(self, root_dir, image_size: int = 256, crop_ranges: List[float] = [1.05, 1.1], ten_crop: bool = True):
        self.dataset = ImageFolder(root_dir)
        
        if ten_crop:
            self.transforms = []
            for crop_range in crop_ranges:
                crop_size = int(image_size * crop_range) 
                transform = transforms.Compose([
                    transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
                    transforms.TenCrop(image_size), # this is a tuple of PIL Images
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), # returns a 4D tensor
                    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
                ])
                self.transforms.append(transform)
        else:
            crop_size = image_size 
            transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
            self.transforms = [transform]
        
            
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        random_transform = random.choice(self.transforms)
        image = random_transform(image)
        if len(image.shape) == 4:  # If TenCrop is applied, we have a batch of crops
            # randomly select one of the crops
            crop_idx = random.randint(0, image.shape[0] - 1)
            image = image[crop_idx]
            
        # map image back to unint8 values
        image = (image * 255).to(torch.long)    
        
        return image, label
    
    def visualize(self, idx):
        image, _ = self.__getitem__(idx) # (3, H, W)
        image = image.permute(1, 2, 0).numpy().astype("uint8") # (H, W, 3)
        
        # convert to PIL Image for visualization
        image = Image.fromarray(image)
        return image
        
        
    def __len__(self):
        return len(self.dataset)
